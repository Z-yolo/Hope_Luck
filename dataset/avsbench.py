"""AVSBench-S4 loader for the released RCC protocol."""

from __future__ import annotations

import csv
import os
import random
import re
from pathlib import Path

import librosa
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES = (
    "helicopter",
    "mynah_bird_singing",
    "typing_on_computer_keyboard",
    "playing_violin",
    "playing_glockenspiel",
    "playing_piano",
    "lions_roaring",
    "baby_laughter",
    "male_speech",
    "lawn_mowing",
    "playing_ukulele",
    "playing_tabla",
    "driving_buses",
    "cap_gun_shooting",
    "chainsawing_trees",
    "playing_acoustic_guitar",
    "cat_meowing",
    "female_singing",
    "ambulance_siren",
    "dog_barking",
    "horse_clip-clop",
    "coyote_howling",
    "race_car",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_key(name: str) -> list[int | str]:
    base = os.path.basename(name)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", base)]


def list_image_files(frame_dir: str | Path) -> list[str]:
    frame_dir = Path(frame_dir)
    if not frame_dir.is_dir():
        return []
    names = [
        entry.name
        for entry in frame_dir.iterdir()
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(names, key=natural_key)


def metadata_class_order(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    labels: list[str] = []
    for row in rows[1:]:
        if len(row) < 3:
            continue
        label = row[2].strip()
        if label and label not in labels:
            labels.append(label)
    return labels


class AVSBenchDataset(Dataset):
    """AVSBench-S4 audio/visual classification view.

    ``data_root`` is the directory containing ``s4_meta_data.csv`` and
    ``s4_data``. The fixed class tuple above prevents silent label permutation
    when a metadata file is reordered.
    """

    class_names = CLASS_NAMES
    class_dict = CLASS_TO_INDEX
    class_number = len(CLASS_NAMES)

    def __init__(self, args, mode: str = "train") -> None:
        if mode not in {"train", "test"}:
            raise ValueError(f"Unsupported AVSBench split: {mode}")
        self.args = args
        self.mode = mode
        self.data_root = Path(args.data_root).expanduser().resolve()
        self.image: list[str] = []
        self.audio: list[str] = []
        self.label: list[int] = []
        self.sample_ids: list[str] = []
        self.sample_id = self.sample_ids

        metadata_path = self.data_root / "s4_meta_data.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"AVSBench metadata not found: {metadata_path}")
        observed = metadata_class_order(metadata_path)
        if observed != list(CLASS_NAMES):
            raise ValueError(
                "AVSBench class order differs from the order used to train the "
                "released checkpoints. Use the official S4 metadata unchanged."
            )

        for class_name in CLASS_NAMES:
            audio_dir = (
                self.data_root / "s4_data" / "audio_wav" / mode / class_name
            )
            if not audio_dir.is_dir():
                continue
            audio_names = sorted(
                [entry.name for entry in audio_dir.iterdir() if entry.is_file() and entry.suffix.lower() == ".wav"],
                key=natural_key,
            )
            for audio_name in audio_names:
                instance = Path(audio_name).stem
                audio_path = audio_dir / audio_name
                visual_path = (
                    self.data_root
                    / "s4_data"
                    / "visual_frames"
                    / mode
                    / class_name
                    / instance
                )
                if visual_path.is_dir() and list_image_files(visual_path):
                    self.audio.append(str(audio_path))
                    self.image.append(str(visual_path))
                    self.label.append(CLASS_TO_INDEX[class_name])
                    self.sample_ids.append(f"{class_name}/{instance}")

        if not self.image:
            raise FileNotFoundError(
                f"No AVSBench pairs were found for split {mode!r} under {self.data_root}."
            )

    def __len__(self) -> int:
        return len(self.image)

    def _load_audio(self, path: str) -> np.ndarray:
        samples, _ = librosa.load(path, sr=22050)
        target_length = 22050 * 3
        if len(samples) == 0:
            samples = np.zeros(target_length, dtype=np.float32)
        waveform = np.tile(samples, 3)[:target_length]
        if len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        waveform = np.clip(waveform.astype(np.float32), -1.0, 1.0)
        spectrogram = librosa.stft(waveform, n_fft=512, hop_length=353)
        return np.log(np.abs(spectrogram) + 1e-7).astype(np.float32)

    def _transform(self):
        if self.mode == "train":
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

    def select_frame_indices(self, frame_count: int, pick_count: int) -> list[int]:
        if frame_count <= 0:
            raise RuntimeError("An AVSBench visual directory contains no images.")
        if frame_count >= pick_count:
            segment = frame_count // pick_count
            indices = []
            for position in range(pick_count):
                start = position * segment
                end = min((position + 1) * segment - 1, frame_count - 1)
                if self.mode == "train":
                    indices.append(random.randint(start, max(start, end)))
                else:
                    indices.append((start + max(start, end)) // 2)
            return indices
        return np.round(np.linspace(0, frame_count - 1, pick_count)).astype(int).tolist()

    def _load_visual(self, directory: str) -> torch.Tensor:
        frame_names = list_image_files(directory)
        pick_count = max(1, int(getattr(self.args, "use_video_frames", 3)))
        indices = self.select_frame_indices(len(frame_names), pick_count)
        transform = self._transform()
        images = torch.zeros((pick_count, 3, 224, 224), dtype=torch.float32)
        for position, frame_index in enumerate(indices):
            path = Path(directory) / frame_names[frame_index]
            with Image.open(path) as image:
                images[position] = transform(image.convert("RGB"))
        return images.permute(1, 0, 2, 3)

    def __getitem__(self, index: int):
        spectrogram = self._load_audio(self.audio[index])
        images = self._load_visual(self.image[index])

        # These two tensors preserve the RNG consumption and six-field batch
        # contract used when the released checkpoints were trained. RCC ignores
        # their values, but retaining them makes reruns protocol-compatible.
        audio_std = float(np.std(spectrogram))
        visual_std = float(images.std().item())
        if audio_std <= 0:
            audio_std = 1.0
        if visual_std <= 0:
            visual_std = 1.0
        audio_noise = np.random.normal(0, audio_std, spectrogram.shape).astype(np.float32)
        visual_noise = np.random.normal(0, visual_std, tuple(images.shape)).astype(np.float32)
        return spectrogram, images, audio_noise, visual_noise, int(self.label[index]), index
