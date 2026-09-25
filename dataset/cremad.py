from __future__ import annotations

import csv
import os
from pathlib import Path

import librosa
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES = ("NEU", "HAP", "SAD", "FEA", "DIS", "ANG")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def frame_sort_key(name: str) -> tuple[int, int | str]:
    stem = Path(name).stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)

def select_frame_name(frame_names):
    index = min(1, len(frame_names) - 1)
    return frame_names[index]


class CREMADDataset(Dataset):
    
    class_names = CLASS_NAMES
    class_dict = CLASS_TO_INDEX

    def __init__(self, args, mode: str = "train") -> None:
        if mode not in {"train", "test"}:
            raise ValueError(f"Unsupported CREMA-D split: {mode}")
        self.args = args
        self.mode = mode
        self.data_root = Path(args.data_root).expanduser().resolve()
        self.audio_root = self.data_root / "AudioWAV"
        self.visual_root = self.data_root
        self.image: list[str] = []
        self.audio: list[str] = []
        self.label: list[int] = []
        self.sample_ids: list[str] = []
        self.sample_id = self.sample_ids
        self.missing_sample_ids: list[str] = []

        split_file = self.data_root / f"{mode}.csv"
        with split_file.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.reader(stream):
                if len(row) < 2:
                    continue
                sample_id, emotion = row[0].strip(), row[1].strip()
                if emotion not in CLASS_TO_INDEX:
                    raise ValueError(f"Unknown CREMA-D label {emotion!r} in {split_file}.")
                audio_path = self.audio_root / f"{sample_id}.wav"
                visual_path = (
                    self.visual_root / f"Image-{int(args.fps):02d}-FPS" / sample_id
                )
                if audio_path.is_file() and visual_path.is_dir():
                    self.audio.append(str(audio_path))
                    self.image.append(str(visual_path))
                    self.label.append(CLASS_TO_INDEX[emotion])
                    self.sample_ids.append(sample_id)
                else:
                    self.missing_sample_ids.append(sample_id)

        if not self.image:
            raise FileNotFoundError(
                "No CREMA-D pairs were found. Expected AudioWAV and "
                f"Image-{int(args.fps):02d}-FPS under {self.data_root}."
            )

    def __len__(self) -> int:
        return len(self.image)

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
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, index: int):
        samples, _ = librosa.load(self.audio[index], sr=22050)
        waveform = np.tile(samples, 20)[: 22050 * 20]
        waveform = np.clip(waveform, -1.0, 1.0)
        spectrogram = librosa.stft(waveform, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7).astype(np.float32)
        
        frame_names = sorted(os.listdir(self.image[index]), key=frame_sort_key)
        frame_name = select_frame_name(frame_names)
        transform = self._transform()
        images = torch.zeros((int(self.args.fps), 3, 224, 224))
        for position in range(int(self.args.fps)):
            path = Path(self.image[index]) / frame_name
            with Image.open(path) as image:
                images[position] = transform(image.convert("RGB"))
        images = images.permute(1, 0, 2, 3)
        return spectrogram, images, int(self.label[index])

