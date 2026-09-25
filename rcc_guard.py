#!/usr/bin/env python3
"""Train or evaluate RCC on CREMA-D and AVSBench.

The public ``train`` command runs calibration and final training in separate
Python processes. The final model is therefore initialized from scratch and
uses only the two probabilities estimated by the calibration process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset, Subset

from models.av_model import RCCGuardModel
from rcc import av
from rcc.metrics import build_evaluation_summary


ROOT = Path(__file__).resolve().parent
CONFIG_PATHS = {
    "cremad": ROOT / "configs" / "cremad.json",
    "avsbench": ROOT / "configs" / "avsbench.json",
}
DATASET_ARGUMENTS = {"cremad": "CREMAD", "avsbench": "avsbench"}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_empty_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def load_config(dataset: str, override: Path | None = None) -> dict[str, Any]:
    path = (override or CONFIG_PATHS[dataset]).expanduser().resolve()
    config = read_json(path)
    if config.get("dataset") != dataset:
        raise ValueError(
            f"Config dataset is {config.get('dataset')!r}, expected {dataset!r}."
        )
    if int(config.get("num_classes", -1)) not in {6, 23}:
        raise ValueError("Only the released 6-class and 23-class protocols are valid.")
    if not isinstance(config.get("parameters"), dict):
        raise ValueError(f"Config has no parameters object: {path}")
    config["_config_path"] = str(path)
    return config


def apply_overrides(parameters: dict[str, Any], assignments: list[str]) -> None:
    protected = {
        "dataset",
        "data_root",
        "output",
        "random_seed",
        "num_classes",
        "rcc_mode",
        "rcc_prob_a2v",
        "rcc_prob_v2a",
        "rcc_calib_json",
        "use_rcc",
        "use_adoption_guard",
    }
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"Expected KEY=VALUE, got {assignment!r}.")
        key, raw_value = assignment.split("=", 1)
        if key in protected:
            raise ValueError(f"{key} is controlled by the two-stage runner.")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        parameters[key] = value


def make_stage_spec(
    config: dict[str, Any],
    data_root: Path,
    output: Path,
    seed: int,
    stage: str,
    device: str,
    probabilities: dict[str, float] | None = None,
) -> dict[str, Any]:
    if stage not in {"calibration", "final"}:
        raise ValueError(stage)
    parameters = dict(config["parameters"])
    calibration = stage == "calibration"
    parameters.update(
        dataset=DATASET_ARGUMENTS[config["dataset"]],
        dataset_name=config["dataset"],
        data_root=str(data_root),
        output=str(output),
        random_seed=int(seed),
        num_classes=int(config["num_classes"]),
        expected_train_samples=int(config["expected_train_samples"]),
        expected_test_samples=int(config["expected_test_samples"]),
        epochs=int(
            config["calibration_epochs"] if calibration else config["epochs"]
        ),
        use_rcc=True,
        rcc_mode="calibrated" if calibration else "manual",
        use_adoption_guard=(
            bool(config["calibration_guard"]) if calibration else True
        ),
        rcc_calib_json=str(output / "rcc_calibration.json") if calibration else "",
        rcc_prob_a2v=(
            -1.0 if calibration else float(probabilities["a2v"])  # type: ignore[index]
        ),
        rcc_prob_v2a=(
            -1.0 if calibration else float(probabilities["v2a"])  # type: ignore[index]
        ),
        device=device,
        modality="full",
        pretrain=False,
    )
    return {
        "schema_version": 1,
        "stage": stage,
        "source_config": config["_config_path"],
        "parameters": parameters,
    }


def read_calibrated_probabilities(path: Path) -> dict[str, float]:
    values = read_json(path)
    probabilities = {
        "a2v": float(values["a2v"]),
        "v2a": float(values["v2a"]),
    }
    for name, value in probabilities.items():
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid calibrated probability {name}={value}.")
    return probabilities


def run_child(spec: dict[str, Any], gpu_ids: str) -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)
    with tempfile.TemporaryDirectory(prefix="rcc_stage_") as directory:
        spec_path = Path(directory) / "stage_spec.json"
        write_json(spec_path, spec)
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "_stage",
            "--spec",
            str(spec_path),
        ]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
        )
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Training stage failed with exit code {return_code}.")


def train_command(cli: argparse.Namespace) -> None:
    config = load_config(cli.dataset, cli.config)
    if cli.calibration_epochs is not None:
        config["calibration_epochs"] = int(cli.calibration_epochs)
    if cli.epochs is not None:
        config["epochs"] = int(cli.epochs)
    if cli.batch_size is not None:
        config["parameters"]["batch_size"] = int(cli.batch_size)
    if cli.num_workers is not None:
        config["parameters"]["num_workers"] = int(cli.num_workers)
    apply_overrides(config["parameters"], cli.set)
    if int(config["calibration_epochs"]) <= int(
        config["parameters"]["rcc_start_epoch"]
    ):
        raise ValueError("Calibration must continue past rcc_start_epoch.")
    if len(set(cli.seeds)) != len(cli.seeds):
        raise ValueError("Duplicate seeds are not allowed.")

    data_root = cli.data_root.expanduser().resolve()
    output = cli.output.expanduser().resolve()
    device = cli.device if cli.dry_run else resolve_device(cli.device)
    plan = []
    for seed in cli.seeds:
        seed_root = output / f"seed{seed}"
        calibration_output = seed_root / "_calib_stage"
        final_output = seed_root / "final_full_train"
        for target in (calibration_output, final_output):
            if target.exists() and (not target.is_dir() or any(target.iterdir())):
                raise FileExistsError(f"Refusing to overwrite existing run: {target}")
        calibration = make_stage_spec(
            config,
            data_root,
            calibration_output,
            seed,
            "calibration",
            device,
        )
        plan.append(
            {
                "seed": seed,
                "calibration": calibration,
                "final": {
                    "stage": "final",
                    "output": str(final_output),
                    "fresh_initialization": True,
                    "probabilities": "read from the last calibration report",
                    "epochs": int(config["epochs"]),
                },
            }
        )
    if cli.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    prepare_empty_directory(output)
    for item in plan:
        seed = int(item["seed"])
        seed_root = output / f"seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        calibration_spec = item["calibration"]
        print(f"[{cli.dataset} seed{seed}] calibration", flush=True)
        run_child(calibration_spec, cli.gpu_ids)
        probabilities = read_calibrated_probabilities(
            Path(calibration_spec["parameters"]["rcc_calib_json"])
        )
        final_spec = make_stage_spec(
            config,
            data_root,
            seed_root / "final_full_train",
            seed,
            "final",
            device,
            probabilities,
        )
        print(
            f"[{cli.dataset} seed{seed}] fresh final: "
            f"p_a2v={probabilities['a2v']:.8f}, p_v2a={probabilities['v2a']:.8f}",
            flush=True,
        )
        run_child(final_spec, cli.gpu_ids)


def build_dataset(args: SimpleNamespace, mode: str):
    if args.dataset_name == "cremad":
        from dataset.cremad import CREMADDataset

        return CREMADDataset(args, mode=mode)
    if args.dataset_name == "avsbench":
        from dataset.avsbench import AVSBenchDataset

        return AVSBenchDataset(args, mode=mode)
    raise ValueError(f"Unsupported dataset: {args.dataset_name}")


def split_calibration(dataset: Dataset, args: SimpleNamespace):
    generator = torch.Generator().manual_seed(int(args.random_seed) + 2027)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    folds = max(1, int(args.rcc_calib_num_folds))
    if folds > 1:
        fold_index = int(args.rcc_calib_fold_index)
        if not 0 <= fold_index < folds:
            raise ValueError("Calibration fold index is out of range.")
        held_indices = np.array_split(np.asarray(indices), folds)[fold_index].tolist()
        held_set = set(held_indices)
        train_indices = [index for index in indices if index not in held_set]
    else:
        count = max(1, int(round(len(dataset) * float(args.rcc_calib_ratio))))
        held_indices, train_indices = indices[:count], indices[count:]
    if not train_indices or not held_indices:
        raise ValueError("Calibration requires nonempty train and held-out subsets.")
    args._rcc_calibration_split_metadata = {
        "strategy": "seeded_random_training_holdout",
        "seed_offset": 2027,
        "train_count": len(train_indices),
        "heldout_count": len(held_indices),
    }
    return Subset(dataset, train_indices), Subset(dataset, held_indices)


def make_loader(
    dataset: Dataset, args: SimpleNamespace, training: bool = False
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=training,
        num_workers=int(args.num_workers),
        pin_memory=args.device == "cuda",
        drop_last=training,
    )


def make_optimizer(args: SimpleNamespace, model: torch.nn.Module):
    if str(args.optimizer).lower() != "sgd":
        raise ValueError("The released protocol supports only SGD.")
    optimizer = optim.SGD(
        model.parameters(),
        lr=float(args.learning_rate),
        momentum=0.9,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(value) for value in args.lr_decay_steps],
        gamma=float(args.lr_decay_ratio),
    )
    return optimizer, scheduler


def stage_command(cli: argparse.Namespace) -> None:
    spec_path = cli.spec.expanduser().resolve()
    spec = read_json(spec_path)
    stage = str(spec["stage"])
    args = SimpleNamespace(**spec["parameters"])
    output = Path(args.output).expanduser().resolve()
    prepare_empty_directory(output)
    av.setup_seed(int(args.random_seed))

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    model = RCCGuardModel(args)
    model.apply(av.weight_init)
    model.to(device)
    if device.type == "cuda":
        model = torch.nn.DataParallel(
            model, device_ids=list(range(torch.cuda.device_count()))
        )
        model.cuda()
    optimizer, scheduler = make_optimizer(args, model)

    train_dataset = build_dataset(args, "train")
    test_dataset = build_dataset(args, "test")
    if len(train_dataset) != int(args.expected_train_samples):
        raise ValueError(
            f"Training split mismatch: expected {args.expected_train_samples}, "
            f"found {len(train_dataset)}."
        )
    if len(test_dataset) != int(args.expected_test_samples):
        raise ValueError(
            f"Test split mismatch: expected {args.expected_test_samples}, "
            f"found {len(test_dataset)}."
        )
    held_dataset = None
    if args.rcc_mode == "calibrated":
        train_dataset, held_dataset = split_calibration(train_dataset, args)
    train_loader = make_loader(train_dataset, args, training=True)
    calibration_train_loader = make_loader(train_dataset, args)
    held_loader = make_loader(held_dataset, args) if held_dataset is not None else None
    test_loader = make_loader(test_dataset, args)
    if not train_loader or held_loader is None and args.rcc_mode == "calibrated":
        raise ValueError("The training or calibration loader is empty.")

    best_accuracy = 0.0
    for epoch in range(int(args.epochs)):
        args.epoch_now = epoch
        if args.rcc_mode == "calibrated":
            av.maybe_update_rcc_calibration(
                args,
                model,
                calibration_train_loader,
                held_loader,
                device,
                epoch,
            )
        loss = av.train_epoch(
            args, epoch, model, device, train_loader, optimizer, scheduler
        )
        summary = av.valid(args, model, device, test_loader)
        accuracy = float(summary["acc_fusion"])

        if (
            stage == "final"
            and accuracy > best_accuracy
            and epoch > int(args.selection_min_epoch)
        ):
            best_accuracy = float(accuracy)
            source_model = model.module if hasattr(model, "module") else model
            torch.save(source_model.state_dict(), output / "best.pth")
            write_json(output / "summary.json", summary)
        message = (
            f"Epoch {epoch:03d} | loss={loss:.4f} "
            f"| fusion={summary['acc_fusion']:.4f} "
            f"| audio={summary['acc_audio']:.4f} "
            f"| visual={summary['acc_visual']:.4f}"
        )
        if stage == "final":
            message += f" | best={best_accuracy:.4f}"
        print(message, flush=True)

    if stage == "calibration":
        calibration_path = Path(args.rcc_calib_json)
        if not calibration_path.is_file():
            raise RuntimeError("Calibration finished without writing probabilities.")
    elif not (output / "best.pth").is_file() or not (
        output / "summary.json"
    ).is_file():
        raise RuntimeError("Final training finished without a selected checkpoint.")


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return requested


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_state(checkpoint: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    metadata: dict[str, Any] = checkpoint if isinstance(checkpoint, dict) else {}
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        state = checkpoint
        metadata = {}
    else:
        raise TypeError("Checkpoint must contain model/state_dict or be a state dict.")
    prefixes = {str(key).startswith("module.") for key in state}
    if len(prefixes) != 1:
        raise ValueError("Checkpoint mixes prefixed and unprefixed state keys.")
    if prefixes == {True}:
        state = {str(key)[7:]: value for key, value in state.items()}
    return dict(state), metadata


class IndexedPrefixDataset(Dataset):
    def __init__(self, dataset: Dataset, count: int) -> None:
        self.dataset = dataset
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        return self.dataset[index], index


@torch.inference_mode()
def infer(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, np.ndarray]:
    indices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    predictions: dict[str, list[np.ndarray]] = {
        "fusion": [],
        "audio": [],
        "visual": [],
    }
    model.eval()
    for batch, sample_indices in loader:
        spectrogram, images, target = av.unpack_av_batch(batch)
        spectrogram = spectrogram.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long().view(-1)
        fusion, audio, visual = model(
            spectrogram.unsqueeze(1).float(), images.float()
        )
        indices.append(sample_indices.numpy().astype(np.int64, copy=False))
        labels.append(target.cpu().numpy().astype(np.int64, copy=False))
        for name, logits in (
            ("fusion", fusion),
            ("audio", audio),
            ("visual", visual),
        ):
            if logits.ndim != 2 or logits.shape[1] != num_classes:
                raise ValueError(
                    f"Checkpoint {name} head does not match the dataset class semantics."
                )
            if not torch.isfinite(logits).all():
                raise ValueError(f"Non-finite {name} logits detected.")
            predictions[name].append(logits.argmax(dim=1).cpu().numpy())
    if not labels:
        raise ValueError("Evaluation loader is empty.")
    result = {
        "sample_indices": np.concatenate(indices),
        "labels": np.concatenate(labels),
    }
    for name, chunks in predictions.items():
        result[f"pred_{name}"] = np.concatenate(chunks).astype(np.int64, copy=False)
    if not np.array_equal(result["sample_indices"], np.arange(len(result["labels"]))):
        raise AssertionError("Evaluation order changed unexpectedly.")
    return result


def evaluate_metrics(result: dict[str, np.ndarray], num_classes: int) -> dict[str, Any]:
    labels = result["labels"]
    if labels.size == 0 or labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError("Evaluation labels are empty or outside the class range.")
    return build_evaluation_summary(
        labels,
        result["pred_audio"],
        result["pred_visual"],
        result["pred_fusion"],
    )


def eval_command(cli: argparse.Namespace) -> None:
    config = load_config(cli.dataset, cli.config)
    parameters = dict(config["parameters"])
    if cli.batch_size is not None:
        parameters["batch_size"] = int(cli.batch_size)
    if cli.num_workers is not None:
        parameters["num_workers"] = int(cli.num_workers)
    parameters.update(
        dataset=DATASET_ARGUMENTS[cli.dataset],
        dataset_name=cli.dataset,
        data_root=str(cli.data_root.expanduser().resolve()),
        num_classes=int(config["num_classes"]),
        random_seed=int(cli.seed),
        modality="full",
        pretrain=False,
    )
    args = SimpleNamespace(**parameters)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli.gpu_ids)
    device_name = resolve_device(cli.device)
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    torch.set_num_threads(int(cli.threads))
    av.setup_seed(int(cli.seed))

    checkpoint_path = cli.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_hash = sha256(checkpoint_path) if cli.checkpoint_sha256 else ""
    if cli.checkpoint_sha256 and checkpoint_hash != cli.checkpoint_sha256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch: expected {cli.checkpoint_sha256}, "
            f"got {checkpoint_hash}."
        )
    state, metadata = checkpoint_state(torch_load(checkpoint_path))
    fusion_name = metadata.get("fusion")
    if fusion_name is not None and str(fusion_name) != args.fusion_method:
        raise ValueError(
            f"Checkpoint fusion={fusion_name!r}, config fusion={args.fusion_method!r}."
        )
    model = RCCGuardModel(args)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    dataset = build_dataset(args, "test")
    available = len(dataset)
    expected = int(config["expected_test_samples"])
    if available != expected:
        raise ValueError(
            f"Test split mismatch: expected {expected}, found {available}."
        )
    all_sample_ids = [str(value) for value in dataset.sample_ids]  # type: ignore[attr-defined]
    if len(all_sample_ids) != available or len(set(all_sample_ids)) != available:
        raise ValueError("Test sample IDs are missing, duplicated, or misaligned.")
    count = available
    output = cli.output.expanduser().resolve()
    prepare_empty_directory(output)
    indexed = IndexedPrefixDataset(dataset, count)
    loader = DataLoader(
        indexed,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    class_names = tuple(dataset.class_names)  # type: ignore[attr-defined]
    result = infer(model, loader, device, len(class_names))
    expected_labels = np.asarray(dataset.label[:count], dtype=np.int64)  # type: ignore[attr-defined]
    if not np.array_equal(result["labels"], expected_labels):
        raise AssertionError("Evaluation labels changed order.")
    metrics = evaluate_metrics(result, len(class_names))
    write_json(output / "summary.json", metrics)
    print(
        f"N={count} | fusion={metrics['acc_fusion']:.6f} "
        f"| audio={metrics['acc_audio']:.6f} "
        f"| visual={metrics['acc_visual']:.6f}",
        flush=True,
    )
    print(f"Saved evaluation to {output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "train", help="Run calibration then fresh final training."
    )
    train.add_argument("--dataset", required=True, choices=tuple(CONFIG_PATHS))
    train.add_argument("--data-root", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--seeds", nargs="+", type=int, default=[0])
    train.add_argument("--gpu-ids", default="0")
    train.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    train.add_argument("--config", type=Path)
    train.add_argument("--calibration-epochs", type=int)
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--num-workers", type=int)
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(handler=train_command)

    evaluate = subparsers.add_parser(
        "eval", help="Strictly evaluate an existing checkpoint."
    )
    evaluate.add_argument("--dataset", required=True, choices=tuple(CONFIG_PATHS))
    evaluate.add_argument("--data-root", required=True, type=Path)
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--checkpoint-sha256", default="")
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    evaluate.add_argument("--gpu-ids", default="0")
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--batch-size", type=int)
    evaluate.add_argument("--num-workers", type=int)
    evaluate.add_argument("--threads", type=int, default=4)
    evaluate.add_argument("--config", type=Path)
    evaluate.set_defaults(handler=eval_command)

    return parser


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "_stage":
        internal = argparse.ArgumentParser(add_help=False)
        internal.add_argument("--spec", required=True, type=Path)
        stage_command(internal.parse_args(sys.argv[2:]))
        return
    parser = build_parser()
    args = parser.parse_args()
    if (
        hasattr(args, "batch_size")
        and args.batch_size is not None
        and args.batch_size < 1
    ):
        parser.error("--batch-size must be positive.")
    if (
        hasattr(args, "num_workers")
        and args.num_workers is not None
        and args.num_workers < 0
    ):
        parser.error("--num-workers must be nonnegative.")
    args.handler(args)


if __name__ == "__main__":
    main()
