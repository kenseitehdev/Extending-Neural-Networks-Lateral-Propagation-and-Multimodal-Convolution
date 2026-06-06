# file: multimodal_har_cnn.py
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_TRAIN_SIZES = (500, 1000, 1500)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_command(command: Sequence[str], cwd: Optional[Path] = None) -> None:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )


def ensure_dataset(base_dir: Path, slug: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    detected = find_dataset_root(base_dir)
    if detected is not None:
        return detected

    zip_name = slug.split("/")[-1] + ".zip"
    zip_path = base_dir / zip_name

    if not zip_path.exists():
        kaggle = shutil.which("kaggle")
        if kaggle is None:
            raise FileNotFoundError(
                "Dataset not found locally and Kaggle CLI is not installed.\n"
                "Install kaggle, configure credentials, then rerun, or download manually."
            )
        run_command([kaggle, "datasets", "download", slug, "-p", str(base_dir)])

    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(base_dir)

    detected = find_dataset_root(base_dir)
    if detected is None:
        raise FileNotFoundError(
            f"Could not find extracted video dataset under {base_dir.resolve()}"
        )
    return detected


def find_dataset_root(base_dir: Path) -> Optional[Path]:
    if not base_dir.exists():
        return None

    best_candidate: Optional[Path] = None
    best_score = -1

    for directory in [base_dir] + [p for p in base_dir.rglob("*") if p.is_dir()]:
        class_dirs = []
        video_count = 0
        for child in directory.iterdir():
            if not child.is_dir():
                continue
            child_videos = [p for p in child.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]
            if child_videos:
                class_dirs.append(child)
                video_count += len(child_videos)
        score = len(class_dirs) * 10_000 + video_count
        if len(class_dirs) >= 2 and score > best_score:
            best_candidate = directory
            best_score = score

    return best_candidate


def discover_videos(dataset_root: Path) -> Tuple[List[Tuple[Path, str]], List[str]]:
    samples: List[Tuple[Path, str]] = []
    class_names: List[str] = []

    for class_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        video_paths = sorted(
            p for p in class_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not video_paths:
            continue
        class_names.append(class_dir.name)
        for video_path in video_paths:
            samples.append((video_path, class_dir.name))

    if len(class_names) < 2:
        raise RuntimeError(
            f"Expected at least 2 class folders under {dataset_root}, found {len(class_names)}"
        )
    if not samples:
        raise RuntimeError(f"No videos found under {dataset_root}")

    return samples, class_names


def safe_label_split(
    samples: Sequence[Tuple[Path, str]],
    test_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    labels = [label for _, label in samples]
    label_counts = Counter(labels)
    stratify = labels if min(label_counts.values()) >= 2 else None
    train_samples, test_samples = train_test_split(
        list(samples),
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return train_samples, test_samples


def balanced_subset(
    samples: Sequence[Tuple[Path, str]],
    target_size: int,
    seed: int,
) -> List[Tuple[Path, str]]:
    if target_size >= len(samples):
        return list(samples)

    rng = random.Random(seed)
    by_label: Dict[str, List[Tuple[Path, str]]] = defaultdict(list)
    for sample in samples:
        by_label[sample[1]].append(sample)

    labels = sorted(by_label)
    total = len(samples)
    target_by_label: Dict[str, int] = {label: 0 for label in labels}

    if target_size >= len(labels):
        for label in labels:
            target_by_label[label] = 1

    current_total = sum(target_by_label.values())
    remaining = target_size - current_total

    if remaining > 0:
        remainders: List[Tuple[float, str]] = []
        for label in labels:
            capacity = len(by_label[label]) - target_by_label[label]
            if capacity <= 0:
                continue
            ideal = (len(by_label[label]) / total) * remaining
            take = min(int(ideal), capacity)
            target_by_label[label] += take
            remainders.append((ideal - int(ideal), label))

        current_total = sum(target_by_label.values())
        extra = target_size - current_total
        if extra > 0:
            for _, label in sorted(remainders, reverse=True):
                if extra <= 0:
                    break
                if target_by_label[label] < len(by_label[label]):
                    target_by_label[label] += 1
                    extra -= 1

            if extra > 0:
                for label in labels:
                    while extra > 0 and target_by_label[label] < len(by_label[label]):
                        target_by_label[label] += 1
                        extra -= 1

    subset: List[Tuple[Path, str]] = []
    for label in labels:
        pool = by_label[label][:]
        rng.shuffle(pool)
        subset.extend(pool[: target_by_label[label]])

    rng.shuffle(subset)
    return subset[:target_size]


def resize_frame(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return frame.astype(np.float32) / 255.0


def sample_video_frames(
    video_path: Path,
    frames_per_clip: int,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        indices = [0] * frames_per_clip
    else:
        indices = np.linspace(0, max(frame_count - 1, 0), frames_per_clip).astype(int).tolist()

    collected: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            if collected:
                frame = (collected[-1] * 255.0).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        rgb = resize_frame(frame, image_size)
        collected.append(rgb)

    cap.release()
    frames = np.stack(collected, axis=0)
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    return tensor


def ffmpeg_audio_available() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_audio_waveform(
    video_path: Path,
    sample_rate: int,
    max_seconds: float,
) -> torch.Tensor:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return torch.zeros(int(sample_rate * max_seconds), dtype=torch.float32)

    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-t",
        f"{max_seconds:.4f}",
        "-f",
        "f32le",
        "pipe:1",
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout:
        return torch.zeros(int(sample_rate * max_seconds), dtype=torch.float32)

    waveform = np.frombuffer(completed.stdout, dtype=np.float32)
    if waveform.size == 0:
        return torch.zeros(int(sample_rate * max_seconds), dtype=torch.float32)

    tensor = torch.from_numpy(waveform.copy())
    target_length = int(sample_rate * max_seconds)
    if tensor.numel() < target_length:
        tensor = F.pad(tensor, (0, target_length - tensor.numel()))
    else:
        tensor = tensor[:target_length]
    return tensor


def waveform_to_log_spectrogram(
    waveform: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    out_size: Tuple[int, int] = (128, 128),
) -> torch.Tensor:
    if waveform.numel() == 0:
        waveform = torch.zeros(16_000, dtype=torch.float32)

    if waveform.abs().max().item() > 0:
        waveform = waveform / waveform.abs().max().clamp(min=1e-6)

    window = torch.hann_window(win_length, device=waveform.device)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    magnitude = stft.abs().pow(2.0)
    log_spec = torch.log1p(magnitude)
    log_spec = log_spec.unsqueeze(0).unsqueeze(0)
    log_spec = F.interpolate(log_spec, size=out_size, mode="bilinear", align_corners=False)
    return log_spec.squeeze(0).contiguous()


class VideoAudioDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[Path, str]],
        label_to_index: Dict[str, int],
        frames_per_clip: int,
        image_size: Tuple[int, int],
        audio_sample_rate: int,
        audio_seconds: float,
    ) -> None:
        self.samples = list(samples)
        self.label_to_index = label_to_index
        self.frames_per_clip = frames_per_clip
        self.image_size = image_size
        self.audio_sample_rate = audio_sample_rate
        self.audio_seconds = audio_seconds

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        video_path, label_name = self.samples[index]
        frames = sample_video_frames(video_path, self.frames_per_clip, self.image_size)
        waveform = extract_audio_waveform(video_path, self.audio_sample_rate, self.audio_seconds)
        audio_spec = waveform_to_log_spectrogram(waveform)
        label = self.label_to_index[label_name]
        return {
            "frames": frames,
            "audio": audio_spec,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(video_path),
        }


class FrameBranch(nn.Module):
    def __init__(self, in_channels: int = 3, feature_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = frames.shape
        x = frames.view(batch_size * time_steps, channels, height, width)
        x = self.features(x)
        x = self.projection(x)
        x = x.view(batch_size, time_steps, -1).mean(dim=1)
        return x


class AudioBranch(nn.Module):
    def __init__(self, in_channels: int = 1, feature_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

    def forward(self, audio_spec: torch.Tensor) -> torch.Tensor:
        x = self.features(audio_spec)
        x = self.projection(x)
        return x


class MultimodalCNN(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int = 128) -> None:
        super().__init__()
        self.frame_branch = FrameBranch(feature_dim=feature_dim)
        self.audio_branch = AudioBranch(feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, frames: torch.Tensor, audio_spec: torch.Tensor) -> torch.Tensor:
        frame_features = self.frame_branch(frames)
        audio_features = self.audio_branch(audio_spec)
        fused = torch.cat([frame_features, audio_features], dim=1)
        logits = self.classifier(fused)
        return logits


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


@dataclass
class CycleMetrics:
    cycle: int
    train_size: int
    epochs: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    avg_confidence: float
    elapsed_seconds: float
    num_test_samples: int


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    frames = batch["frames"].to(device, non_blocking=True)
    audio = batch["audio"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    paths = list(batch["path"])
    return frames, audio, labels, paths


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        frames, audio, labels, _ = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(frames, audio)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

    return EpochMetrics(
        loss=total_loss / max(total_examples, 1),
        accuracy=total_correct / max(total_examples, 1),
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    index_to_label: Dict[int, str],
) -> Tuple[EpochMetrics, float, List[Dict[str, object]], str, List[List[int]]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    confidences: List[float] = []
    true_labels: List[int] = []
    pred_labels: List[int] = []
    prediction_rows: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            frames, audio, labels, paths = move_batch_to_device(batch, device)
            logits = model(frames, audio)
            loss = criterion(logits, labels)
            probabilities = F.softmax(logits, dim=1)
            confidence, predictions = probabilities.max(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)

            confidences.extend(confidence.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
            pred_labels.extend(predictions.cpu().tolist())

            for i in range(labels.size(0)):
                prediction_rows.append(
                    {
                        "path": paths[i],
                        "true_label": index_to_label[int(labels[i].item())],
                        "predicted_label": index_to_label[int(predictions[i].item())],
                        "confidence": float(confidence[i].item()),
                        "correct": int(predictions[i].item() == labels[i].item()),
                    }
                )

    metrics = EpochMetrics(
        loss=total_loss / max(total_examples, 1),
        accuracy=total_correct / max(total_examples, 1),
    )

    report = classification_report(
        true_labels,
        pred_labels,
        labels=sorted(index_to_label.keys()),
        target_names=[index_to_label[i] for i in sorted(index_to_label.keys())],
        zero_division=0,
        digits=4,
    )
    cm = confusion_matrix(
        true_labels,
        pred_labels,
        labels=sorted(index_to_label.keys()),
    ).tolist()

    avg_confidence = float(np.mean(confidences)) if confidences else 0.0
    return metrics, avg_confidence, prediction_rows, report, cm


def save_predictions(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_cycle_metrics(path: Path, rows: Sequence[CycleMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(asdict(rows[0]).keys()) if rows else list(CycleMetrics.__annotations__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def format_seconds(seconds: float) -> str:
    minutes, sec = divmod(seconds, 60.0)
    hours, minutes = divmod(minutes, 60.0)
    if hours >= 1:
        return f"{int(hours):02d}:{int(minutes):02d}:{sec:05.2f}"
    return f"{int(minutes):02d}:{sec:05.2f}"


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def cycle_run(
    cycle_index: int,
    train_size: int,
    train_samples: Sequence[Tuple[Path, str]],
    test_samples: Sequence[Tuple[Path, str]],
    label_to_index: Dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> CycleMetrics:
    sampled_train = balanced_subset(train_samples, target_size=train_size, seed=args.seed + cycle_index)
    train_dataset = VideoAudioDataset(
        samples=sampled_train,
        label_to_index=label_to_index,
        frames_per_clip=args.frames_per_clip,
        image_size=(args.image_width, args.image_height),
        audio_sample_rate=args.audio_sample_rate,
        audio_seconds=args.audio_seconds,
    )
    test_dataset = VideoAudioDataset(
        samples=test_samples,
        label_to_index=label_to_index,
        frames_per_clip=args.frames_per_clip,
        image_size=(args.image_width, args.image_height),
        audio_sample_rate=args.audio_sample_rate,
        audio_seconds=args.audio_seconds,
    )

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers)

    model = MultimodalCNN(num_classes=len(label_to_index), feature_dim=args.feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    start_time = time.perf_counter()
    last_train_metrics = EpochMetrics(loss=0.0, accuracy=0.0)

    for epoch in range(1, args.epochs + 1):
        last_train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(
            f"[cycle {cycle_index}][epoch {epoch}/{args.epochs}] "
            f"train_loss={last_train_metrics.loss:.4f} "
            f"train_acc={last_train_metrics.accuracy:.4f}"
        )

    index_to_label = {v: k for k, v in label_to_index.items()}
    test_metrics, avg_confidence, prediction_rows, report, confusion = evaluate(
        model,
        test_loader,
        criterion,
        device,
        index_to_label,
    )
    elapsed_seconds = time.perf_counter() - start_time

    cycle_dir = output_dir / f"cycle_{cycle_index:02d}_train_{train_size}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_index": label_to_index,
            "args": vars(args),
        },
        cycle_dir / "model.pt",
    )
    save_predictions(cycle_dir / "test_predictions.csv", prediction_rows)

    summary_payload = {
        "cycle": cycle_index,
        "train_size": train_size,
        "epochs": args.epochs,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_hms": format_seconds(elapsed_seconds),
        "train_loss": last_train_metrics.loss,
        "train_accuracy": last_train_metrics.accuracy,
        "test_loss": test_metrics.loss,
        "test_accuracy": test_metrics.accuracy,
        "avg_confidence": avg_confidence,
        "num_train_samples": len(sampled_train),
        "num_test_samples": len(test_samples),
        "confusion_matrix": confusion,
        "labels": [index_to_label[i] for i in sorted(index_to_label)],
    }
    with (cycle_dir / "test_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    with (cycle_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(report)

    print("=" * 90)
    print(
        f"CYCLE {cycle_index} | train_size={train_size} | "
        f"time={format_seconds(elapsed_seconds)} | "
        f"train_loss={last_train_metrics.loss:.4f} | "
        f"train_acc={last_train_metrics.accuracy:.4f} | "
        f"test_loss={test_metrics.loss:.4f} | "
        f"test_acc={test_metrics.accuracy:.4f} | "
        f"avg_confidence={avg_confidence:.4f}"
    )
    print("Test summary:")
    print(report)
    print("=" * 90)

    return CycleMetrics(
        cycle=cycle_index,
        train_size=train_size,
        epochs=args.epochs,
        train_loss=last_train_metrics.loss,
        train_accuracy=last_train_metrics.accuracy,
        test_loss=test_metrics.loss,
        test_accuracy=test_metrics.accuracy,
        avg_confidence=avg_confidence,
        elapsed_seconds=elapsed_seconds,
        num_test_samples=len(test_samples),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hand-rolled multimodal CNN for video activity recognition using frames + audio."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument(
        "--kaggle-slug",
        type=str,
        default="sharjeelmazhar/human-activity-recognition-video-dataset",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "multimodal_har_cnn")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=list(DEFAULT_TRAIN_SIZES))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--audio-seconds", type=float, default=4.0)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-download", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    if args.skip_download:
        dataset_root = find_dataset_root(args.data_dir)
        if dataset_root is None:
            raise FileNotFoundError(
                f"Dataset not found under {args.data_dir.resolve()} and --skip-download was used."
            )
    else:
        dataset_root = ensure_dataset(args.data_dir, args.kaggle_slug)

    samples, class_names = discover_videos(dataset_root)
    label_to_index = {label: idx for idx, label in enumerate(sorted(class_names))}
    train_samples, test_samples = safe_label_split(samples, test_ratio=args.test_ratio, seed=args.seed)

    train_sizes = sorted({size for size in args.train_sizes if size > 0})
    if not train_sizes:
        raise ValueError("Provide at least one positive train size.")

    max_available = len(train_samples)
    adjusted_train_sizes = list(dict.fromkeys(min(size, max_available) for size in train_sizes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary = {
        "dataset_root": str(dataset_root.resolve()),
        "total_samples": len(samples),
        "num_classes": len(class_names),
        "classes": sorted(class_names),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "requested_train_sizes": train_sizes,
        "used_train_sizes": adjusted_train_sizes,
        "device": str(device),
        "ffmpeg_available": ffmpeg_audio_available(),
    }
    with (args.output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2)

    print(json.dumps(dataset_summary, indent=2))

    cycle_metrics: List[CycleMetrics] = []
    for cycle_index, train_size in enumerate(adjusted_train_sizes, start=1):
        metrics = cycle_run(
            cycle_index=cycle_index,
            train_size=train_size,
            train_samples=train_samples,
            test_samples=test_samples,
            label_to_index=label_to_index,
            args=args,
            device=device,
            output_dir=args.output_dir,
        )
        cycle_metrics.append(metrics)

    save_cycle_metrics(args.output_dir / "cycle_metrics.csv", cycle_metrics)

    best_cycle = max(cycle_metrics, key=lambda row: row.test_accuracy)
    final_summary = {
        "best_cycle": asdict(best_cycle),
        "all_cycles": [asdict(row) for row in cycle_metrics],
    }
    with (args.output_dir / "final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print("Final summary:")
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()