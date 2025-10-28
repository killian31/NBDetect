#!/usr/bin/env python3
"""
Benchmark inference latency vs accuracy across multiple input resolutions.

This script loads the nail-biting detector checkpoint, evaluates it on the
test split for several image sizes, and produces a scatter plot where:
    - X axis: average latency per image (milliseconds)
    - Y axis: classification accuracy
    - Marker size: input resolution used during evaluation

The resulting plot is saved to the path provided via --output.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nbdetect.data import (
    NailBitingDataset,
    Record,
    create_transforms,
    load_split_records,
)
from nbdetect.model import build_model, load_checkpoint

IMAGE_SIZES = [128, 224, 384, 512, 720]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark model accuracy vs latency across image sizes."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the trained PyTorch checkpoint (e.g. runs/.../epoch_09.pt).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset"),
        help="Path to dataset root containing train/val/test splits.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used during evaluation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for the DataLoader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device to use (cuda or cpu).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_accuracy_latency.png"),
        help="Destination path for the saved plot.",
    )
    return parser.parse_args()


def prepare_dataloader(
    records: Sequence[Record],
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    _, eval_tf = create_transforms(image_size=image_size)
    dataset = NailBitingDataset(records, transform=eval_tf)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader


@torch.no_grad()
def evaluate_resolution(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    progress_desc: Optional[str] = None,
) -> Tuple[float, float]:
    total_correct = 0
    total_examples = 0
    total_inference_time = 0.0

    is_cuda = device.type == "cuda"

    iterator = (
        tqdm(loader, desc=progress_desc, leave=False) if progress_desc else loader
    )
    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        logits = model(images)
        if is_cuda:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        total_inference_time += elapsed

        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

    accuracy = total_correct / total_examples if total_examples else 0.0
    latency_ms = (
        (total_inference_time / total_examples) * 1000.0 if total_examples else 0.0
    )
    return accuracy, latency_ms


def generate_plot(
    results: Dict[int, Tuple[float, float]],
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-darkgrid")

    fig, ax = plt.subplots(figsize=(8, 6))
    latencies = [results[size][1] for size in IMAGE_SIZES]
    accuracies = [results[size][0] for size in IMAGE_SIZES]

    # Scale marker sizes to keep the scatter plot legible.
    marker_sizes = [max(size / 2, 40) for size in IMAGE_SIZES]

    scatter = ax.scatter(
        latencies,
        accuracies,
        s=marker_sizes,
        c=IMAGE_SIZES,
        cmap="viridis",
        edgecolor="black",
    )

    for size, latency, acc in zip(IMAGE_SIZES, latencies, accuracies):
        ax.annotate(
            f"{size}px",
            (latency, acc),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )

    ax.set_xlabel("Latency per image (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Latency Across Input Resolutions")
    ax.set_xlim(left=0)
    ax.set_ylim(0.0, 1.0)

    cbar = fig.colorbar(scatter, ax=ax, label="Input size (px)")
    cbar.set_ticks(IMAGE_SIZES)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    test_records = load_split_records(args.dataset, "test")

    model = build_model(pretrained=False)
    load_checkpoint(model, args.weights, map_location=device)
    model.to(device)
    model.eval()

    results: Dict[int, Tuple[float, float]] = {}
    pin_memory = device.type == "cuda"

    for image_size in tqdm(IMAGE_SIZES, desc="Benchmark", unit="size"):
        loader = prepare_dataloader(
            test_records,
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        accuracy, latency_ms = evaluate_resolution(
            model,
            loader,
            device,
            progress_desc=f"{image_size}px batches",
        )
        results[image_size] = (accuracy, latency_ms)
        print(
            f"{image_size:>3d}px | accuracy={accuracy:.4f} | latency={latency_ms:.2f} ms"
        )

    generate_plot(results, args.output)
    print(f"Saved accuracy-latency plot to {args.output.resolve()}")


if __name__ == "__main__":
    main()
