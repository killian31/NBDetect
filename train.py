import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from nbdetect.data import NailBitingDataset, create_transforms, load_records, split_records
from nbdetect.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train nail-biting detector with MobileNetV3.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Path to dataset root.")
    parser.add_argument("--output", type=Path, default=Path("runs/latest"), help="Directory to store checkpoints.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--freeze-base", action="store_true", help="Freeze backbone and train classifier only.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="nbdetect")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export the best checkpoint to ONNX after training completes.",
    )
    parser.add_argument("--onnx-output", type=Path, default=None, help="Destination for ONNX model.")
    return parser.parse_args()


def create_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    records = load_records(args.dataset)
    train_records, val_records = split_records(records, val_ratio=args.val_ratio, seed=args.seed)
    train_tf, eval_tf = create_transforms(image_size=args.image_size)
    train_ds = NailBitingDataset(train_records, transform=train_tf)
    val_ds = NailBitingDataset(val_records, transform=eval_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def init_wandb(args: argparse.Namespace):
    if not args.use_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed. Install it or omit --use-wandb.") from exc

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={k: v for k, v in vars(args).items() if k not in {"use_wandb"}},
    )
    return run


def export_best_checkpoint_to_onnx(checkpoint_path: Path, export_path: Path, image_size: int) -> None:
    if not checkpoint_path.exists():
        print("Best checkpoint not found; skipping ONNX export.")
        return
    export_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict)
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        export_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )
    print(f"Exported ONNX model to {export_path}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_examples += labels.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_examples += labels.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Using device: {device}")
    train_loader, val_loader = create_loaders(args)

    model = build_model(pretrained=True, freeze_base=args.freeze_base).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    wandb_run = init_wandb(args)

    best_val_acc = 0.0
    history = {"train": [], "val": []}
    best_ckpt_path = args.output / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_stats = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train"].append({"epoch": epoch, **train_stats})
        history["val"].append({"epoch": epoch, **val_stats})

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_stats['loss']:.4f} acc={train_stats['accuracy']:.3f} | "
            f"val_loss={val_stats['loss']:.4f} acc={val_stats['accuracy']:.3f}"
        )

        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_stats["loss"],
                    "train/accuracy": train_stats["accuracy"],
                    "val/loss": val_stats["loss"],
                    "val/accuracy": val_stats["accuracy"],
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        if val_stats["accuracy"] >= best_val_acc:
            best_val_acc = val_stats["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": best_val_acc,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(f"  ↳ Saved new best checkpoint to {best_ckpt_path}")
            if wandb_run:
                wandb_run.log({"best/val_accuracy": best_val_acc, "best_epoch": epoch})

    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(history, indent=2))
    print(f"Training complete. Best validation accuracy: {best_val_acc:.3f}")
    print(f"Metrics log saved to {metrics_path}")

    if args.export_onnx:
        export_path = args.onnx_output or args.output / "best_model.onnx"
        export_best_checkpoint_to_onnx(best_ckpt_path, export_path, args.image_size)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
