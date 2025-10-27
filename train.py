import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from nbdetect.data import NailBitingDataset, Record, create_transforms, load_split_records
from nbdetect.model import LABEL_TO_INDEX, build_model, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train nail-biting detector with MobileNetV3.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Path to dataset root.")
    parser.add_argument("--output", type=Path, default=Path("runs/latest"), help="Directory to store checkpoints.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
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
    parser.add_argument(
        "--class-balance",
        action="store_true",
        help="Use class-weighted loss to handle label imbalance.",
    )
    parser.add_argument(
        "--lr-decay-milestones",
        type=str,
        default="",
        help="Comma-separated epochs (e.g., 6,9) where LR is divided.",
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=1.0,
        help="Divide LR by this factor at each milestone (>1).",
    )
    return parser.parse_args()


def compute_class_weights(records: List[Record]) -> List[float]:
    counts = Counter(record.label for record in records)
    total = sum(counts.values())
    num_classes = len(LABEL_TO_INDEX)
    if total == 0:
        raise RuntimeError("No records available to compute class weights.")
    weights: List[float] = []
    for label, idx in sorted(LABEL_TO_INDEX.items(), key=lambda item: item[1]):
        count = counts.get(label, 0)
        if count == 0:
            raise RuntimeError(f"No samples found for label '{label}' to compute class weights.")
        weight = total / (num_classes * count)
        weights.append(weight)
    return weights


def create_loaders(
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[List[float]]]:
    train_records = load_split_records(args.dataset, "train")
    val_records = load_split_records(args.dataset, "val")
    test_records = load_split_records(args.dataset, "test")

    class_weights = compute_class_weights(train_records) if args.class_balance else None

    train_tf, eval_tf = create_transforms(image_size=args.image_size)
    train_ds = NailBitingDataset(train_records, transform=train_tf)
    val_ds = NailBitingDataset(val_records, transform=eval_tf)
    test_ds = NailBitingDataset(test_records, transform=eval_tf)

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
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, class_weights


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
    preds_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_examples += labels.size(0)
        preds_all.append(preds.detach().cpu())
        labels_all.append(labels.detach().cpu())

    if preds_all:
        preds_cat = torch.cat(preds_all)
        labels_cat = torch.cat(labels_all)
        positive_idx = LABEL_TO_INDEX["biting"]
        negative_idx = LABEL_TO_INDEX["not_biting"]
        tp = torch.sum((preds_cat == positive_idx) & (labels_cat == positive_idx)).item()
        fp = torch.sum((preds_cat == positive_idx) & (labels_cat == negative_idx)).item()
        fn = torch.sum((preds_cat == negative_idx) & (labels_cat == positive_idx)).item()
        tn = torch.sum((preds_cat == negative_idx) & (labels_cat == negative_idx)).item()
    else:
        tp = fp = fn = tn = 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = total_correct / total_examples if total_examples else 0.0

    return {
        "loss": total_loss / total_examples if total_examples else 0.0,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
    }


def parse_milestones(arg_value: str) -> List[int]:
    if not arg_value.strip():
        return []
    milestones = []
    for token in arg_value.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("Milestones must be positive integers.")
        milestones.append(value)
    return sorted(set(milestones))


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Using device: {device}")
    train_loader, val_loader, test_loader, class_weights = create_loaders(args)
    weight_tensor = (
        torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
    )

    model = build_model(pretrained=True, freeze_base=args.freeze_base).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    if class_weights:
        print(f"Using class weights: {class_weights}")
    else:
        print("Not using class weights.")
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    try:
        milestones = parse_milestones(args.lr_decay_milestones)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    scheduler = None
    if milestones and args.lr_decay_factor > 1.0:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=1.0 / args.lr_decay_factor
        )

    wandb_run = init_wandb(args)

    best_val_fp = float("inf")
    best_val_acc = 0.0
    history = {"train": [], "val": []}
    best_ckpt_path = args.output / "best_model.pt"

    epoch_start_time = None
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_stats = evaluate(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start_time
        remaining_epochs = args.epochs - epoch
        eta_seconds = remaining_epochs * elapsed
        eta_formatted = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

        history["train"].append({"epoch": epoch, **train_stats})
        history["val"].append({"epoch": epoch, **val_stats})

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_stats['loss']:.4f} acc={train_stats['accuracy']:.3f} | "
            f"val_loss={val_stats['loss']:.4f} acc={val_stats['accuracy']:.3f} "
            f"prec={val_stats['precision']:.3f} rec={val_stats['recall']:.3f} f1={val_stats['f1']:.3f} "
            f"fp={val_stats['false_positives']} | epoch_time={elapsed:.1f}s | eta={eta_formatted}"
        )

        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_stats["loss"],
                    "train/accuracy": train_stats["accuracy"],
                    "val/loss": val_stats["loss"],
                    "val/accuracy": val_stats["accuracy"],
                    "val/precision": val_stats["precision"],
                    "val/recall": val_stats["recall"],
                    "val/f1": val_stats["f1"],
                    "val/false_positives": val_stats["false_positives"],
                    "val/false_negatives": val_stats["false_negatives"],
                    "val/false_positive_rate": val_stats["false_positive_rate"],
                    "val/false_negative_rate": val_stats["false_negative_rate"],
                    "lr": current_lr,
                }
            )

        epoch_ckpt_path = args.output / f"epoch_{epoch:02d}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_metrics": val_stats,
                "args": vars(args),
            },
            epoch_ckpt_path,
        )

        if val_stats["false_positives"] < best_val_fp or (
            val_stats["false_positives"] == best_val_fp and val_stats["accuracy"] >= best_val_acc
        ):
            best_val_fp = val_stats["false_positives"]
            best_val_acc = val_stats["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_stats,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(
                f"  ↳ Saved new best checkpoint to {best_ckpt_path} (val_fp={best_val_fp}, acc={best_val_acc:.3f})"
            )
            if wandb_run:
                wandb_run.log(
                    {
                        "best/val_loss": val_stats["loss"],
                        "best/val_accuracy": best_val_acc,
                        "best/val_precision": val_stats["precision"],
                        "best/val_recall": val_stats["recall"],
                        "best/val_f1": val_stats["f1"],
                        "best/val_false_positives": val_stats["false_positives"],
                        "best/val_false_negatives": val_stats["false_negatives"],
                        "best/val_false_positive_rate": val_stats["false_positive_rate"],
                        "best/val_false_negative_rate": val_stats["false_negative_rate"],
                        "best_epoch": epoch,
                    }
                )

    print(f"Training complete. Best validation accuracy: {best_val_acc:.3f}")

    test_stats = evaluate(model, test_loader, criterion, device)
    history["test"] = test_stats

    print(
        "Test set performance | "
        f"loss={test_stats['loss']:.4f} acc={test_stats['accuracy']:.3f} "
        f"prec={test_stats['precision']:.3f} rec={test_stats['recall']:.3f} "
        f"f1={test_stats['f1']:.3f} fp={test_stats['false_positives']}"
    )

    if wandb_run:
        wandb_run.log(
            {
                "test/loss": test_stats["loss"],
                "test/accuracy": test_stats["accuracy"],
                "test/precision": test_stats["precision"],
                "test/recall": test_stats["recall"],
                "test/f1": test_stats["f1"],
                "test/false_positives": test_stats["false_positives"],
            }
        )

    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(history, indent=2))
    print(f"Metrics log saved to {metrics_path}")

    if args.export_onnx:
        best_export_path = args.onnx_output or args.output / "best_model.onnx"
        last_export_path = args.onnx_output or args.output / "last_model.onnx"
        export_best_checkpoint_to_onnx(best_ckpt_path, best_export_path, args.image_size)
        export_best_checkpoint_to_onnx(epoch_ckpt_path, last_export_path, args.image_size)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
