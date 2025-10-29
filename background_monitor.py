#!/usr/bin/env python3
import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from nbdetect.model import INDEX_TO_LABEL, LABEL_TO_INDEX, build_model, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run nail-biting detection in the background and trigger macOS alerts."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to a trained PyTorch checkpoint (.pt).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Image size used during training (e.g. 224).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for triggering an alert.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index passed to cv2.VideoCapture.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.0,
        help="Seconds to wait before showing another alert.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Computation device. Default forces CPU as requested.",
    )
    return parser.parse_args()


def build_transform(image_size: int):
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return v2.Compose(
        [
            v2.ToPILImage(),
            v2.Resize((image_size, image_size)),
            v2.ToTensor(),
            normalize,
        ]
    )


def load_model_for_inference(model_path: Path, device: torch.device) -> torch.nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    model = build_model(pretrained=False)
    load_checkpoint(model, model_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def trigger_alert(probability: float) -> None:
    if sys.platform != "darwin":
        print(
            "\n[info] Alert triggered but macOS dialog is only available on macOS.",
            file=sys.stderr,
        )
        return
    message = f"Nail biting detected with confidence {probability:.2f}. Please take a break!"
    escaped = message.replace("\\", "\\\\").replace('"', r"\"")
    dialog = (
        f'display dialog "{escaped}" buttons {{"I will stop"}} '
        'default button "I will stop" with icon stop'
    )
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to activate',
                "-e",
                dialog,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[warning] Failed to display alert dialog: {exc}", file=sys.stderr)


def install_signal_handlers(cap: cv2.VideoCapture):
    def handle_exit(signum, frame):
        print(f"\nReceived signal {signum}. Shutting down cleanly.")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = load_model_for_inference(args.model, device)
    transform = build_transform(args.input_size)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to access camera index {args.camera}.")

    install_signal_handlers(cap)
    last_alert_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)[0]

            biting_prob = float(probs[LABEL_TO_INDEX["biting"]].item())
            not_biting_prob = float(probs[LABEL_TO_INDEX["not_biting"]].item())
            pred_idx = int(probs.argmax().item())
            pred_label = INDEX_TO_LABEL[pred_idx]

            print(
                f"{time.strftime('%H:%M:%S')} "
                f"biting={biting_prob:.3f} not_biting={not_biting_prob:.3f} "
                f"prediction={pred_label}",
                end="\r",
                flush=True,
            )

            alert_triggered = pred_label == "biting" and biting_prob >= args.threshold
            now = time.monotonic()
            if alert_triggered and (now - last_alert_time) >= args.cooldown:
                last_alert_time = now
                trigger_alert(biting_prob)

            time.sleep(0.05)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nMonitoring stopped.")


if __name__ == "__main__":
    main()
