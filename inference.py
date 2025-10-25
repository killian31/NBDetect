import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

from nbdetect.model import INDEX_TO_LABEL, build_model, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime inference for nail-biting detection.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to PyTorch checkpoint (best_model.pt).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for cv2.VideoCapture.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Probability threshold for alert.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--cooldown", type=float, default=1.5, help="Seconds between repeated alerts.")
    return parser.parse_args()


def build_transform(image_size: int):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )




def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = build_model(pretrained=False)
    load_checkpoint(model, args.weights, map_location=device)
    model.to(device)
    model.eval()

    transform = build_transform(args.image_size)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera.")

    last_alert = 0.0
    label_colors = {0: (0, 200, 0), 1: (50, 30, 230)}

    print("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(probs.argmax())
        pred_label = INDEX_TO_LABEL[pred_idx]
        pred_prob = probs[pred_idx]
        color = label_colors[pred_idx]

        cv2.putText(
            frame,
            f"{pred_label} ({pred_prob:.2f})",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )

        alert_triggered = pred_label == "biting" and pred_prob >= args.threshold
        if alert_triggered and (time.time() - last_alert) >= args.cooldown:
            last_alert = time.time()
            cv2.putText(
                frame,
                "⚠ Nail biting detected!",
                (16, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        border_color = (0, 255, 0) if pred_label == "not_biting" else (80, 40, 255)
        cv2.rectangle(frame, (8, 8), (frame.shape[1] - 8, frame.shape[0] - 8), border_color, 2)

        cv2.imshow("NBDetect Realtime", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
