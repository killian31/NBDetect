import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from nbdetect.model import INDEX_TO_LABEL, build_model, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime inference for nail-biting detection."
    )
    parser.add_argument(
        "--weights", type=Path, help="Path to PyTorch checkpoint (best_model.pt)."
    )
    parser.add_argument("--onnx", type=Path, help="Path to exported ONNX model.")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index for cv2.VideoCapture."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.6, help="Probability threshold for alert."
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--cooldown", type=float, default=1.5, help="Seconds between repeated alerts."
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


def load_onnx_session(onnx_path: Path):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for ONNX models. Install via `pip install onnxruntime`."
        ) from exc
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception:
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
    input_name = session.get_inputs()[0].name
    return session, input_name


def run_onnx_inference(session, input_name: str, tensor: torch.Tensor) -> np.ndarray:
    ort_inputs = {input_name: tensor.numpy()}
    logits = session.run(None, ort_inputs)[0]
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[0]
    return probs


def main() -> None:
    args = parse_args()
    if not args.weights and not args.onnx:
        raise SystemExit(
            "Provide either --weights for PyTorch or --onnx for ONNX inference."
        )

    device = torch.device(args.device)

    use_onnx = args.onnx is not None
    if use_onnx:
        ort_session, ort_input = load_onnx_session(args.onnx)
        model = None
    else:
        if not args.weights:
            raise SystemExit("--weights is required when --onnx is not provided.")
        model = build_model(pretrained=False)
        load_checkpoint(model, args.weights, map_location=device)
        model.to(device)
        model.eval()
        ort_session = None
        ort_input = None

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
        tensor = transform(rgb).unsqueeze(0)

        if use_onnx:
            probs = run_onnx_inference(ort_session, ort_input, tensor)
        else:
            with torch.no_grad():
                logits = model(tensor.to(device))
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
        cv2.rectangle(
            frame, (8, 8), (frame.shape[1] - 8, frame.shape[0] - 8), border_color, 2
        )

        cv2.imshow("NBDetect Realtime", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
