import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import torch
import torch.nn.functional as F
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename
from torchvision import transforms

from nbdetect.model import INDEX_TO_LABEL, LABEL_TO_INDEX, build_model, load_checkpoint


app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPLOAD_DIR = Path("uploaded_models")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

MODEL_LOCK = threading.Lock()
MODEL: Optional[torch.nn.Module] = None
MODEL_PATH: Optional[Path] = None

STATE_LOCK = threading.Lock()
STATE: Dict[str, object] = {
    "status": "idle",
    "message": "Load a trained model to begin monitoring.",
    "model_path": None,
    "threshold": 0.7,
    "probabilities": {"biting": 0.0, "not_biting": 0.0},
    "last_detection": None,
    "alert": False,
}

FRAME_LOCK = threading.Lock()
FRAME_EVENT = threading.Event()
LATEST_FRAME: Optional[bytes] = None

STOP_EVENT: Optional[threading.Event] = None
RESUME_EVENT: Optional[threading.Event] = None
DETECTION_THREAD: Optional[threading.Thread] = None


def _set_state(**kwargs) -> None:
    with STATE_LOCK:
        STATE.update(kwargs)


def _get_state() -> Dict[str, object]:
    with STATE_LOCK:
        return STATE.copy()


def load_model(model_path: Path) -> None:
    global MODEL, MODEL_PATH
    model = build_model(pretrained=False)
    load_checkpoint(model, model_path, map_location=DEVICE)
    model.to(DEVICE)
    model.eval()
    with MODEL_LOCK:
        MODEL = model
        MODEL_PATH = model_path
    _set_state(
        status="ready",
        message="Model loaded. Configure threshold and start monitoring.",
        model_path=str(model_path),
    )


def detection_worker(stop_event: threading.Event, resume_event: threading.Event) -> None:
    global LATEST_FRAME
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        _set_state(status="error", message="Unable to access the default camera.")
        return

    _set_state(status="running", message="Monitoring in progress…")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = TRANSFORM(rgb).unsqueeze(0).to(DEVICE)

        with MODEL_LOCK:
            model = MODEL
        if model is None:
            _set_state(status="error", message="Model not loaded.")
            break

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu()[0]

        biting_prob = probs[LABEL_TO_INDEX["biting"]].item()
        not_biting_prob = probs[LABEL_TO_INDEX["not_biting"]].item()
        pred_idx = int(probs.argmax().item())
        pred_label = INDEX_TO_LABEL[pred_idx]
        threshold = float(_get_state()["threshold"])

        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        overlay_color = (0, 220, 0) if pred_label == "not_biting" else (255, 80, 140)
        cv2.putText(
            frame,
            f"{pred_label}  biting={biting_prob:.2f}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            overlay_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            timestamp,
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        _set_state(
            probabilities={"biting": round(biting_prob, 4), "not_biting": round(not_biting_prob, 4)},
            message="Monitoring in progress…" if pred_label == "not_biting" else "Potential nail biting detected.",
        )

        triggered = pred_label == "biting" and biting_prob >= threshold
        if triggered:
            cv2.putText(
                frame,
                "ALERT",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 0, 255),
                4,
                cv2.LINE_AA,
            )
            _set_state(
                alert=True,
                status="alert",
                message="Nail biting detected! Acknowledge to resume.",
                last_detection=datetime.utcnow().isoformat(),
            )

            with FRAME_LOCK:
                success, jpg = cv2.imencode(".jpg", frame)
                if success:
                    LATEST_FRAME = jpg.tobytes()
                    FRAME_EVENT.set()

            while not stop_event.is_set():
                if resume_event.wait(timeout=0.2):
                    resume_event.clear()
                    _set_state(alert=False, status="running", message="Monitoring resumed.")
                    break
            continue

        with FRAME_LOCK:
            success, jpg = cv2.imencode(".jpg", frame)
            if success:
                LATEST_FRAME = jpg.tobytes()
                FRAME_EVENT.set()

        time.sleep(0.06)

    cap.release()
    _set_state(status="idle", message="Monitoring stopped.", alert=False)


def start_detection(threshold: float) -> None:
    global STOP_EVENT, RESUME_EVENT, DETECTION_THREAD
    if MODEL is None:
        raise RuntimeError("Load a checkpoint before starting monitoring.")
    if DETECTION_THREAD and DETECTION_THREAD.is_alive():
        raise RuntimeError("Monitoring already in progress.")

    STOP_EVENT = threading.Event()
    RESUME_EVENT = threading.Event()
    _set_state(threshold=float(threshold))
    DETECTION_THREAD = threading.Thread(target=detection_worker, args=(STOP_EVENT, RESUME_EVENT), daemon=True)
    DETECTION_THREAD.start()


def stop_detection() -> None:
    global STOP_EVENT, RESUME_EVENT, DETECTION_THREAD
    if STOP_EVENT:
        STOP_EVENT.set()
    if RESUME_EVENT:
        RESUME_EVENT.set()
    if DETECTION_THREAD and DETECTION_THREAD.is_alive():
        DETECTION_THREAD.join(timeout=1.0)
    STOP_EVENT = None
    RESUME_EVENT = None
    DETECTION_THREAD = None
    _set_state(status="idle", message="Monitoring stopped.", alert=False)


@app.get("/")
def ui():
    return render_template("inference_live.html", state=_get_state())


@app.post("/load-model")
def load_model_route():
    data = request.get_json(force=True, silent=True) or {}
    raw_path = data.get("path", "")
    if not raw_path:
        return jsonify({"error": "Model path is required."}), 400
    model_path = Path(raw_path).expanduser()
    if not model_path.exists():
        return jsonify({"error": f"Checkpoint not found at {model_path}."}), 404
    try:
        load_model(model_path)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "ok", "modelPath": str(model_path)})


@app.post("/upload-model")
def upload_model_route():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file uploaded."}), 400
    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename."}), 400
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    save_path = UPLOAD_DIR / f"{timestamp}_{filename}"
    try:
        file.save(save_path)
        load_model(save_path)
    except Exception as exc:
        if save_path.exists():
            save_path.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "ok", "modelPath": str(save_path)})


@app.post("/start")
def start_route():
    data = request.get_json(force=True, silent=True) or {}
    threshold = float(data.get("threshold", 0.7))
    try:
        start_detection(threshold)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "running"})


@app.post("/stop")
def stop_route():
    stop_detection()
    return jsonify({"status": "stopped"})


@app.post("/acknowledge")
def acknowledge_route():
    if RESUME_EVENT:
        RESUME_EVENT.set()
    return jsonify({"status": "resumed"})


@app.post("/threshold")
def threshold_route():
    data = request.get_json(force=True, silent=True) or {}
    value = float(data.get("threshold", 0.7))
    value = max(0.0, min(1.0, value))
    _set_state(threshold=value)
    return jsonify({"status": "ok", "threshold": value})


@app.get("/status")
def status_route():
    return jsonify(_get_state())


@app.get("/video_feed")
def video_feed():
    def generate():
        while True:
            FRAME_EVENT.wait(timeout=1.0)
            with FRAME_LOCK:
                frame = LATEST_FRAME
                FRAME_EVENT.clear()
            if frame:
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            else:
                time.sleep(0.1)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)


if __name__ == "__main__":
    main()
