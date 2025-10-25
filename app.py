import csv
import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from flask import (
    Flask,
    abort,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
)


app = Flask(__name__)

DATASET_ROOT = Path("dataset")
DATASET_ROOT.mkdir(parents=True, exist_ok=True)

ALLOWED_LABELS = {"biting", "not_biting"}
CSV_HEADERS = ["filename", "label", "captured_at", "annotated_at"]

STATE_LOCK = threading.Lock()
PREVIEW_LOCK = threading.Lock()
LATEST_PREVIEW: Optional[bytes] = None
CAPTURE_STATE: Dict[str, Optional[object]] = {
    "status": "idle",
    "session_id": None,
    "session_dir": None,
    "duration": 0.0,
    "started_at": None,
    "captured": 0,
    "message": "",
    "stop_event": None,
    "thread": None,
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    slug = slug.strip("-")
    return slug or "session"


def _set_state(**kwargs) -> None:
    with STATE_LOCK:
        CAPTURE_STATE.update(kwargs)


def _public_state() -> Dict[str, object]:
    with STATE_LOCK:
        state_copy = CAPTURE_STATE.copy()
    progress = 0.0
    remaining = None
    if state_copy["status"] == "running" and state_copy["started_at"] and state_copy["duration"]:
        elapsed = time.time() - float(state_copy["started_at"])
        duration = float(state_copy["duration"])
        progress = max(0.0, min(elapsed / duration, 1.0))
        remaining = max(duration - elapsed, 0.0)

    return {
        "status": state_copy["status"],
        "sessionId": state_copy["session_id"],
        "captured": state_copy["captured"],
        "message": state_copy["message"],
        "progress": progress,
        "remaining": remaining,
        "duration": state_copy["duration"],
    }


def _reset_preview() -> None:
    global LATEST_PREVIEW
    with PREVIEW_LOCK:
        LATEST_PREVIEW = None


def _update_preview(frame: Optional[bytes]) -> None:
    global LATEST_PREVIEW
    with PREVIEW_LOCK:
        LATEST_PREVIEW = frame


def start_capture(session_name: str, duration_minutes: float, interval_seconds: float) -> str:
    if interval_seconds <= 0:
        raise ValueError("Interval must be greater than zero.")
    duration_seconds = duration_minutes * 60
    if duration_seconds <= 0:
        raise ValueError("Duration must be greater than zero.")

    session_slug = slugify(session_name or "session")
    session_id = f"{session_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = DATASET_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    images_dir = session_dir / "images"
    images_dir.mkdir(exist_ok=True)

    captures_log = session_dir / "captures.csv"
    if not captures_log.exists():
        with captures_log.open("w", newline="") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=["filename", "captured_at"])
            writer.writeheader()

    _reset_preview()
    stop_event = threading.Event()
    with STATE_LOCK:
        if CAPTURE_STATE["status"] == "running":
            raise RuntimeError("Capture already in progress.")
        CAPTURE_STATE.update(
            {
                "status": "running",
                "session_id": session_id,
                "session_dir": session_dir,
                "duration": duration_seconds,
                "started_at": time.time(),
                "captured": 0,
                "message": "Capturing...",
                "stop_event": stop_event,
            }
        )

    worker_args = (session_dir, duration_seconds, interval_seconds, stop_event)
    thread = threading.Thread(target=capture_worker, args=worker_args, daemon=True)
    _set_state(thread=thread)
    thread.start()
    return session_id


def capture_worker(
    session_dir: Path, duration_seconds: float, interval_seconds: float, stop_event: threading.Event
) -> None:
    images_dir = session_dir / "images"
    captures_log = session_dir / "captures.csv"
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        _set_state(status="failed", message="Unable to access the default camera.", thread=None, stop_event=None)
        _reset_preview()
        return

    end_time = time.time() + duration_seconds
    shot_idx = 0
    with captures_log.open("a", newline="") as log_file:
        capture_writer = csv.DictWriter(log_file, fieldnames=["filename", "captured_at"])

        while time.time() < end_time and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.2)
                continue

            success_encode, jpeg = cv2.imencode(".jpg", frame)
            if success_encode:
                _update_preview(jpeg.tobytes())

            timestamp = datetime.utcnow()
            filename = f"{shot_idx:05d}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            file_path = images_dir / filename
            if cv2.imwrite(str(file_path), frame):
                capture_writer.writerow({"filename": filename, "captured_at": timestamp.isoformat()})
                log_file.flush()
                shot_idx += 1
                _set_state(captured=shot_idx)

            # Sleep to respect the remaining interval while allowing cancellation.
            remaining_interval = interval_seconds
            while remaining_interval > 0 and not stop_event.is_set():
                chunk = min(0.1, remaining_interval)
                time.sleep(chunk)
                remaining_interval -= chunk

    cap.release()

    if stop_event.is_set():
        _set_state(status="cancelled", message="Capture cancelled.", thread=None, stop_event=None)
        _reset_preview()
    elif shot_idx == 0:
        _set_state(status="failed", message="No frames captured. Please retry.", thread=None, stop_event=None)
        _reset_preview()
    else:
        _set_state(
            status="completed",
            message=f"Capture complete. Saved {shot_idx} frames.",
            thread=None,
            stop_event=None,
        )
        _reset_preview()


def cancel_capture() -> None:
    with STATE_LOCK:
        stop_event = CAPTURE_STATE.get("stop_event")
    if stop_event:
        stop_event.set()


def list_sessions() -> List[Dict[str, object]]:
    sessions = []
    dataset_root = DATASET_ROOT.resolve()
    for session_dir in sorted(dataset_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        images_dir = session_dir / "images"
        if not images_dir.exists():
            continue
        session_id = session_dir.name
        image_files = sorted(images_dir.glob("*.jpg"))
        annotations = _load_annotations(session_dir)
        sessions.append(
            {
                "session_id": session_id,
                "captured": len(image_files),
                "annotated": len(annotations),
                "created_at": datetime.fromtimestamp(session_dir.stat().st_mtime).strftime("%b %d, %Y %H:%M"),
            }
        )
    return sessions


def _ensure_session_dir(session_id: str) -> Path:
    dataset_root = DATASET_ROOT.resolve()
    candidate = (DATASET_ROOT / session_id).resolve()
    if not candidate.exists() or not candidate.is_dir():
        abort(404)
    if dataset_root not in candidate.parents and candidate != dataset_root:
        abort(404)
    return candidate


def _load_annotations(session_dir: Path) -> Dict[str, Dict[str, str]]:
    annotations_path = session_dir / "annotations.csv"
    if not annotations_path.exists():
        return {}
    annotations: Dict[str, Dict[str, str]] = {}
    with annotations_path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            annotations[row["filename"]] = row
    return annotations


def _load_capture_log(session_dir: Path) -> Dict[str, str]:
    captures_path = session_dir / "captures.csv"
    if not captures_path.exists():
        return {}
    captures: Dict[str, str] = {}
    with captures_path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            captures[row["filename"]] = row["captured_at"]
    return captures


def _append_annotation(session_dir: Path, filename: str, label: str) -> None:
    annotations_path = session_dir / "annotations.csv"
    is_new_file = not annotations_path.exists()
    captured_at_map = _load_capture_log(session_dir)
    captured_at = captured_at_map.get(filename, "")
    with annotations_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(
            {
                "filename": filename,
                "label": label,
                "captured_at": captured_at,
                "annotated_at": datetime.utcnow().isoformat(),
            }
        )


@app.get("/")
def index() -> str:
    return render_template("index.html", state=_public_state())


@app.post("/start")
def start_endpoint():
    data = request.get_json(force=True, silent=True) or {}
    try:
        session_name = data.get("sessionName", "")
        duration = float(data.get("durationMinutes", 5))
        interval = float(data.get("intervalSeconds", 5))
        session_id = start_capture(session_name, duration, interval)
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "ok", "sessionId": session_id})


@app.post("/cancel")
def cancel_endpoint():
    cancel_capture()
    return jsonify({"status": "cancelled"})


@app.get("/status")
def status_endpoint():
    return jsonify(_public_state())


@app.get("/sessions")
def sessions_view():
    return render_template("sessions.html", sessions=list_sessions())


@app.get("/preview")
def preview_endpoint():
    with PREVIEW_LOCK:
        frame = LATEST_PREVIEW
    if not frame:
        return Response(status=204)
    response = Response(frame, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.get("/annotate/<session_id>")
def annotate_view(session_id: str):
    session_dir = _ensure_session_dir(session_id)
    images_dir = session_dir / "images"
    image_files = sorted([p.name for p in images_dir.glob("*.jpg")])
    annotations = _load_annotations(session_dir)
    payload = [{"filename": name, "label": annotations.get(name, {}).get("label")} for name in image_files]
    start_index = 0
    for idx, entry in enumerate(payload):
        if not entry["label"]:
            start_index = idx
            break
    else:
        start_index = 0
    return render_template(
        "annotate.html",
        session_id=session_id,
        images_json=json.dumps(payload),
        start_index=start_index,
    )


@app.post("/annotate/<session_id>")
def annotate_endpoint(session_id: str):
    session_dir = _ensure_session_dir(session_id)
    data = request.get_json(force=True, silent=True) or {}
    filename = data.get("filename", "")
    label = data.get("label", "")
    if label not in ALLOWED_LABELS:
        return jsonify({"error": "Invalid label."}), 400
    image_path = (session_dir / "images" / filename).resolve()
    if not image_path.exists() or not image_path.is_file():
        return jsonify({"error": "Image not found."}), 404
    if os.path.commonpath([image_path, session_dir]) != str(session_dir):
        abort(400)
    _append_annotation(session_dir, filename, label)
    return jsonify({"status": "ok"})


@app.get("/dataset/<session_id>/<path:filename>")
def serve_image(session_id: str, filename: str):
    session_dir = _ensure_session_dir(session_id)
    images_dir = session_dir / "images"
    return send_from_directory(images_dir, filename)


def main() -> None:
    app.run(host="0.0.0.0", port=5009, debug=True)


if __name__ == "__main__":
    main()
