# NBDetect

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)

Nail-biting detection, but make it playful: collect your own dataset, train a lightweight MobileNet, and deploy in realtime with either PyTorch or ONNX.

## Highlights
- Webcam-powered data capture UI that writes straight into the `dataset/` tree.
- PyTorch Lightning-free training loop tuned for binary classification with MobileNet V3.
- One-command export to ONNX plus a minimal inference server for web or CLI demos.
- Latency benchmarking utilities to help you pick the snappiest deployment path.

## Quickstart

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Capture your own data
```
python app.py
```
Open `http://localhost:5009` and start a capture session to populate `dataset/train|val|test`.

### Train the detector
```
python train.py --data-root dataset --output runs/exp01 --epochs 30
```
The script saves the best checkpoint and can export ONNX models with `--export-onnx`.

### Run realtime inference
```
python inference.py --weights runs/exp01/best_model.pt
```
Pass `--onnx best_model.onnx` instead to try the ONNX runtime (ensure `onnxruntime` is installed).

### Serve from an API
```
python inference_server.py --weights runs/exp01/best_model.pt
```
Hit `http://localhost:8000/predict` with an image to get probabilities, or point it at an ONNX file.

### Benchmark latency (optional)
```
python benchmark_latency.py --weights runs/exp01/best_model.pt
```
Plots land in `plots/` and help compare PyTorch vs. ONNX throughput.

## Optional goodies
- Weights & Biases logging toggles on automatically when `WANDB_API_KEY` is set.
- Add `--export-onnx` during training to keep both `.pt` and `.onnx` builds in sync.
- `requirements.txt` already includes the extras; comment out `onnxruntime` or `wandb` if you want a slimmer install.
