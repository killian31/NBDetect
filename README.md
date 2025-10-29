# NBDetect

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)

This repo is the source code to train and evaluate a real-time nail-biting detection model, including pretrained weights and inference app.

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
python train.py --data-root dataset --output runs/exp01 --epochs 15
```

### Serve from an API
```
python inference_server.py
```
Visit `http://localhost:5050/` with an image to get probabilities, or point it at an ONNX file.

### Benchmark latency (optional)
```
python benchmark_latency.py --weights runs/exp01/best_model.pt
```
Plots land in `plots/` and help compare PyTorch vs. ONNX throughput.

## Optional goodies
- Weights & Biases logging toggles on automatically when `WANDB_API_KEY` is set.
- Add `--export-onnx` during training to keep both `.pt` and `.onnx` for best model.
- `requirements.txt` already includes the extras; comment out `onnxruntime` or `wandb` if you want a slimmer install.
