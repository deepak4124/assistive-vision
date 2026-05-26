# Assistive Vision Backend and Edge Client

This repository provides an end-to-end framework for deploying powerful Vision-Language Models for assistive hardware (e.g. Raspberry Pi wearable cameras). It uses [Modal](https://modal.com/) to host a scalable serverless GPU API powered by `vLLM`, interacting with a lightweight edge script tracking camera feeds.

## Components 

* **modal_app.py**: A fully functional Modal deployment script. It handles downloading `Qwen/Qwen2-VL-2B-Instruct` directly into the caching layer to prevent cold-start download times. Using the `@app.cls()` standard, the VLM is safely preloaded into an `A10G` instance's VRAM for highly efficient batched decoding inference.
* **edge_client.py**: A conceptual snippet using OpenCV to grab frames from a constrained device (e.g. Pi Camera), encode them into Base64 format, and ship them alongside a prompt to the defined Modal endpoint.

## Usage

### 1. Backend 
Ensure `modal` is authenticated against your workspace:
```bash
python -m pip install modal
modal setup
```

Deploy the serverless API:
```bash
modal deploy modal_app.py
```
This command will provision the infrastructure and output a generic webhook (e.g., `https://<username>--...modal.run`).

### 2. Edge Device Setup
Replace the placeholder `MODAL_ENDPOINT` URL inside [edge_client.py](edge_client.py) with your active webhook.

Install edge dependencies:
```bash
python -m pip install opencv-python requests
```

Capture the frame and dispatch the inference request:
```bash
python edge_client.py
```

### 3. Raspberry Pi Endpoint Test (Thorough)
This section shows how to hit the Modal endpoint from a Raspberry Pi using either `curl` or a small Python script.

#### 3.1. Prerequisites
1. Raspberry Pi OS (64-bit recommended).
2. Python 3.9+ installed.
3. An active Modal endpoint URL (from `modal deploy`).
4. A test image, or a working camera.

#### 3.2. System setup
Update packages and install basic tools:
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv curl
```

Optional: enable the Pi camera from `raspi-config` if you want live captures:
```bash
sudo raspi-config
```
Then navigate to Interface Options and enable the camera, reboot if prompted.

#### 3.3. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

#### 3.4. Choose your input image
You have two options:

Option A: use a local file (recommended for first test)
1. Copy a test image to the Pi, e.g. `test.png`.
2. Make sure the image is not tiny; the model expects reasonable resolution (e.g. 64x64 or larger).

Option B: capture from camera (later step)
1. Use OpenCV in the Python example below to grab a frame.

#### 3.5. Test with curl (base64 from file)
Install the helper if needed:
```bash
sudo apt install -y coreutils
```

Export your endpoint and build a JSON payload:
```bash
export MODAL_ENDPOINT="https://<username>--vision-v2-visionlanguagemodel-generate.modal.run"
IMG_B64=$(base64 -w 0 test.png)
```

Send the request:
```bash
curl -sS "$MODAL_ENDPOINT" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"describe the image\", \"image_b64\": \"$IMG_B64\"}"
```

Expected output:
```json
{"response":"..."}
```

#### 3.6. Test with Python (file or camera)
Install dependencies:
```bash
python -m pip install requests opencv-python
```

Create a test script (example `pi_request.py`):
```python
import base64
import json
import os
import requests

MODAL_ENDPOINT = os.environ.get("MODAL_ENDPOINT")
if not MODAL_ENDPOINT:
	raise RuntimeError("Set MODAL_ENDPOINT in your environment.")

def image_to_base64(path):
	with open(path, "rb") as f:
		return base64.b64encode(f.read()).decode("utf-8")

payload = {
	"prompt": "describe the image",
	"image_b64": image_to_base64("test.png"),
}

resp = requests.post(MODAL_ENDPOINT, json=payload, timeout=180)
resp.raise_for_status()
print(json.dumps(resp.json(), indent=2))
```

Run it:
```bash
export MODAL_ENDPOINT="https://<username>--vision-v2-visionlanguagemodel-generate.modal.run"
python pi_request.py
```

#### 3.7. Live camera capture example (optional)
If you want to capture a live frame on the Pi:
```python
import base64
import cv2
import json
import os
import requests

MODAL_ENDPOINT = os.environ.get("MODAL_ENDPOINT")
if not MODAL_ENDPOINT:
	raise RuntimeError("Set MODAL_ENDPOINT in your environment.")

cap = cv2.VideoCapture(0)
ok, frame = cap.read()
cap.release()
if not ok:
	raise RuntimeError("Failed to read from camera")

_, buf = cv2.imencode(".png", frame)
b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

payload = {
	"prompt": "describe the scene",
	"image_b64": b64,
}

resp = requests.post(MODAL_ENDPOINT, json=payload, timeout=180)
resp.raise_for_status()
print(json.dumps(resp.json(), indent=2))
```

#### 3.8. Troubleshooting
* **Timeouts**: First request can be slower due to cold start or model warmup. Retry with a longer timeout (180s or more).
* **Image too small**: The image processor expects reasonable dimensions. Use at least 64x64.
* **HTTP 400**: Usually means Base64 decoding failed or the image is invalid. Re-encode the file and retry.
* **HTTP 500**: The model failed to run. Check Modal logs and ensure dependencies are pinned in [modal_app.py](modal_app.py).
