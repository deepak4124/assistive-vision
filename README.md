# Assistive Vision Backend and Edge Client

This repository provides an end-to-end framework for deploying powerful Vision-Language Models for assistive hardware (e.g. Raspberry Pi wearable cameras). It uses [Modal](https://modal.com/) to host a scalable serverless GPU API powered by `vLLM`, interacting with a lightweight edge script tracking camera feeds.

## Components 

* **`modal_app.py`**: A fully functional Modal deployment script. It handles downloading `Qwen/Qwen2.5-VL-3B-Instruct` directly into the caching layer to prevent cold-start download times. Using the `@app.cls()` standard, the VLM is safely preloaded into an `A10G` instance's VRAM for highly efficient batched decoding inference.
* **`edge_client.py`**: A conceptual snippet using OpenCV to grab frames from a constraint device (e.g. Pi Camera), encode them into Base64 formats, and ship them alongside a prompt to the defined Modal endpoint.

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
Replace the placeholder `MODAL_ENDPOINT` URL inside `edge_client.py` with your active webhook. 
Install edge dependencies:
```bash
python -m pip install opencv-python requests
```

Capture the frame and dispatch the inference request:
```bash
python edge_client.py
```
