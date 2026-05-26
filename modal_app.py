import modal
from pydantic import BaseModel
import base64
import io
import os

# ---------------------------------------------------------
# Define Modal Application
# ---------------------------------------------------------
app = modal.App("vision-v2")

# Define the precise open-source VLM weight repository to pull
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
HF_SECRET = modal.Secret.from_name("hf-token")

# ---------------------------------------------------------
# Caching & Dependency Setup
# ---------------------------------------------------------
def download_weights():
    from huggingface_hub import snapshot_download
    # Downloads the model weights to the container's cache during the `modal deploy` / build step.
    # This prevents redownloading upon cold starts.
    hf_token = os.environ.get("HF_TOKEN")
    snapshot_download(MODEL_NAME, token=hf_token)

# Set up the environment with required dependencies for vLLM and vision models
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm==0.6.5",
        "transformers==4.48.2",
        "qwen-vl-utils",
        "fastapi[standard]",
        "pydantic",
        "huggingface_hub",
        "pillow"
    )
    .run_function(download_weights, secrets=[HF_SECRET])
)

# ---------------------------------------------------------
# Web API Payload Definition
# ---------------------------------------------------------
class InferencePayload(BaseModel):
    """
    Pydantic schema to strictly validate incoming requests from the edge device.
    Expects a text prompt and an image encoded as a Base64 string.
    """
    prompt: str
    image_b64: str

# ---------------------------------------------------------
# Serverless vLLM Cloud Engine
# ---------------------------------------------------------
@app.cls(image=image, gpu="A10G", scaledown_window=300, secrets=[HF_SECRET])
class VisionLanguageModel:
    @modal.enter()
    def setup(self):
        """
        Executed once upon container start (cold start).
        Initializes the vLLM engine, effectively caching the model weights on the GPU.
        """
        from vllm import LLM, SamplingParams
        
        print(f"Loading {MODEL_NAME} into A10G GPU memory...")
        self.llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,            # Needed for Qwen loading code
            gpu_memory_utilization=0.95,       # Reserve maximum VRAM for weights & KV Cache
            max_model_len=4096,                # Optimize context window length for memory limits
            max_num_seqs=2,                    # Tuned for edge-device batched/concurrent queries
            limit_mm_per_prompt={"image": 1}   # Allow 1 image per inference query
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.2,                   # Low temperature for highly grounded, assistive answers
            max_tokens=256,                    # Max response length (short for quick audio readout)
        )
        print("Engine ready.")

    @modal.fastapi_endpoint(method="POST", docs=True)
    def generate(self, payload: InferencePayload):
        """
        FastAPI-compliant POST endpoint accepting the validated payload.
        Handles image decoding, VLM inference, and error reporting.
        """
        from fastapi import HTTPException
        from PIL import Image
        
        # 1. Base64 Decode & Validation
        try:
            image_bytes = base64.b64decode(payload.image_b64)
            # Decode into a format compatible with the VLM (PIL Image)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            # Handle malformed Base64 safely
            raise HTTPException(status_code=400, detail=f"Invalid Base64 image encoding: {str(e)}")
            
        # 2. Prepare specialized Qwen2.5-VL prompt topology
        # Qwen2.5-VL requires exact start/pad/end vision tokens injected manually 
        # if not using an external chat template orchestrator.
        prompt = "<|vision_start|><|image_pad|><|vision_end|>" + payload.prompt
        
        # 3. vLLM Engine Execution
        try:
            outputs = self.llm.generate(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": img},
                },
                sampling_params=self.sampling_params
            )
            # Extract resulting text sequence
            response_text = outputs[0].outputs[0].text
            return {"response": response_text.strip()}
            
        except Exception as e:
            # Catch internal inference failures (e.g. OOM, parsing issue)
            raise HTTPException(status_code=500, detail=f"vLLM inference failure: {str(e)}")

# Local entry point for testing
if __name__ == "__main__":
    print("This script is designed to be deployed with 'modal deploy'.")
    print("Run: `modal deploy modal_app.py`")
