import base64
import io
import os

import modal
from pydantic import BaseModel

# ---------------------------------------------------------
# Define Modal Application
# ---------------------------------------------------------
app = modal.App("vision-v2")

# Define the precise open-source VLM weight repositories to pull
QWEN_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
# Replaces Phi with a supported VLM that ships a valid image processor.
LLAVA_MISTRAL_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
FALCON_MODEL_ID = "tiiuae/falcon-11B-vlm"
# Replaces Nemotron with a model that ships a valid image processor.
LLAVA_15_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
HF_SECRET = modal.Secret.from_name("hf-token")

# ---------------------------------------------------------
# Caching & Dependency Setup
# ---------------------------------------------------------
def download_weights():
    from huggingface_hub import snapshot_download
    # Downloads the model weights to the container's cache during the `modal deploy` / build step.
    # This prevents redownloading upon cold starts.
    hf_token = os.environ.get("HF_TOKEN")
    for model_id in (
        QWEN_MODEL_ID,
        LLAVA_MISTRAL_MODEL_ID,
        FALCON_MODEL_ID,
        LLAVA_15_MODEL_ID,
    ):
        snapshot_download(model_id, token=hf_token)

# Set up the environment with required dependencies for vLLM and vision models
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm==0.6.5",
        "transformers==4.57.6",
        "accelerate",
        "qwen-vl-utils",
        "fastapi[standard]",
        "pydantic",
        "huggingface_hub",
        "pillow",
        "sentencepiece"
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
class BaseVisionLanguageModel:
    MODEL_ID = ""
    ENGINE = "transformers"
    PROMPT_STYLE = "chat"

    @modal.enter()
    def setup(self):
        """
        Executed once upon container start (cold start).
        Initializes the inference engine and caches the model on GPU memory.
        """
        if self.ENGINE == "vllm":
            from vllm import LLM, SamplingParams

            print(f"Loading {self.MODEL_ID} into GPU memory via vLLM...")
            self.llm = LLM(
                model=self.MODEL_ID,
                trust_remote_code=True,
                gpu_memory_utilization=0.95,
                max_model_len=4096,
                max_num_seqs=2,
                limit_mm_per_prompt={"image": 1},
            )
            self.sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=128,
                repetition_penalty=1.1,
                stop=["<|endoftext|>", "<|im_end|>", "<|assistant|>", "</s>"]
            )
            self.processor = None
            self.model = None
            print("vLLM engine ready.")
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor

        print(f"Loading {self.MODEL_ID} into GPU memory via Transformers...")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Prefer the fast image processor to avoid warnings and speed up preprocessing.
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
            use_fast=True,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.MODEL_ID,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                # Avoid FlashAttention2 requirement when it's not installed.
                attn_implementation="sdpa",
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                # Avoid FlashAttention2 requirement when it's not installed.
                attn_implementation="sdpa",
            )
        self.model.eval()
        self.generation_kwargs = {
            "max_new_tokens": 128,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 4,
        }
        print("Transformers engine ready.")

    def _build_prompt(self, prompt: str) -> str:
        if self.PROMPT_STYLE == "qwen":
            return "<|vision_start|><|image_pad|><|vision_end|>" + prompt

        if self.PROMPT_STYLE == "chat" and self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            return self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

        return prompt

    def _generate_sync(self, payload: InferencePayload):
        from fastapi import HTTPException
        from PIL import Image

        try:
            image_bytes = base64.b64decode(payload.image_b64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid Base64 image encoding: {str(e)}")

        if self.ENGINE == "vllm":
            prompt = self._build_prompt(payload.prompt)
            try:
                outputs = self.llm.generate(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": img},
                    },
                    sampling_params=self.sampling_params,
                )
                response_text = outputs[0].outputs[0].text
                return {"response": response_text.strip()}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"vLLM inference failure: {str(e)}")

        if self.PROMPT_STYLE == "qwen" and self.processor is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": payload.prompt},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            prompt = self._build_prompt(payload.prompt)
        try:
            try:
                inputs = self.processor(images=img, text=prompt, return_tensors="pt")
                decode_with = self.processor
            except TypeError:
                from transformers import AutoImageProcessor, AutoTokenizer

                if not hasattr(self, "image_processor"):
                    self.image_processor = AutoImageProcessor.from_pretrained(
                        self.MODEL_ID,
                        trust_remote_code=True,
                    )
                if not hasattr(self, "text_tokenizer"):
                    self.text_tokenizer = AutoTokenizer.from_pretrained(
                        self.MODEL_ID,
                        trust_remote_code=True,
                    )
                image_inputs = self.image_processor(images=img, return_tensors="pt")
                text_inputs = self.text_tokenizer(prompt, return_tensors="pt")
                inputs = {**image_inputs, **text_inputs}
                decode_with = self.text_tokenizer
                self.tokenizer = self.text_tokenizer

            device = getattr(self.model, "device", None)
            if device is not None:
                inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
            gen_kwargs = dict(self.generation_kwargs)
            if self.tokenizer is not None:
                if getattr(self.tokenizer, "eos_token_id", None) is not None:
                    gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
                if getattr(self.tokenizer, "pad_token_id", None) is not None:
                    gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            output_text = decode_with.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if "input_ids" in inputs:
                prompt_text = decode_with.batch_decode(inputs["input_ids"], skip_special_tokens=True)[0]
                if output_text.startswith(prompt_text):
                    output_text = output_text[len(prompt_text):]
            return {"response": output_text.strip()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transformers inference failure: {str(e)}")

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def generate(self, payload: InferencePayload):
        import anyio

        return await anyio.to_thread.run_sync(self._generate_sync, payload)


@app.cls(image=image, gpu="A10G", scaledown_window=300, secrets=[HF_SECRET])
class QwenVisionLanguageModel(BaseVisionLanguageModel):
    MODEL_ID = QWEN_MODEL_ID
    # vLLM 0.6.5 does not reliably support Qwen2-VL architectures.
    # Use Transformers to avoid startup failures.
    ENGINE = "transformers"
    PROMPT_STYLE = "chat"


@app.cls(image=image, gpu="A10G", scaledown_window=300, secrets=[HF_SECRET])
class LlavaMistralVisionLanguageModel(BaseVisionLanguageModel):
    MODEL_ID = LLAVA_MISTRAL_MODEL_ID
    ENGINE = "transformers"
    PROMPT_STYLE = "chat"




@app.cls(image=image, gpu="A10G", scaledown_window=300, secrets=[HF_SECRET])
class Llava15VisionLanguageModel(BaseVisionLanguageModel):
    MODEL_ID = LLAVA_15_MODEL_ID
    ENGINE = "transformers"
    PROMPT_STYLE = "chat"


@app.cls(image=image, gpu="A10G", scaledown_window=300, secrets=[HF_SECRET])
class FalconVisionLanguageModel(BaseVisionLanguageModel):
    MODEL_ID = FALCON_MODEL_ID
    ENGINE = "transformers"
    PROMPT_STYLE = "chat"



# Local entry point for testing
if __name__ == "__main__":
    print("This script is designed to be deployed with 'modal deploy'.")
    print("Run: `modal deploy modal_app.py`")
