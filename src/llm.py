"""
Local LLM wrapper for microsoft/phi-3-mini-4k-instruct (or any HF causal LM).

This wrapper attempts to load model to GPU (fp16). If model or dependencies are
missing, it will raise a helpful error.

Notes:
  - Transformers + accelerate + bitsandbytes recommended for large models.
  - If you do not want a heavy LLM, you can replace the generate() method
    with a simple local fallback or a dummy response for testing.
"""

from typing import Optional
import torch

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception as e:
    raise ImportError("transformers is required for the LocalLLM. Install with: pip install transformers accelerate bitsandbytes") from e


class LocalLLM:
    def __init__(self, model_name: str = "microsoft/phi-3-mini-4k-instruct", device: str = "cuda"):
        """
        model_name: HF model repo id or local path
        device: 'cuda' or 'cpu'
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"

        print(f"[llm] Loading LLM '{model_name}' to device='{self.device}' (this may take time)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # For GPUs, use fp16 where supported
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16 if self.device == "cuda" else None)
        print("[llm] Model loaded.")

    def generate(self, query: str, context: str, max_length: int = 512) -> str:
        """
        Build a simple prompt using context and query, generate text response.
        """
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # Generate with reasonable defaults. Adjust decoding params as needed.
        outputs = self.model.generate(**inputs, max_new_tokens=max_length)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
