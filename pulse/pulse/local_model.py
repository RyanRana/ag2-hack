"""Shared local model backend for VLM Reasoner + Skeptic.

Loads a multimodal model ONCE and serves two modes:
  - **VLM mode**: image + text prompt → structured JSON response
  - **Text-only mode** (Skeptic): text prompt → structured JSON response

Supported models (in priority order):
  1. OpenGVLab/InternVL2-2B  (~4GB, GPU 4GB+ / CPU) — lighter, default
  2. llava-hf/llava-v1.6-mistral-7b-hf (~14GB, GPU 8GB+) — heavier, better

Falls back gracefully: if no local model can be loaded, returns None
and callers should use the external API path.

No external API calls. Fully offline after weights are downloaded.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
from PIL import Image


class LocalModelBackend:
    """Shared local multimodal model for VLM + Skeptic agents.

    Loads a single model instance that supports both image+text and
    text-only inference. VLM and Skeptic share this — 14GB loaded once.
    """

    # Models to try in order of preference (lighter first)
    MODEL_CANDIDATES = [
        "OpenGVLab/InternVL2-2B",
        "llava-hf/llava-v1.6-mistral-7b-hf",
    ]

    def __init__(
        self,
        model: Any | None = None,
        processor: Any | None = None,
        model_id: str | None = None,
    ) -> None:
        self._model = model
        self._processor = processor
        self._model_id = model_id
        self._device = "cpu"
        self._loaded = model is not None and processor is not None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_id(self) -> str | None:
        return self._model_id

    def load(self, model_id: str | None = None) -> bool:
        """Attempt to load a local multimodal model.

        Returns True if a model was loaded successfully, False otherwise.
        """
        if self._loaded:
            return True

        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        candidates = [model_id] if model_id else self.MODEL_CANDIDATES

        for mid in candidates:
            try:
                self._processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
                # Try AutoModelForCausalLM first (works for LLaVA, InternVL2)
                try:
                    self._model = AutoModelForCausalLM.from_pretrained(
                        mid,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
                except (ValueError, OSError):
                    self._model = AutoModel.from_pretrained(
                        mid,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )

                self._model.eval()
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
                    self._device = "cuda"

                self._model_id = mid
                self._loaded = True
                return True
            except Exception:
                continue

        return False

    def generate_with_image(
        self,
        image: Image.Image | np.ndarray,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """Generate text from image + prompt (VLM mode).

        Parameters
        ----------
        image : PIL Image or numpy array.
        prompt : Text prompt with instructions.
        max_new_tokens : Max tokens to generate.
        temperature : Sampling temperature (low = deterministic).

        Returns
        -------
        Generated text response.
        """
        if not self._loaded:
            raise RuntimeError("No local model loaded. Call load() first.")

        import torch

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0.01,
            )

        # Decode only the generated tokens (skip input tokens)
        input_len = inputs.get("input_ids", torch.tensor([])).shape[-1] if "input_ids" in inputs else 0
        generated = output_ids[0][input_len:]
        return self._processor.decode(generated, skip_special_tokens=True)

    def generate_text_only(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """Generate text from text-only prompt (Skeptic mode).

        Uses the same model but without an image input.
        """
        if not self._loaded:
            raise RuntimeError("No local model loaded. Call load() first.")

        import torch

        inputs = self._processor(
            text=prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0.01,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_len:]
        return self._processor.decode(generated, skip_special_tokens=True)


def parse_structured_output(text: str) -> dict | None:
    """Extract a JSON object from model output.

    The local model is prompted to output JSON, but may wrap it in markdown
    code fences or add preamble text. This function extracts the first valid
    JSON object found.

    Returns None if no valid JSON can be extracted.
    """
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting from markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Try finding first { ... } block
    brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# Module-level singleton for shared access
_shared_backend: LocalModelBackend | None = None


def get_shared_backend() -> LocalModelBackend:
    """Get or create the shared local model backend singleton."""
    global _shared_backend
    if _shared_backend is None:
        _shared_backend = LocalModelBackend()
    return _shared_backend
