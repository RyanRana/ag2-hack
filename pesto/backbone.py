"""DINOv2 backbone — shared feature extraction run ONCE per frame.

This is a pre-processing module (not an AG2 agent). It provides:
  1. Per-patch 768-dim feature embeddings for the anomaly detector.
  2. Self-attention maps for visual explanation overlays.

The backbone runs independently from the existing ML models (MobileNet,
ViT). It does NOT replace them — it supplements them with a separate
feature path (Path B in the architecture diagram).

Model: facebook/dinov2-base (ViT-B/14, 86M params, ~350MB).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


# Patch size for DINOv2-base
_PATCH_SIZE = 14


class DINOv2Backbone:
    """Lazy-loading DINOv2 feature extractor.

    Extracts per-patch CLS + patch token features and attention maps
    from the last transformer layer.
    """

    HF_MODEL_ID = "facebook/dinov2-base"

    def __init__(self, model: Any | None = None, processor: Any | None = None) -> None:
        self._model = model
        self._processor = processor
        self._device = "cpu"

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._processor = AutoImageProcessor.from_pretrained(self.HF_MODEL_ID)
        self._model = AutoModel.from_pretrained(self.HF_MODEL_ID)
        self._model.eval()
        if torch.cuda.is_available():
            self._model = self._model.cuda()
            self._device = "cuda"
        return self._model, self._processor

    def extract_features(
        self,
        image: Image.Image | np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Extract DINOv2 features from a single image (or crop).

        Returns
        -------
        dict with:
          "cls_token"     : (768,) CLS token embedding
          "patch_tokens"  : (N_patches, 768) per-patch embeddings
          "attention_map" : (H_patches, W_patches) mean attention from last layer
          "patch_grid"    : (H_patches, W_patches) tuple
        """
        import torch

        model, processor = self._load()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # Last hidden state: (1, 1+N_patches, 768) — CLS + patch tokens
        last_hidden = outputs.last_hidden_state[0].cpu().numpy()
        cls_token = last_hidden[0]        # (768,)
        patch_tokens = last_hidden[1:]    # (N_patches, 768)

        # Attention from last layer: (1, n_heads, seq_len, seq_len)
        last_attn = outputs.attentions[-1][0].cpu().numpy()  # (n_heads, seq, seq)
        # Mean attention from CLS token to all patch tokens across heads
        cls_attn = last_attn[:, 0, 1:]  # (n_heads, N_patches)
        mean_attn = cls_attn.mean(axis=0)  # (N_patches,)

        # Reshape to spatial grid
        n_patches = patch_tokens.shape[0]
        # DINOv2 resizes input to 518x518 by default → 37x37 patches
        h_patches = int(np.sqrt(n_patches))
        w_patches = n_patches // h_patches
        # Handle non-square patch grids
        if h_patches * w_patches != n_patches:
            h_patches = n_patches
            w_patches = 1
        attn_map = mean_attn[:h_patches * w_patches].reshape(h_patches, w_patches)

        return {
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "attention_map": attn_map,
            "patch_grid": (h_patches, w_patches),
        }

    def extract_per_plant(
        self,
        image_path: str,
        bboxes: list[tuple[int, int, int, int]],
        plant_ids: list[int],
    ) -> dict[int, dict[str, np.ndarray]]:
        """Extract DINOv2 features per plant crop.

        Parameters
        ----------
        image_path : Path to the full field image.
        bboxes     : List of (x1, y1, x2, y2) bounding boxes.
        plant_ids  : Corresponding plant IDs.

        Returns
        -------
        dict plant_id -> feature dict (same keys as extract_features).
        """
        full_img = Image.open(image_path).convert("RGB")
        results: dict[int, dict[str, np.ndarray]] = {}
        for pid, bbox in zip(plant_ids, bboxes):
            crop = full_img.crop(bbox)
            # Skip tiny crops that would produce degenerate features
            if crop.size[0] < _PATCH_SIZE or crop.size[1] < _PATCH_SIZE:
                continue
            results[pid] = self.extract_features(crop)
        return results

    def extract_attention_overlay(
        self,
        image: Image.Image | np.ndarray,
        target_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Extract attention map resized to match the original image dimensions.

        Parameters
        ----------
        image       : Input image.
        target_size : (height, width) to resize the attention map to.
                      Defaults to the input image dimensions.

        Returns
        -------
        (H, W) float32 attention heatmap normalised to [0, 1].
        """
        import cv2

        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            w, h = image.size

        if target_size is None:
            target_size = (h, w)

        feats = self.extract_features(image)
        attn = feats["attention_map"]

        # Resize attention map to target dimensions
        attn_resized = cv2.resize(
            attn.astype(np.float32),
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # Normalise to [0, 1]
        mn, mx = attn_resized.min(), attn_resized.max()
        if mx - mn > 1e-8:
            attn_resized = (attn_resized - mn) / (mx - mn)
        else:
            attn_resized = np.zeros_like(attn_resized)

        return attn_resized
