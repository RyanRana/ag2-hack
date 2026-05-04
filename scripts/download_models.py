"""Pre-cache the HuggingFace models Pesto uses.

Run once before the demo so first-image latency doesn't include downloads.

    python scripts/download_models.py
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download


WEED_REPO = "foduucom/plant-leaf-detection-and-classification"
DISEASE_REPO = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
HEALTH_REPO = "Diginsa/Plant-Disease-Detection-Project"


def main() -> None:
    print(f"[download] {WEED_REPO}/best.pt")
    p = hf_hub_download(repo_id=WEED_REPO, filename="best.pt")
    print(f"   -> {Path(p).resolve()}")

    print(f"[download] {DISEASE_REPO}")
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    AutoImageProcessor.from_pretrained(DISEASE_REPO)
    AutoModelForImageClassification.from_pretrained(DISEASE_REPO)

    print(f"[download] {HEALTH_REPO}")
    AutoImageProcessor.from_pretrained(HEALTH_REPO)
    AutoModelForImageClassification.from_pretrained(HEALTH_REPO)

    print("[done] all model weights cached.")


if __name__ == "__main__":
    main()
