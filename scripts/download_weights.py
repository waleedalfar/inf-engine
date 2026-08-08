"""Download GPT-2 safetensors weights from the HuggingFace Hub into ``weights/``.

This uses ``huggingface_hub.hf_hub_download`` purely as a file fetcher — we only
pull the ``model.safetensors`` artifact. No HuggingFace model class is ever
constructed; the engine loads these files with raw torch (see engine/weights.py).

Usage:
    python scripts/download_weights.py            # downloads gpt2 + gpt2-medium
    python scripts/download_weights.py gpt2       # just one
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
MODEL_FILE = "model.safetensors"


def download(model_name: str) -> Path:
    """Fetch ``model.safetensors`` for ``model_name`` into weights/<model_name>/."""
    target_dir = WEIGHTS_DIR / model_name
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{model_name}] downloading {MODEL_FILE} ...", flush=True)
    path = hf_hub_download(
        repo_id=model_name,
        filename=MODEL_FILE,
        local_dir=str(target_dir),
    )
    size_mb = Path(path).stat().st_size / (1024 * 1024)
    print(f"[{model_name}] -> {path} ({size_mb:.1f} MB)", flush=True)
    return Path(path)


if __name__ == "__main__":
    models = sys.argv[1:] or ["gpt2", "gpt2-medium"]
    for m in models:
        download(m)
