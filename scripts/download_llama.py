"""Download LLaMA safetensors weights from the HuggingFace Hub into weights/.

Requires:
  - A HuggingFace account with access granted to the model repo.
  - Being logged in: run  huggingface-cli login  before this script.

Usage:
    python scripts/download_llama.py                     # LLaMA 3.2 1B (default)
    python scripts/download_llama.py meta-llama/Llama-3.2-3B
    python scripts/download_llama.py meta-llama/Meta-Llama-3-8B
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

# Files we don't need for inference — skip them to save disk space.
_IGNORE = [
    "*.bin",
    "*.pt",
    "original/",
    "tokenizer.model",  # we use HF tokenizer, not sentencepiece
]


def download(repo_id: str) -> Path:
    """Download all safetensors (and config/tokenizer) files for ``repo_id``."""
    target_dir = WEIGHTS_DIR / repo_id.replace("/", "--")
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{repo_id}] downloading into {target_dir} ...", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        ignore_patterns=_IGNORE,
    )
    size_mb = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file()) / 1e6
    print(f"[{repo_id}] done — {size_mb:.0f} MB total", flush=True)
    return target_dir


if __name__ == "__main__":
    # Default: SmolLM2-1.7B (Apache 2.0, no login required).
    # Pass a model id to download a different one:
    #   python scripts/download_llama.py meta-llama/Llama-3.2-1B
    models = sys.argv[1:] or ["HuggingFaceTB/SmolLM2-1.7B"]
    for m in models:
        download(m)
