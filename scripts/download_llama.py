"""Download LLaMA / Qwen3 safetensors weights from the HuggingFace Hub into weights/.

Requires:
  - A HuggingFace account with access granted to the model repo.
  - Being logged in: run  huggingface-cli login  before this script.

Usage:
    python scripts/download_llama.py                     # Qwen3-14B (default)
    python scripts/download_llama.py qwen3-8b
    python scripts/download_llama.py qwen3-30b-a3b
    python scripts/download_llama.py Qwen/Qwen3-4B       # full HF repo id also works

Shorthand names (case-insensitive):
    qwen3-0.6b   qwen3-1.7b   qwen3-4b    qwen3-8b
    qwen3-14b    qwen3-32b    qwen3-30b-a3b
    smollm2-1.7b
    llama-3.2-1b  llama-3.2-3b  llama-3-8b
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

# Shorthand name → HuggingFace repo id.
KNOWN_MODELS: dict[str, str] = {
    "qwen3-0.6b":    "Qwen/Qwen3-0.6B",
    "qwen3-1.7b":    "Qwen/Qwen3-1.7B",
    "qwen3-4b":      "Qwen/Qwen3-4B",
    "qwen3-8b":      "Qwen/Qwen3-8B",
    "qwen3-14b":     "Qwen/Qwen3-14B",
    "qwen3-32b":     "Qwen/Qwen3-32B",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    "smollm2-1.7b":  "HuggingFaceTB/SmolLM2-1.7B",
    "llama-3.2-1b":  "meta-llama/Llama-3.2-1B",
    "llama-3.2-3b":  "meta-llama/Llama-3.2-3B",
    "llama-3-8b":    "meta-llama/Meta-Llama-3-8B",
}


def resolve(name: str) -> str:
    """Return a full HF repo id, resolving shorthand names."""
    return KNOWN_MODELS.get(name.lower(), name)


def download(name: str) -> Path:
    """Download all safetensors (and config/tokenizer) files for ``name``."""
    repo_id = resolve(name)
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
    names = sys.argv[1:] or ["qwen3-14b"]
    for n in names:
        download(n)
