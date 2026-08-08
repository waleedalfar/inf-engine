"""Load GPT-2 weights directly from HuggingFace safetensors with raw torch.

We never instantiate a HuggingFace model class. We open the ``.safetensors``
file ourselves and pull named tensors into a plain dict.

Two GPT-2-specific facts a reader must know to follow the rest of the engine:

1. **Conv1D, not Linear.** GPT-2's projections use HuggingFace's ``Conv1D`` layer,
   which stores its weight as ``(d_in, d_out)`` and computes ``y = x @ W + b``.
   This is the *transpose* of ``nn.Linear`` (which stores ``(d_out, d_in)`` and
   computes ``y = x @ Wᵀ + b``). We therefore matmul with the weight *as stored*,
   with no transpose. Getting this wrong silently corrupts every projection and
   the correctness gate will fail.

2. **No causal-mask tensor in the file.** The lower-triangular mask GPT-2 uses is
   a non-persistent buffer, so it is absent from the checkpoint. We build it
   ourselves in ``attention.py``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from engine.config import GPT2Config


class GPT2Weights:
    """Named GPT-2 tensors loaded from safetensors, kept on one device/dtype.

    All tensors are returned exactly as stored on disk (no transposes), so the
    Conv1D weights are ``(d_in, d_out)`` and are used as ``x @ W + b``.
    """

    def __init__(self, tensors: dict[str, torch.Tensor], config: GPT2Config) -> None:
        self._t = tensors
        self.config = config

    # -- whole-model tensors ------------------------------------------------
    @property
    def wte(self) -> torch.Tensor:
        """Token embedding table. Shape: (vocab_size, d_model)."""
        return self._t["wte.weight"]

    @property
    def wpe(self) -> torch.Tensor:
        """Positional embedding table. Shape: (n_ctx, d_model)."""
        return self._t["wpe.weight"]

    @property
    def ln_f_weight(self) -> torch.Tensor:
        """Final LayerNorm gain. Shape: (d_model,)."""
        return self._t["ln_f.weight"]

    @property
    def ln_f_bias(self) -> torch.Tensor:
        """Final LayerNorm bias. Shape: (d_model,)."""
        return self._t["ln_f.bias"]

    # -- per-block tensors --------------------------------------------------
    def block(self, i: int) -> dict[str, torch.Tensor]:
        """Return the tensors for transformer block ``i``.

        Keys and shapes (d=d_model, f=d_mlp=4d):
            ln_1.{weight,bias}      (d,)
            attn.c_attn.weight      (d, 3d)     -> used as x @ W + b
            attn.c_attn.bias        (3d,)
            attn.c_proj.weight      (d, d)
            attn.c_proj.bias        (d,)
            ln_2.{weight,bias}      (d,)
            mlp.c_fc.weight         (d, f)
            mlp.c_fc.bias           (f,)
            mlp.c_proj.weight       (f, d)
            mlp.c_proj.bias         (d,)
        """
        p = f"h.{i}."
        return {k[len(p):]: v for k, v in self._t.items() if k.startswith(p)}


def load_weights(
    safetensors_path: str | Path,
    config: GPT2Config,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> GPT2Weights:
    """Load all GPT-2 tensors from a safetensors file into ``GPT2Weights``.

    Args:
        safetensors_path: Path to ``model.safetensors``.
        config: Model config the file is expected to match.
        device: Target device for the tensors (e.g. "cuda", "cpu").
        dtype: Target dtype. fp32 for the correctness gate to match HF defaults.

    Returns:
        A ``GPT2Weights`` holding every tensor on ``device`` with ``dtype``.

    Raises:
        FileNotFoundError: if the path does not exist.
        KeyError: if an expected top-level tensor is missing.
    """
    path = Path(safetensors_path)
    if not path.is_file():
        raise FileNotFoundError(f"weights not found: {path}")

    tensors: dict[str, torch.Tensor] = {}
    # safe_open memory-maps the file; we copy each slice to the target device/dtype.
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118 (safetensors handle has no .items)
            tensors[key] = f.get_tensor(key).to(device=device, dtype=dtype)

    for required in ("wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"):
        if required not in tensors:
            raise KeyError(f"missing tensor '{required}' in {path}")

    return GPT2Weights(tensors, config)
