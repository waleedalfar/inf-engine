"""Phase 5 K3 benchmark: INT8 weight quantization — quality, memory, throughput.

Three questions:
  1. Quality: how much does per-column INT8 weight quant raise perplexity on held-out text?
  2. Memory: how much smaller are the quantized linear weights? (INT8 vs FP32 -> 4x.)
  3. Throughput: INT8 dequant-matmul vs torch FP16 matmul at a decode-shaped GEMM.

Usage:
    python bench/bench_int8.py --model gpt2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.config import get_config  # noqa: E402
from engine.kernels.quant import int8_matmul, quantize_weight_int8  # noqa: E402
from engine.model import GPT2Model  # noqa: E402
from engine.tokenizer import GPT2Tokenizer  # noqa: E402
from engine.weights import GPT2Weights, load_weights  # noqa: E402

# Held-out text (not used to tune anything) for the perplexity quality gate.
HELDOUT = (
    "Machine learning systems increasingly run at a scale where the bottleneck is not the "
    "mathematics of the model but the movement of bytes through the memory hierarchy. A "
    "transformer spends most of its inference time waiting on memory, not computing, and the "
    "art of fast inference is the art of not moving data you do not need to move. Quantization "
    "is one lever: by storing weights in eight bits instead of thirty-two, an engine streams a "
    "quarter of the bytes from high-bandwidth memory, and for a decoder running one token at a "
    "time that traffic is the dominant cost. The price is precision, and the question that "
    "matters is whether the lost precision changes the model's predictions enough to matter."
)
# Linear (Conv1D) weights we quantize; embeddings and LayerNorm stay full precision.
LINEAR_SUFFIXES = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
MB = 1024 * 1024


def perplexity(model: GPT2Model, ids: torch.Tensor) -> float:
    """Teacher-forced perplexity = exp(mean next-token NLL)."""
    logits = model.forward(ids)                                  # (1, T, V)
    logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)     # (T-1, V)
    targets = ids[0, 1:]                                         # (T-1,)
    nll = -logp[torch.arange(targets.shape[0]), targets].mean()
    return float(torch.exp(nll))


def quantized_weights(w: GPT2Weights) -> tuple[GPT2Weights, int, int]:
    """Return a copy with linear weights quant->dequant'd, plus (fp32_bytes, int8_bytes)."""
    new = dict(w._t)
    fp32_bytes = int8_bytes = 0
    for key, tensor in w._t.items():
        if key.endswith(LINEAR_SUFFIXES):
            wq, scale = quantize_weight_int8(tensor)
            new[key] = wq.float() * scale[None, :]               # simulate int8 round-trip
            fp32_bytes += tensor.numel() * 4
            int8_bytes += wq.numel() * 1 + scale.numel() * 4     # int8 weights + fp32 scales
    return GPT2Weights(new, w.config), fp32_bytes, int8_bytes


def _time(fn, warmup=10, runs=50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(runs):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / runs / 1e3


def run(model_name: str) -> dict:
    device = "cuda"
    config = get_config(model_name)
    wpath = REPO_ROOT / "weights" / model_name / "model.safetensors"
    weights = load_weights(wpath, config, device=device, dtype=torch.float32)
    model = GPT2Model(weights, config)
    tok = GPT2Tokenizer()
    ids = torch.tensor([tok.encode(HELDOUT)], device=device)

    ppl_fp32 = perplexity(model, ids)
    qweights, fp32_b, int8_b = quantized_weights(weights)
    ppl_int8 = perplexity(GPT2Model(qweights, config), ids)

    # throughput: decode-shaped GEMM (c_attn): M tokens x K=d_model -> N=3*d_model
    m, k, n = 16, config.d_model, 3 * config.d_model
    a = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(k, n, device=device, dtype=torch.float32) * 0.1
    wq, scale = quantize_weight_int8(w)
    w_fp16 = w.to(torch.float16)
    t_int8 = _time(lambda: int8_matmul(a, wq, scale))
    t_fp16 = _time(lambda: a @ w_fp16)

    results = {
        "model": model_name,
        "tokens": ids.shape[1],
        "ppl_fp32": ppl_fp32, "ppl_int8": ppl_int8,
        "ppl_delta_pct": 100 * (ppl_int8 - ppl_fp32) / ppl_fp32,
        "linear_fp32_mib": fp32_b / MB, "linear_int8_mib": int8_b / MB,
        "weight_mem_reduction": fp32_b / int8_b,
        "gemm_shape": [m, k, n],
        "int8_matmul_us": t_int8 * 1e6, "torch_fp16_us": t_fp16 * 1e6,
        "matmul_speedup": t_fp16 / t_int8,
    }
    print(f"perplexity: fp32={ppl_fp32:.3f}  int8={ppl_int8:.3f}  "
          f"(+{results['ppl_delta_pct']:.2f}%)")
    print(f"linear weights: fp32={fp32_b/MB:.0f} MiB  int8={int8_b/MB:.0f} MiB  "
          f"({results['weight_mem_reduction']:.2f}x smaller)")
    print(f"GEMM {m}x{k}x{n}: int8 {t_int8*1e6:.1f}us  torch fp16 {t_fp16*1e6:.1f}us  "
          f"({results['matmul_speedup']:.2f}x)")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", choices=["gpt2", "gpt2-medium"])
    args = parser.parse_args()
    torch.manual_seed(0)
    results = run(args.model)
    out_dir = REPO_ROOT / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"int8_{args.model}.json").write_text(json.dumps(results, indent=2))
