# Qwen3 Inference Engine

A Qwen3-8B interactive coding assistant built on a **from-scratch LLM inference engine** —
no HuggingFace model classes, no vLLM, no TensorRT. Every layer, kernel, and memory layout
is implemented in raw PyTorch and Triton so every byte and every flop is accounted for.

---

## Quick start

```bash
# Download Qwen3-8B (runs quantized INT4 by default — ~4 GB VRAM)
.venv/bin/python scripts/download_llama.py qwen3-8b

# Launch the interactive coding assistant
.venv/bin/python main.py --model-dir weights/Qwen--Qwen3-8B

# Review past conversations
.venv/bin/python main.py --show-history        # all sessions
.venv/bin/python main.py --show-history 5      # last 5 sessions
```

In-chat commands: `/clear` `/tools` `/history` `/exit`

---

## What makes it fast

### Fused INT4 W4A16 Triton kernel

The core bottleneck in transformer decode is weight streaming — every token reads the
entire model from HBM. The fused kernel (`engine/kernels/quant.py`) eliminates the
dequantize-then-matmul pattern:

| Path | HBM reads/token (8B) | Extra |
|------|---------------------:|------:|
| bf16 matmul | 16 GB | 1× |
| INT4 dequant → bf16 → matmul | 36 GB | 2.25× excess |
| **Fused INT4 kernel (this engine)** | **~16 GB** | **1×** |

The kernel unpacks nibbles on-the-fly inside a Triton tile, applies per-group float32
scale, and accumulates directly into float32 — the intermediate bf16 weight tensor
never touches HBM.

Packing layout: `packed[k//2, n]` encodes two int4 weights per byte (high nibble =
even row, low nibble = odd row). Decode uses arithmetic right-shift sign extension:
`high = p >> 4`, `low = ((p & 0x0F).to(int8) << 4) >> 4`.

### FlashAttention2 prefill (SDPA)

`engine/llama_attention.py` dispatches all attention through `F.scaled_dot_product_attention`,
which selects FlashAttention2 on sm_80+ GPUs. FA2 eliminates the O(N²) score matrix:
at 8K context, the naive score tensor is `8192² × 32 heads × 2 bytes ≈ 4 GB` — FA2
uses O(N) HBM instead. Single-token decode also goes through SDPA (FA2 decode mode),
ensuring speculative decoding's verify and decode steps use the same kernel, preserving
the mathematical correctness guarantee.

Add `--compile` to wrap `model.forward` with `torch.compile(mode='reduce-overhead')`
for an additional 10–30% decode speedup after a one-time ~60s compilation.

### Paged KV cache

`PagedLlamaKVCache` allocates KV memory in fixed-size blocks (like virtual memory pages)
so sequences of different lengths can share the same physical pool without padding or
fragmentation. `LlamaPagedEngine` runs continuous batching on top: finished sequences
evict their blocks every step, and queued requests fill the freed slots immediately.

**Result:** 1.24× throughput over static batching on variable-length workloads, and O(1)
memory overhead per step regardless of batch size.

### Speculative decoding

`SpeculativeDecoder` runs a small draft model K steps ahead, then verifies all K tokens
in a single target-model forward pass. When the draft is right, you get K tokens for
the cost of ~1. Typical acceptance rate 70–90% on coding tasks → **2–4× effective tok/s**.

### GQA-aware KV cache

Qwen3-8B uses 32 query heads but only 8 KV heads (GQA). `LlamaStaticKVCache` stores only
the 8 KV heads — 4× less KV memory than a naive MHA cache.

---

## Architecture

### Agentic loop (`main.py` + `engine/agent.py`)

`VerboseAgentLoop` drives multi-turn tool-calling:

1. Format full conversation history as Qwen3 ChatML
2. Generate tokens (streaming to stdout)
3. Detect `<tool_call>` blocks → dispatch registered Python functions
4. Inject tool results as `tool` messages → repeat

Tools available to the model: `read_file`, `write_file`, `list_dir`, `search_files`,
`run_shell`, `run_python`. All sandboxed to the `--workspace` directory.

Sessions are appended to `~/.qwen3_history.jsonl` on exit.

### INT4 quantization dispatch

```
linear(x, weight)
  └─ hasattr(weight, "fused_linear")  → _Int4Weight.fused_linear(x)
                                            └─ int4_matmul() → Triton kernel
  └─ else                             → x @ weight.T
```

`quantize_llama()` replaces all projection weights (q/k/v/o/gate/up/down) with
`_Int4Weight` objects in-place. The attention, MLP, and block code never change —
the dispatch is transparent.

For Qwen3-8B (d_model=4096): weights are quantized directly on GPU (bf16 fits in 16 GB).

### Model loading

```
detect_config(model_dir)     # parse Qwen3-8B / 14B / etc. from directory name
load_llama_weights()         # sharded safetensors, no HuggingFace
quantize_llama()             # INT4 W4A16, group_size=128
```

Auto-quantize is enabled for any dense model with d_model ≥ 4096. Pass `--no-quantize`
to run in bf16 (requires ~16 GB free VRAM for 8B).

---

## Setup

Requires **WSL2 Ubuntu** with the repo on the Linux filesystem (Triton JIT on Blackwell
sm_120 needs the Linux kernel's `perf_event_open` and CPython headers).

```bash
python3 -m venv --without-pip .venv
curl -fsSL https://bootstrap.pypa.io/get-pip.py | ./.venv/bin/python
./.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
./.venv/bin/pip install -r requirements.txt
```

Verify CUDA:
```bash
./.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"
# → True (12, 0)
```

Download weights:
```bash
# Qwen3-8B (~16 GB bf16, ~4 GB INT4 after quant)
./.venv/bin/python scripts/download_llama.py qwen3-8b

# Other supported sizes
./.venv/bin/python scripts/download_llama.py qwen3-0.6b   # 0.6B, good for testing
./.venv/bin/python scripts/download_llama.py qwen3-4b
./.venv/bin/python scripts/download_llama.py qwen3-14b    # requires CPU-first quant path
```

Run tests:
```bash
./.venv/bin/pytest tests/ -v   # 165 passed, 6 skipped
```

---

## File layout

```
engine/
  config.py              LlamaConfig dataclass + Qwen3 presets (0.6B–32B + MoE)
  layers.py              rms_norm, silu, linear (INT4-dispatch), RoPE
  llama_weights.py       sharded safetensors loader
  llama_attention.py     GQA + RoPE + optional QK-norm (Qwen3)
  llama_mlp.py           SwiGLU MLP
  llama_block.py         pre-norm block (dense or lazy MoE import)
  llama_model.py         LlamaModel + generate_cached
  llama_moe.py           Qwen3-MoE router + expert dispatch
  llama_moe_model.py     MoE weight loader
  moe_offload.py         ExpertOffloadManager (pinned CPU → VRAM)
  disk_expert_manager.py DiskExpertManager (stream experts from safetensors)
  kv_cache.py            LlamaStaticKVCache (GQA-aware, pre-allocated)
  paged_cache.py         BlockManager + PagedLlamaKVCache
  llama_paged_engine.py  LlamaPagedEngine (continuous batching)
  speculative.py         SpeculativeDecoder + SpecStats
  quantize.py            INT4 W4A16 — _Int4Weight, quantize_llama, quantized_to_device
  qwen_tokenizer.py      tiktoken BPE tokenizer (no HuggingFace)
  chat.py                format_messages() — Qwen3 ChatML
  tool_parser.py         extract_tool_calls(), strip_thinking()
  agent.py               AgentLoop, Tool, AgentResult
  sampling.py            greedy / top-k / top-p
  kernels/
    softmax.py           Triton fused softmax
    flash_attention.py   Triton Flash-Attention (tiled online softmax, O(N) memory)
    quant.py             Triton INT8 W8A16 + fused INT4 W4A16 matmul kernel

scripts/
  download_llama.py      fetch Qwen3 checkpoints by shorthand name
  generate_llama.py      quick non-interactive generation script
  inspect_model.py       print weight shapes + dtypes for a loaded checkpoint

tests/
  test_llama_forward.py  LLaMA forward pass correctness
  test_int4_quant.py     INT4 quantization + fused kernel (8 tests)
  test_paged_cache.py    paged KV cache + continuous batching
  test_speculative.py    speculative decoding correctness
  test_qwen3_dense.py    Qwen3 dense model + tokenizer
  test_moe.py            Qwen3-MoE dispatch + expert offload
  test_agent.py          chat template + tool parser + agent loop (56 tests)
  test_phase5_kernels.py Triton kernel correctness

main.py                  interactive coding assistant (VerboseAgentLoop + tools)
```

---

## Implementation phases (reference)

| # | What was built |
|---|---|
| 1 | GPT-2 forward pass, KV cache, static/continuous batching, Triton kernels |
| 2 | LLaMA: RoPE, RMSNorm, SwiGLU, GQA |
| 3 | INT4 W4A16 quantization (per-group symmetric) |
| 4 | Paged KV cache + continuous batching |
| 5 | Speculative decoding |
| 6 | Qwen3 dense + tokenizer; Qwen3-MoE + expert offload |
| 7 | Qwen3 ChatML + agentic loop + coding assistant |
| ✦ | Fused INT4 Triton kernel (this session) |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`ROADMAP.md`](ROADMAP.md) for deeper notes.
