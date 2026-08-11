# Architecture

---

## Overview

The engine is a pure-function transformer stack: every forward pass is a deterministic function
of token ids and loaded weights. There is no `nn.Module` graph, no hidden state, and no framework
abstraction layer. The only mutable state — the KV cache — is layered on top explicitly.

The engine started as a GPT-2 implementation and was extended through six phases to support
LLaMA-family dense models, Qwen3 (dense + MoE), INT4 quantization, paged memory, speculative
decoding, and (Phase 7) a full agentic loop with structured tool calling.

---

## Data flow — dense decode

```
text ──► QwenTokenizer / GPT2Tokenizer ──► input_ids (B, T)
                                                │
         model.safetensors ──► load_*_weights() ──► LlamaWeights / QuantizedLlamaWeights
                                                │
                                         LlamaModel.forward(input_ids, cache, start_pos)
                                                │
         embed_tokens[input_ids]                            # (B, T, d_model)
                │
         for i in 0..n_layer-1:  llama_block
                │   ├─ x = x + llama_attention(rms_norm(x))   # RoPE, GQA, KV cache
                │   └─ x = x + swiglu_mlp(rms_norm(x))        # 3-weight SwiGLU
                │              OR moe_mlp(rms_norm(x))         # Qwen3-MoE path
                │
         rms_norm (final)                                   # (B, T, d_model)
                │
         logits = x @ lm_head.T                            # (B, T, vocab_size)
                │
         sample_next_token (greedy / top-k / top-p)        # (B, 1)
```

---

## Module map

### Core GPT-2 layer (original engine, still present)

| File | Responsibility |
|---|---|
| `engine/config.py` | `GPT2Config`, `LlamaConfig` — all model dims in one place |
| `engine/weights.py` | GPT-2 safetensors → named tensors (Conv1D layout) |
| `engine/tokenizer.py` | GPT-2 BPE via tiktoken |
| `engine/layers.py` | `rms_norm`, `silu`, `linear`, `gelu`, `conv1d`, `precompute_rope_freqs`, `apply_rope` |
| `engine/attention.py` | GPT-2 multi-head attention |
| `engine/mlp.py` | GPT-2 GELU MLP |
| `engine/block.py` | GPT-2 pre-norm transformer block |
| `engine/model.py` | GPT-2 full model + `generate` / `generate_cached` |
| `engine/sampling.py` | Greedy / top-k / top-p — shared by GPT-2 and LLaMA |
| `engine/kv_cache.py` | `LlamaStaticKVCache`, `DynamicKVCache`, `SlotKVCache` |
| `engine/batching.py` | Static batching: left-pad, lockstep decode |
| `engine/continuous.py` | Continuous batching engine, FCFS/SJF schedulers |
| `engine/kernels/` | Triton: fused softmax, Flash-Attention fwd, INT8 W8A16 matmul |

### LLaMA / Qwen3 stack (Phases 1, 5, 6)

| File | Responsibility |
|---|---|
| `engine/llama_weights.py` | `LlamaWeights` — HF safetensors, sharded load, `layer(i)`, `expert_weights(i, eid)` |
| `engine/llama_attention.py` | GQA attention + RoPE + optional QK-norm (Qwen3) |
| `engine/llama_mlp.py` | `swiglu_mlp` — dense SwiGLU (gate×silu×up→down) |
| `engine/llama_moe.py` | `moe_mlp` — router + top-K expert dispatch + shared expert (Qwen3-MoE) |
| `engine/llama_block.py` | Pre-norm block: attention sub-layer + MLP sub-layer (dense or MoE) |
| `engine/llama_model.py` | `LlamaModel` — embeddings, block stack, final norm, `generate` / `generate_cached` |
| `engine/config.py` | `LlamaConfig` with all fields: GQA, RoPE, QK-norm, MoE params, is_moe property |
| `engine/qwen_tokenizer.py` | `QwenTokenizer` — tiktoken BPE from `qwen.tiktoken` or `tokenizer.json`, no HF dep |

### Memory + serving (Phases 2, 3, 4)

| File | Responsibility |
|---|---|
| `engine/quantize.py` | `quantize_llama` — INT4 W4A16 per-group, dequant on demand; covers dense + MoE experts |
| `engine/paged_cache.py` | `BlockManager` (physical block pool) + `PagedLlamaKVCache` (non-contiguous KV) |
| `engine/llama_paged_engine.py` | `LlamaPagedEngine` — continuous batching with paged cache |
| `engine/speculative.py` | `SpeculativeDecoder` + `SpecStats` — draft K tokens → verify K+1 in one target pass |
| `engine/moe_offload.py` | `ExpertOffloadManager` — pinned CPU expert tensors, sync VRAM fetch per token |

### Agentic layer (Phase 7 — planned)

| File | Responsibility |
|---|---|
| `engine/chat.py` | `format_messages(messages, tools)` — OpenAI-style dict list → Qwen3 ChatML string |
| `engine/tool_parser.py` | `extract_tool_calls`, `has_tool_call`, `format_tool_result` — `<tool_call>` JSON parsing |
| `engine/agent.py` | `AgentLoop`, `Tool`, `AgentResult` — multi-turn generate → parse → execute → inject loop |

---

## KV cache design

`llama_attention` and `llama_block` take an optional `cache` and `start_pos`. The causal mask
is an absolute-position comparison (`key_pos ≤ query_pos`), which reduces exactly to the
standard lower-triangular when `start_pos=0` with no cache — so the uncached path is
byte-identical to cached.

**Static cache** (`LlamaStaticKVCache`): pre-allocated `(n_layer, B, n_kv_heads, max_seq, head_dim)`.
Simple but wasteful — each sequence reserves `max_seq` slots from the start.

**Paged cache** (`PagedLlamaKVCache`): `BlockManager` manages a pool of fixed-size physical blocks
(e.g., 16 tokens each). Each sequence gets only the blocks it actually uses; blocks return to the
pool the moment a sequence completes. ≤1 block/sequence fragmentation vs full `max_seq` waste.

---

## MoE dispatch

When `config.is_moe` is True, `llama_block` calls `moe_mlp` instead of `swiglu_mlp`:

```
moe_mlp(x, weights, config):
    shared_out = swiglu_expert(x, shared_expert_weights)          # always computed
    router_w   = softmax(x @ gate.T)                              # (S, n_experts)
    topk_w, topk_ids = topk(router_w, K)                          # K = n_experts_per_tok
    topk_w = topk_w / topk_w.sum(dim=-1)                          # renormalize to 1
    routed_out = Σ_k  topk_w_k · swiglu_expert(x, expert_k_weights)
    return shared_out + routed_out
```

Expert weights for all 128 experts live in the block weight dict returned by `LlamaWeights.layer(i)`.
For large models that don't fit in VRAM, `ExpertOffloadManager` stores all expert tensors in
pinned CPU RAM and copies only the active K experts to VRAM per token.

---

## Speculative decoding

`SpeculativeDecoder(draft, target, n_draft=K)` maintains a state invariant:
"cache filled to position L; last token at position L not yet in cache."

Each step:
1. Draft: run small model K times from `last_tok` to produce K draft tokens + probability distributions
2. Verify: run target model ONCE on `[last_tok, draft_0, …, draft_{K-1}]` (K+1 tokens, one forward)
3. Accept/reject: token `x̃` accepted with prob `min(1, p_target(x̃) / p_draft(x̃))`
4. Rejection at position j: sample correction from `max(0, p_target − p_draft)/Z`; rollback caches to `L+j+1`
5. All-accept: sync draft cache by processing `draft_{K-1}` at `L+K`; emit K+bonus tokens

Greedy speculative decode is bit-exact identical to standard greedy — the correctness gate.

---

## Agentic loop (Phase 7)

The agent loop wires the inference engine into a multi-turn tool-calling system:

```
AgentLoop.run(messages):
    while turns < max_turns:
        prompt = format_messages(history, tool_schemas)    # ChatML string
        ids    = tokenizer.encode(prompt)
        out    = model.generate_cached(ids, eos=im_end_id) # extend KV cache, don't re-prefill
        text   = tokenizer.decode(out)
        if has_tool_call(text):
            calls  = extract_tool_calls(text)
            for call in calls:
                result = tool.fn(**call.arguments)         # execute registered Python fn
                history.append(format_tool_result(...))    # inject as "tool" role message
        else:
            return AgentResult(final_text=text, ...)
```

**Cache reuse across turns:** the KV cache is extended across turns rather than rebuilt from scratch.
Only the new tokens (tool result + assistant open-turn marker) are prefilled each iteration.
This is critical for long agentic sessions — re-encoding the full history each turn would be O(n²).

**Tool schema:** each `Tool` carries a JSON Schema `parameters` dict injected into the system prompt
by `format_messages`, so Qwen3 knows what arguments to pass.

---

## INT4 quantization

`quantize_llama(model, group_size=128)` replaces every linear projection with a packed INT4
(`_Int4Weight`: packed nibbles + per-group scale). `QuantizedLlamaWeights.layer(i)` dequantizes
on demand — only one layer's weights in full float at a time. For MoE models this covers all
128 × 3 expert matrices per layer plus the shared expert, giving ~4× VRAM reduction.

---

## Dependency boundary

`engine/` imports only `torch`, `safetensors`, `tiktoken`, `numpy`.
`transformers` stays in `tests/` as the correctness oracle. This is a hard project invariant.

---

## Hardware target

- **GPU:** Blackwell sm_120 — 16 GB VRAM, 896 GB/s memory bandwidth
- **Environment:** WSL2 Ubuntu on Windows 11
- **Roofline:** ~44 TFLOP/s FP32, memory-bandwidth-bound at all practical batch sizes for 7B+

**Decode throughput estimate (7B INT8, batch=1):**
7 GB weights / 896 GB/s ≈ 7.8 ms/token → ~128 tok/s

With speculative decoding (draft = Qwen3-0.6B): 2–4× on chat workloads → ~250–500 tok/s effective.
