# Inference Engine Scale-Up Roadmap

**Hardware target:** RTX 5070 Ti — 16 GB VRAM, 896 GB/s memory bandwidth (Blackwell).

---

## Phase Status

| # | Phase | Status | Tests |
|---|---|---|---|
| 1 | LLaMA Architecture Support | ✅ Done | `tests/test_llama_smollm2.py` |
| 2 | INT4 Quantization (W4A16) | ✅ Done | `tests/test_int4_quant.py` |
| 3 | Paged KV Cache + Continuous Batching | ✅ Done | `tests/test_paged_cache.py` |
| 4 | Speculative Decoding | ✅ Done | `tests/test_speculative.py` |
| 5 | Qwen3 Dense + Tokenizer | ✅ Done | `tests/test_qwen3_dense.py` |
| 6 | Qwen3-MoE + Expert CPU Offload | ✅ Done | `tests/test_moe.py` |
| 7 | Chat Template + Full Agent Loop | ✅ Done | `tests/test_agent.py` |

Run the full suite (excluding the real-weights gate):
```bash
pytest tests/ --ignore=tests/test_llama_smollm2.py
```

---

## Completed Phases — Summary

### Phase 1 — LLaMA Architecture
Added RoPE, RMSNorm, SwiGLU, and GQA on top of the GPT-2 engine. GPT-2 code untouched.
Key files: `engine/llama_weights.py`, `llama_attention.py`, `llama_mlp.py`, `llama_block.py`, `llama_model.py`.
Correctness gate: max |Δlogit| = 6e-5 vs HuggingFace on SmolLM2-1.7B.

### Phase 2 — INT4 Quantization
Per-group symmetric INT4 (group_size=128). JIT dequant one layer at a time → peak overhead = 1 layer.
Memory: 14B bf16 = 28 GB → 14B INT4 = 7 GB (fits on 16 GB with KV budget).
Key files: `engine/quantize.py`, `engine/kernels/quant.py`.

### Phase 3 — Paged KV Cache
Fixed-size physical blocks (default 16 tokens). Sequences claim only blocks they use; blocks return to pool on completion. Eliminates KV fragmentation for long concurrent sessions.
Key files: `engine/paged_cache.py`, `engine/llama_paged_engine.py`.

### Phase 4 — Speculative Decoding
Draft K tokens with a small model → verify K+1 in one target forward. Greedy output is bit-exact identical to standard greedy. Typical gain: 2–4× throughput on chat workloads.
Key files: `engine/speculative.py`.

### Phase 5 — Qwen3 Dense + Tokenizer
Only change from LLaMA 3: per-head RMSNorm on Q and K before RoPE (`qk_norm=True`).
Added configs: `QWEN3_0_6B`, `1_7B`, `4B`, `8B`, `14B`, `32B`.
Added `QwenTokenizer` (tiktoken BPE, no HuggingFace dep).
Key files: `engine/config.py`, `llama_attention.py`, `qwen_tokenizer.py`.

```python
from engine import load_llama_weights, QWEN3_8B, LlamaModel, QwenTokenizer
weights = load_llama_weights("/path/to/Qwen3-8B", QWEN3_8B, dtype=torch.bfloat16)
model   = LlamaModel(weights, QWEN3_8B)
tok     = QwenTokenizer("/path/to/Qwen3-8B")
print(tok.decode(model.generate(tok.encode("Hello!"), 50)))
```

### Phase 6 — Qwen3-MoE + Expert CPU Offload
Qwen3-30B-A3B: 128 routed experts + 1 shared expert per layer, top-8 active per token (3B active / 30B total).
`ExpertOffloadManager` stores all expert weights as pinned CPU tensors; fetches only active experts to VRAM per token — fits a 30B MoE on 16 GB.

```
output = shared_expert(x) + Σ_k  w_k · routed_expert_k(x)
```

Added config: `QWEN3_30B_A3B` (128 experts, top-8, d_model=2048, 48 layers — verify from `config.json`).
Key files: `engine/llama_moe.py`, `engine/moe_offload.py`, `engine/config.py`, `engine/quantize.py`.

```python
from engine import load_llama_weights, QWEN3_30B_A3B, LlamaModel, ExpertOffloadManager
weights = load_llama_weights("/path/to/Qwen3-30B-A3B", QWEN3_30B_A3B, dtype=torch.bfloat16)
model   = LlamaModel(weights, QWEN3_30B_A3B)
# CPU offload: expert weights on RAM, only active K experts go to VRAM per token
mgr = ExpertOffloadManager.from_weights(weights, QWEN3_30B_A3B, device="cuda")
```

---

## Phase 7 — Chat Template + Full Agent Loop ✅ DONE

Closes the last gap between the inference engine and a production agentic system:
Qwen3 ChatML formatting → structured tool-call parsing → multi-turn agent loop.

### New files

| File | Exports |
|---|---|
| `engine/chat.py` | `format_messages(messages, tools, enable_thinking)` |
| `engine/tool_parser.py` | `ToolCall`, `extract_tool_calls`, `has_tool_call`, `strip_thinking`, `format_tool_result` |
| `engine/agent.py` | `Tool`, `AgentResult`, `AgentLoop` |

### How to use

```python
from engine import AgentLoop, Tool, QwenTokenizer, load_llama_weights, QWEN3_8B, LlamaModel
from engine.kv_cache import LlamaStaticKVCache

weights = load_llama_weights("/path/to/Qwen3-8B", QWEN3_8B, dtype=torch.bfloat16)
model   = LlamaModel(weights, QWEN3_8B)
tok     = QwenTokenizer("/path/to/Qwen3-8B")

agent = AgentLoop(
    model=model, tokenizer=tok,
    cache_factory=lambda: LlamaStaticKVCache(QWEN3_8B, batch=1, max_seq=4096, device="cuda"),
    tools=[Tool(
        name="add",
        description="Add two numbers",
        parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
        fn=lambda a, b: a + b,
    )],
    max_turns=8,
    enable_thinking=False,   # inject empty <think> block to skip CoT (faster)
)
result = agent.run([{"role": "user", "content": "What is 17 + 25?"}])
print(result.final_text)       # "The answer is 42."
print(result.n_turns)          # 2 (one tool-call turn + final-answer turn)
print(result.tool_calls_made)  # [ToolCall(name='add', arguments={'a': 17, 'b': 25})]
```

### Tests — `tests/test_agent.py` (56 tests, no real weights)

56 tests across 6 classes:
- `TestFormatMessages` (11) — ChatML role tags, tool schema injection, thinking mode
- `TestExtractToolCalls` (8) — single/multi/malformed block parsing
- `TestHasToolCall` (3) — presence detection
- `TestStripThinking` (5) — think-block removal
- `TestFormatToolResult` (4) — str/dict/number/list serialization
- `TestAgentNoToolCall` / `TestAgentSingleToolCall` / `TestAgentMultipleTurns` / `TestAgentErrorHandling` / `TestAgentResultStructure` / `TestGenerateToEos` (25) — full loop behaviour, error handling, real MockLlamaModel forward calls

```bash
pytest tests/test_agent.py -v          # 56 passed
pytest tests/ --ignore=tests/test_llama_smollm2.py   # 179 passed
```
