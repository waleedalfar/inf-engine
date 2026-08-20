# CLI Reference

This project has two entry points: `main.py` for a single interactive session in your
terminal, and `server.py` for serving many concurrent sessions over HTTP with continuous
batching. See [`main.py`](#cli-reference--mainpy) below, or jump to
[`server.py`](#cli-reference--serverpy).

# CLI Reference — `main.py`

```
python main.py [OPTIONS]
```

Interactive Qwen3 coding assistant. The model streams responses token-by-token,
calls tools (read/write/run) autonomously, and saves each session to a history file.

---

## Required

| Flag | Description |
|------|-------------|
| `--model-dir PATH` | Directory containing model weights and tokenizer. Must include safetensors files and either `qwen.tiktoken` or `tokenizer.json`. Not required when using `--show-history`. |

---

## Model & inference

| Flag | Default | Description |
|------|---------|-------------|
| `--device {cuda,cpu}` | `cuda` if available | Device to run inference on. |
| `--quantize` | auto | Force INT4 W4A16 quantization on load. Auto-enabled for dense models with `d_model ≥ 4096` (8B, 14B, 32B). |
| `--no-quantize` | — | Disable auto-quantization. Runs in bf16. Requires ~16 GB free VRAM for 8B. |
| `--max-new-tokens N` | `1024` | Maximum tokens the model may generate per agent turn. Includes tool-call responses. |
| `--max-ctx N` | `8192` | KV cache size in tokens. Qwen3 supports up to 32768. Larger values use more VRAM (~0.5 GB per 4096 tokens for Qwen3-8B INT4). When a conversation exceeds this, the oldest turns are silently dropped. |
| `--thinking` | off | Enable Qwen3 chain-of-thought (`<think>` blocks). Produces more accurate answers at the cost of extra tokens and latency. Off by default. |
| `--compile` | off | Wrap `model.forward` with `torch.compile(mode='reduce-overhead')`. The first response takes ~60s to compile; subsequent calls are 10–30% faster. Recommended for long interactive sessions. |
| `--repetition-penalty N` | `1.1` | Flat, one-time penalty applied to tokens seen in the last 256 generated tokens (llama.cpp-style `repeat_penalty` + `repeat_last_n`). 1.0 = off. Values 1.05–1.3 cover mild to strong suppression. Only counts generated tokens, not the prompt. |

---

## Speculative decoding

Run a small draft model in parallel to predict tokens; the large model verifies them in batches. Typically gives **2–3× throughput** on coding tasks.

| Flag | Default | Description |
|------|---------|-------------|
| `--draft-model-dir PATH` | off | Path to a small draft model (e.g. `weights/Qwen--Qwen3-0.6B`). Enables speculative decoding when set. |
| `--n-draft N` | `4` | Tokens the draft model speculates per step. Higher values help when the draft is accurate; lower values are safer when topics vary. |

```bash
# Example: 0.6B draft + 8B target
python main.py --model-dir weights/Qwen--Qwen3-8B \
               --draft-model-dir weights/Qwen--Qwen3-0.6B \
               --n-draft 4
```

Each response prints: `[N tokens, X.XX tok/s | spec accept YY%, Z.Z tok/step]`

---

## Agent & workspace

| Flag | Default | Description |
|------|---------|-------------|
| `--workspace PATH` | `.` (current dir) | Root directory the agent can read, write, and run shell commands in. All tool paths are sandboxed to this directory — attempts to escape with `../` are blocked. |
| `--system TEXT` | built-in | Override the system prompt. The default instructs the model to use tools without asking, keep responses short, and verify work after writing files. |
| `--max-turns N` | `8` | Maximum tool-call iterations per user message before the agent gives up and returns its last response. |

---

## MoE expert offload (Qwen3-30B-A3B only)

| Flag | Default | Description |
|------|---------|-------------|
| `--expert-offload {disk,ram}` | `disk` | How to store the 128 routed experts. `disk` streams expert weights from safetensors on each forward (slow first token, low RAM). `ram` loads all experts into pinned CPU RAM upfront (fast, but needs ~57 GB RAM). |
| `--cache-mb MB` | `2000` | VRAM budget for the expert cache in MB. Recently used experts are kept in VRAM to avoid re-loading. 1800 MB covers all 8 active experts for one full forward pass. |

---

## Session history

| Flag | Default | Description |
|------|---------|-------------|
| `--history-file PATH` | `~/.qwen3_history.jsonl` | File where sessions are appended on exit. Each line is one JSON session record containing timestamp, model name, workspace, and all user/assistant messages. |
| `--show-history [N]` | — | Print past sessions and exit immediately (no model loaded). Omit `N` for all sessions; pass `N` to show only the last N. |

```bash
python main.py --show-history        # all past sessions
python main.py --show-history 5      # last 5 sessions
python main.py --show-history --history-file ~/work/history.jsonl
```

---

## In-chat commands

Once the assistant is running, type these at the `You:` prompt:

| Command | Description |
|---------|-------------|
| `/history` | Print all messages in the current session (user + assistant, first 200 chars each). |
| `/clear` | Wipe the current conversation and start fresh. Keeps the system prompt. |
| `/tools` | List the tools available to the model with their descriptions. |
| `/exit` | Quit and save the session to history. Also triggered by `Ctrl+C` or `Ctrl+D`. |

---

## Common examples

```bash
# Minimal: 8B model, current directory as workspace
python main.py --model-dir weights/Qwen--Qwen3-8B

# Point at a specific project
python main.py --model-dir weights/Qwen--Qwen3-8B \
               --workspace ~/myproject

# Longer context window (uses more VRAM)
python main.py --model-dir weights/Qwen--Qwen3-8B \
               --max-ctx 16384

# Faster output with speculative decoding
python main.py --model-dir weights/Qwen--Qwen3-8B \
               --draft-model-dir weights/Qwen--Qwen3-0.6B

# Enable chain-of-thought for hard problems
python main.py --model-dir weights/Qwen--Qwen3-8B \
               --thinking --max-new-tokens 4096

# Run in bf16 (no quantization, needs ~16 GB free VRAM)
python main.py --model-dir weights/Qwen--Qwen3-8B \
               --no-quantize

# Custom system prompt
python main.py --model-dir weights/Qwen--Qwen3-8B \
               --system "You are a Rust expert. Always write idiomatic Rust."

# Review what you asked the model last week
python main.py --show-history 10
```

---

# CLI Reference — `server.py`

```
python server.py --model-dir PATH [OPTIONS]
```

Multi-session HTTP server. One loaded model, one background "engine thread" running
`LlamaPagedEngine` with continuous batching (`engine/paged_session.py`) — concurrent HTTP
conversations get their decode steps batched together instead of running one-at-a-time like
`main.py`'s single-session `AgentLoop`.

## Required

| Flag | Description |
|------|-------------|
| `--model-dir PATH` | Directory containing model weights and tokenizer (same requirement as `main.py`). |

## Model & inference

| Flag | Default | Description |
|------|---------|-------------|
| `--workspace PATH` | `.` | Root directory tools can read/write/run in (sandboxed, shared by all sessions). |
| `--host` | `0.0.0.0` | Bind address. |
| `--port` | `8000` | Bind port. |
| `--device {cuda,cpu}` | `cuda` if available | Device to run inference on. |
| `--quantize` / `--no-quantize` | auto | Same auto-detect/override behavior as `main.py`'s flags. |
| `--max-ctx N` | `8192` | Per-session context budget in tokens. |
| `--max-new-tokens N` | `1024` | Max tokens generated per agent turn, per session. |
| `--max-turns N` | `8` | Max tool-call iterations per user message before a session gives up. |
| `--thinking` | off | Enable Qwen3 `<think>` blocks. |
| `--repetition-penalty N` | `1.1` | Same as `main.py`'s flag. |

## KV pool (shared across all concurrent sessions)

| Flag | Default | Description |
|------|---------|-------------|
| `--n-total-blocks N` | `4096` | Physical KV blocks in the pool. With the default `--block-size`, that's 65536 tokens of total capacity shared across every active session. |
| `--block-size N` | `16` | Tokens per physical block. |

```bash
python server.py --model-dir weights/Qwen--Qwen3-8B --port 8000

curl -N -X POST http://localhost:8000/v1/chat \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "list the files here"}]}'
```

Response body is NDJSON (one JSON object per line): a `session` event first (echoes the
session id — pass it back as `"session_id"` in a follow-up request to continue the
conversation), then `token` events as text streams in, then either a terminal `done` or
`error` event.

`GET /healthz` returns `{"status": "loading"}` before the model finishes loading, then
`{"status": "ok", "active_sessions": N}`.
