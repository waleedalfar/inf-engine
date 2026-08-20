"""Qwen3 interactive coding assistant using the custom inference engine.

Usage:
    python main.py --model-dir weights/Qwen--Qwen3-8B
    python main.py --model-dir weights/Qwen--Qwen3-8B --workspace ~/myproject
    python main.py --model-dir weights/Qwen--Qwen3-8B --system "You are a senior Python engineer."
    python main.py --show-history          # print all past sessions and exit
    python main.py --show-history 5        # print last 5 sessions and exit

In-chat commands:
    /clear    Reset conversation (keeps system prompt)
    /tools    List available tools
    /history  Print current session so far
    /exit     Quit (also Ctrl+C or Ctrl+D)

Sessions are appended to ~/.qwen3_history.jsonl on exit.

The model directory must contain either:
  - tokenizer.json (HuggingFace format)  OR  qwen.tiktoken
  - model.safetensors  OR  model-*.safetensors shards + model.safetensors.index.json
"""

import argparse
import datetime
import json
import os
import re
import subprocess
from pathlib import Path
import torch

from engine.config import (
    QWEN3_0_6B, QWEN3_1_7B, QWEN3_4B, QWEN3_8B, QWEN3_14B, QWEN3_32B,
    QWEN3_30B_A3B, LlamaConfig,
)
from engine.kv_cache import LlamaStaticKVCache
from engine.llama_model import LlamaModel
from engine.llama_moe_model import load_moe_weights, load_moe_weights_disk
from engine.llama_weights import load_llama_weights
from engine.quantize import quantize_llama, quantized_to_device
from engine.qwen_tokenizer import QwenTokenizer
from engine.agent import AgentLoop, Tool
from engine.sampling import SamplingConfig, sample_next_token
from engine.speculative import SpeculativeDecoder

# ---------------------------------------------------------------------------
# Session history helpers
# ---------------------------------------------------------------------------

_DEFAULT_HISTORY = Path("~/.qwen3_history.jsonl")


def _save_session(
    history_file: Path,
    model_name: str,
    workspace: Path,
    messages: list[dict],
) -> None:
    """Append the completed session to the history JSONL file."""
    convo = [m for m in messages if m["role"] in ("user", "assistant")]
    if not convo:
        return
    record = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "workspace": str(workspace),
        "messages": convo,
    }
    try:
        history_file.parent.mkdir(parents=True, exist_ok=True)
        with history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


def _show_history(history_file: Path, last_n: int | None = None) -> None:
    """Print past sessions from the history file."""
    if not history_file.exists():
        print("No history yet.")
        return
    sessions = []
    with history_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    sessions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not sessions:
        print("No history yet.")
        return
    if last_n is not None:
        sessions = sessions[-last_n:]
    for s in sessions:
        print(f"\n=== {s.get('ts', '?')}  |  {s.get('model', '?')}  |  {s.get('workspace', '?')} ===")
        for m in s.get("messages", []):
            role = m["role"]
            content = m.get("content", "")
            if role == "assistant":
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            preview = content[:300].replace("\n", " ")
            if len(content) > 300:
                preview += "..."
            label = "You:   " if role == "user" else "Agent: "
            print(f"  {label}{preview}")
    print()


# Maps directory-name fragments → config, largest first so "Qwen3-14B"
# doesn't accidentally match before "Qwen3-1.7B".
_DIR_TO_CONFIG: list[tuple[str, LlamaConfig]] = [
    ("Qwen3-30B-A3B", QWEN3_30B_A3B),
    ("Qwen3-32B",     QWEN3_32B),
    ("Qwen3-14B",     QWEN3_14B),
    ("Qwen3-8B",      QWEN3_8B),
    ("Qwen3-4B",      QWEN3_4B),
    ("Qwen3-1.7B",    QWEN3_1_7B),
    ("Qwen3-0.6B",    QWEN3_0_6B),
]


def detect_config(model_dir: str) -> LlamaConfig:
    name = Path(model_dir).name
    for fragment, cfg in _DIR_TO_CONFIG:
        if fragment.lower() in name.lower():
            print(f"Auto-detected config: {cfg.name}")
            return cfg
    raise ValueError(
        f"Cannot detect Qwen3 config from directory name '{name}'.\n"
        f"Known fragments: {[f for f, _ in _DIR_TO_CONFIG]}\n"
        "Rename the directory or add a case to _DIR_TO_CONFIG in main.py."
    )


# ---------------------------------------------------------------------------
# Sandbox helper
# ---------------------------------------------------------------------------

def _resolve_safe(workspace: Path, path: str) -> Path:
    """Resolve path relative to workspace, raise if it escapes the sandbox."""
    resolved = (workspace / path).resolve()
    if not str(resolved).startswith(str(workspace.resolve())):
        raise PermissionError(f"Path '{path}' escapes the workspace. Access denied.")
    return resolved


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def make_tools(workspace: Path) -> list[Tool]:
    """Build the tool list, closing over the workspace path."""

    def read_file(path: str) -> str:
        try:
            return _resolve_safe(workspace, path).read_text(encoding="utf-8")
        except PermissionError as e:
            return f"Error: {e}"
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        except Exception as e:
            return f"Error: {e}"

    def write_file(path: str, content: str) -> str:
        try:
            target = _resolve_safe(workspace, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Written {len(content)} chars to {path}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"

    def list_dir(path: str = ".") -> str:
        try:
            target = _resolve_safe(workspace, path)
            if not target.is_dir():
                return f"Error: not a directory: {path}"
            entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
            lines = []
            for entry in entries:
                tag = "" if entry.is_dir() else ""
                lines.append(f"{tag} {entry.name}")
            return "\n".join(lines) if lines else "(empty directory)"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"

    def search_files(pattern: str, path: str = ".") -> str:
        try:
            target = _resolve_safe(workspace, path)
            result = subprocess.run(
                ["grep", "-rn", "--include=*", pattern, str(target)],
                capture_output=True, text=True, timeout=15,
            )
            output = result.stdout.strip()
            # Strip the workspace prefix from paths for cleaner output.
            ws_str = str(workspace.resolve()) + "/"
            output = output.replace(ws_str, "")
            return output if output else "(no matches)"
        except PermissionError as e:
            return f"Error: {e}"
        except subprocess.TimeoutExpired:
            return "Error: search timed out"
        except Exception as e:
            return f"Error: {e}"

    def run_shell(command: str) -> str:
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,  # noqa: S602
                cwd=str(workspace), timeout=60,
            )
            out = result.stdout.strip()
            err = result.stderr.strip()
            parts = []
            if out:
                parts.append(out)
            if err:
                parts.append(f"[stderr]\n{err}")
            if not parts:
                parts.append(f"(exit code {result.returncode})")
            return "\n".join(parts)
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 60s"
        except Exception as e:
            return f"Error: {e}"

    def run_python(code: str) -> str:
        """Execute a Python snippet inline and return stdout."""
        import io, contextlib, traceback
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<agent>", "exec"), {})  # noqa: S102
            return buf.getvalue() or "(no output)"
        except Exception:
            return traceback.format_exc()

    return [
        Tool(
            name="read_file",
            description="Read the contents of a file in the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace root."},
                },
                "required": ["path"],
            },
            fn=read_file,
        ),
        Tool(
            name="write_file",
            description="Write (create or overwrite) a file in the workspace. Creates parent directories as needed.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace root."},
                    "content": {"type": "string", "description": "Full content to write."},
                },
                "required": ["path", "content"],
            },
            fn=write_file,
        ),
        Tool(
            name="list_dir",
            description="List files and subdirectories inside a workspace directory.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path relative to workspace root. Defaults to '.' (root)."},
                },
                "required": [],
            },
            fn=list_dir,
        ),
        Tool(
            name="search_files",
            description="Search for a text pattern across files in the workspace (grep).",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text or regex pattern to search for."},
                    "path": {"type": "string", "description": "Directory to search in, relative to workspace root. Defaults to '.' (all files)."},
                },
                "required": ["pattern"],
            },
            fn=search_files,
        ),
        Tool(
            name="run_shell",
            description="Run a shell command inside the workspace directory. Use this to run tests, install packages, execute scripts, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                },
                "required": ["command"],
            },
            fn=run_shell,
        ),
        Tool(
            name="run_python",
            description="Execute a Python code snippet inline and return its output. Useful for quick calculations or logic checks without creating a file.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source code to execute."},
                },
                "required": ["code"],
            },
            fn=run_python,
        ),
    ]


# ---------------------------------------------------------------------------
# Verbose agent subclass — prints tool calls as they happen
# ---------------------------------------------------------------------------

class VerboseAgentLoop(AgentLoop):
    def _execute_tool(self, call):
        print(f"\n  → [tool: {call.name}]", flush=True)
        return super()._execute_tool(call)

    def _generate_to_eos(self, ids, cache):
        """Stream each token to stdout as it is produced."""
        device = self.device
        n = len(ids)

        ids_t = torch.tensor([ids], device=device)
        pos_t = torch.arange(n, device=device)

        logits = self.model.forward(ids_t, cache=cache, start_pos=0, position_ids=pos_t)
        next_tok = sample_next_token(logits[:, -1:, :], self.sampling).item()

        generated: list[int] = [next_tok]
        pos = n

        # Reset cache stats so we measure per-generation hit rate.
        expert_cache = self._get_expert_cache()
        if expert_cache is not None:
            expert_cache.reset_stats()

        import time
        print("Agent: ", end="", flush=True)
        t0 = time.perf_counter()

        while next_tok != self.eos_token_id and len(generated) < self.max_new_tokens:
            piece = self.tokenizer.decode([next_tok], skip_special_tokens=True)
            print(piece, end="", flush=True)

            tok_t = torch.tensor([[next_tok]], device=device)
            pos_s = torch.tensor([pos], device=device)
            logits = self.model.forward(tok_t, cache=cache, start_pos=pos, position_ids=pos_s)
            pos += 1
            next_tok = sample_next_token(logits[:, -1:, :], self.sampling).item()
            generated.append(next_tok)

        elapsed = time.perf_counter() - t0
        n_gen = len(generated)
        tps = n_gen / elapsed if elapsed > 0 else 0
        cache_info = ""
        if expert_cache is not None and (expert_cache.hits + expert_cache.misses) > 0:
            cache_info = (
                f"  expert cache: {expert_cache.hit_rate:.0%} hit rate "
                f"({expert_cache.hits}H/{expert_cache.misses}M, "
                f"{expert_cache.used_mb:.0f} MB)"
            )
        print(f"\n  [{n_gen} tokens, {tps:.2f} tok/s]{cache_info}", flush=True)
        return generated

    def _get_expert_cache(self):
        try:
            return getattr(self.model.offload_mgr, "_cache", None)
        except AttributeError:
            return None


class SpeculativeVerboseAgentLoop(VerboseAgentLoop):
    """VerboseAgentLoop that uses a small draft model to speed up generation.

    Wraps SpeculativeDecoder so each generate call runs draft-then-verify
    instead of single-token decoding. Prints acceptance rate with tok/s.
    """

    def __init__(self, *args, spec_decoder: SpeculativeDecoder, draft_cache_factory, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec_decoder = spec_decoder
        self.draft_cache_factory = draft_cache_factory

    def _generate_to_eos(self, ids: list[int], cache) -> list[int]:
        import time
        device = self.device
        ids_t = torch.tensor([ids], device=device)          # (1, T_p)

        draft_cache = self.draft_cache_factory()

        print("Agent: ", end="", flush=True)
        t0 = time.perf_counter()

        out_t, stats = self.spec_decoder.generate(
            ids_t,
            max_new_tokens=self.max_new_tokens,
            draft_cache=draft_cache,
            target_cache=cache,
            sampling=self.sampling,
            eos_token=self.eos_token_id,
        )

        elapsed = time.perf_counter() - t0
        gen_ids = out_t[0, len(ids):].tolist()

        # Decode and print the completed output (batch print — spec decoding
        # generates multiple tokens per step so streaming is not straightforward).
        trim = gen_ids[:-1] if gen_ids and gen_ids[-1] == self.eos_token_id else gen_ids
        text = self.tokenizer.decode(trim, skip_special_tokens=True)
        print(text, flush=True)

        n_gen = len(gen_ids)
        tps = n_gen / elapsed if elapsed > 0 else 0
        print(
            f"\n  [{n_gen} tokens, {tps:.2f} tok/s | "
            f"spec accept {stats.acceptance_rate:.0%}, "
            f"{stats.tokens_per_step:.1f} tok/step]",
            flush=True,
        )
        return gen_ids


# ---------------------------------------------------------------------------
# Model + tokenizer loading
# ---------------------------------------------------------------------------

def load_model(
    model_dir: str,
    config: LlamaConfig,
    device: str,
    dtype: torch.dtype,
    quantize: bool = False,
    expert_offload: str = "disk",
    cache_mb: float = 2_000,
):
    print(f"Loading weights from {model_dir} ...")
    if config.is_moe:
        if expert_offload == "ram":
            print("MoE model — RAM expert offload (needs ~57 GB RAM for 30B-A3B)")
            model = load_moe_weights(model_dir, config, device=device, dtype=dtype)
        else:
            print(f"MoE model — disk expert offload ({cache_mb:.0f} MB VRAM expert cache)")
            model = load_moe_weights_disk(model_dir, config, device=device, dtype=dtype, cache_mb=cache_mb)
    else:
        # 14B+ in bf16 exceeds 16 GB VRAM — load to CPU, quantize there, move INT4 to GPU.
        # 8B in bf16 (~16 GB) fits and quantizes faster directly on the GPU.
        cpu_first = quantize and device == "cuda" and config.d_model > 4096
        if cpu_first:
            print("Loading to CPU for quantization (bf16 too large for VRAM on 14B+) ...")
            weights = load_llama_weights(model_dir, config, device="cpu", dtype=dtype)
            model = LlamaModel(weights, config)
            print("Quantizing to INT4 W4A16 on CPU ...")
            model = quantize_llama(model)
            print(f"Moving INT4 weights to {device} ...")
            model = quantized_to_device(model, device)
        else:
            weights = load_llama_weights(model_dir, config, device=device, dtype=dtype)
            model = LlamaModel(weights, config)
            if quantize:
                print("Quantizing to INT4 W4A16 ...")
                model = quantize_llama(model)
        if quantize:
            vram = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0
            print(f"Quantization done — {vram:.1f} GB VRAM in use")
    print(f"Model ready: {config.name} on {device}")
    return model


def make_cache_factory(config: LlamaConfig, device: str, dtype: torch.dtype, max_seq: int = 8192):
    def factory():
        return LlamaStaticKVCache(config, batch=1, max_seq=max_seq, device=device, dtype=dtype)
    return factory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Qwen3 interactive coding assistant")
    parser.add_argument(
        "--model-dir", default=None,
        help="Path to the model directory (safetensors + tokenizer files). "
             "Required except when using --show-history.",
    )
    parser.add_argument(
        "--workspace", default=".",
        help="Directory the agent can read/write/run commands in (default: current directory)",
    )
    parser.add_argument(
        "--system", default=None,
        help="Optional system prompt override",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=1024,
        help="Max tokens to generate per agent turn (default: 1024)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=8,
        help="Max tool-call iterations per user message (default: 8)",
    )
    parser.add_argument(
        "--max-ctx", type=int, default=8192,
        help="KV cache size in tokens (default: 8192). Qwen3 supports up to 32768. "
             "Larger values use more VRAM (Qwen3-8B INT4: ~0.5 GB per 4096 tokens of KV).",
    )
    parser.add_argument(
        "--thinking", action="store_true",
        help="Enable Qwen3 chain-of-thought (<think> blocks). Off by default for speed.",
    )
    parser.add_argument(
        "--draft-model-dir", default=None,
        help="Path to a small draft model for speculative decoding "
             "(e.g. weights/Qwen--Qwen3-0.6B). If set, enables speculative decoding "
             "which typically gives 2-3× throughput on coding tasks.",
    )
    parser.add_argument(
        "--n-draft", type=int, default=4,
        help="Number of draft tokens to speculate per step (default: 4).",
    )
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument(
        "--quantize", dest="quantize", action="store_const", const=True,
        help="Quantize weights to INT4 W4A16 after loading. "
             "Auto-enabled for dense models with d_model >= 4096 (8B, 14B, 32B).",
    )
    quant_group.add_argument(
        "--no-quantize", dest="quantize", action="store_const", const=False,
        help="Disable INT4 quantization even for large models.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--cache-mb", type=float, default=2_000,
        help="VRAM budget for the expert VRAM cache in MB (default: 2000). "
             "Caches recently used experts so repeated tokens skip disk reads. "
             "48 layers × 8 experts × 4.7 MB ≈ 1800 MB for full first-token coverage.",
    )
    parser.add_argument(
        "--expert-offload", choices=["disk", "ram"], default="disk",
        help="MoE expert offload mode: 'disk' streams from safetensors (needs ~0 RAM), "
             "'ram' loads all experts into pinned CPU RAM (needs ~57 GB for 30B-A3B). "
             "Default: disk.",
    )
    parser.add_argument(
        "--history-file", default=str(_DEFAULT_HISTORY),
        help=f"Path to the JSONL session log (default: {_DEFAULT_HISTORY}). "
             "Each session is appended as one JSON line on exit.",
    )
    parser.add_argument(
        "--show-history", metavar="N", nargs="?", const=0, type=int,
        help="Print past sessions and exit. Optionally pass N to show only the last N sessions.",
    )
    parser.set_defaults(quantize=None)  # None = auto-detect based on model size
    args = parser.parse_args()

    history_file = Path(args.history_file).expanduser().resolve()

    if args.show_history is not None:
        last_n = args.show_history if args.show_history > 0 else None
        _show_history(history_file, last_n=last_n)
        return

    if args.model_dir is None:
        parser.error("--model-dir is required (except with --show-history)")

    try:
        import readline  # noqa: F401 — enables arrow-key navigation and input history
    except ImportError:
        pass

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        raise SystemExit(f"Workspace not found: {workspace}")

    dtype = torch.bfloat16
    config = detect_config(args.model_dir)

    # Auto-enable quantize for large dense models (8B, 14B, 32B) if not explicitly set.
    if args.quantize is None:
        args.quantize = not config.is_moe and config.d_model >= 4096
        if args.quantize:
            print(f"Auto-enabling INT4 quantization for {config.name} "
                  f"(d_model={config.d_model}). Pass --no-quantize to disable.")

    tokenizer = QwenTokenizer(args.model_dir)
    model = load_model(
        args.model_dir, config, args.device, dtype,
        quantize=args.quantize,
        expert_offload=args.expert_offload,
        cache_mb=args.cache_mb,
    )
    cache_factory = make_cache_factory(config, args.device, dtype, max_seq=args.max_ctx)

    # Optional speculative decoding: load a small draft model.
    draft_model = None
    draft_config = None
    if args.draft_model_dir:
        draft_config = detect_config(args.draft_model_dir)
        print(f"Loading draft model {draft_config.name} for speculative decoding ...")
        draft_model = load_model(args.draft_model_dir, draft_config, args.device, dtype, quantize=False)

    tools = make_tools(workspace)

    # Build the system prompt. User-supplied --system overrides the default.
    system_prompt = args.system or (
        f"You are a coding assistant. The user's workspace is at {workspace}.\n\n"
        "RULES — follow these exactly:\n"
        "1. NEVER print code or file contents directly in your response. "
        "Always use write_file to save code to the workspace.\n"
        "2. When asked to create, build, generate, or write anything, use write_file immediately. "
        "Do not ask for clarification first.\n"
        "3. After writing files, use run_shell to verify the work "
        "(e.g. open the file, run tests, execute the script).\n"
        "4. Use list_dir or read_file to understand existing files before modifying them.\n"
        "5. Keep responses short — one sentence saying what you did and which file(s) were written."
    )

    agent_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        cache_factory=cache_factory,
        tools=tools,
        sampling=SamplingConfig(temperature=0.6, top_p=0.95),
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.thinking,
        device=args.device,
        max_ctx=args.max_ctx,
    )

    if draft_model is not None:
        spec_decoder = SpeculativeDecoder(draft=draft_model, target=model, n_draft=args.n_draft)
        draft_cache_factory = make_cache_factory(draft_config, args.device, dtype, max_seq=args.max_ctx)
        agent = SpeculativeVerboseAgentLoop(
            **agent_kwargs,
            spec_decoder=spec_decoder,
            draft_cache_factory=draft_cache_factory,
        )
        print(f"Speculative decoding enabled: {draft_config.name} draft, {args.n_draft} tokens/step")
    else:
        agent = VerboseAgentLoop(**agent_kwargs)

    print(f"\nWorkspace: {workspace}")
    print("Type /exit to quit, /clear to reset, /tools to list tools, /history to review.\n")

    messages = [{"role": "system", "content": system_prompt}]

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not user_input:
                continue

            if user_input == "/exit":
                print("Bye.")
                break

            if user_input == "/clear":
                messages = [messages[0]]  # keep system prompt
                print("Conversation cleared.\n")
                continue

            if user_input == "/tools":
                for t in tools:
                    print(f"  {t.name}: {t.description}")
                print()
                continue

            if user_input == "/history":
                convo = [m for m in messages if m["role"] in ("user", "assistant")]
                if not convo:
                    print("  (no conversation yet)\n")
                else:
                    for m in convo:
                        preview = m.get("content", "")[:200].replace("\n", " ")
                        print(f"  [{m['role']}] {preview}")
                    print()
                continue

            messages.append({"role": "user", "content": user_input})
            print()  # blank line before streamed output
            result = agent.run(messages)
            messages = result.messages  # accumulate full history for next turn
            print()
    finally:
        _save_session(history_file, config.name, workspace, messages)
        print(f"(session saved to {history_file})")


if __name__ == "__main__":
    main()
