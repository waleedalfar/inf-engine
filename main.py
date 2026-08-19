"""Qwen3 interactive coding assistant using the custom inference engine.

Usage:
    python main.py --model-dir weights/Qwen--Qwen3-0.6B
    python main.py --model-dir weights/Qwen--Qwen3-0.6B --workspace ~/myproject
    python main.py --model-dir weights/Qwen--Qwen3-1.7B --system "You are a senior Python engineer."

In-chat commands:
    /clear    Reset conversation (keeps system prompt)
    /tools    List available tools
    /history  Print conversation so far
    /exit     Quit (also Ctrl+C or Ctrl+D)

The model directory must contain either:
  - tokenizer.json (HuggingFace format)  OR  qwen.tiktoken
  - model.safetensors  OR  model-*.safetensors shards + model.safetensors.index.json
"""

import argparse
import os
import subprocess
from pathlib import Path
import torch

from engine.config import (
    QWEN3_0_6B, QWEN3_1_7B, QWEN3_4B, QWEN3_8B, QWEN3_14B, QWEN3_32B,
    LlamaConfig,
)
from engine.kv_cache import LlamaStaticKVCache
from engine.llama_model import LlamaModel
from engine.llama_weights import load_llama_weights
from engine.qwen_tokenizer import QwenTokenizer
from engine.agent import AgentLoop, Tool
from engine.sampling import SamplingConfig

# Maps directory-name fragments → config, largest first so "Qwen3-14B"
# doesn't accidentally match before "Qwen3-1.7B".
_DIR_TO_CONFIG: list[tuple[str, LlamaConfig]] = [
    ("Qwen3-32B",  QWEN3_32B),
    ("Qwen3-14B",  QWEN3_14B),
    ("Qwen3-8B",   QWEN3_8B),
    ("Qwen3-4B",   QWEN3_4B),
    ("Qwen3-1.7B", QWEN3_1_7B),
    ("Qwen3-0.6B", QWEN3_0_6B),
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
        print(f"  → [tool: {call.name}]", flush=True)
        return super()._execute_tool(call)


# ---------------------------------------------------------------------------
# Model + tokenizer loading
# ---------------------------------------------------------------------------

def load_model(model_dir: str, config: LlamaConfig, device: str, dtype: torch.dtype):
    print(f"Loading weights from {model_dir} ...")
    weights = load_llama_weights(model_dir, config, device=device, dtype=dtype)
    model = LlamaModel(weights, config)
    print(f"Model ready: {config.name} on {device}")
    return model


def make_cache_factory(config: LlamaConfig, device: str, dtype: torch.dtype, max_seq: int = 4096):
    def factory():
        return LlamaStaticKVCache(config, batch=1, max_seq=max_seq, device=device, dtype=dtype)
    return factory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Qwen3 interactive coding assistant")
    parser.add_argument(
        "--model-dir", required=True,
        help="Path to the model directory (safetensors + tokenizer files)",
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
        "--thinking", action="store_true",
        help="Enable Qwen3 chain-of-thought (<think> blocks). Off by default for speed.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    try:
        import readline  # noqa: F401 — enables arrow-key navigation and input history
    except ImportError:
        pass

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        raise SystemExit(f"Workspace not found: {workspace}")

    dtype = torch.bfloat16
    config = detect_config(args.model_dir)
    tokenizer = QwenTokenizer(args.model_dir)
    model = load_model(args.model_dir, config, args.device, dtype)
    cache_factory = make_cache_factory(config, args.device, dtype)
    tools = make_tools(workspace)

    # Build the system prompt. User-supplied --system overrides the default.
    system_prompt = args.system or (
        f"You are a coding assistant with access to the user's workspace at {workspace}. "
        "You can read files, write files, list directories, search for text, and run shell commands "
        "— all sandboxed to that directory. "
        "When asked to implement something, read relevant existing files first, then write clean code. "
        "Always run tests or execute the code to verify your work before reporting success."
    )

    agent = VerboseAgentLoop(
        model=model,
        tokenizer=tokenizer,
        cache_factory=cache_factory,
        tools=tools,
        sampling=SamplingConfig(temperature=0.6, top_p=0.95),
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.thinking,
        device=args.device,
    )

    print(f"\nWorkspace: {workspace}")
    print("Type /exit to quit, /clear to reset, /tools to list tools, /history to review.\n")

    messages = [{"role": "system", "content": system_prompt}]

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
            if len(messages) <= 1:
                print("  (no conversation yet)\n")
            for m in messages:
                if m["role"] == "system":
                    continue
                preview = m.get("content", "")[:200].replace("\n", " ")
                print(f"  [{m['role']}] {preview}")
            print()
            continue

        messages.append({"role": "user", "content": user_input})
        result = agent.run(messages)
        messages = result.messages  # accumulate full history for next turn

        print(f"\nAgent: {result.final_text}\n")


if __name__ == "__main__":
    main()
