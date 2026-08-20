"""Parse Qwen3 <tool_call> blocks from model output.

Qwen3's tool-calling output format:
    <tool_call>
    {"name": "fn_name", "arguments": {"arg1": val1, ...}}
    </tool_call>

Multiple tool calls may appear in a single response.  Blocks with
invalid JSON or a missing "name" key are silently dropped so the
agent loop can keep running even if the model produces a partial call.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# JSON only allows \n \r \t \\ \" \/ \uXXXX — models sometimes emit \xNN.
_INVALID_HEX_ESCAPE = re.compile(r"\\x([0-9a-fA-F]{2})")


@dataclass
class ToolCall:
    """A parsed tool call from the model's output."""

    name: str
    arguments: dict = field(default_factory=dict)


def _repair_json(raw: str) -> str:
    r"""Replace \xNN hex escapes (invalid in JSON) with \\xNN (escaped backslash + x)."""
    return _INVALID_HEX_ESCAPE.sub(r"\\\\x\1", raw)


def extract_tool_calls(text: str) -> list[ToolCall]:
    """Find and parse every ``<tool_call>…</tool_call>`` block in *text*.

    On JSONDecodeError, attempts to repair common model mistakes (e.g. bare
    ``\\xNN`` hex escapes that are invalid in JSON) before giving up.
    Blocks that still fail to parse are skipped with a warning printed to
    stderr so the agent loop can keep running.
    """
    import sys
    calls: list[ToolCall] = []
    for m in _TOOL_CALL_RE.finditer(text):
        raw = m.group(1)
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            repaired = _repair_json(raw)
            try:
                data = json.loads(repaired)
            except (json.JSONDecodeError, ValueError):
                print(
                    f"  [tool_parser] dropped malformed tool call (JSON parse failed)\n"
                    f"  {raw[:120]}{'...' if len(raw) > 120 else ''}",
                    file=sys.stderr,
                )
                continue
        if "name" not in data:
            continue
        calls.append(ToolCall(name=data["name"], arguments=data.get("arguments", {})))
    return calls


def has_tool_call(text: str) -> bool:
    """Return True if *text* contains at least one ``<tool_call>`` block."""
    return bool(_TOOL_CALL_RE.search(text))


def count_tool_call_blocks(text: str) -> int:
    """Count ``<tool_call>...</tool_call>`` blocks in *text*, valid or not.

    Compare against ``len(extract_tool_calls(text))`` to detect calls that
    were present but dropped for failing to parse — the caller can then
    feed an error back to the model instead of silently losing the call.
    """
    return len(_TOOL_CALL_RE.findall(text))


def strip_thinking(text: str) -> str:
    """Remove all ``<think>…</think>`` blocks and return the stripped string.

    Used to get the visible / tool-dispatch portion of the model's response
    when running in thinking mode.  The raw text (including think blocks) is
    kept in the conversation history so the model can see its own reasoning
    in subsequent turns.
    """
    return _THINK_RE.sub("", text).strip()


def format_tool_result(name: str, result: Any) -> str:
    """Serialise a tool's return value as a string for the tool-role message body.

    Plain strings are returned as-is.  Everything else is JSON-encoded so the
    model receives a deterministic, schema-consistent representation.
    """
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)
