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


@dataclass
class ToolCall:
    """A parsed tool call from the model's output."""

    name: str
    arguments: dict = field(default_factory=dict)


def extract_tool_calls(text: str) -> list[ToolCall]:
    """Find and parse every ``<tool_call>…</tool_call>`` block in *text*.

    Blocks with malformed JSON or without a ``name`` key are skipped.
    """
    calls: list[ToolCall] = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            data = json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        if "name" not in data:
            continue
        calls.append(ToolCall(name=data["name"], arguments=data.get("arguments", {})))
    return calls


def has_tool_call(text: str) -> bool:
    """Return True if *text* contains at least one ``<tool_call>`` block."""
    return bool(_TOOL_CALL_RE.search(text))


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
