"""Qwen3 ChatML formatter: OpenAI-style message dicts → tokenizable string.

Format per message:
    <|im_start|>role\\ncontent<|im_end|>\\n

Roles:
    system    — system prompt (tool schema injected here when tools are provided)
    user      — human turn
    assistant — model turn (may contain <tool_call> blocks or thinking <think> tags)
    tool      — tool execution result wrapped in <tool_response> tags

The formatted string always ends with an open assistant turn:
    <|im_start|>assistant\\n

When enable_thinking=False (default), the open turn is prefixed with an empty
think block so the model skips the reasoning phase:
    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n

This is the soft non-thinking mode from Qwen3: the model sees a closed <think>
block and proceeds directly to the response. Tokens are part of the prompt
(prefill), so they don't appear in the generated output.
"""

from __future__ import annotations

import json
from typing import Any

_DEFAULT_SYSTEM = (
    "You are Qwen, made by Alibaba Cloud. You are a helpful assistant."
)

_TOOL_SECTION = """\n\n# Tools\n
You may call one or more functions to assist with the user query.\n
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_json}
</tools>

For each function call, return a json object with function name and arguments \
within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def _tool_to_schema(tool: Any) -> dict:
    """Convert a Tool dataclass or plain dict to the Qwen3 JSON Schema format."""
    if isinstance(tool, dict):
        # Caller passed a raw schema dict; use as-is if it has 'function' key.
        if "function" in tool:
            return tool
        # Otherwise wrap it.
        return {"type": "function", "function": tool}
    # engine.agent.Tool dataclass
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _build_tool_section(tools: list) -> str:
    """Build the tool-description block that is appended to the system prompt."""
    schemas = [_tool_to_schema(t) for t in tools]
    tool_json = "\n".join(json.dumps(s, ensure_ascii=False) for s in schemas)
    return _TOOL_SECTION.format(tool_json=tool_json)


def format_messages(
    messages: list[dict],
    tools: list | None = None,
    enable_thinking: bool = False,
) -> str:
    """Format an OpenAI-style message list to a Qwen3 ChatML string.

    Args:
        messages:        List of ``{"role": str, "content": str}`` dicts.
                         Supported roles: ``system``, ``user``, ``assistant``, ``tool``.
                         For tool results, include ``"name"`` for clarity (not required).
        tools:           Optional list of ``Tool`` dataclasses or JSON-Schema dicts.
                         When provided, the tool signatures are injected into the
                         system prompt using the Qwen3 tool-calling convention.
        enable_thinking: When ``False`` (default) the open assistant turn is prefixed
                         with ``<think>\\n\\n</think>\\n\\n`` to suppress chain-of-thought
                         and speed up tool-calling inference. Set ``True`` to let the
                         model reason before answering.

    Returns:
        ChatML string ready to be tokenized and fed to the model.  Ends with an
        open assistant turn so the model can continue generating.
    """
    msgs = list(messages)

    # Inject a default system message when tools are provided and none exists.
    if tools and (not msgs or msgs[0]["role"] != "system"):
        msgs = [{"role": "system", "content": _DEFAULT_SYSTEM}] + msgs

    parts: list[str] = []
    for msg in msgs:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            if tools:
                content = content + _build_tool_section(tools)
            parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")

        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")

        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")

        elif role == "tool":
            # Tool result: wrap in <tool_response> inside a tool-role turn.
            parts.append(
                f"<|im_start|>tool\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
            )

    # Open assistant turn for the model to continue.
    if enable_thinking:
        parts.append("<|im_start|>assistant\n")
    else:
        # Empty think block forces the model to skip chain-of-thought.
        parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")

    return "".join(parts)
