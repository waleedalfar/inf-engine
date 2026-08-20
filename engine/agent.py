"""Multi-turn agentic loop: generate → parse → execute → inject → repeat.

Usage::

    from engine.agent import AgentLoop, Tool

    tools = [
        Tool(
            name="add",
            description="Return the sum of two numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
            fn=lambda a, b: a + b,
        )
    ]

    loop = AgentLoop(
        model=llama_model,
        tokenizer=qwen_tokenizer,
        cache_factory=lambda: LlamaStaticKVCache(config, batch=1, max_seq=4096),
        tools=tools,
        max_turns=8,
        max_new_tokens=1024,
        device="cuda",
    )

    result = loop.run([{"role": "user", "content": "What is 42 + 58?"}])
    print(result.final_text)   # "The answer is 100."

The loop ends when the model produces a response with no ``<tool_call>``
block, or when ``max_turns`` is exhausted.  Each turn rebuilds the KV cache
from the full conversation history so far — O(n²) in token cost but always
correct and free of position-tracking bugs.  For sessions that stay under
~4K tokens this is negligible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from engine.chat import format_messages
from engine.sampling import SamplingConfig, sample_next_token
from engine.tool_parser import (
    ToolCall,
    extract_tool_calls,
    format_tool_result,
    has_tool_call,
    strip_thinking,
)


def _truncate_history(
    history: list[dict],
    token_limit: int,
    tokenizer: Any,
    tool_list: list,
    enable_thinking: bool,
) -> list[dict]:
    """Drop middle turns until the formatted prompt fits within token_limit.

    Keeps the system message (index 0) and trims oldest non-system messages
    first, preserving the most recent context. Stops as soon as it fits or
    only one message remains.
    """
    # Separate system message from the rest.
    sys_msgs = [m for m in history if m["role"] == "system"]
    convo = [m for m in history if m["role"] != "system"]

    while len(convo) > 1:
        prompt = format_messages(sys_msgs + convo, tool_list or None, enable_thinking)
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        if len(ids) <= token_limit:
            break
        convo.pop(0)  # drop oldest non-system message

    if sys_msgs:
        print(
            f"  [context truncated: keeping system + last {len(convo)} messages]",
            flush=True,
        )
    return sys_msgs + convo


@dataclass
class Tool:
    """A Python function exposed to the model as a callable tool.

    Args:
        name:        Identifier used in ``<tool_call>`` blocks.
        description: Natural-language description injected into the system prompt.
        parameters:  JSON Schema ``object`` describing the function's arguments.
        fn:          Called with ``**arguments``; return value is serialised and
                     fed back to the model as the tool result.
    """

    name: str
    description: str
    parameters: dict
    fn: Callable = field(repr=False)


@dataclass
class AgentResult:
    """Completed agentic run."""

    final_text: str
    """Visible response text (thinking blocks stripped)."""

    messages: list[dict]
    """Full conversation history including all tool calls and results."""

    n_turns: int
    """Number of generate iterations performed."""

    tool_calls_made: list[ToolCall]
    """All tool calls dispatched across all turns, in order."""


class AgentLoop:
    """Drive multi-turn tool-using inference with Qwen3.

    Args:
        model:          ``LlamaModel`` (or quantized / MoE variant).
        tokenizer:      ``QwenTokenizer`` with ``encode(text)`` → ``list[int]``,
                        ``decode(ids)`` → ``str``, and ``im_end_id`` property.
        cache_factory:  Zero-argument callable that returns a fresh KV cache
                        (e.g. ``LlamaStaticKVCache`` or ``PagedLlamaKVCache``).
                        Called at the start of every generate turn.
        tools:          List of :class:`Tool` instances to expose to the model.
        sampling:       :class:`~engine.sampling.SamplingConfig` — defaults to greedy.
        max_turns:      Hard cap on generate iterations (tool-call turns + final answer).
        max_new_tokens: Maximum tokens the model may generate per turn.
        enable_thinking: Pass ``True`` to let Qwen3 produce ``<think>`` reasoning.
                         ``False`` (default) injects an empty think block so the model
                         skips the reasoning phase and proceeds directly to the answer.
        device:         Torch device string for prompt tensors.  Defaults to
                        ``"cuda"`` when a GPU is available, else ``"cpu"``.
        eos_token_id:   Token id that signals end-of-turn.  Defaults to
                        ``tokenizer.im_end_id`` (``<|im_end|>``, id 151645 for Qwen3).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        cache_factory: Callable,
        tools: list[Tool] | None = None,
        sampling: SamplingConfig | None = None,
        max_turns: int = 10,
        max_new_tokens: int = 512,
        enable_thinking: bool = False,
        device: str | None = None,
        eos_token_id: int | None = None,
        max_ctx: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cache_factory = cache_factory
        self.tools = {t.name: t for t in (tools or [])}
        self.sampling = sampling or SamplingConfig()
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if eos_token_id is None:
            eos_token_id = getattr(tokenizer, "im_end_id", 151645)
        self.eos_token_id = eos_token_id
        self.max_ctx = max_ctx

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, messages: list[dict]) -> AgentResult:
        """Execute the agentic loop starting from *messages*.

        Args:
            messages: OpenAI-style message list.  Typically starts with a
                      ``{"role": "user", "content": "..."}`` dict.  A system
                      message is injected automatically when tools are provided
                      and none is present.

        Returns:
            :class:`AgentResult` with the final visible answer, the full
            history, the turn count, and every tool call that was dispatched.
        """
        history: list[dict] = list(messages)
        all_tool_calls: list[ToolCall] = []
        n_turns = 0
        tool_list = list(self.tools.values())

        for turn in range(self.max_turns):
            n_turns = turn + 1

            # Build full prompt from entire history each turn.
            prompt = format_messages(history, tool_list or None, self.enable_thinking)
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)

            # Graceful overflow: drop middle turns when prompt would overflow cache.
            if self.max_ctx is not None:
                hard_limit = self.max_ctx - self.max_new_tokens
                if len(ids) > hard_limit and len(history) > 2:
                    history = _truncate_history(
                        history, hard_limit, self.tokenizer, tool_list, self.enable_thinking
                    )
                    prompt = format_messages(history, tool_list or None, self.enable_thinking)
                    ids = self.tokenizer.encode(prompt, add_special_tokens=True)

            # Fresh cache — rebuilt from full history (always correct).
            cache = self.cache_factory()

            with torch.no_grad():
                gen_ids = self._generate_to_eos(ids, cache)

            raw_text = self._decode(gen_ids)
            visible_text = strip_thinking(raw_text)

            if not has_tool_call(visible_text):
                # Final response — no tool dispatch.
                history.append({"role": "assistant", "content": raw_text})
                return AgentResult(
                    final_text=visible_text,
                    messages=history,
                    n_turns=n_turns,
                    tool_calls_made=all_tool_calls,
                )

            # Tool call(s) present — dispatch and inject results.
            calls = extract_tool_calls(visible_text)
            history.append({"role": "assistant", "content": raw_text})

            for call in calls:
                all_tool_calls.append(call)
                result_str = self._execute_tool(call)
                history.append(
                    {
                        "role": "tool",
                        "content": result_str,
                        "name": call.name,
                    }
                )

        # Max turns reached without a clean exit.
        last_content = history[-1].get("content", "") if history else ""
        return AgentResult(
            final_text=strip_thinking(last_content),
            messages=history,
            n_turns=n_turns,
            tool_calls_made=all_tool_calls,
        )

    # ------------------------------------------------------------------
    # Internal helpers — overridable in tests
    # ------------------------------------------------------------------

    def _generate_to_eos(self, ids: list[int], cache: Any) -> list[int]:
        """Prefill *ids* then decode until EOS or ``max_new_tokens``.

        Returns the generated token ids (not including the prompt).
        This method is a plain instance method rather than a closure so
        tests can monkey-patch it with ``loop._generate_to_eos = fake_fn``
        to inject preset responses without touching tokenization.
        """
        device = self.device
        n = len(ids)

        ids_t = torch.tensor([ids], device=device)
        pos_t = torch.arange(n, device=device)

        # Prefill: write all prompt tokens into the cache.
        logits = self.model.forward(
            ids_t, cache=cache, start_pos=0, position_ids=pos_t
        )

        # Sample the first generated token from the last prefill position.
        ctx_t = ids_t                                                       # (1, T_p)
        next_tok = sample_next_token(logits[:, -1:, :], self.sampling, ctx_t).item()

        generated: list[int] = [next_tok]
        pos = n  # the position where the first gen token lives

        while next_tok != self.eos_token_id and len(generated) < self.max_new_tokens:
            tok_t = torch.tensor([[next_tok]], device=device)
            ctx_t = torch.cat([ctx_t, tok_t], dim=1)                       # (1, T_p + step)
            pos_s = torch.tensor([pos], device=device)
            logits = self.model.forward(
                tok_t, cache=cache, start_pos=pos, position_ids=pos_s
            )
            pos += 1
            next_tok = sample_next_token(logits[:, -1:, :], self.sampling, ctx_t).item()
            generated.append(next_tok)

        return generated

    def _decode(self, gen_ids: list[int]) -> str:
        """Decode generated ids → text, stripping the EOS token."""
        # Exclude the trailing EOS so it doesn't appear in the returned text.
        ids = gen_ids[:-1] if gen_ids and gen_ids[-1] == self.eos_token_id else gen_ids
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def _execute_tool(self, call: ToolCall) -> str:
        """Dispatch a parsed tool call and return a result string.

        Unknown tool names and runtime exceptions both produce an error
        string that is fed back to the model as the tool result.  The loop
        never raises — a bad tool call becomes model-visible feedback.
        """
        if call.name not in self.tools:
            available = ", ".join(self.tools.keys()) or "<none>"
            return f"Error: unknown tool '{call.name}'. Available: {available}"
        try:
            result = self.tools[call.name].fn(**call.arguments)
            return format_tool_result(call.name, result)
        except Exception as exc:  # noqa: BLE001
            return f"Error: {type(exc).__name__}: {exc}"
