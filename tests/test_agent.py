"""Phase 7 — comprehensive tests for chat, tool_parser, and agent loop.

All tests use synthetic fixtures; no real weights are required.
Run via WSL:
    wsl bash -c "cd /home/waleed/mlproj && source .venv/bin/activate && pytest tests/test_agent.py -v"
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from engine.agent import AgentLoop, AgentResult, Tool
from engine.chat import format_messages
from engine.tool_parser import (
    ToolCall,
    extract_tool_calls,
    format_tool_result,
    has_tool_call,
    strip_thinking,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Minimal character-level tokenizer for deterministic test encoding/decoding.

    Token 0 is reserved for <|im_end|>.  Every other character gets a unique id
    assigned on first use.  Special strings encountered during encode are handled
    atomically so ``<|im_end|>`` is always token 0 regardless of context.
    """

    IM_END_ID = 0
    _SPECIAL = "<|im_end|>"

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {self._SPECIAL: 0}
        self._rev: dict[int, str] = {0: self._SPECIAL}
        self._next = 1

    def _get_id(self, c: str) -> int:
        if c not in self._vocab:
            self._vocab[c] = self._next
            self._rev[self._next] = c
            self._next += 1
        return self._vocab[c]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        while text:
            if text.startswith(self._SPECIAL):
                ids.append(0)
                text = text[len(self._SPECIAL):]
            else:
                ids.append(self._get_id(text[0]))
                text = text[1:]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        parts: list[str] = []
        for i in ids:
            if i == 0:
                if not skip_special_tokens:
                    parts.append(self._SPECIAL)
            else:
                parts.append(self._rev.get(i, "?"))
        return "".join(parts)

    @property
    def im_end_id(self) -> int:
        return self.IM_END_ID


class _MockLlamaModel:
    """Returns preset token ids one per forward() call (order: prefill then decode)."""

    def __init__(self, flat_token_sequence: list[int], vocab_size: int = 300) -> None:
        self._seq = flat_token_sequence
        self._ptr = 0
        self._vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Any = None,
        start_pos: int = 0,
        position_ids: Any = None,
        attn_mask: Any = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        logits = torch.full((B, T, self._vocab_size), float("-inf"))
        tok = self._seq[self._ptr] if self._ptr < len(self._seq) else 0
        logits[:, -1, tok] = 0.0  # argmax selects tok
        self._ptr += 1
        return logits


@pytest.fixture()
def tok() -> _MockTokenizer:
    return _MockTokenizer()


def _make_agent(
    tokenizer: _MockTokenizer,
    responses: list[str],
    tools: list[Tool] | None = None,
    max_turns: int = 5,
    enable_thinking: bool = False,
) -> AgentLoop:
    """Build an AgentLoop with _generate_to_eos replaced by preset string responses."""
    response_idx = [0]

    def fake_generate(ids: list[int], cache: Any) -> list[int]:
        idx = response_idx[0]
        response_idx[0] += 1
        text = responses[idx] if idx < len(responses) else ""
        return tokenizer.encode(text, add_special_tokens=False)

    loop = AgentLoop(
        model=None,
        tokenizer=tokenizer,
        cache_factory=lambda: None,
        tools=tools or [],
        max_turns=max_turns,
        enable_thinking=enable_thinking,
        device="cpu",
    )
    loop._generate_to_eos = fake_generate  # type: ignore[method-assign]
    return loop


def _add_tool() -> Tool:
    return Tool(
        name="add",
        description="Add two numbers.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        fn=lambda a, b: a + b,
    )


def _echo_tool() -> Tool:
    return Tool(
        name="echo",
        description="Echo back the input.",
        parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
        fn=lambda msg: msg,
    )


def _exploding_tool() -> Tool:
    def _boom(**_):
        raise RuntimeError("kaboom")

    return Tool(
        name="boom",
        description="Always raises.",
        parameters={"type": "object", "properties": {}},
        fn=_boom,
    )


# ===========================================================================
# Section 1 — format_messages (chat template)
# ===========================================================================


class TestFormatMessages:
    def test_user_message_wrapped_in_im_tags(self):
        out = format_messages([{"role": "user", "content": "Hello"}])
        assert "<|im_start|>user\nHello<|im_end|>" in out

    def test_system_message_wrapped_in_im_tags(self):
        out = format_messages([
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ])
        assert "<|im_start|>system\nBe helpful.<|im_end|>" in out

    def test_assistant_message_wrapped_in_im_tags(self):
        out = format_messages([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there!"},
            {"role": "user", "content": "Go on"},
        ])
        assert "<|im_start|>assistant\nHello there!<|im_end|>" in out

    def test_tool_message_wrapped_in_tool_response_tags(self):
        out = format_messages([
            {"role": "user", "content": "?"},
            {"role": "assistant", "content": "calling"},
            {"role": "tool", "content": "42", "name": "add"},
        ])
        assert "<|im_start|>tool\n<tool_response>\n42\n</tool_response><|im_end|>" in out

    def test_messages_appear_in_correct_order(self):
        out = format_messages([
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USR"},
            {"role": "assistant", "content": "AST"},
        ])
        assert out.index("SYS") < out.index("USR") < out.index("AST")

    def test_open_assistant_turn_no_thinking(self):
        """Default mode: open turn includes empty think block."""
        out = format_messages([{"role": "user", "content": "Q"}])
        # Must end with the assistant prefix followed by the empty think block.
        assert out.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def test_open_assistant_turn_enable_thinking(self):
        """Thinking mode: open turn has no injected think block."""
        out = format_messages([{"role": "user", "content": "Q"}], enable_thinking=True)
        assert out.endswith("<|im_start|>assistant\n")
        assert "<think>" not in out

    def test_tools_schema_injected_in_system_prompt(self):
        tool = _add_tool()
        out = format_messages(
            [{"role": "user", "content": "calc"}],
            tools=[tool],
        )
        assert '"name": "add"' in out
        assert "<tools>" in out
        assert "</tools>" in out

    def test_default_system_injected_when_tools_but_no_system(self):
        """When tools are provided but messages have no system message, a default is added."""
        tool = _echo_tool()
        out = format_messages([{"role": "user", "content": "hi"}], tools=[tool])
        assert "<|im_start|>system\n" in out
        # The injected system must appear before the user turn.
        assert out.index("<|im_start|>system") < out.index("<|im_start|>user")

    def test_explicit_system_not_duplicated_when_tools_provided(self):
        """An existing system message is augmented with tool info — not duplicated."""
        tool = _add_tool()
        out = format_messages(
            [
                {"role": "system", "content": "Custom system."},
                {"role": "user", "content": "calc"},
            ],
            tools=[tool],
        )
        assert out.count("<|im_start|>system") == 1
        assert "Custom system." in out
        assert '"name": "add"' in out

    def test_no_tools_no_tool_section(self):
        out = format_messages([{"role": "user", "content": "hi"}])
        assert "<tools>" not in out
        assert "<tool_call>" not in out


# ===========================================================================
# Section 2 — tool_parser
# ===========================================================================


class TestExtractToolCalls:
    def test_single_valid_call_parsed(self):
        text = '<tool_call>\n{"name": "add", "arguments": {"a": 1, "b": 2}}\n</tool_call>'
        calls = extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "add"
        assert calls[0].arguments == {"a": 1, "b": 2}

    def test_multiple_calls_in_one_response(self):
        text = (
            '<tool_call>{"name": "f1", "arguments": {}}</tool_call>\n'
            '<tool_call>{"name": "f2", "arguments": {"x": "y"}}</tool_call>'
        )
        calls = extract_tool_calls(text)
        assert len(calls) == 2
        assert calls[0].name == "f1"
        assert calls[1].name == "f2"
        assert calls[1].arguments == {"x": "y"}

    def test_no_tool_call_blocks_returns_empty_list(self):
        assert extract_tool_calls("Just a plain answer.") == []

    def test_malformed_json_block_skipped(self):
        text = (
            '<tool_call>NOT JSON</tool_call>'
            '<tool_call>{"name": "ok", "arguments": {}}</tool_call>'
        )
        calls = extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "ok"

    def test_hex_escape_in_content_repaired(self):
        # Models sometimes emit \xNN in JSON strings (invalid JSON; only \uXXXX is allowed).
        # The parser must repair and recover rather than silently drop the tool call.
        text = '<tool_call>{"name": "write_file", "arguments": {"path": "x.py", "content": "b\'\\x00\'"}}</tool_call>'
        calls = extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "write_file"
        assert calls[0].arguments["path"] == "x.py"
        # After repair, \x00 becomes \\x00 in the string value (backslash + x00)
        assert "x00" in calls[0].arguments["content"]

    def test_missing_name_key_skipped(self):
        text = '<tool_call>{"arguments": {"a": 1}}</tool_call>'
        assert extract_tool_calls(text) == []

    def test_nested_arguments_parsed(self):
        text = '<tool_call>{"name": "fn", "arguments": {"opts": {"x": [1, 2]}}}</tool_call>'
        calls = extract_tool_calls(text)
        assert calls[0].arguments == {"opts": {"x": [1, 2]}}

    def test_extra_json_keys_ignored(self):
        text = '<tool_call>{"name": "fn", "arguments": {}, "extra": "ignored"}</tool_call>'
        calls = extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "fn"

    def test_whitespace_inside_block_handled(self):
        text = '<tool_call>   \n  {"name": "ws", "arguments": {}}  \n</tool_call>'
        calls = extract_tool_calls(text)
        assert len(calls) == 1


class TestHasToolCall:
    def test_returns_true_when_block_present(self):
        assert has_tool_call('<tool_call>{"name": "f"}</tool_call>')

    def test_returns_false_without_block(self):
        assert not has_tool_call("Plain text response.")

    def test_returns_false_for_partial_tag(self):
        assert not has_tool_call("<tool_call>no closing tag")


class TestStripThinking:
    def test_removes_think_block(self):
        text = "<think>internal reasoning here</think>Final answer."
        assert strip_thinking(text) == "Final answer."

    def test_preserves_content_after_think_block(self):
        text = "<think>thinking</think>The sky is blue."
        result = strip_thinking(text)
        assert "The sky is blue." in result
        assert "<think>" not in result

    def test_handles_multiline_think_block(self):
        text = "<think>\nline1\nline2\n</think>\nResult."
        assert strip_thinking(text) == "Result."

    def test_no_think_block_returns_original(self):
        text = "No thinking here."
        assert strip_thinking(text) == text

    def test_strips_multiple_think_blocks(self):
        text = "<think>a</think> middle <think>b</think>end"
        result = strip_thinking(text)
        assert "a" not in result
        assert "b" not in result
        assert "end" in result


class TestFormatToolResult:
    def test_string_returned_as_is(self):
        assert format_tool_result("any_tool", "result text") == "result text"

    def test_dict_json_serialized(self):
        result = {"temperature": 72, "unit": "F"}
        out = format_tool_result("weather", result)
        parsed = json.loads(out)
        assert parsed == result

    def test_number_serialized(self):
        out = format_tool_result("add", 100)
        assert out == "100"

    def test_list_serialized(self):
        out = format_tool_result("items", [1, 2, 3])
        assert json.loads(out) == [1, 2, 3]


# ===========================================================================
# Section 3 — AgentLoop (monkey-patched generate)
# ===========================================================================


class TestAgentNoToolCall:
    def test_returns_final_text_when_no_tool_call(self, tok):
        loop = _make_agent(tok, responses=["Hello, I can help with that."])
        result = loop.run([{"role": "user", "content": "Hi"}])
        assert result.final_text == "Hello, I can help with that."

    def test_n_turns_is_one_on_direct_answer(self, tok):
        loop = _make_agent(tok, responses=["Direct answer."])
        result = loop.run([{"role": "user", "content": "?"}])
        assert result.n_turns == 1

    def test_tool_calls_made_empty_on_no_dispatch(self, tok):
        loop = _make_agent(tok, responses=["Simple answer."])
        result = loop.run([{"role": "user", "content": "?"}])
        assert result.tool_calls_made == []

    def test_final_text_strips_thinking(self, tok):
        loop = _make_agent(tok, responses=["<think>internal</think>Visible answer."])
        result = loop.run([{"role": "user", "content": "think"}])
        assert "<think>" not in result.final_text
        assert "Visible answer." in result.final_text


class TestAgentSingleToolCall:
    def test_tool_dispatched_with_correct_name(self, tok):
        call_log: list[dict] = []

        def recording_add(a, b):
            call_log.append({"a": a, "b": b})
            return a + b

        tool = Tool(
            name="add",
            description="Add two numbers.",
            parameters={"type": "object", "properties": {}},
            fn=recording_add,
        )
        tc = '<tool_call>{"name": "add", "arguments": {"a": 3, "b": 4}}</tool_call>'
        loop = _make_agent(tok, responses=[tc, "The result is 7."], tools=[tool])
        result = loop.run([{"role": "user", "content": "add 3 and 4"}])

        assert len(call_log) == 1
        assert call_log[0] == {"a": 3, "b": 4}
        assert result.final_text == "The result is 7."

    def test_tool_result_injected_as_tool_role_message(self, tok):
        tool = _add_tool()
        tc = '<tool_call>{"name": "add", "arguments": {"a": 1, "b": 2}}</tool_call>'
        loop = _make_agent(tok, responses=[tc, "Sum is 3."], tools=[tool])
        result = loop.run([{"role": "user", "content": "calc"}])

        tool_msgs = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert "3" in tool_msgs[0]["content"]

    def test_tool_call_recorded_in_tool_calls_made(self, tok):
        tool = _add_tool()
        tc = '<tool_call>{"name": "add", "arguments": {"a": 10, "b": 20}}</tool_call>'
        loop = _make_agent(tok, responses=[tc, "Done."], tools=[tool])
        result = loop.run([{"role": "user", "content": "calc"}])

        assert len(result.tool_calls_made) == 1
        assert result.tool_calls_made[0].name == "add"
        assert result.tool_calls_made[0].arguments == {"a": 10, "b": 20}

    def test_n_turns_is_two_for_single_tool_call(self, tok):
        tool = _add_tool()
        tc = '<tool_call>{"name": "add", "arguments": {"a": 1, "b": 1}}</tool_call>'
        loop = _make_agent(tok, responses=[tc, "Result: 2."], tools=[tool])
        result = loop.run([{"role": "user", "content": "add"}])
        assert result.n_turns == 2

    def test_history_contains_all_roles(self, tok):
        tool = _add_tool()
        tc = '<tool_call>{"name": "add", "arguments": {"a": 5, "b": 5}}</tool_call>'
        loop = _make_agent(tok, responses=[tc, "10."], tools=[tool])
        result = loop.run([{"role": "user", "content": "go"}])

        roles = [m["role"] for m in result.messages]
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" in roles


class TestAgentMultipleTurns:
    def test_two_sequential_tool_calls_dispatched(self, tok):
        calls: list[str] = []

        def tracker(**kw):
            calls.append("called")
            return "ok"

        tool = Tool(name="f", description=".", parameters={"type": "object", "properties": {}}, fn=tracker)
        tc = '<tool_call>{"name": "f", "arguments": {}}</tool_call>'
        loop = _make_agent(tok, responses=[tc, tc, "Done."], tools=[tool])
        result = loop.run([{"role": "user", "content": "go"}])

        assert len(calls) == 2
        assert result.n_turns == 3
        assert len(result.tool_calls_made) == 2

    def test_multiple_tool_calls_in_single_response(self, tok):
        """Model emits two <tool_call> blocks in one response."""
        add = _add_tool()
        echo = _echo_tool()
        tc = (
            '<tool_call>{"name": "add", "arguments": {"a": 1, "b": 2}}</tool_call>'
            '<tool_call>{"name": "echo", "arguments": {"msg": "hi"}}</tool_call>'
        )
        loop = _make_agent(tok, responses=[tc, "All done."], tools=[add, echo])
        result = loop.run([{"role": "user", "content": "both"}])

        assert len(result.tool_calls_made) == 2
        names = [c.name for c in result.tool_calls_made]
        assert "add" in names
        assert "echo" in names

    def test_tool_result_content_visible_to_next_turn(self, tok):
        """The tool result must appear in result.messages so the model sees it."""
        tool = _echo_tool()
        tc = '<tool_call>{"name": "echo", "arguments": {"msg": "hello"}}</tool_call>'
        loop = _make_agent(tok, responses=[tc, "Got: hello."], tools=[tool])
        result = loop.run([{"role": "user", "content": "echo hi"}])

        tool_msgs = [m for m in result.messages if m["role"] == "tool"]
        assert any("hello" in m["content"] for m in tool_msgs)


class TestAgentErrorHandling:
    def test_unknown_tool_returns_error_string_as_tool_result(self, tok):
        tc = '<tool_call>{"name": "nonexistent", "arguments": {}}</tool_call>'
        loop = _make_agent(tok, responses=[tc, "Noted."], tools=[])
        result = loop.run([{"role": "user", "content": "?"}])

        tool_msgs = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert "Error" in tool_msgs[0]["content"]
        assert "nonexistent" in tool_msgs[0]["content"]

    def test_tool_exception_produces_error_string_not_crash(self, tok):
        tc = '<tool_call>{"name": "boom", "arguments": {}}</tool_call>'
        loop = _make_agent(tok, responses=[tc, "Noted."], tools=[_exploding_tool()])
        result = loop.run([{"role": "user", "content": "kaboom"}])

        tool_msgs = [m for m in result.messages if m["role"] == "tool"]
        assert "Error" in tool_msgs[0]["content"]
        assert "kaboom" in tool_msgs[0]["content"]

    def test_max_turns_stops_loop_before_final_answer(self, tok):
        """If the model keeps calling tools, the loop stops at max_turns."""
        tool = _echo_tool()
        tc = '<tool_call>{"name": "echo", "arguments": {"msg": "x"}}</tool_call>'
        # Provide 10 tool-call responses but set max_turns=3.
        loop = _make_agent(tok, responses=[tc] * 10, tools=[tool], max_turns=3)
        result = loop.run([{"role": "user", "content": "loop"}])
        assert result.n_turns == 3

    def test_tool_returns_dict_serialized_in_tool_message(self, tok):
        tool = Tool(
            name="info",
            description="Returns a dict.",
            parameters={"type": "object", "properties": {}},
            fn=lambda: {"status": "ok", "value": 42},
        )
        tc = '<tool_call>{"name": "info", "arguments": {}}</tool_call>'
        loop = _make_agent(tok, responses=[tc, "Info received."], tools=[tool])
        result = loop.run([{"role": "user", "content": "get info"}])

        tool_msgs = [m for m in result.messages if m["role"] == "tool"]
        parsed = json.loads(tool_msgs[0]["content"])
        assert parsed == {"status": "ok", "value": 42}


class TestAgentResultStructure:
    def test_result_is_agent_result_instance(self, tok):
        loop = _make_agent(tok, responses=["Hi."])
        result = loop.run([{"role": "user", "content": "hello"}])
        assert isinstance(result, AgentResult)

    def test_all_fields_present(self, tok):
        loop = _make_agent(tok, responses=["Hi."])
        result = loop.run([{"role": "user", "content": "hello"}])
        assert hasattr(result, "final_text")
        assert hasattr(result, "messages")
        assert hasattr(result, "n_turns")
        assert hasattr(result, "tool_calls_made")

    def test_messages_includes_original_user_message(self, tok):
        loop = _make_agent(tok, responses=["Response."])
        result = loop.run([{"role": "user", "content": "Original message"}])
        user_msgs = [m for m in result.messages if m["role"] == "user"]
        assert any("Original message" in m["content"] for m in user_msgs)

    def test_enable_thinking_does_not_crash(self, tok):
        loop = _make_agent(tok, responses=["<think>r</think>Answer."], enable_thinking=True)
        result = loop.run([{"role": "user", "content": "think"}])
        assert "Answer." in result.final_text

    def test_agent_with_system_message(self, tok):
        loop = _make_agent(tok, responses=["Done."])
        result = loop.run([
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ])
        assert result.final_text == "Done."
        assert result.messages[0]["role"] == "system"


# ===========================================================================
# Section 4 — _generate_to_eos integration (real MockLlamaModel)
# ===========================================================================


class TestGenerateToEos:
    """Test _generate_to_eos using the _MockLlamaModel (no monkey-patching)."""

    def test_generate_stops_at_eos(self, tok):
        # 65='A', 66='B', 67='C', 0=EOS
        model = _MockLlamaModel([65, 66, 67, 0])
        loop = AgentLoop(
            model=model,
            tokenizer=tok,
            cache_factory=lambda: None,
            eos_token_id=0,
            device="cpu",
        )
        gen_ids = loop._generate_to_eos([10, 11], None)
        assert gen_ids[-1] == 0
        assert 65 in gen_ids and 66 in gen_ids and 67 in gen_ids

    def test_generate_stops_at_max_new_tokens(self, tok):
        # All tokens are 42 (never EOS=0), so max_new_tokens must cap generation.
        model = _MockLlamaModel([42] * 1000)
        loop = AgentLoop(
            model=model,
            tokenizer=tok,
            cache_factory=lambda: None,
            eos_token_id=0,
            max_new_tokens=5,
            device="cpu",
        )
        gen_ids = loop._generate_to_eos([10], None)
        assert len(gen_ids) <= 5

    def test_decode_strips_trailing_eos(self, tok):
        """_decode removes the EOS token so it doesn't appear in returned text."""
        loop = AgentLoop(
            model=None,
            tokenizer=tok,
            cache_factory=lambda: None,
            eos_token_id=0,
            device="cpu",
        )
        # Encode some text then append EOS
        ids = tok.encode("hello") + [0]  # 0 = EOS
        decoded = loop._decode(ids)
        assert "<|im_end|>" not in decoded
        assert "hello" in decoded

    def test_generate_to_eos_full_round_trip(self, tok):
        """Prefill prompt → decode to EOS → text contains expected content."""
        # Pre-register the characters we'll use so they get stable IDs.
        _ = tok.encode("ok!")  # registers 'o', 'k', '!'
        ok_ids = tok.encode("ok!")
        model = _MockLlamaModel(ok_ids + [0])  # generate "ok!" then EOS

        loop = AgentLoop(
            model=model,
            tokenizer=tok,
            cache_factory=lambda: None,
            eos_token_id=0,
            device="cpu",
        )
        prompt_ids = tok.encode("prompt")
        gen_ids = loop._generate_to_eos(prompt_ids, None)
        text = loop._decode(gen_ids)
        assert "ok!" in text
