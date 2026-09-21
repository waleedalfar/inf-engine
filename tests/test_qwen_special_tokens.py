"""Added-token ids must come from the checkpoint, not from a stale table.

Regression test for a bug that was silent in every benchmark and wrong in every
agentic run: `_load_special_tokens` only honoured entries flagged
``"special": true`` in tokenizer_config.json, but Qwen3 marks `<think>`,
`</think>`, `<tool_response>` and `</tool_response>` as ``false``. Those four
therefore kept hardcoded ids that were off by six or inverted, so

  * `<think>` encoded as 151668, which is really `</think>`,
  * tool results were wrapped in `<|fim_prefix|>` / `<|fim_middle|>`,
  * and a genuine `<think>` (151667) generated under --thinking hit
    `KeyError: Invalid token for decoding: 151667`.

HF's `special` flag means "strip me on decode", not "this token exists".
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from engine.qwen_tokenizer import QwenTokenizer

_WEIGHTS = Path("weights/Qwen--Qwen3-8B")
needs_weights = pytest.mark.skipif(
    not (_WEIGHTS / "tokenizer_config.json").is_file(),
    reason="needs a real Qwen3 checkpoint",
)


def test_non_special_added_tokens_are_still_registered():
    """A token with special=false must be encodable, and must not be stripped."""
    with tempfile.TemporaryDirectory() as d:
        md = Path(d)
        (md / "tokenizer_config.json").write_text(json.dumps({
            "added_tokens_decoder": {
                "151643": {"content": "<|endoftext|>", "special": True},
                "151645": {"content": "<|im_end|>", "special": True},
                "151667": {"content": "<think>", "special": False},
                "151668": {"content": "</think>", "special": False},
            }
        }))
        tokens, strippable = QwenTokenizer._load_special_tokens(md)

    # Every added token is registered, whatever its `special` flag.
    assert tokens["<think>"] == 151667
    assert tokens["</think>"] == 151668
    # Flagged ones are stripped on decode; the unflagged ones are not. (Other
    # defaults the fixture omits stay strippable, hence no exact-set assertion.)
    assert {151643, 151645} <= strippable
    assert 151667 not in strippable
    assert 151668 not in strippable


def test_checkpoint_wins_over_a_stale_default_id():
    """Defaults fill gaps, but never override the checkpoint — and never alias it.

    Two properties: the checkpoint's id for a name wins, and a default whose id
    the checkpoint gave to a *different* name is dropped rather than left as a
    second name for that id (tiktoken cannot take two names for one id).
    """
    with tempfile.TemporaryDirectory() as d:
        md = Path(d)
        (md / "tokenizer_config.json").write_text(json.dumps({
            "added_tokens_decoder": {
                # Real Qwen3 layout: the ids the stale table got wrong.
                "151665": {"content": "<tool_response>", "special": False},
                "151667": {"content": "<think>", "special": False},
                "151668": {"content": "</think>", "special": False},
            }
        }))
        tokens, _ = QwenTokenizer._load_special_tokens(md)

    assert tokens["<think>"] == 151667
    assert tokens["</think>"] == 151668
    assert tokens["<tool_response>"] == 151665
    # Defaults the checkpoint did not mention still fill in.
    assert tokens["<|im_start|>"] == 151644
    # No id maps from two different names.
    ids = list(tokens.values())
    assert len(ids) == len(set(ids)), "duplicate id would break tiktoken"


@needs_weights
def test_ids_match_the_checkpoint_and_round_trip():
    tok = QwenTokenizer(str(_WEIGHTS))
    truth = {
        info["content"]: int(i)
        for i, info in json.load(
            open(_WEIGHTS / "tokenizer_config.json"))["added_tokens_decoder"].items()
    }
    for name, want in truth.items():
        assert tok.encode(name, add_special_tokens=True) == [want], name
        assert tok.decode([want], skip_special_tokens=False) == name, name

    # The exact id and the exact call that used to raise KeyError.
    assert tok.decode([151667], skip_special_tokens=False) == "<think>"
    # `<think>` survives skip_special_tokens; `<|im_end|>` does not.
    assert tok.decode([151667], skip_special_tokens=True) == "<think>"
    assert tok.decode([151645], skip_special_tokens=True) == ""
