"""GPT-2 BPE tokenizer via ``tiktoken``.

We use ``tiktoken``'s ``gpt2`` encoding rather than a HuggingFace tokenizer so the
engine package has no HuggingFace dependency at all. The two produce identical
token ids for GPT-2; ``tests/`` asserts that parity explicitly so the
token-for-token correctness gate is trustworthy.
"""

from __future__ import annotations

import tiktoken


class GPT2Tokenizer:
    """Thin wrapper over tiktoken's GPT-2 BPE encoding."""

    def __init__(self) -> None:
        self._enc = tiktoken.get_encoding("gpt2")

    @property
    def eot_token(self) -> int:
        """End-of-text token id (50256 for GPT-2)."""
        return self._enc.eot_token

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token ids."""
        return self._enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back to text."""
        return self._enc.decode(ids)
