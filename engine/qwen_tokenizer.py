"""Qwen3 BPE tokenizer via tiktoken — no HuggingFace dependency.

Supports two on-disk formats found in Qwen model directories:
  • ``qwen.tiktoken``   — tiktoken's native base64-encoded vocab (Qwen2 and some Qwen3).
  • ``tokenizer.json`` — HuggingFace fast-tokenizer JSON (Qwen3 primary format).

Special tokens (``<|im_start|>``, ``<|im_end|>``, ``<tool_call>``, etc.) are loaded
from ``tokenizer_config.json`` when present, falling back to the Qwen3 defaults.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import tiktoken

# Qwen3 uses the same regex pattern as Qwen2 / tiktoken cl100k.
_QWEN_PAT_STR = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
    r"[^\r\n\p{L}\p{N}]?\p{L}+|"
    r"\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
    r"\s*[\r\n]+|"
    r"\s+(?!\S)|"
    r"\s+"
)

# Published Qwen3 special-token IDs (consistent across all dense variants).
_QWEN3_SPECIAL_TOKENS: dict[str, int] = {
    "<|endoftext|>": 151_643,
    "<|im_start|>": 151_644,
    "<|im_end|>": 151_645,
    "<|object_ref_start|>": 151_646,
    "<|object_ref_end|>": 151_647,
    "<|box_start|>": 151_648,
    "<|box_end|>": 151_649,
    "<|quad_start|>": 151_650,
    "<|quad_end|>": 151_651,
    "<|vision_start|>": 151_652,
    "<|vision_end|>": 151_653,
    "<|vision_pad|>": 151_654,
    "<|image_pad|>": 151_655,
    "<|video_pad|>": 151_656,
    "<tool_call>": 151_657,
    "</tool_call>": 151_658,
    "<tool_response>": 151_659,
    "</tool_response>": 151_660,
    "<think>": 151_668,
    "</think>": 151_669,
}


def _bytes_to_unicode_inverse() -> dict[int, int]:
    """Build the inverse of GPT-2's bytes_to_unicode mapping: unicode_ord → byte_value.

    GPT-2-style BPE stores token bytes as printable Unicode characters so that
    vocab strings are valid UTF-8 text.  The mapping is a bijection over 256 values;
    this function returns its inverse so that token strings can be converted back
    to raw bytes when loading ``tokenizer.json``.
    """
    # These byte values map to themselves (they are already printable ASCII / Latin-1).
    printable = (
        list(range(ord("!"), ord("~") + 1))      # 33..126
        + list(range(ord("¡"), ord("¬") + 1))    # 161..172
        + list(range(ord("®"), ord("ÿ") + 1))    # 174..255
    )
    unicode_ords = list(printable)               # forward mapping target
    byte_vals = list(printable)                  # identity for printable bytes

    # The 68 remaining bytes (0..32, 127..160, 173) map to 256, 257, …
    n = 0
    for b in range(256):
        if b not in printable:
            byte_vals.append(b)
            unicode_ords.append(256 + n)
            n += 1

    return dict(zip(unicode_ords, byte_vals))   # unicode_ord → byte_value


_U2B = _bytes_to_unicode_inverse()


class QwenTokenizer:
    """Qwen3 tokenizer backed by tiktoken.

    Loads from a model directory that contains either ``qwen.tiktoken``
    (tiktoken native) or ``tokenizer.json`` (HuggingFace fast-tokenizer format).
    Special tokens are read from ``tokenizer_config.json`` when present.

    Args:
        model_dir: Directory containing the tokenizer files.
    """

    def __init__(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)

        special_tokens = self._load_special_tokens(model_dir)

        tiktoken_file = model_dir / "qwen.tiktoken"
        json_file = model_dir / "tokenizer.json"

        if tiktoken_file.is_file():
            self._enc = self._from_tiktoken_file(tiktoken_file, special_tokens)
        elif json_file.is_file():
            self._enc = self._from_json_file(json_file, special_tokens)
        else:
            raise FileNotFoundError(
                f"No tokenizer found in {model_dir}. "
                "Expected 'qwen.tiktoken' or 'tokenizer.json'. "
                "Download the full model directory from HuggingFace."
            )

        self._special_tokens: dict[str, int] = special_tokens
        self._special_token_ids: frozenset[int] = frozenset(special_tokens.values())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def eos_token_id(self) -> int:
        """``<|im_end|>`` — end of assistant turn (151 645)."""
        return self._special_tokens["<|im_end|>"]

    @property
    def im_start_id(self) -> int:
        """``<|im_start|>`` token id (151 644)."""
        return self._special_tokens["<|im_start|>"]

    @property
    def im_end_id(self) -> int:
        """``<|im_end|>`` token id (151 645)."""
        return self._special_tokens["<|im_end|>"]

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Encode ``text`` to token ids.

        Args:
            text:               Input string (may contain special tokens in angle brackets).
            add_special_tokens: When True, special tokens in the text are tokenised as
                                single tokens rather than split into sub-pieces.

        Returns:
            List of integer token ids.
        """
        allowed = (
            frozenset(self._special_tokens.keys())
            if add_special_tokens
            else frozenset()
        )
        return self._enc.encode(text, allowed_special=allowed)

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token ids back to a string.

        Args:
            ids:                  Token ids to decode.
            skip_special_tokens:  When True, special tokens (``<|im_start|>`` etc.)
                                  are omitted from the output.

        Returns:
            Decoded string.
        """
        if skip_special_tokens:
            ids = [i for i in ids if i not in self._special_token_ids]
        return self._enc.decode(ids)

    def __len__(self) -> int:
        """Total vocabulary size including special tokens."""
        return self._enc.n_vocab

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_special_tokens(model_dir: Path) -> dict[str, int]:
        """Build special-token dict from tokenizer_config.json, falling back to defaults."""
        result = dict(_QWEN3_SPECIAL_TOKENS)
        config_path = model_dir / "tokenizer_config.json"
        if config_path.is_file():
            with open(config_path) as f:
                config = json.load(f)
            for id_str, info in config.get("added_tokens_decoder", {}).items():
                if info.get("special", False):
                    result[info["content"]] = int(id_str)
        return result

    @staticmethod
    def _load_tiktoken_bpe(path: Path) -> dict[bytes, int]:
        """Parse a tiktoken BPE vocab file: one ``base64_token rank`` per line."""
        ranks: dict[bytes, int] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token_b64, rank_str = line.split()
                ranks[base64.b64decode(token_b64)] = int(rank_str)
        return ranks

    @staticmethod
    def _from_tiktoken_file(
        path: Path, special_tokens: dict[str, int]
    ) -> tiktoken.Encoding:
        """Load from the native tiktoken binary vocab format."""
        ranks = QwenTokenizer._load_tiktoken_bpe(path)
        return tiktoken.Encoding(
            name="qwen",
            pat_str=_QWEN_PAT_STR,
            mergeable_ranks=ranks,
            special_tokens=special_tokens,
        )

    @staticmethod
    def _from_json_file(
        path: Path, special_tokens: dict[str, int]
    ) -> tiktoken.Encoding:
        """Load from HuggingFace tokenizer.json (GPT-2 bytes-to-unicode encoding)."""
        with open(path) as f:
            data = json.load(f)

        vocab: dict[str, int] = data["model"]["vocab"]

        # Reconstruct bytes → id mapping by reversing the bytes_to_unicode encoding.
        ranks: dict[bytes, int] = {}
        for token_str, token_id in vocab.items():
            token_bytes = bytes(_U2B.get(ord(c), ord(c)) for c in token_str)
            ranks[token_bytes] = token_id

        return tiktoken.Encoding(
            name="qwen",
            pat_str=_QWEN_PAT_STR,
            mergeable_ranks=ranks,
            special_tokens=special_tokens,
        )
