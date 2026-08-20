"""Model configuration for the Qwen3 inference engine.

Every dimension the engine uses is named and typed here — no bare integer
literals in model code. Supported families: LLaMA 3, SmolLM2, Qwen3 (dense
and MoE).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LlamaConfig:
    """Hyperparameters for a LLaMA-family model (LLaMA 3, Mistral, Qwen, etc.).

    Attributes:
        name:                 HuggingFace repo id used to locate weights.
        vocab_size:           Vocabulary size (128 256 for LLaMA 3).
        n_ctx:                Maximum sequence length (rope supports beyond this
                              with scaling, but we cap the precomputed table here).
        d_model:              Residual stream width (``hidden_size``).
        n_layer:              Number of transformer blocks.
        n_head:               Query head count (``num_attention_heads``).
        n_kv_heads:           Key/value head count for GQA (< ``n_head`` reduces
                              KV-cache size by ``n_kv_groups``×).
        intermediate_size:    SwiGLU hidden width (not 4×d_model for most models).
        rope_theta:           Base frequency for RoPE (500 000 for LLaMA 3).
        norm_eps:             Epsilon for RMSNorm.
        tie_word_embeddings:  When True ``lm_head`` shares weights with
                              ``embed_tokens`` (LLaMA 3.2 1B / 3B).
    """

    name: str
    vocab_size: int
    n_ctx: int
    d_model: int
    n_layer: int
    n_head: int
    n_kv_heads: int
    intermediate_size: int
    rope_theta: float = 500_000.0
    norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    qk_norm: bool = False
    # Explicit head dimension — Qwen3 uses head_dim=128 regardless of d_model/n_head.
    # 0 means derive automatically as d_model // n_head (correct for LLaMA).
    head_dim_override: int = 0
    # MoE fields (all default to 0 → dense model; set n_experts > 0 to enable MoE path).
    n_experts: int = 0                         # total routed expert count per layer
    n_experts_per_tok: int = 0                 # top-K experts activated per token
    moe_intermediate_size: int = 0             # per-routed-expert FFN hidden width
    shared_expert_intermediate_size: int = 0   # always-active shared expert hidden width

    @property
    def is_moe(self) -> bool:
        """True when this config describes a Mixture-of-Experts model."""
        return self.n_experts > 0

    @property
    def head_dim(self) -> int:
        if self.head_dim_override > 0:
            return self.head_dim_override
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model={self.d_model} not divisible by n_head={self.n_head}")
        return self.d_model // self.n_head

    @property
    def n_kv_groups(self) -> int:
        """Query heads per KV head (GQA grouping factor; 1 = standard MHA)."""
        if self.n_head % self.n_kv_heads != 0:
            raise ValueError(f"n_head={self.n_head} not divisible by n_kv_heads={self.n_kv_heads}")
        return self.n_head // self.n_kv_heads


# LLaMA 3.2 / 3 published configs.
LLAMA_3_2_1B = LlamaConfig(
    name="meta-llama/Llama-3.2-1B",
    vocab_size=128_256,
    n_ctx=131_072,
    d_model=2048,
    n_layer=16,
    n_head=32,
    n_kv_heads=8,
    intermediate_size=8192,
    rope_theta=500_000.0,
    norm_eps=1e-5,
    tie_word_embeddings=True,
)

LLAMA_3_2_3B = LlamaConfig(
    name="meta-llama/Llama-3.2-3B",
    vocab_size=128_256,
    n_ctx=131_072,
    d_model=3072,
    n_layer=28,
    n_head=24,
    n_kv_heads=8,
    intermediate_size=8192,
    rope_theta=500_000.0,
    norm_eps=1e-5,
    tie_word_embeddings=True,
)

LLAMA_3_8B = LlamaConfig(
    name="meta-llama/Meta-Llama-3-8B",
    vocab_size=128_256,
    n_ctx=8192,
    d_model=4096,
    n_layer=32,
    n_head=32,
    n_kv_heads=8,
    intermediate_size=14_336,
    rope_theta=500_000.0,
    norm_eps=1e-5,
    tie_word_embeddings=False,
)

# SmolLM2: Apache 2.0, no gating, identical LlamaForCausalLM weight format.
# Good for validating the architecture without needing a Meta license.
SMOLLM2_1_7B = LlamaConfig(
    name="HuggingFaceTB/SmolLM2-1.7B",
    vocab_size=49_152,
    n_ctx=8192,
    d_model=2048,
    n_layer=24,
    n_head=32,
    n_kv_heads=32,  # full MHA — actual config.json has num_key_value_heads=32
    intermediate_size=8192,
    rope_theta=130_000.0,
    norm_eps=1e-5,
    tie_word_embeddings=True,
)

# Qwen3 dense family.  All variants share:
#   vocab_size=151 936, rope_theta=1 000 000, norm_eps=1e-6, qk_norm=True.
# intermediate_size values verified from each model's config.json on HuggingFace.
QWEN3_0_6B = LlamaConfig(
    name="Qwen/Qwen3-0.6B",
    vocab_size=151_936,
    n_ctx=32_768,
    d_model=1024,
    n_layer=28,
    n_head=16,
    n_kv_heads=8,
    intermediate_size=3072,
    rope_theta=1_000_000.0,
    norm_eps=1e-6,
    tie_word_embeddings=True,
    qk_norm=True,
    head_dim_override=128,
)

QWEN3_1_7B = LlamaConfig(
    name="Qwen/Qwen3-1.7B",
    vocab_size=151_936,
    n_ctx=32_768,
    d_model=2048,
    n_layer=28,
    n_head=16,
    n_kv_heads=8,
    intermediate_size=8192,
    rope_theta=1_000_000.0,
    norm_eps=1e-6,
    tie_word_embeddings=True,
    qk_norm=True,
    head_dim_override=128,
)

QWEN3_4B = LlamaConfig(
    name="Qwen/Qwen3-4B",
    vocab_size=151_936,
    n_ctx=32_768,
    d_model=2560,
    n_layer=36,
    n_head=32,
    n_kv_heads=8,
    intermediate_size=9728,
    rope_theta=1_000_000.0,
    norm_eps=1e-6,
    qk_norm=True,
    head_dim_override=128,
)

QWEN3_8B = LlamaConfig(
    name="Qwen/Qwen3-8B",
    vocab_size=151_936,
    n_ctx=32_768,
    d_model=4096,
    n_layer=36,
    n_head=32,
    n_kv_heads=8,
    intermediate_size=22_016,
    rope_theta=1_000_000.0,
    norm_eps=1e-6,
    qk_norm=True,
    head_dim_override=128,
)

QWEN3_14B = LlamaConfig(
    name="Qwen/Qwen3-14B",
    vocab_size=151_936,
    n_ctx=32_768,
    d_model=5120,
    n_layer=40,
    n_head=40,
    n_kv_heads=8,
    intermediate_size=17_920,
    rope_theta=1_000_000.0,
    norm_eps=1e-6,
    qk_norm=True,
    head_dim_override=128,
)

QWEN3_32B = LlamaConfig(
    name="Qwen/Qwen3-32B",
    vocab_size=151_936,
    n_ctx=32_768,
    d_model=5120,
    n_layer=64,
    n_head=64,
    n_kv_heads=8,
    intermediate_size=25_600,
    rope_theta=1_000_000.0,
    norm_eps=1e-6,
    qk_norm=True,
    head_dim_override=128,
)

# Qwen3-30B-A3B: 30B total parameters, ~3B active per token.
# Verified against weights/Qwen--Qwen3-30B-A3B/config.json:
#   q_proj.weight (4096, 2048) → n_head=32, head_dim=128
#   k_proj.weight (512,  2048) → n_kv_heads=4, head_dim=128
#   experts.0.gate_proj.weight (768, 2048) → moe_intermediate_size=768
#   NO shared expert in this model — pure top-K routing only.
QWEN3_30B_A3B = LlamaConfig(
    name="Qwen/Qwen3-30B-A3B",
    vocab_size=151_936,
    n_ctx=40_960,
    d_model=2048,
    n_layer=48,
    n_head=32,
    n_kv_heads=4,
    intermediate_size=6144,      # dense FFN size from config.json (unused — is_moe=True)
    rope_theta=1_000_000.0,
    norm_eps=1e-6,
    qk_norm=True,
    head_dim_override=128,
    n_experts=128,
    n_experts_per_tok=8,
    moe_intermediate_size=768,
    shared_expert_intermediate_size=0,  # no shared expert in Qwen3-30B-A3B
)


LLAMA_CONFIGS: dict[str, LlamaConfig] = {
    LLAMA_3_2_1B.name: LLAMA_3_2_1B,
    LLAMA_3_2_3B.name: LLAMA_3_2_3B,
    LLAMA_3_8B.name: LLAMA_3_8B,
    SMOLLM2_1_7B.name: SMOLLM2_1_7B,
    QWEN3_0_6B.name: QWEN3_0_6B,
    QWEN3_1_7B.name: QWEN3_1_7B,
    QWEN3_4B.name: QWEN3_4B,
    QWEN3_8B.name: QWEN3_8B,
    QWEN3_14B.name: QWEN3_14B,
    QWEN3_32B.name: QWEN3_32B,
    QWEN3_30B_A3B.name: QWEN3_30B_A3B,
}


def get_llama_config(name: str) -> LlamaConfig:
    if name not in LLAMA_CONFIGS:
        raise KeyError(f"unknown llama model '{name}'; known: {sorted(LLAMA_CONFIGS)}")
    return LLAMA_CONFIGS[name]


