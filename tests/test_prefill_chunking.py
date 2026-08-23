"""Prefill chunking: same answer, bounded activation memory.

Activation memory scales with tokens in flight, not with what the KV cache
holds. At 24k tokens Qwen3-8B's fused gate_up projection alone is a 2.16 GB
tensor and the SwiGLU product another 1.08 GB, which put a 16 GB card at 90%
occupancy and turned prefill into allocator thrash (100 s for 24k tokens).

Chunking bounds that at ``prefill_chunk`` tokens. Each chunk attends over
everything already cached — llama_attention's offset-causal case — so the result
must be *identical* to a single pass, not merely close.

Run:
    pytest tests/test_prefill_chunking.py -v
"""

from __future__ import annotations

import pytest
import torch

from engine.config import LlamaConfig
from engine.llama_model import LlamaModel
from engine.llama_paged_engine import LlamaPagedEngine, LlamaRequest
from engine.llama_weights import LlamaWeights
from engine.sampling import SamplingConfig, SamplingMode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def _config() -> LlamaConfig:
    return LlamaConfig(
        name="test-chunked-prefill",
        vocab_size=64,
        n_ctx=512,
        d_model=32,
        n_layer=2,
        n_head=4,
        n_kv_heads=2,
        intermediate_size=64,
        rope_theta=10_000.0,
        norm_eps=1e-5,
        qk_norm=True,
    )


def _model(config: LlamaConfig, seed: int = 0) -> LlamaModel:
    d, f = config.d_model, config.intermediate_size
    h = config.n_kv_heads * config.head_dim
    torch.manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, dtype=DTYPE, device=DEVICE)

    tensors = {
        "model.embed_tokens.weight": rand(config.vocab_size, d),
        "model.norm.weight": torch.ones(d, dtype=DTYPE, device=DEVICE),
    }
    for i in range(config.n_layer):
        p = f"model.layers.{i}."
        tensors |= {
            p + "input_layernorm.weight": torch.ones(d, dtype=DTYPE, device=DEVICE),
            p + "post_attention_layernorm.weight": torch.ones(d, dtype=DTYPE, device=DEVICE),
            p + "self_attn.q_proj.weight": rand(config.n_head * config.head_dim, d),
            p + "self_attn.k_proj.weight": rand(h, d),
            p + "self_attn.v_proj.weight": rand(h, d),
            p + "self_attn.o_proj.weight": rand(d, config.n_head * config.head_dim),
            p + "mlp.gate_proj.weight": rand(f, d),
            p + "mlp.up_proj.weight": rand(f, d),
            p + "mlp.down_proj.weight": rand(d, f),
            p + "self_attn.q_norm.weight": rand(config.head_dim).abs(),
            p + "self_attn.k_norm.weight": rand(config.head_dim).abs(),
        }
    return LlamaModel(LlamaWeights(tensors, config), config)


def _generate(model, cfg, prompt, chunk, max_new=12):
    engine = LlamaPagedEngine(
        model, n_total_blocks=200, block_size=16, eos_token=None,
        sampling=SamplingConfig(mode=SamplingMode.GREEDY),
        enable_cuda_graphs=False, prefill_chunk=chunk,
    )
    req = LlamaRequest(req_id=0, prompt_ids=prompt, max_new_tokens=max_new)
    engine.run_offline([req])
    return req.generated


@pytest.mark.parametrize("prompt_len", [1, 15, 16, 17, 64, 100])
@pytest.mark.parametrize("chunk", [8, 16, 32])
def test_chunked_prefill_matches_single_pass(prompt_len, chunk):
    """Identical tokens, not merely close.

    Prompt lengths straddle the block size (16) and chunk boundaries, since an
    off-by-one in position ids or slot allocation shows up exactly there.
    """
    cfg = _config()
    model = _model(cfg, seed=3)
    torch.manual_seed(0)
    prompt = torch.randint(0, cfg.vocab_size, (prompt_len,)).tolist()

    single = _generate(model, cfg, prompt, chunk=0)
    chunked = _generate(model, cfg, prompt, chunk=chunk)
    assert chunked == single, (
        f"prompt_len={prompt_len} chunk={chunk}\n"
        f"  single-pass: {single}\n  chunked:     {chunked}"
    )


def test_chunk_larger_than_prompt_is_a_single_pass():
    cfg = _config()
    model = _model(cfg, seed=4)
    prompt = [1, 2, 3, 4, 5]
    assert _generate(model, cfg, prompt, chunk=4096) == _generate(model, cfg, prompt, chunk=0)


def test_default_engine_chunks_prefill():
    """The default must actually be chunked — that is the point of the feature."""
    cfg = _config()
    engine = LlamaPagedEngine(_model(cfg), n_total_blocks=64, block_size=16)
    assert engine.prefill_chunk > 0


@pytest.mark.skipif(DEVICE != "cuda", reason="peak-memory accounting needs CUDA")
def test_chunking_lowers_peak_memory():
    """Bounding tokens in flight must actually reduce the peak, not just tidy code."""
    cfg = LlamaConfig(**{**_config().__dict__, "n_ctx": 4096,
                         "d_model": 256, "intermediate_size": 2048})
    model = _model(cfg, seed=5)
    torch.manual_seed(0)
    prompt = torch.randint(0, cfg.vocab_size, (2048,)).tolist()

    peaks = {}
    for chunk in (0, 128):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _generate(model, cfg, prompt, chunk=chunk, max_new=2)
        peaks[chunk] = torch.cuda.max_memory_allocated()

    assert peaks[128] < peaks[0], (
        f"chunked peak {peaks[128]/1e6:.0f} MB not below single-pass "
        f"{peaks[0]/1e6:.0f} MB"
    )
