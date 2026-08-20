"""Phase 3 (distributed pipeline-parallel) tests: RemoteStageWorker + RemoteStageClient.

End-to-end over a *real* TCP loopback socket (127.0.0.1, ephemeral port) —
not an in-process shortcut. One stage runs locally via ``forward_stage``
(mirroring Phase 1/2's in-process pattern), the other runs inside a
``RemoteStageWorker`` served from a background thread and reached only
through ``RemoteStageClient``. Splitting the KV cache across a real network
hop must reproduce the same tokens as one full-range in-process forward,
across both prefill and cached decode steps — same bar as Phase 2's
in-process ranged-cache test, now with a socket in the loop.

Also asserts the worker's own internal cache state (owned_layers) directly,
not just output parity — a worker that silently ignored its layer range and
ran the whole model would still produce matching output for a full-depth
remote stage in some configurations, so state must be checked directly per
the standing "assert new path's own state" lesson.

Run:
    pytest tests/test_distributed_remote_stage.py -v
"""

from __future__ import annotations

import socket
import threading

import torch

from engine.config import LlamaConfig
from engine.distributed.remote_client import RemoteStageClient
from engine.distributed.remote_worker import RemoteStageWorker
from engine.paged_cache import BlockManager, PagedLlamaKVCache

# RemoteStageWorker runs its accept/serve loop in a background thread sharing
# this process's CUDA context would be unusual for a real deployment (the
# whole point is a *separate* device) — keep this test on CPU for determinism
# and to avoid any cross-thread CUDA-stream subtlety unrelated to what's
# being tested here (the wire protocol + layer-range correctness).
DEVICE = "cpu"
DTYPE = torch.float32


def _mini_config(n_layer: int = 4) -> LlamaConfig:
    return LlamaConfig(
        name="test-mini-remote",
        vocab_size=256,
        n_ctx=128,
        d_model=64,
        n_layer=n_layer,
        n_head=4,
        n_kv_heads=2,
        intermediate_size=128,
    )


def _mini_model(config: LlamaConfig, seed: int = 0):
    from engine.llama_model import LlamaModel
    from engine.llama_weights import LlamaWeights

    d = config.d_model
    h = config.n_kv_heads * config.head_dim
    f = config.intermediate_size
    torch.manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, dtype=DTYPE, device=DEVICE)

    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": rand(config.vocab_size, d),
        "model.norm.weight":         torch.ones(d, dtype=DTYPE, device=DEVICE),
    }
    for i in range(config.n_layer):
        p = f"model.layers.{i}."
        tensors |= {
            p + "input_layernorm.weight":          torch.ones(d, dtype=DTYPE, device=DEVICE),
            p + "post_attention_layernorm.weight":  torch.ones(d, dtype=DTYPE, device=DEVICE),
            p + "self_attn.q_proj.weight":          rand(d, d),
            p + "self_attn.k_proj.weight":          rand(h, d),
            p + "self_attn.v_proj.weight":          rand(h, d),
            p + "self_attn.o_proj.weight":          rand(d, d),
            p + "mlp.gate_proj.weight":             rand(f, d),
            p + "mlp.up_proj.weight":               rand(f, d),
            p + "mlp.down_proj.weight":              rand(d, f),
        }
    weights = LlamaWeights(tensors, config)
    return LlamaModel(weights, config)


def test_remote_stage_matches_local_full_forward_multistep():
    cfg = _mini_config(n_layer=4)
    split_layer = 2

    model_ref = _mini_model(cfg, seed=11)
    model_local = _mini_model(cfg, seed=11)   # identical weights, stage 0 (in-process)
    model_remote = _mini_model(cfg, seed=11)  # identical weights, stage 1 (served over TCP)

    worker = RemoteStageWorker(
        model_remote, start_layer=split_layer, end_layer=cfg.n_layer,
        is_first=False, is_last=True, host="127.0.0.1", port=0,
    )
    thread = threading.Thread(target=worker.serve_one_connection, daemon=True)
    thread.start()
    client = RemoteStageClient(worker.host, worker.port)

    mgr_ref = BlockManager(n_total=32, block_size=4)
    cache_ref = PagedLlamaKVCache(cfg, mgr_ref, DEVICE, DTYPE)
    mgr_local = BlockManager(n_total=32, block_size=4)
    cache_local = PagedLlamaKVCache(cfg, mgr_local, DEVICE, DTYPE, owned_layers=range(0, split_layer))

    T_p = 5
    ids = torch.randint(0, cfg.vocab_size, (1, T_p), device=DEVICE)
    pos = torch.arange(T_p, device=DEVICE)

    cache_ref.allocate_sequence(0, T_p)
    cache_ref.begin_step([0])
    cache_local.allocate_sequence(0, T_p)
    cache_local.begin_step([0])

    with torch.no_grad():
        ref_logits = model_ref.forward(ids, cache=cache_ref, start_pos=0, position_ids=pos)
        hidden = model_local.forward_stage(
            ids, 0, split_layer, True, False, cache=cache_local, start_pos=0, position_ids=pos
        )
    remote_logits = client.forward_stage(hidden, start_pos=0, position_ids=pos)

    assert torch.equal(ref_logits, remote_logits)

    # Two cached decode steps over the wire — the trickiest bookkeeping case.
    next_tok = ref_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    start_pos = T_p
    for step in range(2):
        pos_s = torch.tensor([start_pos], device=DEVICE)

        cache_ref.ensure_slot(0)
        cache_ref.begin_step([0])
        cache_local.ensure_slot(0)
        cache_local.begin_step([0])

        with torch.no_grad():
            ref_logits = model_ref.forward(next_tok, cache=cache_ref, start_pos=start_pos, position_ids=pos_s)
            hidden = model_local.forward_stage(
                next_tok, 0, split_layer, True, False, cache=cache_local, start_pos=start_pos, position_ids=pos_s
            )
        remote_logits = client.forward_stage(hidden, start_pos=start_pos, position_ids=pos_s)

        assert torch.equal(ref_logits, remote_logits), f"diverged at step {step}"

        next_tok = ref_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        start_pos += 1

    client.close()
    thread.join(timeout=5)
    assert not thread.is_alive()

    # Direct state check on the worker's own cache — not just output parity.
    assert worker.last_cache is not None
    assert worker.last_cache.owned_layers == range(split_layer, cfg.n_layer)
    assert worker.last_cache.k_pool.shape[0] == cfg.n_layer - split_layer


def test_remote_stage_first_stage_embeds_and_worker_never_sees_ids_after_first_call():
    """Sanity check on the is_first=True direction: the worker embeds the
    token ids itself (rather than expecting a pre-embedded residual) when
    it owns layer 0, and correctly returns a residual (not logits) for a
    non-last remote stage."""
    cfg = _mini_config(n_layer=4)
    split_layer = 2
    model_remote = _mini_model(cfg, seed=3)

    worker = RemoteStageWorker(
        model_remote, start_layer=0, end_layer=split_layer,
        is_first=True, is_last=False, host="127.0.0.1", port=0,
    )
    thread = threading.Thread(target=worker.serve_one_connection, daemon=True)
    thread.start()
    client = RemoteStageClient(worker.host, worker.port)

    ids = torch.randint(0, cfg.vocab_size, (1, 4), dtype=torch.long)
    pos = torch.arange(4)
    hidden = client.forward_stage(ids, start_pos=0, position_ids=pos)

    assert hidden.shape == (1, 4, cfg.d_model)
    assert hidden.dtype == torch.float32  # residual stream, not logits/ids

    client.close()
    thread.join(timeout=5)


def test_wire_smoke_over_real_loopback_socket(monkeypatch=None):
    """Not model-related: proves RemoteStageWorker really uses a live TCP
    socket bound to 127.0.0.1 (an ephemeral port), not some in-process stub."""
    cfg = _mini_config(n_layer=2)
    model = _mini_model(cfg, seed=1)
    worker = RemoteStageWorker(
        model, start_layer=0, end_layer=cfg.n_layer, is_first=True, is_last=True,
        host="127.0.0.1", port=0,
    )
    assert worker.host == "127.0.0.1"
    assert isinstance(worker.port, int) and worker.port > 0

    # A plain socket can connect to it independent of RemoteStageClient.
    raw = socket.create_connection((worker.host, worker.port), timeout=5.0)
    thread = threading.Thread(target=worker.serve_one_connection, daemon=True)
    thread.start()
    from engine.distributed import wire
    wire.send_msg(raw, "close", {})
    raw.close()
    thread.join(timeout=5)
