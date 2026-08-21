"""Profile the components of a single CUDA-graph decode step.

Breaks down: graph replay vs sampling vs gen_ctx bookkeeping vs per-step
Python overhead, so we can tell whether the kernel or the eager tail is the
bottleneck.

Usage:
    python -m bench.graph_step_profile --model-dir weights/Qwen--Qwen3-8B
"""

from __future__ import annotations

import argparse
import time

import torch


_PROMPT_LEN = 777
_N_MEASURE = 200
_WARMUP = 20


def _sync_time(fn, n: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000  # ms per call


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    args = p.parse_args()

    import main as cli
    from engine.llama_paged_engine import LlamaPagedEngine, LlamaRequest
    from engine.sampling import SamplingConfig, SamplingMode

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("Requires CUDA.")

    dtype = torch.bfloat16
    config = cli.detect_config(args.model_dir)
    model = cli.load_model(args.model_dir, config, device, dtype, quantize=True)

    sampling = SamplingConfig(mode=SamplingMode.GREEDY)

    # Build a warm paged engine with graph already captured.
    block_size = 16
    n_blocks = (_PROMPT_LEN + _N_MEASURE + block_size) // block_size + 64
    engine = LlamaPagedEngine(
        model, n_total_blocks=n_blocks, block_size=block_size,
        eos_token=None, sampling=sampling, enable_cuda_graphs=True,
    )

    # Prefill to establish KV state.
    prompt_ids = list(range(1, _PROMPT_LEN + 1))
    req = LlamaRequest(req_id=0, prompt_ids=prompt_ids, max_new_tokens=_N_MEASURE + _WARMUP + 10)
    engine.submit(req)
    engine.step()  # prefill only (no active sequences yet after this for decode)

    # At this point engine._active has the seq. Run warmup decode steps to trigger
    # graph capture, then time the individual pieces.
    print(f"Warming up {_WARMUP} decode steps (triggers graph capture) ...")
    for _ in range(_WARMUP):
        engine.step()

    # Now grab the captured graph internals.
    assert engine._graphs, "No CUDA graph captured — something went wrong"
    bucket = min(engine._graphs.keys(), key=lambda k: k[0] * 1000 + k[1])
    cg = engine._graphs[bucket]
    print(f"Using captured graph bucket: batch={cg.bucket_size}, len={cg.capture_len}")

    seq_ids = list(engine._active.keys())
    assert len(seq_ids) == 1

    # ---- Measure 1: raw graph replay only ----
    def _replay_only():
        cg.graph.replay()

    ms_replay = _sync_time(_replay_only, _N_MEASURE)

    # ---- Measure 2: replay + sampling (no context_ids) ----
    def _replay_sample():
        cg.graph.replay()
        from engine.sampling import sample_next_token
        sample_next_token(cg.logits[:1, -1, :], sampling)

    ms_replay_sample = _sync_time(_replay_sample, _N_MEASURE)

    # ---- Measure 3: full _decode_step_graphed equivalent ----
    def _full_step():
        engine._fill_graph_inputs(cg, seq_ids)
        cg.graph.replay()
        from engine.sampling import sample_next_token
        nxt = sample_next_token(cg.logits[:1, -1, :], sampling)
        _ = nxt.view(-1).tolist()  # GPU→CPU sync

    ms_full_step = _sync_time(_full_step, _N_MEASURE)

    # ---- Measure 4: full engine.step() call ----
    # Re-submit so we have live state.
    # (The above measurements don't advance seq state, so we need a fresh engine.)
    engine2 = LlamaPagedEngine(
        model, n_total_blocks=n_blocks, block_size=block_size,
        eos_token=None, sampling=sampling, enable_cuda_graphs=True,
    )
    req2 = LlamaRequest(req_id=0, prompt_ids=prompt_ids, max_new_tokens=_N_MEASURE + _WARMUP + 10)
    engine2.submit(req2)
    engine2.step()
    for _ in range(_WARMUP):
        engine2.step()

    def _engine_step():
        engine2.step()

    ms_engine_step = _sync_time(_engine_step, _N_MEASURE)

    print(f"\n=== Decode step breakdown at prompt_len={_PROMPT_LEN}, batch=1 ===")
    print(f"  graph replay only:           {ms_replay:.3f} ms  ({1000/ms_replay:.1f} tok/s ceiling)")
    print(f"  replay + sampling:           {ms_replay_sample:.3f} ms  ({1000/ms_replay_sample:.1f} tok/s)")
    print(f"  replay + sample + CPU sync:  {ms_full_step:.3f} ms  ({1000/ms_full_step:.1f} tok/s)")
    print(f"  full engine.step():          {ms_engine_step:.3f} ms  ({1000/ms_engine_step:.1f} tok/s)")
    print()
    print(f"  overhead breakdown:")
    print(f"    fill_graph_inputs:         {ms_full_step - ms_replay_sample:.3f} ms")
    print(f"    sampling (CUDA op):        {ms_replay_sample - ms_replay:.3f} ms")
    print(f"    engine bookkeeping:        {ms_engine_step - ms_full_step:.3f} ms")
    print(f"    raw replay:                {ms_replay:.3f} ms")


if __name__ == "__main__":
    main()
