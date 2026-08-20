"""Real network-hop benchmark — Phase 3 decision gate for the distributed
pipeline-parallel plan (see the plan file referenced from
memory/project_distributed_pipeline.md).

Measures pure wire round-trip time and throughput (socket send/recv +
(de)serialize overhead of ``engine/distributed/wire.py``) for decode-sized
(tiny, frequent) and prefill-sized (large, infrequent) payloads. No model is
loaded — this isolates the network hop's cost from per-layer compute time,
which is the number needed to decide whether Phase 5's async bubble-free
scheduling is worth its complexity, and whether decode-time activation
quantization (currently scoped to prefill-only) should be reconsidered.

This must be run across the two *physical* machines to mean anything —
looping back through 127.0.0.1 only proves the script works; it does not
measure real LAN latency.

Usage:
    # On the Mac (or whichever machine hosts the "remote" pipeline stage):
    python -m bench.lan_hop_bench server --host 0.0.0.0 --port 9876

    # On the 5070 Ti box:
    python -m bench.lan_hop_bench client --host <mac-ip> --port 9876 \\
        --d-model 4096 --decode-trials 500 --prefill-tokens 512 --prefill-trials 20
"""

from __future__ import annotations

import argparse
import socket
import time

import torch

from engine.distributed import wire


def run_server(host: str, port: int, sock: socket.socket | None = None) -> None:
    """Accept one client and echo back every tensor it sends until "close".

    ``sock``, if given, must already be bound+listening (used by tests to
    avoid a bind/accept race against a background thread); otherwise a new
    socket is created, bound, and closed on exit.
    """
    owns_socket = sock is None
    if sock is None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(1)
    print(f"[lan_hop_bench] listening on {sock.getsockname()}, waiting for one client...")
    conn, addr = sock.accept()
    print(f"[lan_hop_bench] client connected from {addr}")
    try:
        while True:
            msg_type, _meta, tensor = wire.recv_msg(conn)
            if msg_type == "close":
                break
            wire.send_msg(conn, "echo", {}, tensor)
    finally:
        conn.close()
        if owns_socket:
            sock.close()


def _time_roundtrips(sock: socket.socket, tensor: torch.Tensor, n_trials: int) -> list[float]:
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        wire.send_msg(sock, "forward", {}, tensor)
        wire.recv_msg(sock)
        times.append(time.perf_counter() - t0)
    return times


def _summarize(label: str, times: list[float], payload_bytes: int) -> None:
    times_sorted = sorted(times)
    n = len(times_sorted)
    mean_s = sum(times_sorted) / n
    p50 = times_sorted[n // 2]
    p99 = times_sorted[min(n - 1, int(n * 0.99))]
    mb = payload_bytes / 1e6
    print(
        f"[{label}] n={n} payload={mb:.3f}MB  "
        f"mean={mean_s * 1000:.3f}ms  p50={p50 * 1000:.3f}ms  p99={p99 * 1000:.3f}ms  "
        f"throughput={mb / mean_s:.1f} MB/s"
    )


def run_client(
    host: str,
    port: int,
    d_model: int,
    decode_trials: int,
    prefill_tokens: int,
    prefill_trials: int,
) -> None:
    sock = socket.create_connection((host, port), timeout=30.0)
    try:
        decode_payload = torch.randn(1, 1, d_model, dtype=torch.float32)
        decode_times = _time_roundtrips(sock, decode_payload, decode_trials)
        _summarize("decode (1 token)", decode_times, decode_payload.numel() * 4)

        prefill_payload = torch.randn(1, prefill_tokens, d_model, dtype=torch.float32)
        prefill_times = _time_roundtrips(sock, prefill_payload, prefill_trials)
        _summarize(f"prefill ({prefill_tokens} tokens)", prefill_times, prefill_payload.numel() * 4)
    finally:
        wire.send_msg(sock, "close", {})
        sock.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    ps = sub.add_parser("server", help="Run on the remote (e.g. Mac) side.")
    ps.add_argument("--host", default="0.0.0.0")
    ps.add_argument("--port", type=int, default=9876)

    pc = sub.add_parser("client", help="Run on the 5070 Ti box; connects to the server.")
    pc.add_argument("--host", required=True, help="Server's LAN IP.")
    pc.add_argument("--port", type=int, default=9876)
    pc.add_argument("--d-model", type=int, default=4096, help="Target model's d_model (Qwen3-8B=4096).")
    pc.add_argument("--decode-trials", type=int, default=500)
    pc.add_argument("--prefill-tokens", type=int, default=512)
    pc.add_argument("--prefill-trials", type=int, default=20)

    args = p.parse_args()
    if args.mode == "server":
        run_server(args.host, args.port)
    else:
        run_client(
            args.host, args.port, args.d_model,
            args.decode_trials, args.prefill_tokens, args.prefill_trials,
        )


if __name__ == "__main__":
    main()
