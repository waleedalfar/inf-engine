"""Phase 3 (distributed pipeline-parallel) smoke test: bench/lan_hop_bench.py.

Only proves the benchmark script's server/client plumbing works over a real
socket (127.0.0.1) — it does NOT measure real LAN latency. The actual Phase 3
decision-gate numbers must come from running this script across the two
physical machines (see the module docstring in bench/lan_hop_bench.py);
that real cross-machine run is outside what this environment can execute.

Run:
    pytest tests/test_lan_hop_bench_smoke.py -v
"""

from __future__ import annotations

import socket
import threading

import torch

from bench.lan_hop_bench import _time_roundtrips, run_server
from engine.distributed import wire


def test_bench_server_echoes_tensor_over_loopback():
    srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv_sock.bind(("127.0.0.1", 0))
    srv_sock.listen(1)
    host, port = srv_sock.getsockname()

    thread = threading.Thread(target=run_server, args=(host, port, srv_sock), daemon=True)
    thread.start()

    client_sock = socket.create_connection((host, port), timeout=5.0)
    payload = torch.randn(1, 1, 64)
    times = _time_roundtrips(client_sock, payload, n_trials=5)

    assert len(times) == 5
    assert all(t >= 0 for t in times)

    wire.send_msg(client_sock, "close", {})
    client_sock.close()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_bench_server_echo_is_bit_identical():
    srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv_sock.bind(("127.0.0.1", 0))
    srv_sock.listen(1)
    host, port = srv_sock.getsockname()

    thread = threading.Thread(target=run_server, args=(host, port, srv_sock), daemon=True)
    thread.start()

    client_sock = socket.create_connection((host, port), timeout=5.0)
    payload = torch.randn(1, 8, 32)
    wire.send_msg(client_sock, "forward", {}, payload)
    msg_type, _meta, echoed = wire.recv_msg(client_sock)

    assert msg_type == "echo"
    assert torch.equal(echoed, payload)

    wire.send_msg(client_sock, "close", {})
    client_sock.close()
    thread.join(timeout=5)
