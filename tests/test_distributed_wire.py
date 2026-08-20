"""Phase 3 (distributed pipeline-parallel) tests: engine/distributed/wire.py.

Exercises the binary frame protocol over real socket pairs (not mocked) —
send_msg on one end, recv_msg on the other, assert bit-identical tensor
round-trip and exact metadata round-trip, for both the "carries a tensor"
and "control message, no tensor" cases, and for a non-float dtype (token
ids) since numpy dtype-string round-tripping is exactly the kind of thing
that silently corrupts if a dtype is ever encoded/decoded inconsistently.

Run:
    pytest tests/test_distributed_wire.py -v
"""

from __future__ import annotations

import socket

import pytest
import torch

from engine.distributed import wire


@pytest.fixture()
def sock_pair():
    a, b = socket.socketpair()
    yield a, b
    a.close()
    b.close()


def test_roundtrip_float_tensor_and_meta(sock_pair):
    a, b = sock_pair
    t = torch.randn(2, 3, 4)
    wire.send_msg(a, "forward", {"start_pos": 5, "position_ids": [5, 6, 7]}, t)

    msg_type, meta, recv_t = wire.recv_msg(b)

    assert msg_type == "forward"
    assert meta == {"start_pos": 5, "position_ids": [5, 6, 7]}
    assert recv_t is not None
    assert recv_t.shape == t.shape
    assert recv_t.dtype == t.dtype
    assert torch.equal(recv_t, t)


def test_roundtrip_no_tensor_control_message(sock_pair):
    a, b = sock_pair
    wire.send_msg(a, "close", {})

    msg_type, meta, recv_t = wire.recv_msg(b)

    assert msg_type == "close"
    assert meta == {}
    assert recv_t is None


def test_roundtrip_int_dtype_preserved(sock_pair):
    """Token ids are int64 — a dtype-string bug would silently produce floats."""
    a, b = sock_pair
    ids = torch.randint(0, 1000, (1, 7), dtype=torch.long)
    wire.send_msg(a, "forward", {"start_pos": 0, "position_ids": None}, ids)

    _msg_type, meta, recv_t = wire.recv_msg(b)

    assert meta["position_ids"] is None
    assert recv_t.dtype == torch.long
    assert torch.equal(recv_t, ids)


def test_multiple_frames_in_sequence(sock_pair):
    """Frames must not bleed into each other on a persistent connection."""
    a, b = sock_pair
    t1 = torch.randn(1, 1, 8)
    t2 = torch.randn(1, 4, 8)

    wire.send_msg(a, "forward", {"start_pos": 0}, t1)
    wire.send_msg(a, "forward", {"start_pos": 1}, t2)

    _mt1, meta1, r1 = wire.recv_msg(b)
    _mt2, meta2, r2 = wire.recv_msg(b)

    assert meta1["start_pos"] == 0 and torch.equal(r1, t1)
    assert meta2["start_pos"] == 1 and torch.equal(r2, t2)


def test_recv_exact_raises_on_early_close(sock_pair):
    a, b = sock_pair
    a.close()
    with pytest.raises(ConnectionError):
        wire.recv_msg(b)
