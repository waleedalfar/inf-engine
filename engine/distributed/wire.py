"""Binary frame protocol for shipping pipeline-stage activations over TCP.

Raw persistent TCP + fixed binary framing, not gRPC/protobuf: at decode time
this is potentially hundreds of tiny round-trips per second, so per-hop
framing overhead matters more than protocol flexibility.

Frame layout (all integers big-endian unsigned 32-bit)::

    [4B header_len][header_len bytes of JSON][4B tensor_len][tensor_len bytes]

The JSON header carries ``msg_type`` (str), a caller-defined ``meta`` dict
(e.g. ``start_pos``, ``position_ids``), and ``tensor_meta`` (``{"dtype",
"shape"}``, or ``None`` when the message carries no tensor — e.g. "close").
Tensor bytes are the tensor's raw row-major buffer (``numpy.tobytes()``);
dtype/shape in the header are enough to reconstruct it losslessly on the
other end without a serialization framework.
"""

from __future__ import annotations

import json
import socket
import struct

import numpy as np
import torch

_LEN = struct.Struct(">I")  # 4-byte big-endian unsigned length prefix


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Block until exactly ``n`` bytes are read; raise if the peer closes early."""
    if n == 0:
        return b""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(f"peer closed connection with {remaining} bytes still expected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_msg(
    sock: socket.socket,
    msg_type: str,
    meta: dict,
    tensor: torch.Tensor | None = None,
) -> None:
    """Encode and send one frame. ``tensor`` may be omitted (e.g. control messages)."""
    tensor_meta: dict | None = None
    tensor_bytes = b""
    if tensor is not None:
        arr = tensor.detach().to("cpu").contiguous().numpy()
        tensor_bytes = arr.tobytes()
        tensor_meta = {"dtype": str(arr.dtype), "shape": list(arr.shape)}

    header = {"msg_type": msg_type, "meta": meta, "tensor_meta": tensor_meta}
    header_bytes = json.dumps(header).encode("utf-8")

    frame = (
        _LEN.pack(len(header_bytes)) + header_bytes
        + _LEN.pack(len(tensor_bytes)) + tensor_bytes
    )
    sock.sendall(frame)


def recv_msg(sock: socket.socket) -> tuple[str, dict, torch.Tensor | None]:
    """Block until one full frame arrives; returns ``(msg_type, meta, tensor)``.

    ``tensor`` is ``None`` when the sender's frame carried no tensor.
    """
    (header_len,) = _LEN.unpack(recv_exact(sock, 4))
    header = json.loads(recv_exact(sock, header_len))
    (tensor_len,) = _LEN.unpack(recv_exact(sock, 4))
    tensor_bytes = recv_exact(sock, tensor_len)

    tensor: torch.Tensor | None = None
    tensor_meta = header["tensor_meta"]
    if tensor_meta is not None:
        arr = np.frombuffer(tensor_bytes, dtype=tensor_meta["dtype"]).reshape(tensor_meta["shape"])
        tensor = torch.from_numpy(arr.copy())  # copy: frombuffer's array is read-only

    return header["msg_type"], header["meta"], tensor
