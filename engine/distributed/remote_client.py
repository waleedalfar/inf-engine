"""Synchronous client for one ``RemoteStageWorker`` connection.

Every call blocks until the remote stage's response arrives. Phase 5 will
wrap this in a future/callback-based async API so the local GPU stage can
keep working on other sessions while a remote hop for one session is in
flight; this synchronous client is what Phase 4's synchronous
``DistributedPagedEngine`` builds on first.
"""

from __future__ import annotations

import socket

import torch

from engine.distributed import wire


class RemoteStageClient:
    def __init__(self, host: str, port: int, timeout: float | None = 30.0) -> None:
        self._sock = socket.create_connection((host, port), timeout=timeout)

    def forward_stage(
        self,
        x: torch.Tensor,
        start_pos: int,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Ship ``x`` to the remote stage and block for its output tensor."""
        meta = {
            "start_pos": start_pos,
            "position_ids": position_ids.tolist() if position_ids is not None else None,
        }
        wire.send_msg(self._sock, "forward", meta, x)
        msg_type, _meta, tensor = wire.recv_msg(self._sock)
        if msg_type != "forward_result":
            raise RuntimeError(f"unexpected response msg_type {msg_type!r}")
        return tensor

    def close(self) -> None:
        try:
            wire.send_msg(self._sock, "close", {})
        finally:
            self._sock.close()
