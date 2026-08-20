"""Serves one pipeline stage's forward_stage over a persistent TCP connection.

Single-owner-thread pattern (mirrors ``PagedSessionManager`` in
``paged_session.py``): one thread accepts a connection and exclusively
drives this stage's model + KV cache state for that connection's lifetime.
No other thread touches ``self.model``/the per-connection cache.

Phase 3 scope: one connection, one sequence, synchronous request/response —
no continuous batching yet. Phase 4's ``DistributedPagedEngine`` will reuse
this stage-serving shape (own layers, own ranged KV cache, network hop for
the residual stream) rather than duplicating it, generalized to many
concurrent sequences.
"""

from __future__ import annotations

import socket

import torch

from engine.distributed import wire
from engine.llama_model import LlamaModel
from engine.paged_cache import BlockManager, PagedLlamaKVCache


class RemoteStageWorker:
    """Owns layers ``[start_layer, end_layer)`` of ``model``; serves them over TCP.

    Args:
        model:           Loaded model (already sliced to just the layers this
                          worker needs, if memory-constrained — the forward
                          only touches ``[start_layer, end_layer)`` regardless).
        start_layer, end_layer: This stage's layer range (matches the
                          ``forward_stage``/``PagedLlamaKVCache.owned_layers``
                          convention: end-exclusive).
        is_first, is_last: Whether this stage owns the embedding lookup /
                          final norm+lm_head.
        host, port:       Bind address. ``port=0`` picks an ephemeral free
                          port — read it back via ``self.port`` after construction.
        n_total_blocks, block_size: Sized for a single sequence's KV cache
                          (Phase 3 has no continuous batching).
    """

    def __init__(
        self,
        model: LlamaModel,
        start_layer: int,
        end_layer: int,
        is_first: bool,
        is_last: bool,
        host: str = "0.0.0.0",
        port: int = 0,
        n_total_blocks: int = 256,
        block_size: int = 16,
    ) -> None:
        self.model = model
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.is_first = is_first
        self.is_last = is_last
        self.device = str(model.w.embed_tokens.device)
        self.dtype = model.w.embed_tokens.dtype
        self.n_total_blocks = n_total_blocks
        self.block_size = block_size
        # Set once per served connection; exposed for tests/introspection to
        # assert this worker actually built a layer-ranged cache, not a
        # silently-full-range one (state check, not just output parity).
        self.last_cache: PagedLlamaKVCache | None = None

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(1)
        self.host, self.port = self._sock.getsockname()

    def serve_one_connection(self) -> None:
        """Accept exactly one connection and serve it until the client closes it.

        Blocking — call from a dedicated thread/process.
        """
        conn, _addr = self._sock.accept()
        try:
            self._handle_connection(conn)
        finally:
            conn.close()

    def close(self) -> None:
        self._sock.close()

    def _handle_connection(self, conn: socket.socket) -> None:
        manager = BlockManager(self.n_total_blocks, self.block_size)
        cache = PagedLlamaKVCache(
            self.model.config, manager, self.device, self.dtype,
            owned_layers=range(self.start_layer, self.end_layer),
        )
        self.last_cache = cache
        seq_id = 0
        allocated = False

        while True:
            msg_type, meta, tensor = wire.recv_msg(conn)
            if msg_type == "close":
                return
            if msg_type != "forward":
                raise ValueError(f"unexpected msg_type {msg_type!r}")

            x = tensor.to(self.device)
            start_pos = meta["start_pos"]
            position_ids = (
                torch.tensor(meta["position_ids"], device=self.device)
                if meta.get("position_ids") is not None else None
            )

            if not allocated:
                cache.allocate_sequence(seq_id, x.shape[1])
                allocated = True
            cache.begin_step([seq_id])

            with torch.no_grad():
                out = self.model.forward_stage(
                    x, self.start_layer, self.end_layer, self.is_first, self.is_last,
                    cache=cache, start_pos=start_pos, position_ids=position_ids,
                )
            cache.ensure_slot(seq_id)  # grow the block table for the *next* forward, if needed

            wire.send_msg(conn, "forward_result", {}, out)
