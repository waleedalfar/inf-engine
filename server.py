"""Multi-session HTTP server for the continuous-batching inference engine.

One loaded model is shared by every session through PagedSessionManager
(engine/paged_session.py): concurrent requests get their decode steps
batched together by LlamaPagedEngine instead of running one-request-at-a-time
like main.py's AgentLoop. A single background "engine thread" owns the
engine/manager (see paged_session.py's ownership-model docstring); HTTP
handlers only ever call PagedSessionManager.submit(), which is safe to call
from any thread.

Usage:
    python server.py --model-dir weights/Qwen--Qwen3-8B
    python server.py --model-dir weights/Qwen--Qwen3-8B --workspace ~/myproject --port 8000

Client usage (NDJSON-streamed events: token / tool_exec / done / error):
    curl -N -X POST http://localhost:8000/v1/chat \\
        -H "Content-Type: application/json" \\
        -d '{"messages": [{"role": "user", "content": "list the files here"}]}'
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import threading
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from engine.llama_paged_engine import LlamaPagedEngine
from engine.paged_session import PagedSessionManager
from engine.qwen_tokenizer import QwenTokenizer
from engine.sampling import SamplingConfig, SamplingMode
from main import detect_config, load_model, make_tools

app = FastAPI(title="LLaMA continuous-batching server")

_manager: PagedSessionManager | None = None
_stop_event = threading.Event()
_engine_thread: threading.Thread | None = None
_next_session_id = itertools.count(1)


class ChatRequest(BaseModel):
    messages: list[dict]
    session_id: int | None = None


@app.post("/v1/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    if _manager is None:
        raise HTTPException(503, "model not loaded")

    session_id = req.session_id if req.session_id is not None else next(_next_session_id)
    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()

    def emit(event: dict) -> None:
        # Called from the engine thread — hop back onto the request's event loop.
        loop.call_soon_threadsafe(events.put_nowait, event)

    _manager.submit(session_id, req.messages, emit)

    async def stream():
        yield json.dumps({"type": "session", "session_id": session_id}) + "\n"
        while True:
            event = await events.get()
            yield json.dumps(event) + "\n"
            if event["type"] in ("done", "error"):
                break

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/healthz")
async def healthz() -> dict:
    if _manager is None:
        return {"status": "loading"}
    return {"status": "ok", "active_sessions": len(_manager._sessions)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-session continuous-batching inference server")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max-ctx", type=int, default=8192,
        help="Per-session context budget in tokens (default: 8192).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument(
        "--n-total-blocks", type=int, default=4096,
        help="Physical KV blocks in the pool, shared across all concurrent sessions "
             "(default: 4096; with --block-size 16 that's 65536 tokens of capacity).",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument("--quantize", dest="quantize", action="store_const", const=True)
    quant_group.add_argument("--no-quantize", dest="quantize", action="store_const", const=False)
    parser.set_defaults(quantize=None)  # None = auto-detect based on model size
    return parser.parse_args()


def main() -> None:
    global _manager, _engine_thread

    args = _parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        raise SystemExit(f"Workspace not found: {workspace}")

    dtype = torch.bfloat16
    config = detect_config(args.model_dir)

    quantize = args.quantize
    if quantize is None:
        quantize = not config.is_moe and config.d_model >= 4096
        if quantize:
            print(f"Auto-enabling INT4 quantization for {config.name} (d_model={config.d_model})")

    tokenizer = QwenTokenizer(args.model_dir)
    model = load_model(args.model_dir, config, args.device, dtype, quantize=quantize)
    tools = make_tools(workspace)

    sampling = SamplingConfig(
        mode=SamplingMode.TOP_P, temperature=0.6, top_p=0.95,
        repetition_penalty=args.repetition_penalty,
    )
    engine = LlamaPagedEngine(
        model,
        n_total_blocks=args.n_total_blocks,
        block_size=args.block_size,
        eos_token=tokenizer.eos_token_id,
        sampling=sampling,
    )
    _manager = PagedSessionManager(
        engine, tokenizer, tools=tools,
        max_turns=args.max_turns, max_new_tokens=args.max_new_tokens,
        enable_thinking=args.thinking, max_ctx=args.max_ctx,
    )

    _engine_thread = threading.Thread(target=_manager.run_forever, args=(_stop_event,), daemon=True)
    _engine_thread.start()

    print(f"Workspace: {workspace}")
    print(f"Serving {config.name} on {args.host}:{args.port} "
          f"(KV pool: {args.n_total_blocks * args.block_size} tokens)")

    import uvicorn
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        _stop_event.set()
        _engine_thread.join(timeout=5)


if __name__ == "__main__":
    main()
