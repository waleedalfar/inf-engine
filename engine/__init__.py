"""Minimal LLM inference engine built from scratch (no HuggingFace, no vLLM).

GPT-2 (original engine):
    GPT2Config, get_config        -- model hyperparameters
    load_weights, GPT2Weights     -- raw safetensors loading
    GPT2Model                     -- forward pass + generation
    GPT2Tokenizer                 -- GPT-2 BPE via tiktoken

LLaMA family (scale-up):
    LlamaConfig, get_llama_config -- LLaMA / Mistral / Qwen hyperparameters
    load_llama_weights            -- sharded safetensors loading
    LlamaWeights                  -- named LLaMA tensors
    LlamaModel                    -- forward pass + cached generation
    LlamaStaticKVCache            -- GQA-aware static KV cache (stores n_kv_heads)
    BlockManager                  -- physical block pool for paged allocation
    PagedLlamaKVCache             -- paged KV cache (block-based, non-contiguous)
    LlamaPagedEngine              -- continuous batching engine with paged cache
    LlamaRequest                  -- request dataclass for LlamaPagedEngine
    SpeculativeDecoder            -- draft+verify speculative decoding (Phase 4)
    SpecStats                     -- acceptance statistics for speculative decoding

Qwen3 dense (Phase 5):
    QWEN3_0_6B .. QWEN3_32B      -- pre-built Qwen3 config constants
    QwenTokenizer                 -- tiktoken-based Qwen BPE tokenizer (no HF dep)

Qwen3 MoE (Phase 6):
    QWEN3_30B_A3B                 -- Qwen3-30B-A3B config (128 experts, top-8)
    ExpertOffloadManager          -- CPU-pinned expert cache with VRAM fetch/evict

Agentic loop (Phase 7):
    format_messages               -- OpenAI-style message list → Qwen3 ChatML string
    extract_tool_calls            -- parse <tool_call> blocks from model output
    has_tool_call                 -- quick check for tool call presence
    strip_thinking                -- remove <think>…</think> blocks from output
    format_tool_result            -- serialise a tool return value for injection
    Tool                          -- tool descriptor (name, description, schema, fn)
    AgentResult                   -- completed run with final text + history + stats
    AgentLoop                     -- multi-turn generate → parse → execute loop

Shared:
    SamplingConfig, SamplingMode  -- greedy / top-k / top-p decoding

See ROADMAP.md for the scale-up plan and phase status.
"""

from engine.batching import generate_batched, left_pad
from engine.config import (
    GPT2Config,
    LlamaConfig,
    QWEN3_0_6B,
    QWEN3_1_7B,
    QWEN3_4B,
    QWEN3_8B,
    QWEN3_14B,
    QWEN3_32B,
    QWEN3_30B_A3B,
    get_config,
    get_llama_config,
)
from engine.continuous import ContinuousBatchingEngine, Policy, Request
from engine.kv_cache import (
    DynamicKVCache,
    KVCache,
    LlamaStaticKVCache,
    SlotKVCache,
    StaticKVCache,
    kv_cache_bytes,
)
from engine.llama_model import LlamaModel
from engine.llama_paged_engine import LlamaPagedEngine, LlamaRequest
from engine.llama_weights import LlamaWeights, load_llama_weights
from engine.model import GPT2Model
from engine.paged_cache import BlockManager, PagedLlamaKVCache
from engine.agent import AgentLoop, AgentResult, Tool
from engine.chat import format_messages
from engine.moe_offload import ExpertOffloadManager
from engine.qwen_tokenizer import QwenTokenizer
from engine.sampling import SamplingConfig, SamplingMode
from engine.speculative import SpecStats, SpeculativeDecoder
from engine.tokenizer import GPT2Tokenizer
from engine.tool_parser import (
    ToolCall,
    extract_tool_calls,
    format_tool_result,
    has_tool_call,
    strip_thinking,
)
from engine.weights import GPT2Weights, load_weights

__all__ = [
    # GPT-2
    "GPT2Config",
    "get_config",
    "GPT2Model",
    "GPT2Tokenizer",
    "GPT2Weights",
    "load_weights",
    # LLaMA — static path
    "LlamaConfig",
    "get_llama_config",
    "LlamaModel",
    "LlamaWeights",
    "load_llama_weights",
    "LlamaStaticKVCache",
    # LLaMA — paged path (Phase 3)
    "BlockManager",
    "PagedLlamaKVCache",
    "LlamaPagedEngine",
    "LlamaRequest",
    # LLaMA — speculative decoding (Phase 4)
    "SpeculativeDecoder",
    "SpecStats",
    # Qwen3 dense (Phase 5)
    "QWEN3_0_6B",
    "QWEN3_1_7B",
    "QWEN3_4B",
    "QWEN3_8B",
    "QWEN3_14B",
    "QWEN3_32B",
    "QwenTokenizer",
    # Qwen3 MoE (Phase 6)
    "QWEN3_30B_A3B",
    "ExpertOffloadManager",
    # Agentic loop (Phase 7)
    "format_messages",
    "extract_tool_calls",
    "has_tool_call",
    "strip_thinking",
    "format_tool_result",
    "ToolCall",
    "Tool",
    "AgentResult",
    "AgentLoop",
    # Shared
    "SamplingConfig",
    "SamplingMode",
    "KVCache",
    "StaticKVCache",
    "DynamicKVCache",
    "kv_cache_bytes",
    "generate_batched",
    "left_pad",
    "SlotKVCache",
    "ContinuousBatchingEngine",
    "Policy",
    "Request",
]
