"""Qwen3 inference engine built from scratch (no HuggingFace, no vLLM).

LLaMA / Qwen3 dense:
    LlamaConfig, get_llama_config -- hyperparameters
    load_llama_weights            -- sharded safetensors loading
    LlamaWeights                  -- named tensors
    LlamaModel                    -- forward pass + cached generation
    LlamaStaticKVCache            -- GQA-aware static KV cache

Paged / continuous batching:
    BlockManager                  -- physical block pool for paged allocation
    PagedLlamaKVCache             -- block-based non-contiguous KV cache
    LlamaPagedEngine              -- continuous batching engine
    LlamaRequest                  -- request dataclass for LlamaPagedEngine

Speculative decoding:
    SpeculativeDecoder            -- draft+verify for 2-4x throughput
    SpecStats                     -- acceptance statistics

Qwen3 configs:
    QWEN3_0_6B .. QWEN3_32B      -- dense model constants
    QWEN3_30B_A3B                 -- MoE config (128 experts, top-8)
    QwenTokenizer                 -- tiktoken-based BPE tokenizer (no HF dep)

Qwen3 MoE:
    ExpertOffloadManager          -- CPU-pinned expert cache with VRAM fetch
    DiskExpertManager             -- disk-streaming expert loader with VRAM LRU cache

Agentic loop:
    format_messages               -- message list → Qwen3 ChatML string
    extract_tool_calls            -- parse <tool_call> blocks from model output
    has_tool_call, strip_thinking, format_tool_result
    Tool, AgentResult, AgentLoop

Shared:
    SamplingConfig, SamplingMode  -- greedy / top-k / top-p decoding
"""

from engine.config import (
    LlamaConfig,
    QWEN3_0_6B,
    QWEN3_1_7B,
    QWEN3_4B,
    QWEN3_8B,
    QWEN3_14B,
    QWEN3_32B,
    QWEN3_30B_A3B,
    get_llama_config,
)
from engine.kv_cache import LlamaStaticKVCache
from engine.llama_model import LlamaModel
from engine.llama_paged_engine import LlamaPagedEngine, LlamaRequest
from engine.llama_weights import LlamaWeights, load_llama_weights
from engine.paged_cache import BlockManager, PagedLlamaKVCache
from engine.agent import AgentLoop, AgentResult, Tool
from engine.chat import format_messages
from engine.disk_expert_manager import DiskExpertManager
from engine.moe_offload import ExpertOffloadManager
from engine.qwen_tokenizer import QwenTokenizer
from engine.sampling import SamplingConfig, SamplingMode
from engine.speculative import SpecStats, SpeculativeDecoder
from engine.tool_parser import (
    ToolCall,
    extract_tool_calls,
    format_tool_result,
    has_tool_call,
    strip_thinking,
)

__all__ = [
    # Config
    "LlamaConfig",
    "get_llama_config",
    # Dense model
    "LlamaModel",
    "LlamaWeights",
    "load_llama_weights",
    "LlamaStaticKVCache",
    # Paged engine
    "BlockManager",
    "PagedLlamaKVCache",
    "LlamaPagedEngine",
    "LlamaRequest",
    # Speculative decoding
    "SpeculativeDecoder",
    "SpecStats",
    # Qwen3 configs
    "QWEN3_0_6B",
    "QWEN3_1_7B",
    "QWEN3_4B",
    "QWEN3_8B",
    "QWEN3_14B",
    "QWEN3_32B",
    "QWEN3_30B_A3B",
    "QwenTokenizer",
    # MoE
    "ExpertOffloadManager",
    "DiskExpertManager",
    # Agentic loop
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
]
