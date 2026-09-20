"""
Token Optimiser — Configuration

UPDATED 2026-09-19:
- T1 endpoint: 192.168.1.10:8081 (llama-server, Vulkan, RX 480)
- T1 model: Phi-3.5-mini-instruct Q5_K_M (3.0GB, fits 4GB VRAM with 4K context)
- T2 expanded: 40+ free models (NVIDIA Nemotron, DeepSeek V4 Flash, Poolside, etc.)
- New: Free tier priority mode — T2 free cloud before T1 when quality is better
- New: NVIDIA NIM integration (integrate.api.nvidia.com/v1)
- New: DeepSeek V4 Flash free via OpenRouter AND NVIDIA NIM
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenOptimiserConfig:
    enabled: bool = True
    learning_log_path: str = "token_optimiser/learning_log.jsonl"
    confidence_calibration: dict = field(default_factory=dict)
    priority_budget: int = 1000
    max_t1_daily_savings: int = 5000
    recent_decisions: list = field(default_factory=list)
    learning_log: list = field(default_factory=list)

    # Free tier priority: T2 free cloud tried BEFORE T1 local
    free_tier_priority: bool = True

    # Category → preferred free model mapping
    category_free_model_preference: dict = field(default_factory=lambda: {
        "code_gen_small": "poolside/laguna-s-2.1:free",
        "code_gen_medium": "poolside/laguna-s-2.1:free",
        "code_gen_large": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "documentation": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "research": "deepseek/deepseek-v4-flash-0731:free",
        "testing": "poolside/laguna-s-2.1:free",
        "infrastructure": "nvidia/nemotron-3-super-120b-a12b:free",
        "simple_transform": "nvidia/nemotron-3-super-120b-a12b:free",
        "architecture": "nvidia/nemotron-3-ultra-550b-a55b:free",
    })

    tier_backends: dict = field(default_factory=lambda: {
        "T1": {
            "url": "http://192.168.1.10:8081",
            "model": "Phi-3.5-mini-instruct-Q5_K_M",
            "fallback_model": "qwen2.5-coder-3b-instruct",
            "max_prompt_tokens": 4096,
            "enabled": True,
            "tweak_notes": "Local llama-server on RX 480 (Vulkan). Phi-3.5-mini Q5_K_M = 3.0GB VRAM, ~25-35 tok/s. Best quality for 4GB card.",
            "quality_rating": 0.75,
            "vram_usage_gb": 3.0,
            "context_tokens": 4096,
            "alternative_models": {
                "code": "Qwen2.5-Coder-7B-Instruct-IQ3_S (3.0GB, best code)",
                "reasoning": "Phi-3.5-mini-instruct-Q5_K_M (3.0GB, best reasoning)",
                "agentic": "Qwen3.5-4B-Q4_K_M (2.4GB, tool calling)",
                "fast": "Llama-3.2-3B-Instruct-Q4_K_M (2.0GB, fastest)",
            },
        },
        "T2": {
            "gateways": [
                {
                    "name": "OpenRouter",
                    "url": "https://openrouter.ai/api/v1",
                    "type": "openrouter",
                    "api_key_env": "OPENROUTER_API_KEY",
                    "weight": 2.0,
                    "enabled": True,
                },
                {
                    "name": "NVIDIA_NIM",
                    "url": "https://integrate.api.nvidia.com/v1",
                    "type": "nvidia_nim",
                    "api_key_env": "NVIDIA_API_KEY",
                    "weight": 1.5,
                    "enabled": True,
                },
                {
                    "name": "OmniRoute",
                    "url": "http://127.0.0.1:20128/v1",
                    "type": "omniroute",
                    "weight": 1.0,
                    "enabled": True,
                },
                {
                    "name": "NOUS",
                    "url": "https://api.nousresearch.com/v1",
                    "type": "nous",
                    "api_key_env": "NOUS_API_KEY",
                    "weight": 0.5,
                    "enabled": True,
                },
            ],
            "free_models": [
                "nvidia/nemotron-3-ultra-550b-a55b:free",
                "nvidia/nemotron-3-super-120b-a12b:free",
                "nvidia/nemotron-3.5-lightning:free",
                "deepseek/deepseek-v4-flash-0731:free",
                "poolside/laguna-s-2.1:free",
                "poolside/laguna-xs-2.1:free",
                "inclusionai/ling-3.0-flash-fin:free",
                "inclusionai/ling-3.0-flash-vl:free",
                "inclusionai/ling-3.0-flash-sante:free",
                "cohere/north-mini-code:free",
                "nex-agi/nex-n2.5-pro:free",
                "nex-agi/nex-n2.5-mini:free",
                "thinkingmachines/inkling:free",
                "dots-studio/dots-3-note-preview:free",
                "liquid/lfm-2.5-2.6b:free",
                "z-ai/glm-5.2:free",
                "qwen/qwen3.8-27b:free",
                "openrouter/free",
            ],
            "nvidia_free_models": [
                "deepseek-ai/deepseek-v4-flash-0731",
                "deepseek-ai/deepseek-r1",
                "nvidia/nemotron-3-ultra-550b-a55b",
                "nvidia/nemotron-3-super-120b-a12b",
                "nvidia/nemotron-3.5-lightning",
                "nvidia/nvidia-nemotron-nano-9b-v2",
                "meta/llama-3.3-70b-instruct",
                "meta/llama-3.1-405b-instruct",
                "meta/llama-3.2-3b-instruct",
                "qwen/qwen2.5-coder-32b-instruct",
            ],
            "max_prompt_tokens": 32000,
            "quality_rating": 0.9,
        },
        "T3": {
            "providers": [
                {
                    "name": "Grok (SuperGrok)",
                    "url": "https://grok.api.x.ai/v1",
                    "type": "grok",
                    "api_key_env": "GROK_API_KEY",
                    "model": "grok-2-1212",
                    "max_prompt_tokens": 32000,
                    "enabled": False,
                },
            ],
            "default_provider": "Grok (SuperGrok)",
        },
    })

    security_confidence_cap: int = 55
    category_confidence_caps: dict = field(default_factory=lambda: {"security": 55})
    threshold_t1_min: int = 80
    threshold_t2_min: int = 60
    threshold_t3_always_below: int = 60
    hybrid_band_enabled: bool = True
    ask_human_band: bool = False
    ambiguous_task_asks_human: bool = True
    scope_word_thresholds: dict = field(default_factory=lambda: {"small": 30, "medium": 100})
    scope_prefix_pattern: str = r"\[(?:scope|size|scope_size)?[:\s]*(\w+)\]"
    scope_prefix_values: dict = field(default_factory=lambda: {
        "small": "small", "s": "small", "sm": "small",
        "medium": "medium", "m": "medium", "med": "medium",
        "large": "large", "l": "large", "lg": "large",
    })

    # Parallelization scoring
    parallelize_threshold: float = 0.7

    # Orchestrator-aware routing
    orchestrator_routing: bool = True
    min_review_tier: str = "T2"
    min_orchestrator_tier: str = "T2"
    enforce_review: bool = True

    def _ts(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "threshold_t1_min": self.threshold_t1_min,
            "threshold_t2_min": self.threshold_t2_min,
            "threshold_t3_always_below": self.threshold_t3_always_below,
            "hybrid_band_enabled": self.hybrid_band_enabled,
            "ask_human_band": self.ask_human_band,
            "ambiguous_task_asks_human": self.ambiguous_task_asks_human,
            "security_confidence_cap": self.security_confidence_cap,
            "category_confidence_caps": self.category_confidence_caps or {},
            "scope_word_thresholds": self.scope_word_thresholds,
            "scope_prefix_pattern": self.scope_prefix_pattern,
            "scope_prefix_values": self.scope_prefix_values,
            "confidence_calibration": self.confidence_calibration,
            "tier_backends": self.tier_backends,
            "priority_budget": self.priority_budget,
            "max_t1_daily_savings": self.max_t1_daily_savings,
            "recent_decisions": self.recent_decisions,
            "learning_log": self.learning_log,
            "free_tier_priority": self.free_tier_priority,
            "category_free_model_preference": self.category_free_model_preference,
            "parallelize_threshold": self.parallelize_threshold,
            "orchestrator_routing": self.orchestrator_routing,
            "min_review_tier": self.min_review_tier,
            "min_orchestrator_tier": self.min_orchestrator_tier,
            "enforce_review": self.enforce_review,
        }


config = TokenOptimiserConfig()


def _ts() -> str:
    return config._ts()


def to_dict() -> dict:
    return config.to_dict()


def set_enabled(enabled: bool):
    config.enabled = enabled


def is_enabled() -> bool:
    return config.enabled


def update_calibration(category: str, delta: int):
    if category not in config.confidence_calibration:
        config.confidence_calibration[category] = 0
    config.confidence_calibration[category] += delta


def reset_calibration():
    config.confidence_calibration.clear()
