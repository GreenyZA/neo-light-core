"""
Token Optimiser — Configuration

UPDATED 2026-10-04 (CHG-2026-030) — corrected against the wire, not from memory:
|- T1 endpoint: localhost:8081 (llama-server, Vulkan, RX 480)
|- T1 model: qwen2.5-3b-instruct Q4_K_M, one slot, n_ctx 8192 (CHG-2026-064).
|  The server accepts any model name and silently substitutes, so this name
|  has to match the alias or the logs lie.
|- T1 fallback: NONE. Coder-3B is on disk as a swap-in, not a fallback.
|- T1b: ENABLED. localhost:8082 serves Qwen2.5-7B-Instruct-Q4_K_M
|  via llama-server (CPU-only, ngl=0, n_ctx=8192). Systemd: llama-t1b.service.
|  Privacy short-circuit routes strictest tasks here (air-gapped, local-only).
|- Credentials: token_optimiser never loaded ~/.hermes/.env, so from inside this
|  process every free tier looked unavailable and decisions escalated to T3.
|  Fixed in credentials.py (auto-loads on import; env always wins).
|- Calibration: the writer stores outcome={"outcome": "worked"}; the reader
|  looked for "outcome_type". Every category took a permanent -10 penalty.
|  Fixed in monica_integration.py.
|- T3 remains opt-in (enabled: False) per WarPuppy's standing "No SuperGrok".

UPDATED 2026-09-19:
|- T2 expanded: 40+ free models (NVIDIA Nemotron, DeepSeek V4 Flash, Poolside, etc.)
|- New: Free tier priority mode — T2 free cloud before T1 when quality is better
|- New: NVIDIA NIM integration (integrate.api.nvidia.com/v1)
|- New: DeepSeek V4 Flash free via OpenRouter AND NVIDIA NIM
|- T1b was: secondary local tier on Lenovo M710Q for larger CPU models when
  privacy requires local + more knowledge than T1's 3B.
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
        "privacy": "local-only",
    })

    tier_backends: dict = field(default_factory=lambda: {
        "T1": {
            # VERIFIED 2026-11-26 (CHG-2026-064). /v1/models id=qwen2.5-3b-instruct,
            # n_ctx=8192, n_params=3085938688. One slot. f16 KV, flash-attn on,
            # batch 2048/512, Vulkan0, ngl 99. Peak VRAM about 2625 MiB.
            # Generation about 66 tok/s. A 1053-token prompt about 380 tok/s.
            # The server accepts any model name and substitutes what is loaded.
            "url": "http://localhost:8081",
            "model": "qwen2.5-3b-instruct",
            # Coder-3B is on disk. It is a swap-in, not the resident model.
            "fallback_model": None,
            "max_prompt_tokens": 8192,
            "enabled": True,
            "tweak_notes": "Local llama-server on RX 480 (Vulkan). Qwen2.5-3B-Instruct Q4_K_M, one slot, n_ctx 8192, f16 KV, flash-attn on. Measured 2026-11-26: about 66 tok/s generation, about 380 tok/s on a 1053-token prompt, peak VRAM about 2.6 GB. 3B-class: route substantive work to T2 free.",
            "quality_rating": 0.70,
            "vram_usage_gb": 2.6,
            "context_tokens": 8192,
            "alternative_models": {
                "code": "Qwen2.5-Coder-3B-Instruct-Q4_K_M (on disk, not loaded)",
                "reasoning": "qwen2.5-3b-instruct Q4_K_M (loaded on :8081)",
                "spare": "Phi-4-mini-instruct-Q4_K_M (on disk, cold spare)",
                "larger": "Qwen2.5-7B-Instruct-Q4_K_M (T1b CPU :8082, do not offload)",
            },
        },
        "T1b": {
            # ENABLED 2026-11-26. Live /v1/models on localhost:8082 returned
            # id qwen2.5-7b-instruct, n_ctx 8192, Q4_K_M.
            # llama-server, CPU-only (ngl=0). Systemd: llama-t1b.service.
            # Privacy short-circuit routes strictest tasks here.
            "url": "http://localhost:8082",
            "model": "qwen2.5-7b-instruct",
            "fallback_model": None,
            "max_prompt_tokens": 8192,
            "enabled": True,
            "tweak_notes": "ENABLED 2026-11-26: Qwen2.5-7B-Instruct Q4_K_M on localhost:8082, n_ctx 8192, CPU-only (ngl=0). Systemd: llama-t1b.service.",
            "quality_rating": 0.75,
            "vram_usage_gb": 0,
            "context_tokens": 8192,
            "alternative_models": {
                "code": "qwen2.5-7b-instruct (CPU, 7B, better code)",
                "chat": "qwen2.5-7b-instruct (CPU, more knowledge than T1 3B)",
                "reasoning": "qwen2.5-7b-instruct (CPU, 7B for complex analysis)",
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
                "stealth/crouton-alpha",
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
    category_confidence_caps: dict = field(default_factory=lambda: {
        "security": 55,
        "architecture": 65,
        "privacy": 55,
    })

    # Privacy/offline keywords that ALWAYS force T1 routing regardless of free_tier_priority.
    # When privacy_force_t1 is True and any of these keywords appear in the task,
    # the router short-circuits to T1 (local, air-gapped, never leaves the machine).
    privacy_offline_keywords: list = field(default_factory=lambda: [
        "offline", "airgap", "air-gapped", "air gapped", "no internet",
        "local only", "on-premise", "on premise", "must stay local",
        "data residency", "sovereign", "leave this machine", "never leave",
        "stay on device", "no cloud", "without internet",
    ])

    privacy_force_t1: bool = True

    # Keywords that indicate the user needs a larger model than T1's 3B.
    # When combined with privacy/offline keywords (force T1), routes to T1b (Lenovo 7B CPU) instead.
    t1b_keywords: list = field(default_factory=lambda: [
        "deep analysis", "7b", "larger model", "more knowledge",
        "longer context", "complex analysis", "deep reasoning", "detailed output",
    ])
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

    # Learning log calibration
    calibration_enabled: bool = True
    calibration_min_samples: int = 5

    # Parallelization scoring
    parallelize_threshold: float = 0.7

    # Orchestrator-aware routing
    orchestrator_routing: bool = True
    min_review_tier: str = "T2"
    min_orchestrator_tier: str = "T2"
    enforce_review: bool = True

    # BOA/ITIL governance mode for monica_integration.optimise
    # "off": never use BOA, pure router (backward compat, no overhead)
    # "auto": use lightweight BOA (classify_itil + validate_swarm_plan) for non-trivial/ops tasks
    # "force_boa": always attempt lightweight BOA governance for binding tier + swarm
    governance_mode: str = "auto"

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
            "privacy_offline_keywords": self.privacy_offline_keywords,
            "privacy_force_t1": self.privacy_force_t1,
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
            "governance_mode": self.governance_mode,
            "t1b_keywords": self.t1b_keywords,
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
