# Warpuppy's Token Optimiser Config Overrides
# This file is loaded by token_optimiser/config.py to override defaults

from config import TokenOptimiserConfig

config = TokenOptimiserConfig(
    # Warpuppy: Local T1 on Acer (RX 480) via llama-server
    tier_backends={
        "T1": {
            "url": "http://192.168.1.10:8081",
            "model": "Phi-3.5-mini-instruct-Q5_K_M",
            "fallback_model": "qwen2.5-coder-3b-instruct",
            "max_prompt_tokens": 4096,
            "enabled": True,
            "tweak_notes": "Local llama-server on RX 480 (Vulkan). Phi-3.5-mini Q5_K_M = 3.0GB VRAM, ~25-35 tok/s.",
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
                    "enabled": True,
                },
            ],
            "default_provider": "Grok (SuperGrok)",
        },
    },

    # Warpuppy preferences
    free_tier_priority=True,
    privacy_force_t1=True,
    security_confidence_cap=55,
    category_confidence_caps={
        "security": 55,
        "architecture": 65,
        "privacy": 55,
    },

    # Learning log calibration
    calibration_enabled=True,
    calibration_min_samples=5,

    # Parallelization
    parallelize_threshold=0.7,

    # Orchestrator-aware routing
    orchestrator_routing=True,
    min_review_tier="T2",
    min_orchestrator_tier="T2",
    enforce_review=True,

    # BOA/ITIL governance mode
    governance_mode="auto",
)