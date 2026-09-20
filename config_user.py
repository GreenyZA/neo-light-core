# Phantom's Token Optimiser Config Overrides
# No GPU — T1 disabled. Heavy NVIDIA free tier usage. Grok media models on T3.

from config import TokenOptimiserConfig

config = TokenOptimiserConfig(
    tier_backends={
        "T1": {
            "enabled": False,  # No GPU, no local tier
        },
        "T2": {
            "gateways": [
                {
                    "name": "NVIDIA_NIM",
                    "url": "https://integrate.api.nvidia.com/v1",
                    "type": "nvidia_nim",
                    "api_key_env": "NVIDIA_API_KEY",
                    "weight": 2.0,  # Primary for phantom
                    "enabled": True,
                },
                {
                    "name": "OpenRouter",
                    "url": "https://openrouter.ai/api/v1",
                    "type": "openrouter",
                    "api_key_env": "OPENROUTER_API_KEY",
                    "weight": 1.5,
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
                "cohere/north-mini-code:free",
                "openrouter/free",
            ],
            "nvidia_free_models": [
                "deepseek-ai/deepseek-v4-flash-0731",
                "nvidia/nemotron-3-ultra-550b-a55b",
                "nvidia/nemotron-3-super-120b-a12b",
                "meta/llama-3.3-70b-instruct",
                "qwen/qwen2.5-coder-32b-instruct",
            ],
            "max_prompt_tokens": 32000,
            "quality_rating": 0.9,
        },
        "T3": {
            "providers": [
                {
                    "name": "Grok Imagine (Image Gen)",
                    "url": "https://grok.api.x.ai/v1",
                    "type": "grok",
                    "api_key_env": "GROK_API_KEY",
                    "model": "grok-2-image",
                    "max_prompt_tokens": 8192,
                    "enabled": True,
                },
                {
                    "name": "Grok Video",
                    "url": "https://grok.api.x.ai/v1",
                    "type": "grok",
                    "api_key_env": "GROK_API_KEY",
                    "model": "grok-2-video",
                    "max_prompt_tokens": 8192,
                    "enabled": True,
                },
                {
                    "name": "Grok Compose",
                    "url": "https://grok.api.x.ai/v1",
                    "type": "grok",
                    "api_key_env": "GROK_API_KEY",
                    "model": "grok-2-compose",
                    "max_prompt_tokens": 8192,
                    "enabled": True,
                },
            ],
            "default_provider": "Grok Imagine (Image Gen)",
        },
    },

    # Phantom preferences
    free_tier_priority=True,
    privacy_force_t1=False,  # No local tier, so privacy routes to T2
    security_confidence_cap=55,
    category_confidence_caps={
        "security": 55,
        "architecture": 65,
        "privacy": 55,
        "media_generation": 70,  # New cap for Grok media tasks
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
