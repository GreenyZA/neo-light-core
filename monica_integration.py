#!/usr/bin/env python3
"""
Token Optimiser — Monica Integration (Thin Wrapper)

UPDATED 2026-09-19:
- Free tier priority mode enabled by default
- T2 free cloud now stronger than T1 local for most tasks
- NVIDIA NIM, DeepSeek V4 Flash, Poolside Laguna integrated
"""

import sys
import os
from typing import Optional, Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def optimise(task: str, context: Optional[dict] = None) -> Dict[str, Any]:
    """
    Assess a task and return a routing decision with parallelization info.
    """
    from token_optimiser.router import TokenRouter
    router = TokenRouter()
    decision = router.assess(task, context or {})
    result = decision.to_dict()
    result["parallelize"] = decision.parallelize
    result["parallelize_score"] = decision.parallelize_score
    return result


def optimise_batch(tasks: List[str], context: Optional[dict] = None) -> List[Dict[str, Any]]:
    from token_optimiser.router import TokenRouter
    router = TokenRouter()
    return [router.assess(task, context or {}).to_dict() for task in tasks]


def optimiser_enabled() -> bool:
    from token_optimiser.config import config
    return bool(config.enabled)


def record_outcome(task_id: str, outcome: dict) -> bool:
    from token_optimiser.router import LearningLog
    log = LearningLog()
    return log.append_outcome(task_id, outcome)


def get_learning_log(limit: int = 50) -> List[dict]:
    from token_optimiser.router import LearningLog
    log = LearningLog()
    return log.get_recent(limit)


def get_orchestrator_routing(task_type: str, confidence: int) -> Dict[str, Any]:
    """
    Return an orchestrator-aware RoutingDecision for the given task_type.

    Always uses route_orchestrator_task under the hood, guaranteeing that
    reviews and orchestrator tasks land on a capable tier.  Same task_type +
    same confidence + same config ⇒ same decision (deterministic).
    """
    from token_optimiser.router import route_orchestrator_task
    decision = route_orchestrator_task(task_type, confidence)
    return decision.to_dict()


def should_review(task_type: str) -> bool:
    """
    Return True if the given task_type requires a review pass.

    Deterministic: depends only on the task_type and config.enforce_review.
    """
    from token_optimiser.config import config
    if not config.enforce_review:
        return False
    task_lower = task_type.lower()
    review_patterns = ("review", "complex_worker", "multi_worker", "orchestrate")
    return any(p in task_lower for p in review_patterns)


# ── System Prompt Text ──────────────────────────────────────────────────────────

SYSTEM_PROMPT_INSTRUCTIONS = """
## Token Optimiser (Always-On Pre-Dispatch Filter)

Before dispatching ANY worker, you must route the task through the token
optimiser. This is mandatory — every agent call passes through it.

### How to call it

    from token_optimiser.monica_integration import optimise, optimiser_enabled
    decision = optimise(task_description, {"estimated_lines": 50})

### Tier meanings (UPDATED 2026-09-19)

- T1: Local llama-server (qwen2.5-coder-3b-instruct Q4_K_M on RX 480). Fast, free, private.
  Used ONLY for: privacy-sensitive tasks, offline mode, or when free cloud is unavailable.
  Quality: 0.6/1.0 — decent for simple tasks, weak on complex reasoning.

- T2: FREE CLOUD — NOW THE DEFAULT for most tasks (free_tier_priority=True).
  Models available (all $0):
    - NVIDIA Nemotron 3 Ultra (550B MoE, 55B active) — frontier reasoning
    - NVIDIA Nemotron 3 Super (120B MoE, 12B active) — agentic
    - DeepSeek V4 Flash 0731 (284B MoE, 13B active) — research, analysis
    - Poolside Laguna S 2.1 (118B MoE, 8B active) — coding
    - Cohere North Mini Code — agentic coding
    - Nex AGI Nex-N2.5-Pro — verified outcomes
  Quality: 0.95/1.0 — STRONGER than T1 for most tasks.

- T3: Grok (SuperGrok). Expensive. Reserved for: security review, architecture decisions,
  final judgment, tasks where free cloud would give wrong answers.

- ASK: Ambiguous tasks (3+ categories) or in hybrid band. Pause and ask user.

### Free Tier Priority Mode (ENABLED)

When free_tier_priority=True (default), the optimizer routes to T2 (free cloud)
BEFORE T1 (local) because the free models are FAR superior to the local 3B model.

T1 is only used for:
1. Privacy-sensitive tasks (data must stay local)
2. Offline mode (no internet)
3. Free cloud is rate-limited/unavailable

For ALL other tasks, T2 free cloud is the FIRST choice.

### Recording outcomes

After the worker completes:

    from token_optimiser.monica_integration import record_outcome
    record_outcome(decision["task_id"], {
        "outcome_type": "worked",
        "details": "Worker completed successfully"
    })

### Rules

1. ALWAYS call optimise() before dispatching any worker — no exceptions.
2. If tier is "ASK", do NOT dispatch. Ask the user first.
3. If the optimiser is disabled, route to T2 and note it.
4. Record outcomes after every dispatched task.
5. The routing decision is visible to the worker — share it so the worker can adjust
   (T1: be concise. T2: use full context, the model is smart. T3: thorough, critical).
"""
