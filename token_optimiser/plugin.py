"""
Token Optimiser — Hermes Agent Plugin

Registers the token-optimiser as a tool available to agents.
When enabled, intercepts task routing and applies T1/T2/T3 budget routing.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

# ── Hermes plugin integration (soft import) ────────────────────────────────────
# The token-optimiser plugin registers tools with a Hermes Agent session.
# If Hermes Agent is not installed (e.g. during unit tests), the plugin is
# still importable — the @tool decorators simply won't register.

try:
    from hermes_agent.plugins.plugin_utils import PluginTool, tool
    HERMES_AVAILABLE = True
except ImportError:
    HERMES_AVAILABLE = False
    PluginTool = None

    # No-op decorator so @tool(...) still works in tests / standalone
    def tool(**kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)


# ── Tool definitions ──────────────────────────────────────────────────────────

@tool(
    name="token_optimiser_assess",
    description=(
        "Assess a task and return a routing decision (T1/T2/T3) with confidence score, "
        "rationale, and scope analysis. Use before delegating any non-trivial work to "
        "determine which model tier should handle it."
    ),
    parameter_schema={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The task description to assess",
            },
            "context": {
                "type": "object",
                "description": "Optional context: estimated_lines, domain_hints, etc.",
                "properties": {
                    "estimated_lines": {"type": "integer", "description": "Estimated lines of code / output"},
                    "domain": {"type": "string", "description": "Known domain hint (python, typescript, security, etc.)"},
                },
            },
        },
        "required": ["task"],
    },
)
def token_optimiser_assess(task: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Assess a task and produce a routing decision.
    Returns tier, confidence, rationale, matched categories, and scope.
    """
    from token_optimiser.router import TokenRouter, LearningLog

    router = TokenRouter()
    log = LearningLog()

    decision = router.assess(task, context or {})
    log.append(decision)

    return {
        "tier": decision.tier,
        "confidence": decision.confidence,
        "rationale": decision.rationale,
        "scope": decision.scope,
        "matched_categories": decision.matched_categories,
        "task": decision.task,
    }


@tool(
    name="token_optimiser_plan",
    description=(
        "Produce a Token Budget Plan for a multi-step task. "
        "Returns the plan with goal, good-enough criteria, cheapest viable tier, "
        "confidence estimate, and risk assessment."
    ),
    parameter_schema={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The multi-step task to plan for",
            },
            "subtasks": {
                "type": "array",
                "description": "Optional list of known subtasks",
                "items": {"type": "string"},
            },
        },
        "required": ["task"],
    },
)
def token_optimiser_plan(task: str, subtasks: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Produce a Token Budget Plan for a multi-step task.
    """
    from token_optimiser.router import TokenRouter, LearningLog

    router = TokenRouter()
    log = LearningLog()

    # Assess the overall task
    decision = router.assess(task)

    # If subtasks provided, assess each
    subtask_decisions = []
    if subtasks:
        for st in subtasks:
            sd = router.assess(st)
            subtask_decisions.append({
                "task": st,
                "tier": sd.tier,
                "confidence": sd.confidence,
            })

    # Determine the highest tier needed across subtasks (if any)
    tier_order = {"T1": 0, "T2": 1, "T3": 2}
    if subtask_decisions:
        max_tier_idx = max(tier_order.get(d["tier"], 1) for d in subtask_decisions)
        overall_tier = ["T1", "T2", "T3"][min(max_tier_idx, 2)]
    else:
        overall_tier = decision.tier

    plan = {
        "task": task,
        "overall_tier": overall_tier,
        "overall_confidence": decision.confidence,
        "budget_plan": {
            "real_goal": f"Complete: {task[:100]}",
            "good_enough": "Task completed to spec, no Critical/High findings in review",
            "cheapest_viable_tier": f"{overall_tier}",
            "confidence_percent": decision.confidence,
            "risk_if_cheap_fails": (
                "Retry at next tier up. If T1 fails, try T2. "
                "If T2 fails, escalate to T3 with stripped prompt."
            ),
        },
        "subtask_routes": subtask_decisions if subtask_decisions else None,
        "rationale": decision.rationale,
    }

    log.append(decision)
    return plan


@tool(
    name="token_optimiser_toggle",
    description=(
        "Enable or disable the token optimiser. When disabled, all routing defaults "
        "to T2 (free cloud). Use to instantly switch off the optimiser if it's "
        "causing issues."
    ),
    parameter_schema={
        "type": "object",
        "properties": {
            "enabled": {
                "type": "boolean",
                "description": "True to enable, False to disable",
            },
        },
        "required": ["enabled"],
    },
)
def token_optimiser_toggle(enabled: bool) -> Dict[str, Any]:
    """
    Enable or disable the token optimiser.
    """
    from token_optimiser.config import config

    old_state = config.enabled
    config.enabled = enabled

    return {
        "previous_state": old_state,
        "new_state": config.enabled,
        "status": "enabled" if enabled else "disabled",
        "note": "When disabled, all routing falls back to T2 baseline",
    }


@tool(
    name="token_optimiser_status",
    description=(
        "Get current token optimiser status: enabled/disabled, recent learning log entries, "
        "routing statistics."
    ),
    parameter_schema={
        "type": "object",
        "properties": {},
    },
)
def token_optimiser_status() -> Dict[str, Any]:
    """
    Get token optimiser status and recent activity.
    """
    from token_optimiser.config import config
    from token_optimiser.router import LearningLog

    log = LearningLog()
    recent = log.get_recent(10)

    tier_counts = {"T1": 0, "T2": 0, "T3": 0}
    for entry in recent:
        tier = entry.get("tier", "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    return {
        "enabled": config.enabled,
        "recent_routes": recent,
        "tier_counts": tier_counts,
        "total_routes": len(recent),
        "learning_log_path": str(log.path),
    }


@tool(
    name="token_optimiser_feedback",
    description=(
        "Record feedback on a routing decision after the task completes. "
        "This feeds the learning log and improves future confidence calibration."
    ),
    parameter_schema={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The task that was routed",
            },
            "outcome": {
                "type": "string",
                "description": "How it went: worked / partial / failed / needed_escalation",
            },
            "notes": {
                "type": "string",
                "description": "Any observations about why the routing was right or wrong",
            },
        },
        "required": ["task", "outcome"],
    },
)
def token_optimiser_feedback(task: str, outcome: str, notes: Optional[str] = None) -> Dict[str, Any]:
    """
    Record feedback on a routing decision.
    """
    from token_optimiser.router import TokenRouter, LearningLog

    router = TokenRouter()
    log = LearningLog()

    # Re-assess to get the original decision
    decision = router.assess(task)

    outcome_map = {
        "worked": {"quality": "good", "delta": 0},
        "partial": {"quality": "acceptable", "delta": -5},
        "failed": {"quality": "poor", "delta": -15},
        "needed_escalation": {"quality": "poor", "delta": -10},
    }

    outcome_data = outcome_map.get(outcome, {"quality": "unknown", "delta": 0})

    log.append(decision, outcome={
        "outcome": outcome,
        "quality": outcome_data["quality"],
        "confidence_delta": outcome_data["delta"],
        "notes": notes or "",
    })

    return {
        "logged": True,
        "task": task,
        "outcome": outcome,
        "original_confidence": decision.confidence,
        "adjustment_delta": outcome_data["delta"],
    }
