#!/usr/bin/env python3
"""
Token Optimiser — Mock Backends for Testing

Provides simulated T1, T2, and T3 backends so the system can be tested
without hitting real APIs. Each mock backend returns deterministic responses
that simulate the tier's behavior.

UPDATED 2026-09-19:
- T2 mock now simulates frontier free models (Nemotron 3 Ultra, DeepSeek V4 Flash)
- T2 quality bumped to 0.95 (free tier is now stronger than local 3B)
- T1 quality remains 0.6 (3B model is limited)
"""

import json
import time
import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime


# ── Mock Backend Base ──────────────────────────────────────────────────────────

@dataclass
class MockBackend:
    name: str
    tier: str
    response_time: float
    quality: float

    def generate(self, task: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class T1MockBackend(MockBackend):
    """Mock T1 backend — local llama-server (qwen2.5-coder-3b-instruct Q4_K_M)."""

    name: str = "T1 Mock (llama-server qwen2.5-coder-3b)"
    tier: str = "T1"
    response_time: float = 0.5
    quality: float = 0.6  # 3B model — decent but limited

    def generate(self, task: str, **kwargs) -> Dict[str, Any]:
        time.sleep(self.response_time)

        task_lower = task.lower()
        if any(kw in task_lower for kw in ["security", "auth", "encrypt", "vulnerability"]):
            return {
                "tier": "T1",
                "backend": "ollama",
                "model": "qwen2.5-coder-3b-instruct",
                "response": (
                    "[T1 SIMULATED] I can give you a basic answer but this involves "
                    "security — a local 3B model isn't reliable for auth/crypto. "
                    "Consider escalating to T3 for a proper review."
                ),
                "quality": "low_on_security",
                "simulated": True,
                "notes": "T1 is not suitable for security tasks",
            }

        if any(kw in task_lower for kw in ["refactor", "rewrite", "large", "architect"]):
            return {
                "tier": "T1",
                "backend": "ollama",
                "model": "qwen2.5-coder-3b-instruct",
                "response": (
                    "[T1 SIMULATED] Here's a quick approach: [basic outline]. "
                    "For a full refactor I'd need more context than a 3B model handles well."
                ),
                "quality": "partial",
                "simulated": True,
                "notes": "T1 provides outline only for large tasks",
            }

        return {
            "tier": "T1",
            "backend": "ollama",
            "model": "qwen2.5-coder-3b-instruct",
            "response": (
                f"[T1 SIMULATED] Here's the result for: {task[:80]}... "
                f"[concise, direct answer — T1 is good at straightforward tasks]"
            ),
            "quality": "good_for_simple",
            "simulated": True,
            "notes": "T1 handles simple tasks well",
        }


@dataclass
class T2MockBackend(MockBackend):
    """
    Mock T2 backend — free cloud frontier models (2026).
    Simulates: NVIDIA Nemotron 3 Ultra, DeepSeek V4 Flash, Poolside Laguna S 2.1
    These are FAR superior to the local 3B model.
    """

    name: str = "T2 Mock (Free Cloud — Nemotron/DeepSeek/Poolside)"
    tier: str = "T2"
    response_time: float = 1.5  # Slightly slower than T1 (network)
    quality: float = 0.95  # Frontier free models — excellent

    def generate(self, task: str, **kwargs) -> Dict[str, Any]:
        time.sleep(self.response_time)

        task_lower = task.lower()

        # Simulate model selection based on task
        if any(kw in task_lower for kw in ["code", "write", "implement", "class", "function"]):
            model_used = "poolside/laguna-s-2.1:free"
            model_desc = "Laguna S 2.1 (118B MoE, 8B active, Terminal-Bench 70.2%)"
        elif any(kw in task_lower for kw in ["research", "analyze", "compare", "evaluate"]):
            model_used = "deepseek/deepseek-v4-flash-0731:free"
            model_desc = "DeepSeek V4 Flash (284B MoE, 13B active, 1M context)"
        elif any(kw in task_lower for kw in ["security", "auth", "encrypt"]):
            # Security still routes to T3, but if it somehow gets here
            return {
                "tier": "T2",
                "backend": "free_cloud",
                "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
                "response": (
                    "[T2 SIMULATED] Here's a decent analysis but security tasks "
                    "should be escalated to T3 for full review."
                ),
                "quality": "good_with_caveat",
                "simulated": True,
                "notes": "T2 is decent but T3 preferred for production security",
            }
        else:
            model_used = "nvidia/nemotron-3-ultra-550b-a55b:free"
            model_desc = "Nemotron 3 Ultra (550B MoE, 55B active, frontier reasoning)"

        return {
            "tier": "T2",
            "backend": "free_cloud",
            "model": model_used,
            "model_desc": model_desc,
            "response": (
                f"[T2 SIMULATED] Frontier model response for: {task[:80]}... "
                f"[comprehensive, well-structured answer from {model_desc} "
                f"— free tier is now stronger than local 3B for most tasks]"
            ),
            "quality": "excellent",
            "simulated": True,
            "notes": f"T2 free cloud using {model_desc}",
        }


@dataclass
class T3MockBackend(MockBackend):
    """Mock T3 backend — high tier. Best quality, slow, expensive."""

    name: str = "T3 Mock (High Tier)"
    tier: str = "T3"
    response_time: float = 2.0
    quality: float = 0.98

    def generate(self, task: str, **kwargs) -> Dict[str, Any]:
        time.sleep(self.response_time)

        task_lower = task.lower()
        if "typo" in task_lower or "fix typo" in task_lower:
            return {
                "tier": "T3",
                "backend": "high_tier",
                "model": "grok",
                "response": (
                    "[T3 SIMULATED] Overkill for a typo fix, but here's the corrected text: "
                    f"[fix for: {task[:60]}]"
                ),
                "quality": "excellent",
                "simulated": True,
                "notes": "T3 is overkill for simple tasks but still excellent",
                "efficiency_note": "This task would have been better on T1 — consider routing",
            }

        if any(kw in task_lower for kw in ["security", "auth", "encrypt", "vulnerability"]):
            return {
                "tier": "T3",
                "backend": "high_tier",
                "model": "grok",
                "response": (
                    f"[T3 SIMULATED] Full security analysis for: {task[:100]}... "
                    "\n\n#### Threat Model\n"
                    "[detailed threat modeling with attack vectors and mitigations]\n\n"
                    "#### Security Review\n"
                    "[thorough review of auth/crypto/secrets handling]\n\n"
                    "#### Recommendations\n"
                    "[specific, actionable security recommendations]\n\n"
                    "#### Code Examples\n"
                    "[secure implementation examples with explanations]"
                ),
                "quality": "excellent",
                "simulated": True,
                "notes": "T3 provided thorough security analysis",
            }

        return {
            "tier": "T3",
            "backend": "high_tier",
            "model": "grok",
            "response": (
                f"[T3 SIMULATED] Excellent response for: {task[:80]}... "
                "[comprehensive, well-structured answer with full context and edge cases]"
            ),
            "quality": "excellent",
            "simulated": True,
            "notes": "T3 provided top-quality response",
        }


# ── Mock Backend History ────────────────────────────────────────────────────────

class MockBackendHistory:
    """Tracks backend execution history and gateway availability for analytics."""

    def __init__(self):
        self.history: list = []
        self.gateway_states: dict = {
            "T1": {"available": True, "model": "qwen2.5-coder-3b-instruct", "base_url": "http://192.168.1.10:8081"},
            "T2": {"available": True, "providers": ["OpenRouter", "NVIDIA_NIM", "OmniRoute", "NOUS"]},
            "T3": {"available": False, "providers": ["Grok"]},
        }

    def record(self, tier: str, task: str, context: dict = None, confidence: int = 0,
               outcome: str = None, quality: str = None) -> dict:
        entry = {
            "tier": tier,
            "task": task[:100] if isinstance(task, str) else str(task)[:100],
            "task_full": task if isinstance(task, str) else str(task),
            "timestamp": datetime.now().isoformat(),
            "context": context or {},
        }
        if confidence is not None:
            entry["confidence"] = confidence
        if outcome:
            entry["outcome"] = outcome
        if quality:
            entry["quality"] = quality
        self.history.append(entry)
        return entry

    def record_decision(self, decision_id: str, data: dict) -> dict:
        entry = {"id": decision_id, "timestamp": datetime.now().isoformat()}
        entry.update(data or {})
        self.history.append(entry)
        return entry

    def record_gateway_state(self, tier: str, available: bool, details: dict = None):
        self.gateway_states[tier] = {"available": available, "details": details or {}}

    def get_history(self, tier: str = None, limit: int = 50) -> list:
        if tier:
            return [h for h in self.history if h.get("tier") == tier][-limit:]
        return self.history[-limit:]

    def get_recent(self, n: int = 10) -> list:
        return self.history[-n:]

    def get_tier_counts(self) -> dict:
        counts = {"T1": 0, "T2": 0, "T3": 0}
        for h in self.history:
            t = h.get("tier", "unknown")
            if t in counts:
                counts[t] += 1
        return counts

    def clear(self):
        self.history = []
        self.gateway_states = {
            "T1": {"available": True, "model": "qwen2.5-coder-3b-instruct", "base_url": "http://192.168.1.10:8081"},
            "T2": {"available": True, "providers": ["OpenRouter", "NVIDIA_NIM", "OmniRoute", "NOUS"]},
            "T3": {"available": False, "providers": ["Grok"]},
        }


# ── Mock Gateway ────────────────────────────────────────────────────────────────

class MockGateway:
    """
    Simulates a multi-tier gateway that routes to the appropriate mock backend.
    Updated with 2026 free tier models.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.mock_mode = self.config.get("mock_mode", True)
        self.history: list = []

        self.backends = {
            "T1": T1MockBackend(),
            "T2": T2MockBackend(),
            "T3": T3MockBackend(),
        }

    def execute(self, tier: str, task: str, **kwargs) -> Dict[str, Any]:
        if not self.mock_mode:
            return {
                "error": "Mock mode is disabled. Configure mock_mode=True to test.",
                "tier": tier,
                "task": task[:50],
            }

        backend = self.backends.get(tier)
        if not backend:
            return {
                "error": f"No backend for tier {tier}",
                "tier": tier,
                "task": task[:50],
            }

        result = backend.generate(task, **kwargs)
        result["execution_time"] = backend.response_time
        result["task_preview"] = task[:80]

        self.history.append(result)
        return result

    def get_history(self, tier: Optional[str] = None, limit: int = 20) -> list:
        if tier:
            filtered = [h for h in self.history if h.get("tier") == tier]
            return filtered[-limit:]
        return self.history[-limit:]

    def get_tier_stats(self) -> Dict[str, Any]:
        stats = {"T1": 0, "T2": 0, "T3": 0}
        quality_counts = {"good": 0, "partial": 0, "low": 0, "excellent": 0}

        for entry in self.history:
            tier = entry.get("tier", "unknown")
            stats[tier] = stats.get(tier, 0) + 1

            quality = entry.get("quality", "unknown")
            if "excellent" in quality:
                quality_counts["excellent"] += 1
            elif "good" in quality:
                quality_counts["good"] += 1
            elif "partial" in quality:
                quality_counts["partial"] += 1
            elif "low" in quality:
                quality_counts["low"] += 1

        total = sum(stats.values())
        return {
            "total_executions": total,
            "by_tier": stats,
            "quality_distribution": quality_counts,
        }


# ── End-to-end test harness ────────────────────────────────────────────────────

class OptimiserTestHarness:
    """Full end-to-end test harness: assess → route → execute → feedback."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.gateway = MockGateway(config)
        self.results: list = []

    def run_task(self, task: str, context: Optional[dict] = None) -> Dict[str, Any]:
        from token_optimiser.router import TokenRouter, LearningLog
        from token_optimiser.config import config as default_config

        router = TokenRouter()
        log = LearningLog()

        assessment = router.assess(task, context)
        execution = self.gateway.execute(assessment.tier, task, **context or {})

        result = {
            "task": task,
            "assessment": assessment.to_dict(),
            "execution": execution,
            "tier_used": assessment.tier,
            "confidence": assessment.confidence,
            "timestamp": time.time(),
        }
        self.results.append(result)
        log.append(assessment, outcome={
            "outcome": "worked" if "error" not in execution else "failed",
            "quality": execution.get("quality", "unknown"),
        })

        return result

    def run_scenario(self, name: str, tasks: list) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {name}")
        print(f"{'='*60}")

        for task in tasks:
            result = self.run_task(task)
            print(f"\nTask: {task[:80]}")
            print(f"  → Assessed: {result['assessment']['confidence']}% confidence")
            print(f"  → Routed to: {result['tier_used']}")
            print(f"  → Executed on: {result['execution'].get('backend', 'N/A')}")
            print(f"  → Quality: {result['execution'].get('quality', 'N/A')}")

        stats = self.gateway.get_tier_stats()
        print(f"\n--- Scenario Stats ---")
        print(f"Total: {stats['total_executions']} tasks")
        print(f"T1: {stats['by_tier'].get('T1', 0)} | T2: {stats['by_tier'].get('T2', 0)} | T3: {stats['by_tier'].get('T3', 0)}")

        return {
            "scenario": name,
            "tasks": len(tasks),
            "results": self.results[-len(tasks):],
            "gateway_stats": stats,
        }

    def print_summary(self):
        print(f"\n{'='*60}")
        print("FULL TEST SUMMARY")
        print(f"{'='*60}")

        stats = self.gateway.get_tier_stats()
        print(f"\nTotal executions: {stats['total_executions']}")

        print(f"\nBy tier:")
        for tier, count in stats["by_tier"].items():
            print(f"  {tier}: {count} tasks")

        print(f"\nQuality distribution:")
        for quality, count in stats["quality_distribution"].items():
            print(f"  {quality}: {count} tasks")

        if self.results:
            print(f"\nLearning log entries: {len(self.results)}")
            print(f"Mock history entries: {len(self.gateway.history)}")
