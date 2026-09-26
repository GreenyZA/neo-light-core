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

    def generate(self, task: str, seat_id: str = None, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class T1MockBackend(MockBackend):
    """Mock T1 backend — local llama-server (qwen2.5-coder-3b-instruct Q4_K_M)."""

    name: str = "T1 Mock (llama-server qwen2.5-coder-3b)"
    tier: str = "T1"
    response_time: float = 0.5
    quality: float = 0.6  # 3B model — decent but limited

    def generate(self, task: str, seat_id: str = None, **kwargs) -> Dict[str, Any]:
        time.sleep(self.response_time)

        # Strip framing prefix
        if "Question: " in task:
            task_preview = task.split("Question: ", 1)[-1]
        else:
            task_preview = task[-80:] if len(task) > 80 else task

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
                f"[T1 SIMULATED] Here's the result for: {task_preview}... "
                f"[concise, direct answer — T1 is good at straightforward tasks]"
            ),
            "quality": "good_for_simple",
            "simulated": True,
            "notes": "T1 handles simple tasks well",
        }


@dataclass
class T1PrivacyBackend(T1MockBackend):
    """
    T1 privacy/offline backend — simulates local-only processing.
    No cloud calls, no data leaves the machine.
    For privacy-sensitive questions that must stay local.
    """

    name: str = "T1 Privacy (Offline Local — qwen2.5-coder-3b)"
    tier: str = "T1"
    response_time: float = 0.3
    quality: float = 0.55

    def generate(self, task: str, seat_id: str = None, **kwargs) -> Dict[str, Any]:
        time.sleep(self.response_time)
        return {
            "tier": "T1",
            "backend": "privacy_offline",
            "model": "qwen2.5-coder-3b-instruct",
            "response": (
                "[T1 PRIVACY — LOCAL ONLY] Offline response for: "
                f"'{task[-60:]}...' | Seat: {seat_id} | "
                "No data leaves this machine. Using local 3B model — "
                "good for non-sensitive tasks, limited on complex reasoning."
            ),
            "quality": "privacy_limited",
            "simulated": False,
            "seat_id": seat_id,
            "privacy_mode": True,
            "notes": "T1 Privacy: fully offline, no cloud calls",
        }


@dataclass
class T2MockBackend(MockBackend):
    """
    Mock T2 backend — free cloud frontier models (2026).
    Simulates: NVIDIA Nemotron 3 Ultra, DeepSeek V4 Flash, Poolside Laguna S 2.1
    These are FAR superior to the local 3B model.

    Seat-specific responses: each board persona gets a tailored answer
    reflecting their focus (Architect, Engineer, Researcher, Reviewer, Budget, Executive).
    """

    name: str = "T2 Mock (Free Cloud — Nemotron/DeepSeek/Poolside)"
    tier: str = "T2"
    response_time: float = 1.5  # Slightly slower than T1 (network)
    quality: float = 0.95  # Frontier free models — excellent

    # Seat-specific response templates keyed by seat_id
    SEAT_TEMPLATES: Dict[str, str] = field(default_factory=lambda: {
        "architect": (
            "[T2 — {model}] ## System Design\n\n"
            "For the question '{task_preview}', I'd recommend a layered architecture:\n\n"
            "1. **Service mesh**: Use a sidecar proxy for cross-cutting concerns (auth, retries, observability).\n"
            "2. **Bounded contexts**: Split into independent deployable units — each owns its data.\n"
            "3. **API gateway**: Single entry point with rate limiting, auth enforcement, and request routing.\n"
            "4. **Async messaging**: Use an event bus (NATS/Kafka) for inter-service communication.\n\n"
            "Trade-off: eventual consistency vs. strong consistency — choose per subdomain. "
            "Long-term maintainability favors explicit boundaries over shared schemas."
        ),
        "engineer": (
            "[T2 — {model}] ## Implementation\n\n"
            "For '{task_preview}', here's the practical path:\n\n"
            "```python\n"
            "# Minimal viable implementation\n"
            "from fastapi import FastAPI, Depends\n"
            "from auth import verify_token  # your auth provider\n\n"
            "app = FastAPI()\n\n"
            "@app.get('/resource')\n"
            "async def get_resource(user=Depends(verify_token)):\n"
            "    return {'data': 'protected', 'user': user.sub}\n"
            "```\n\n"
            "Stack: FastAPI + Uvicorn + PyJWT (or better: validate JWKS from IdP). "
            "Performance: ~2ms overhead per request with JWT validation. "
            "Keep middleware thin — offload heavy auth to sidecar."
        ),
        "researcher": (
            "[T2 — {model}] ## Evidence & Benchmarks\n\n"
            "Regarding '{task_preview}':\n\n"
            "- **OWASP Top 10 2025**: Broken Access Control remains #1 risk. "
            "Implement RBAC/ABAC with deny-by-default.\n"
            "- **NIST SP 800-63B**: Recommends FIDO2/WebAuthn over passwords for new systems.\n"
            "- **Benchmarks**: Ed25519 signatures at ~50μs/verify on modern hardware vs RSA-2048 at ~300μs.\n"
            "- **Precedent**: GitHub, Stripe all use short-lived JWTs (5-15 min) with refresh rotation.\n\n"
            "Citation: Demonstrated that token-based auth scales horizontally without shared state — "
            "critical for microservice topology."
        ),
        "reviewer": (
            "[T2 — {model}] ## Risk Assessment\n\n"
            "For '{task_preview}', here are the failure modes to address:\n\n"
            "⚠️ **Token theft**: Short-lived tokens help, but stolen tokens are valid until expiry. "
            "Mitigate with binding (TLS token binding or DPoP).\n\n"
            "⚠️ **Secret sprawl**: API keys in env vars leak via /proc, crash logs, or debug endpoints. "
            "Use a secrets manager (HashiCorp Vault, AWS Secrets Manager) with rotation.\n\n"
            "⚠️ **Downstream cascade**: If auth service is down, all services fail. "
            "Cache JWKS locally with TTL + stale-while-revalidate.\n\n"
            "⚠️ **Edge case**: Clock skew between services causes JWT exp validation failures. "
            "Allow 30-60s leeway max, log drift incidents."
        ),
        "budget": (
            "[T2 — {model}] ## Cost Analysis\n\n"
            "Routing '{task_preview}' — cost-optimal tier selection:\n\n"
            "| Tier | Cost | Suitability |\n"
            "|------|------|-------------|\n"
            "| T1 (local 3B) | ~$0 | Partial — outlines only, no security depth |\n"
            "| **T2 (free cloud)** | **$0** | **Best fit — frontier free models handle this well** |\n"
            "| T3 (premium) | $$ | Overkill unless compliance audit required |\n\n"
            "**Recommendation**: T2 free tier (Nemotron/DeepSeek) — quality=0.95, cost=$0. "
            "Reserve T3 for compliance-sensitive or production-security reviews."
        ),
        "executive": (
            "[T2 — {model}] ## Board Recommendation\n\n"
            "Synthesizing for '{task_preview}':\n\n"
            "**Decision**: Proceed with T2 free-tier implementation.\n\n"
            "Priority order:\n"
            "1. Immediate: Set up API gateway with JWT validation (Engineer's path)\n"
            "2. This sprint: Define bounded contexts and async messaging (Architect's view)\n"
            "3. Before launch: Security review for the 4 risk vectors identified (Reviewer)\n"
            "4. Ongoing: Monitor T2 quality metrics — escalate to T3 only if confidence drops\n\n"
            "Expected outcome: Secure, maintainable, zero-cost-infrastructure solution."
        ),
    })

    def generate(self, task: str, seat_id: str = None, **kwargs) -> Dict[str, Any]:
        time.sleep(self.response_time)

        task_lower = task.lower()
        # Strip framing prefix if present
        if "Question: " in task:
            task_preview = task.split("Question: ", 1)[-1]
        else:
            task_preview = task[-80:] if len(task) > 80 else task

        # Simulate model selection based on task type
        if any(kw in task_lower for kw in ["code", "write", "implement", "class", "function"]):
            model_used = "poolside/laguna-s-2.1:free"
            model_short = "Laguna S 2.1"
        elif any(kw in task_lower for kw in ["research", "analyze", "compare", "evaluate"]):
            model_used = "deepseek/deepseek-v4-flash-0731:free"
            model_short = "DeepSeek V4 Flash"
        else:
            model_used = "nvidia/nemotron-3-ultra-550b-a55b:free"
            model_short = "Nemotron 3 Ultra"

        # Get seat-specific template or fall back to generic
        template = self.SEAT_TEMPLATES.get(seat_id) if seat_id else None
        if template is None:
            template = (
                "[T2 — {model}] Frontier model response for: '{task_preview}'... "
                "[comprehensive, well-structured answer from {model} — "
                "free tier is now stronger than local 3B for most tasks]"
            )

        response = template.format(model=model_short, task_preview=task_preview)

        return {
            "tier": "T2",
            "backend": "free_cloud",
            "model": model_used,
            "response": response,
            "quality": "excellent",
            "simulated": True,
            "seat_id": seat_id,
            "notes": f"T2 free cloud using {model_short} — seat-specific response",
        }


@dataclass
class T3MockBackend(MockBackend):
    """Mock T3 backend — high tier. Best quality, slow, expensive."""

    name: str = "T3 Mock (High Tier)"
    tier: str = "T3"
    response_time: float = 2.0
    quality: float = 0.98

    # Seat-specific response templates keyed by seat_id
    SEAT_TEMPLATES: Dict[str, str] = field(default_factory=lambda: {
        "architect": (
            "[T3 — grok] ## Premium Architecture Review\n\n"
            "Comprehensive analysis of '{task_preview}':\n\n"
            "### 1. Domain-Driven Design\n"
            "- Identify bounded contexts and aggregate roots\n"
            "- Define ubiquitous language with all stakeholders\n\n"
            "### 2. C4 Model Documentation\n"
            "- Context diagram → Container diagram → Component diagram\n"
            "- Use Structurizr DSL for living architecture docs\n\n"
            "### 3. Resilience Patterns\n"
            "- Circuit breaker with exponential backoff\n"
            "- Bulkhead isolation for critical paths\n"
            "- Chaos engineering: verify failure modes before production\n\n"
            "### 4. Long-term Evolution\n"
            "- Plan for versioning strategy (URL vs header vs content-type)\n"
            "- Event sourcing for audit-critical paths\n"
            "- CQRS when read/write patterns diverge significantly"
        ),
        "engineer": (
            "[T3 — grok] ## Production Implementation\n\n"
            "Full implementation for '{task_preview}':\n\n"
            "```python\n"
            "# FastAPI with full middleware stack\n"
            "from fastapi import FastAPI, Depends, Request\n"
            "from fastapi.middleware.cors import CORSMiddleware\n"
            "from opentelemetry import trace\n"
            "from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor\n\n"
            "app = FastAPI()\n\n"
            "# OpenTelemetry auto-instrumentation\n"
            "FastAPIInstrumentor.instrument_app(app)\n\n"
            "# Middleware: correlation ID → metrics → auth\n"
            "app.add_middleware(CORSMiddleware, allow_origins=['*'])\n\n"
            "@app.middleware('http')\n"
            "async def add_correlation_id(request: Request, call_next):\n"
            "    cid = request.headers.get('X-Correlation-ID', str(uuid4()))\n"
            "    with trace.get_tracer(__name__).start_as_current_span('request'):\n"
            "        response = await call_next(request)\n"
            "    response.headers['X-Correlation-ID'] = cid\n"
            "    return response\n"
            "```\n\n"
            "### Performance Targets\n"
            "- P50 < 10ms, P99 < 100ms\n"
            "- Connection pooling: min 5, max 20\n"
            "- Health checks: /health, /ready, /startup"
        ),
        "researcher": (
            "[T3 — grok] ## Deep Research Briefing\n\n"
            "Evidence-based analysis of '{task_preview}':\n\n"
            "### Academic Foundations\n"
            "- Fowler (2002): 'Microservices' — original definition and trade-offs\n"
            "- Richardson Maturity Model: Level 2+ required for RESTful microservices\n"
            "- CAP Theorem: partition tolerance is mandatory → choose CP or AP\n\n"
            "### Industry Benchmarks (2025)\n"
            "- DORA metrics: elite deploy frequency = on-demand (multiple/day)\n"
            "- Google SRE: error budget policy — 0.1% downtime acceptable\n"
            "- AWS Well-Architected: 6 pillars including performance efficiency\n\n"
            "### Recent Findings\n"
            "- 'Microservices Premium' (Brandenburg 2024): 40% overhead for small teams\n"
            "- Monolith-first approach validated for <10 engineers\n"
            "- Event-driven architectures reduce coupling by 60% (measured via fan-out)"
        ),
        "reviewer": (
            "[T3 — grok] ## Critical Risk Assessment\n\n"
            "Thorough review of '{task_preview}':\n\n"
            "### 🔴 Critical Risks\n"
            "1. **Distributed monolith**: Services tightly coupled via sync HTTP → cascading failures\n"
            "2. **Data inconsistency**: No saga pattern → partial writes corrupt state\n"
            "3. **Observability gap**: Missing distributed traces → blind to cross-service issues\n\n"
            "### 🟡 Moderate Risks\n"
            "4. **Secret rotation**: No automated rotation → stale credentials in production\n"
            "5. **API versioning**: Breaking changes without deprecation window\n"
            "6. **Resource limits**: No memory/CPU limits → noisy neighbor kills node\n\n"
            "### 🟢 Mitigations Required\n"
            "- Implement saga orchestrator for multi-step operations\n"
            "- Add distributed tracing (OpenTelemetry) before launch\n"
            "- Define SLOs with error budgets per service"
        ),
        "budget": (
            "[T3 — grok] ## Cost-Benefit Analysis\n\n"
            "Financial assessment of '{task_preview}':\n\n"
            "| Approach | Monthly Cost | Quality | ROI |\n"
            "|----------|-------------|---------|-----|\n"
            "| T1 Local | $0 | 0.60 | Low — limited capability |\n"
            "| T2 Free Cloud | $0 | 0.95 | **High — best value** |\n"
            "| T3 Premium | ~$50-200 | 0.98 | Medium — diminishing returns |\n\n"
            "### Recommendation\n"
            "T3 is justified ONLY for:\n"
            "- Compliance/audit requirements (SOC2, HIPAA)\n"
            "- Production security reviews\n"
            "- High-stakes architectural decisions\n\n"
            "For routine development: T2 free tier provides 97% of T3 quality at $0 cost."
        ),
        "executive": (
            "[T3 — grok] ## Executive Decision Brief\n\n"
            "Strategic recommendation for '{task_preview}':\n\n"
            "### Decision Matrix\n"
            "- **Risk Level**: Medium (new architecture pattern)\n"
            "- **Time to Value**: 2-3 sprints for MVP\n"
            "- **Team Readiness**: Requires upskilling on distributed systems\n\n"
            "### Recommended Path\n"
            "1. **Phase 1** (Sprint 1-2): Modular monolith with clear boundaries\n"
            "2. **Phase 2** (Sprint 3-4): Extract highest-churn service to microservice\n"
            "3. **Phase 3** (Sprint 5+): Gradual decomposition based on metrics\n\n"
            "### Success Criteria\n"
            "- Independent deployability of extracted services\n"
            "- Zero-downtime deployments via blue-green\n"
            "- P99 latency < 200ms for cross-service calls\n\n"
            "**Verdict**: Proceed with phased approach. Avoid big-bang rewrite."
        ),
    })

    def generate(self, task: str, seat_id: str = None, **kwargs) -> Dict[str, Any]:
        time.sleep(self.response_time)

        task_lower = task.lower()
        # Strip framing prefix if present (e.g. "You are the Architect... Question: ")
        if "Question: " in task:
            task_preview = task.split("Question: ", 1)[-1]
        else:
            task_preview = task[-80:] if len(task) > 80 else task

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

        # Get seat-specific template or fall back to generic
        template = self.SEAT_TEMPLATES.get(seat_id) if seat_id else None
        if template is None:
            if any(kw in task_lower for kw in ["security", "auth", "encrypt", "vulnerability"]):
                template = (
                    "[T3 — grok] Full security analysis for: '{task_preview}'...\n\n"
                    "#### Threat Model\n"
                    "[detailed threat modeling with attack vectors and mitigations]\n\n"
                    "#### Security Review\n"
                    "[thorough review of auth/crypto/secrets handling]\n\n"
                    "#### Recommendations\n"
                    "[specific, actionable security recommendations]\n\n"
                    "#### Code Examples\n"
                    "[secure implementation examples with explanations]"
                )
            else:
                template = (
                    "[T3 — grok] Excellent response for: '{task_preview}'... "
                    "[comprehensive, well-structured answer with full context and edge cases]"
                )

        response = template.format(task_preview=task_preview)

        return {
            "tier": "T3",
            "backend": "high_tier",
            "model": "grok",
            "response": response,
            "quality": "excellent",
            "simulated": True,
            "seat_id": seat_id,
            "notes": "T3 premium tier — seat-specific response",
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
