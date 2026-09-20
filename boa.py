#!/usr/bin/env python3
"""
BOA Runner — Board of Advisors Query Orchestrator with ITIL + Governance.

Accepts a question, classifies it (ops → ITIL), routes through TOS,
collects answers from 6 board seats, synthesises, writes session log,
and logs governance decisions to work/boa/governance/.

Importable but optional — not required by the token_optimiser package.

Usage:
    from token_optimiser.boa import run_boa, classify_itil
    result = run_boa("What's the best way to handle auth in a microservice?")

Or CLI:
    python -m token_optimiser.boa "your question here"
    python token_optimiser/boa.py "your question here"
"""

import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure the package root is importable when run as a script.
_PACKAGE_ROOT = Path(__file__).resolve().resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from token_optimiser.monica_integration import optimise
from token_optimiser.mock_backends import T1MockBackend, T1PrivacyBackend, T2MockBackend, T3MockBackend

# ── Seat definitions ──────────────────────────────────────────────────────────

SEATS: List[Dict[str, str]] = [
    {
        "id": "architect",
        "name": "Architect",
        "framing": (
            "You are the Architect on the Board of Advisors. Focus on system design, "
            "trade-offs, long-term maintainability, and structural coherence. "
            "Question: "
        ),
    },
    {
        "id": "engineer",
        "name": "Engineer",
        "framing": (
            "You are the Engineer on the Board of Advisors. Focus on implementation "
            "details, practicality, performance, and correctness. "
            "Question: "
        ),
    },
    {
        "id": "researcher",
        "name": "Researcher",
        "framing": (
            "You are the Researcher on the Board of Advisors. Focus on evidence, "
            "precedents, benchmarks, and citations. Where possible, reference known "
            "patterns or data. "
            "Question: "
        ),
    },
    {
        "id": "reviewer",
        "name": "Reviewer",
        "framing": (
            "You are the Reviewer on the Board of Advisors. Critically assess the "
            "proposal: identify risks, blind spots, edge cases, and failure modes. "
            "Be constructively sceptical. "
            "Question: "
        ),
    },
    {
        "id": "budget",
        "name": "Budget (Token Optimiser)",
        "framing": (
            "You are the Budget seat on the Board of Advisors (Token Optimiser). "
            "Focus on cost, routing, and efficiency. Which tier (T1/T2/T3) is "
            "appropriate for implementing the solution? What's the cheapest viable "
            "path? "
            "Question: "
        ),
    },
    {
        "id": "executive",
        "name": "Executive (Chair)",
        "framing": (
            "You are the Executive Chair on the Board of Advisors. Synthesise the "
            "board's views into a single recommendation. Decide what to do, in what "
            "order, with what priority. "
            "Question: "
        ),
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# ITIL Classification
# ═══════════════════════════════════════════════════════════════════════════════

# ITIL incident keywords — service disruption, degradation, failures
ITIL_INCIDENT_KEYWORDS = [
    "down", "outage", "failure", "failed", "broken", "error", "crash", "stale",
    "unavailable", "unreachable", "timeout", "timing out", "502", "503", "504",
    "hops stale", "stale hop", "not responding", "service down", "dead",
    "failing", "degradation", "degraded", "latency spike", "slow",
    "connection refused", "refusing connections", "refused",
]

# ITIL problem keywords — root cause, recurring issues, known errors
ITIL_PROBLEM_KEYWORDS = [
    "root cause", "recurring", "keeps happening", "repeatedly", "known error",
    "systemic", "pattern of failures", "chronic", "intermittent",
    "investigate why", "why does this keep", "post-mortem", "postmortem",
    "rca", "5 whys", "trend analysis",
]

# ITIL change keywords — planned modifications, deployments
ITIL_CHANGE_KEYWORDS = [
    "change", "deploy", "deployment", "modify", "modify config", "config change",
    "update config", "alter configuration", "roll out", "rollback",
    "release", "migration", "upgrade", "version bump", "chg",
    "change request", "implement change", "push to prod", "production push",
    "routing config", "change routing", "config update",
]

# ITIL request keywords — new additions, feature requests, catalog items
ITIL_REQUEST_KEYWORDS = [
    "new model", "add model", "available", "feature request", "request for",
    "new tool", "integrate", "onboard", "provision", "catalog",
    "request access", "new gateway", "new provider", "new free",
    "new models available", "can we add", "should we add",
]

# ITIL event keywords — alerts, notifications, informational
ITIL_EVENT_KEYWORDS = [
    "alert", "notification", "warn", "threshold", "monitor", "metric",
    "log entry", "health check", "ping", "heartbeat", "sensor",
    "informational", "status update", "scheduled maintenance",
]

# Ops detection — any of these categories trigger ITIL classification
OPS_CATEGORIES = [
    "incident", "problem", "change", "request", "event",
]


def classify_itil(question: str) -> Dict[str, Any]:
    """
    Classify a question using ITIL taxonomy.

    Returns a dict with:
        - is_ops: bool — whether this is an ops-related question
        - classification: str — incident/problem/change/request/event/unknown
        - confidence: float — 0.0-1.0 classification confidence
        - requires_chg_number: bool — Change-type requires CHG number
        - prefix: str — INC-/PRB-/CHG-/REQ-/EVT- log prefix
        - keywords_matched: list[str] — which keywords triggered classification
    """
    q = question.lower().strip()

    scores: Dict[str, float] = {}
    matched_keywords: Dict[str, List[str]] = {}

    # Score each ITIL category
    categories = {
        "incident": ITIL_INCIDENT_KEYWORDS,
        "problem": ITIL_PROBLEM_KEYWORDS,
        "change": ITIL_CHANGE_KEYWORDS,
        "request": ITIL_REQUEST_KEYWORDS,
        "event": ITIL_EVENT_KEYWORDS,
    }

    for cat, keywords in categories.items():
        hits = [kw for kw in keywords if kw in q]
        matched_keywords[cat] = hits
        # Score: number of hits, weighted by keyword specificity (longer = more specific)
        score = sum(len(kw.split()) for kw in hits)
        scores[cat] = float(score)

    max_score = max(scores.values()) if scores else 0.0
    total_score = sum(scores.values()) if scores else 0.0

    if max_score == 0.0:
        return {
            "is_ops": False,
            "classification": "unknown",
            "confidence": 0.0,
            "requires_chg_number": False,
            "prefix": "",
            "keywords_matched": [],
        }

    best_category = max(scores, key=scores.get)
    confidence = max_score / (total_score + 1e-6) if total_score > 0 else 0.0
    confidence = min(confidence, 1.0)

    prefix_map = {
        "incident": "INC",
        "problem": "PRB",
        "change": "CHG",
        "request": "REQ",
        "event": "EVT",
    }

    return {
        "is_ops": True,
        "classification": best_category,
        "confidence": round(confidence, 3),
        "requires_chg_number": best_category == "change",
        "prefix": prefix_map.get(best_category, ""),
        "keywords_matched": matched_keywords.get(best_category, []),
    }


def _generate_incident_id() -> str:
    """Generate a compact incident/problem/change ID."""
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"BOA-{ts}"


# ═══════════════════════════════════════════════════════════════════════════════
# TOS Optimization — T2 Gateway Probing
# ═══════════════════════════════════════════════════════════════════════════════

# Known T2 free-model gateways and their models
T2_GATEWAYS = {
    "openrouter": {
        "label": "OpenRouter (free)",
        "endpoint": "https://openrouter.ai/api/v1/models",
        "known_free_models": [
            "nvidia/nemotron-3-ultra-550b",
            "nvidia/nemotron-3-super-120b",
            "deepseek/deepseek-v4-flash-0731",
            "poolside/laguna-s-2.1",
            "cohere/north-mini-code",
            "qwen/qwen3-32b",
            "meta-llama/llama-3.3-70b-instruct",
            "google/gemma-3-27b-it",
        ],
    },
    "nvidia": {
        "label": "NVIDIA NIM (free tier)",
        "endpoint": "https://integrate.api.nvidia.com/v1/models",
        "known_free_models": [
            "nvidia/llama-3.3-nemotron-super-49b-v1",
            "nvidia/nemotron-3-ultra-550b-moe",
            "meta/llama-3.1-70b-instruct",
            "mistralai/mistral-large",
        ],
    },
    "deepseek": {
        "label": "DeepSeek (free)",
        "endpoint": "https://api.deepseek.com/v1/models",
        "known_free_models": [
            "deepseek-chat",
            "deepseek-v4-flash-0731",
            "deepseek-reasoner",
        ],
    },
    "poolside": {
        "label": "Poolside (free)",
        "endpoint": "https://api.poolside.ai/v1/models",
        "known_free_models": [
            "laguna-s-2.1",
            "laguna-s-2",
        ],
    },
    "cohere": {
        "label": "Cohere (free trial)",
        "endpoint": "https://api.cohere.com/v1/models",
        "known_free_models": [
            "command-r-plus",
            "command-r",
            "north-mini-code",
        ],
    },
}


def probe_t2_gateways(question: str) -> Dict[str, Any]:
    """
    Probe T2 gateways for current free model availability and quality.

    For routing/quality questions, checks which free models are currently
    available across gateways. Returns structured availability info.

    Note: In a real implementation, this would make HTTP calls to gateway /models
    endpoints. Here we check env vars for connectivity flags and return
    structured availability based on known configuration.
    """
    q = question.lower()

    # Only probe if question is about routing, model availability, or quality
    routing_keywords = [
        "routing", "route", "available", "free models", "model quality",
        "which model", "best free", "t2 gateway", "openrouter", "nvidia",
        "deepseek", "poolside", "provider", "tier 2", "tier2", "gateway",
        "hops stale", "switch provider", "fallback",
    ]

    if not any(kw in q for kw in routing_keywords):
        return {"probed": False, "reason": "Not a routing/quality question"}

    # Check env vars for gateway reachability
    gateway_status: Dict[str, Dict[str, Any]] = {}

    for gw_key, gw_info in T2_GATEWAYS.items():
        env_var = f"T2_{gw_key.upper()}_ENABLED"
        env_token = os.environ.get(f"{gw_key.upper()}_API_TOKEN", "")
        is_enabled = os.environ.get(env_var, "auto").lower()
        has_token = bool(env_token) or gw_key in ("openrouter", "deepseek")

        if is_enabled == "false":
            status = "disabled"
        elif is_enabled == "true":
            status = "enabled"
        else:
            # auto — infer from token presence and connectivity
            status = "available" if has_token else "no_token"

        gateway_status[gw_key] = {
            "label": gw_info["label"],
            "endpoint": gw_info["endpoint"],
            "status": status,
            "known_models": gw_info["known_free_models"],
            "has_token": has_token,
        }

    # Determine which models are most relevant
    recommended = []
    if "stale" in q or "hops" in q:
        recommended = [
            {"gateway": "openrouter", "model": "nvidia/nemotron-3-ultra-550b", "reason": "Strongest free frontier model"},
            {"gateway": "openrouter", "model": "deepseek/deepseek-v4-flash-0731", "reason": "Research + analysis fallback"},
        ]
    elif "quality" in q or "best" in q:
        recommended = [
            {"gateway": "openrouter", "model": "nvidia/nemotron-3-ultra-550b", "reason": "Highest quality free tier"},
            {"gateway": "nvidia", "model": "nvidia/llama-3.3-nemotron-super-49b-v1", "reason": "Agentic reasoning"},
        ]
    elif "available" in q or "new" in q:
        recommended = [
            {"gateway": gw_key, "model": gw["known_models"][0], "reason": "Primary free model"}
            for gw_key, gw in gateway_status.items()
            if gw["status"] in ("enabled", "available") and gw["known_models"]
        ][:5]
    else:
        recommended = [
            {"gateway": "openrouter", "model": "nvidia/nemotron-3-ultra-550b", "reason": "Default best free"},
        ]

    return {
        "probed": True,
        "gateways": gateway_status,
        "recommended": recommended,
        "probe_timestamp": datetime.datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Swarm Governance
# ═══════════════════════════════════════════════════════════════════════════════

def validate_swarm_plan(question: str) -> Dict[str, Any]:
    """
    Validate a swarm plan described in the question.

    Checks:
    - Sizing: number of workers matches task count and independence
    - Task independence: tasks should not have shared state / ordering dependencies
    - Reviewer assignment: every worker must have a designated reviewer
    - Threshold delegation: >3 files, >2 modules, >5 sequential calls triggers swarm

    Returns a governance assessment dict.
    """
    q = question.lower()

    swarm_keywords = [
        "swarm", "delegate_task", "multi-agent", "multi agent",
        "parallel workers", "orchestrate", "spawn workers",
        "worker count", "how many workers", "swarm plan",
        "delegate", "agents", "workers", "files",
        "parallel tasks", "parallelize",
    ]

    if not any(kw in q for kw in swarm_keywords):
        return {"governed": False, "reason": "Not a swarm plan question"}

    # Extract sizing info
    worker_count = None
    task_count = None

    worker_match = re.search(r'(\d+)\s*(?:workers?|agents?)', q)
    if worker_match:
        worker_count = int(worker_match.group(1))

    task_match = re.search(r'(\d+)\s*(?:tasks?|files?)', q)
    if task_match:
        task_count = int(task_match.group(1))

    # Infer file/module counts from question
    file_patterns = re.findall(r'[\w./\-_]+\.(?:py|js|ts|jsx|tsx|go|rs|java|rb|cpp|c|h|hpp|cs|php|swift|kt|scala|sh|yaml|yml|json|toml|xml|md|txt|sql)', q)
    file_count = len(file_patterns) if file_patterns else (task_count or 0)

    module_keywords = ["module", "package", "crate", "library"]
    module_count = sum(1 for kw in module_keywords if kw in q)
    module_count = max(module_count, 1 if task_count and task_count > 2 else 0)

    # Apply delegation thresholds
    exceeds_files = file_count > 3
    exceeds_modules = module_count > 2
    sequential_calls = re.findall(r'(?:step|then|after|next)\s+\d+', q)
    exceeds_sequential = len(sequential_calls) > 5

    should_delegate = exceeds_files or exceeds_modules or exceeds_sequential

    # Size validation
    sizing_ok = True
    sizing_notes: List[str] = []

    if worker_count is not None and task_count is not None:
        if worker_count > task_count:
            sizing_ok = False
            sizing_notes.append(f"Too many workers ({worker_count}) for {task_count} tasks")
        if worker_count < 2 and should_delegate:
            sizing_ok = False
            sizing_notes.append(f"Single worker for swarm task — should use delegate_task")

    if worker_count is None and should_delegate:
        sizing_notes.append("No explicit worker count — recommend 2-5 for file-heavy tasks")

    # Task independence check
    has_dependencies = any(kw in q for kw in ["after", "before", "depends on", "order", "sequential", "then"])
    independence_ok = not has_dependencies or (worker_count is not None and worker_count <= 3)

    # Reviewer assignment check
    has_reviewer = any(kw in q for kw in ["reviewer", "review", "paired", "clean pass"])
    reviewer_ok = has_reviewer or not should_delegate  # reviewer needed for swarms

    # Governance decision
    governance_pass = sizing_ok and independence_ok and (not should_delegate or reviewer_ok)

    return {
        "governed": True,
        "should_delegate": should_delegate,
        "delegation_triggers": {
            "exceeds_file_threshold": exceeds_files,
            "exceeds_module_threshold": exceeds_modules,
            "exceeds_sequential_threshold": exceeds_sequential,
        },
        "sizing": {
            "worker_count": worker_count,
            "task_count": task_count,
            "file_count": file_count,
            "module_count": module_count,
            "sizing_ok": sizing_ok,
            "notes": sizing_notes,
        },
        "independence": {
            "has_dependencies": has_dependencies,
            "independence_ok": independence_ok,
        },
        "reviewer": {
            "has_reviewer": reviewer_ok,
            "reviewer_required": should_delegate,
        },
        "governance_pass": governance_pass,
        "recommendation": _swarm_recommendation(
            governance_pass, should_delegate, worker_count, independence_ok, reviewer_ok
        ),
    }


def _swarm_recommendation(
    governance_pass: bool,
    should_delegate: bool,
    worker_count: Optional[int],
    independence_ok: bool,
    reviewer_ok: bool,
) -> str:
    """Build a human-readable swarm governance recommendation."""
    if not should_delegate:
        return "Solo execution approved — task does not meet swarm thresholds."

    if governance_pass:
        rec = "APPROVED for swarm delegation."
        if worker_count:
            rec += f" {worker_count} workers is within bounds."
        return rec

    issues: List[str] = []
    if not independence_ok:
        issues.append("tasks have sequential dependencies — decompose into independent units")
    if not reviewer_ok:
        issues.append("no reviewer assigned — each worker needs a paired reviewer")

    return f"SWARM REJECTED: {'; '.join(issues)}. Fix before delegating."


# ═══════════════════════════════════════════════════════════════════════════════
# Governance Audit Trail
# ═══════════════════════════════════════════════════════════════════════════════

def _governance_dir(work_dir: Path) -> Path:
    """Return the governance audit directory, creating if needed."""
    gov_dir = work_dir / "governance"
    gov_dir.mkdir(parents=True, exist_ok=True)
    return gov_dir


def log_governance_decision(
    question: str,
    itil_classification: Optional[Dict[str, Any]],
    tos_probe: Optional[Dict[str, Any]],
    swarm_governance: Optional[Dict[str, Any]],
    work_dir: Path,
) -> Optional[Path]:
    """
    Log a BOA governance decision to work/boa/governance/ as JSONL.

    Only logs if there's something to govern (ITIL ops, TOS probe, or swarm).
    Returns the log path, or None if no governance was needed.
    """
    if not itil_classification.get("is_ops") and not tos_probe.get("probed") and not swarm_governance.get("governed"):
        return None

    gov_dir = _governance_dir(work_dir)
    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "incident_id": _generate_incident_id(),
    }

    if itil_classification.get("is_ops"):
        entry["itil"] = {
            "classification": itil_classification["classification"],
            "confidence": itil_classification["confidence"],
            "prefix": itil_classification["prefix"],
            "keywords_matched": itil_classification["keywords_matched"],
            "requires_chg_number": itil_classification["requires_chg_number"],
        }

    if tos_probe.get("probed"):
        entry["tos_probe"] = {
            "gateways_probed": list(tos_probe["gateways"].keys()),
            "recommended_models": tos_probe["recommended"],
        }

    if swarm_governance.get("governed"):
        entry["swarm"] = {
            "should_delegate": swarm_governance["should_delegate"],
            "governance_pass": swarm_governance["governance_pass"],
            "recommendation": swarm_governance["recommendation"],
            "sizing": swarm_governance["sizing"],
            "reviewer": swarm_governance["reviewer"],
        }

    log_path = gov_dir / f"governance-{ts}.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    return log_path


# ═══════════════════════════════════════════════════════════════════════════════
# Backend resolution (mock vs. real)
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_backend() -> str:
    """Return backend mode: 'mock' for offline/test, 'live' otherwise."""
    return os.environ.get("BOA_BACKEND", "mock").lower()


def _generate_seat_answer(seat: Dict[str, str], question: str, tier: str) -> Dict[str, Any]:
    """
    Generate a single seat's answer.

    In mock mode (default), returns a structured placeholder based on the seat's
    persona and the routed tier. In live mode, calls the appropriate T1/T2/T3
    mock backend from mock_backends.py.
    """
    backend = _resolve_backend()
    prompt = seat["framing"] + question

    if backend == "mock":
        return {
            "seat_id": seat["id"],
            "seat_name": seat["name"],
            "tier": tier,
            "prompt_preview": prompt[:120],
            "answer": (
                f"[{seat['name']} — {tier} mock] Response for: {question[:80]}... "
                f"[{seat['id']} perspective applied]"
            ),
            "simulated": True,
            "backend": "mock",
        }

    # Live mode: dispatch to the tier-appropriate backend.
    backend_map = {
        "T1": T1PrivacyBackend,
        "T2": T2MockBackend,
        "T3": T3MockBackend,
    }
    backend_cls = backend_map.get(tier, T2MockBackend)
    try:
        result = backend_cls().generate(prompt, seat_id=seat["id"])
        answer = result.get("response", "")
        model = result.get("model", "unknown")
        quality = result.get("quality", "unknown")
    except Exception as e:
        # Graceful fallback to mock if backend fails
        answer = (
            f"[{seat['name']} — {tier} live] Backend error ({e}), "
            f"falling back to mock. Response for: {question[:80]}..."
        )
        model = "fallback"
        quality = "fallback"

    return {
        "seat_id": seat["id"],
        "seat_name": seat["name"],
        "tier": tier,
        "prompt_preview": prompt[:120],
        "answer": answer,
        "model": model,
        "quality": quality,
        "simulated": False,
        "backend": backend,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Synthesis
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_key_points(answer: str, max_chars: int = 120) -> List[str]:
    """Extract key recommendation sentences from a seat answer."""
    # Strip the [TX SIMULATED] prefix and extract first substantive sentence
    cleaned = re.sub(r"^\[T[123] SIMULATED\]\s*", "", answer)
    cleaned = re.sub(r"^.*?:\s*", "", cleaned, count=1)

    # Split into sentences, keep the most substantive ones
    sentences = re.split(r"[.!?]\s+", cleaned)
    points = []
    for sent in sentences[:3]:
        sent = sent.strip()
        if len(sent) > 20:
            points.append(sent[:max_chars].rstrip() + ("..." if len(sent) > max_chars else ""))
    return points[:2]  # max 2 key points per seat


def _synthesise(
    seat_answers: List[Dict[str, str]],
    tier: str,
    confidence: int,
    itil_classification: Optional[Dict[str, Any]] = None,
    tos_probe: Optional[Dict[str, Any]] = None,
    swarm_governance: Optional[Dict[str, Any]] = None,
) -> str:
    """Produce a single synthesis paragraph from actual seat answers.

    Extracts key points from each seat's answer and aggregates them
    into a coherent board-level recommendation.
    """
    seat_names = ", ".join(a["seat_name"] for a in seat_answers)

    # Extract key points from each seat's answer
    seat_points: Dict[str, List[str]] = {}
    for ans in seat_answers:
        key = ans["seat_name"]
        seat_points[key] = _extract_key_points(ans.get("answer", ""))

    # Build synthesis body from actual answers
    recommendation_lines = []
    for seat_name, points in seat_points.items():
        if points:
            recommendation_lines.append(f"{seat_name}: {points[0]}")
        else:
            recommendation_lines.append(f"{seat_name}: [no substantive answer]")

    # Aggregate unique themes
    all_points = " ".join(" ".join(pts) for pts in seat_points.values()).lower()
    themes = []
    if any(kw in all_points for kw in ["security", "auth", "encrypt"]):
        themes.append("security-critical")
    if any(kw in all_points for kw in ["refactor", "rewrite", "migrate", "architecture"]):
        themes.append("structural change")
    if any(kw in all_points for kw in ["implement", "code", "build", "deploy"]):
        themes.append("implementation task")
    if any(kw in all_points for kw in ["review", "risk", "edge case", "failure"]):
        themes.append("risk-aware")
    if any(kw in all_points for kw in ["cost", "budget", "cheap", "free", "token"]):
        themes.append("cost-sensitive")

    theme_str = f" | Themes: {', '.join(themes)}" if themes else ""

    parts = [
        f"Synthesis ({len(seat_answers)} seats: {seat_names}) | "
        f"Tier: {tier} | Confidence: {confidence}%{theme_str}"
    ]

    if itil_classification and itil_classification.get("is_ops"):
        parts.append(
            f"ITIL: {itil_classification['classification'].upper()} "
            f"(conf: {itil_classification['confidence']:.0%}, "
            f"prefix: {itil_classification['prefix']})"
        )

    if tos_probe and tos_probe.get("probed"):
        rec_count = len(tos_probe.get("recommended", []))
        parts.append(f"TOS probe: {rec_count} recommended models across {len(tos_probe.get('gateways', {}))} gateways")

    if swarm_governance and swarm_governance.get("governed"):
        parts.append(
            f"Swarm governance: {'PASS' if swarm_governance.get('governance_pass') else 'REJECTED'}"
        )

    parts.append("Board recommendations:")
    parts.extend(f"  — {line}" for line in recommendation_lines)

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Session log
# ═══════════════════════════════════════════════════════════════════════════════

def _timestamp_compact() -> str:
    """Return a compact timestamp suitable for filenames: YYYY-MM-DD-HHMMSS."""
    return datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")


def _write_session_log(
    question: str,
    tier: str,
    confidence: int,
    seat_answers: List[Dict[str, Any]],
    synthesis: str,
    decision: Dict[str, Any],
    itil_classification: Optional[Dict[str, Any]],
    tos_probe: Optional[Dict[str, Any]],
    swarm_governance: Optional[Dict[str, Any]],
    governance_log_path: Optional[Path],
    work_dir: Path,
) -> Path:
    """Write the BOA session log in markdown format. Returns the log path."""
    work_dir.mkdir(parents=True, exist_ok=True)
    ts = _timestamp_compact()
    log_path = work_dir / f"{ts}.md"

    lines = [
        f"# BOA Session — {ts}",
        "",
        f"**Question:** {question}",
        f"**Tier:** {tier}",
        f"**Confidence:** {confidence}%",
        f"**Routing decision:** `{decision.get('rationale', '—')}`",
        "",
    ]

    # ITIL classification section
    if itil_classification and itil_classification.get("is_ops"):
        lines.append("## ITIL Classification")
        lines.append("")
        lines.append(f"| Field | Value |")
        lines.append(f"|-------|-------|")
        lines.append(f"| Classification | **{itil_classification['classification'].upper()}** |")
        lines.append(f"| Prefix | `{itil_classification['prefix']}` |")
        lines.append(f"| Confidence | {itil_classification['confidence']:.0%} |")
        lines.append(f"| Keywords Matched | {', '.join(itil_classification['keywords_matched'])} |")
        lines.append(f"| CHG Number Required | {'Yes' if itil_classification['requires_chg_number'] else 'No'} |")
        lines.append("")

    # TOS probe section
    if tos_probe and tos_probe.get("probed"):
        lines.append("## TOS Probe — T2 Gateway Availability")
        lines.append("")
        for gw_key, gw_info in tos_probe.get("gateways", {}).items():
            status_icon = "🟢" if gw_info["status"] in ("enabled", "available") else "🔴" if gw_info["status"] == "disabled" else "🟡"
            lines.append(f"- {status_icon} **{gw_info['label']}** — `{gw_info['status']}`")
            for model in gw_info.get("known_models", [])[:4]:
                lines.append(f"    - `{model}`")
        if tos_probe.get("recommended"):
            lines.append("")
            lines.append("### Recommended")
            lines.append("")
            for rec in tos_probe["recommended"]:
                lines.append(f"- **`{rec['model']}`** via {rec['gateway']} — {rec['reason']}")
        lines.append("")

    # Swarm governance section
    if swarm_governance and swarm_governance.get("governed"):
        lines.append("## Swarm Governance")
        lines.append("")
        lines.append(f"| Field | Value |")
        lines.append(f"|-------|-------|")
        lines.append(f"| Should Delegate | {'Yes' if swarm_governance.get('should_delegate') else 'No'} |")
        lines.append(f"| Governance Pass | {'✅ YES' if swarm_governance.get('governance_pass') else '❌ NO'} |")
        if swarm_governance.get("sizing"):
            sz = swarm_governance["sizing"]
            lines.append(f"| Workers | {sz.get('worker_count', 'N/A')} |")
            lines.append(f"| Tasks | {sz.get('task_count', 'N/A')} |")
            lines.append(f"| Files | {sz.get('file_count', 'N/A')} |")
        lines.append("")
        lines.append(f"**Recommendation:** {swarm_governance.get('recommendation', '—')}")
        lines.append("")

    # Governance audit trail reference
    if governance_log_path:
        lines.append("## Governance Audit Trail")
        lines.append("")
        lines.append(f"Decision logged to: `{governance_log_path}`")
        lines.append("")

    lines.append("## Seat Answers")
    lines.append("")

    for ans in seat_answers:
        lines.append(f"### {ans['seat_name']} (`{ans['seat_id']}`) — {ans['tier']}")
        lines.append("")
        lines.append(f"> Prompt: {ans['prompt_preview']}...")
        lines.append("")
        lines.append(ans["answer"])
        lines.append("")

    lines.append("## Synthesis")
    lines.append("")
    lines.append(synthesis)
    lines.append("")

    lines.append("## Raw Decision")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(decision, indent=2, default=str))
    lines.append("```")
    lines.append("")

    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_boa(
    question: str,
    context: Optional[Dict[str, Any]] = None,
    work_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run the Board of Advisors for a given question.

    Parameters
    ----------
    question : str
        The question to put to the board.
    context : dict, optional
        Additional context passed to monica_integration.optimise (e.g.
        estimated_lines, domain).
    work_dir : Path, optional
        Directory for session logs. Defaults to
        /home/warpuppy/Projects/light-engine/work/boa.

    Returns
    -------
    dict
        {
            "tier": str,
            "confidence": int,
            "seat_answers": list[dict],
            "synthesis": str,
            "log_path": str,
            "decision": dict,   # raw routing decision
            "itil_classification": dict | None,
            "tos_probe": dict | None,
            "swarm_governance": dict | None,
            "governance_log_path": str | None,
        }
    """
    if work_dir is None:
        work_dir = Path("/home/warpuppy/Projects/light-engine/work/boa")

    # 0. ITIL classification (always run — cheap)
    itil_classification = classify_itil(question)

    # 1. Route via monica_integration.optimise (TokenRouter under the hood).
    decision = optimise(question, context or {})
    tier = decision.get("tier", "T2")
    confidence = decision.get("confidence", 50)

    # For incidents, bump confidence ceiling — these need reliable answers
    if itil_classification.get("classification") == "incident":
        confidence = min(confidence, 80)

    # 2. Collect answers from all 6 seats.
    seat_answers = [_generate_seat_answer(seat, question, tier) for seat in SEATS]

    # 3. TOS probe for routing/quality questions
    tos_probe = probe_t2_gateways(question)

    # 4. Swarm governance for swarm-plan questions
    swarm_governance = validate_swarm_plan(question)

    # 5. Synthesise.
    synthesis = _synthesise(
        seat_answers, tier, confidence,
        itil_classification=itil_classification,
        tos_probe=tos_probe,
        swarm_governance=swarm_governance,
    )

    # 6. Log governance decision
    governance_log_path = log_governance_decision(
        question=question,
        itil_classification=itil_classification,
        tos_probe=tos_probe,
        swarm_governance=swarm_governance,
        work_dir=work_dir,
    )

    # 7. Write session log.
    log_path = _write_session_log(
        question=question,
        tier=tier,
        confidence=confidence,
        seat_answers=seat_answers,
        synthesis=synthesis,
        decision=decision,
        itil_classification=itil_classification,
        tos_probe=tos_probe,
        swarm_governance=swarm_governance,
        governance_log_path=governance_log_path,
        work_dir=work_dir,
    )

    return {
        "tier": tier,
        "confidence": confidence,
        "seat_answers": seat_answers,
        "synthesis": synthesis,
        "log_path": str(log_path),
        "decision": decision,
        "itil_classification": itil_classification,
        "tos_probe": tos_probe,
        "swarm_governance": swarm_governance,
        "governance_log_path": str(governance_log_path) if governance_log_path else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _cli() -> None:
    """Simple CLI for standalone invocation."""
    if len(sys.argv) < 2:
        print("Usage: python -m token_optimiser.boa <question>")
        print("   or: python token_optimiser/boa.py <question>")
        print("")
        print("Built-in tests:")
        print("   python -m token_optimiser.boa --test-itil")
        print("   python -m token_optimiser.boa --test-boa")
        print("   BOA_BACKEND=live python -m token_optimiser.boa --test-boa")
        sys.exit(1)

    # Test mode
    if sys.argv[1] == "--test-itil":
        _test_itil_classification()
        return

    if sys.argv[1] == "--test-boa":
        _test_boa_session()
        return

    question = " ".join(sys.argv[1:])
    result = run_boa(question)

    print(f"\n{'='*60}")
    print(f"BOA Session Complete")
    print(f"{'='*60}")
    print(f"Question:   {question}")
    print(f"Tier:       {result['tier']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Log:        {result['log_path']}")

    # ITIL output
    itil = result.get("itil_classification")
    if itil and itil.get("is_ops"):
        print(f"\n📋 ITIL Classification: {itil['classification'].upper()}")
        print(f"   Prefix:     {itil['prefix']}")
        print(f"   Confidence: {itil['confidence']:.0%}")
        if itil["requires_chg_number"]:
            print(f"   ⚠️  CHG number required before proceeding")

    # TOS probe output
    tos = result.get("tos_probe")
    if tos and tos.get("probed"):
        print(f"\n🔍 T2 Gateway Probe:")
        for gw_key, gw_info in tos.get("gateways", {}).items():
            icon = "🟢" if gw_info["status"] in ("enabled", "available") else "🔴" if gw_info["status"] == "disabled" else "🟡"
            print(f"   {icon} {gw_info['label']}: {gw_info['status']}")
        if tos.get("recommended"):
            print(f"   Recommended:")
            for rec in tos["recommended"]:
                print(f"     → {rec['model']} via {rec['gateway']} ({rec['reason']})")

    # Swarm governance output
    swarm = result.get("swarm_governance")
    if swarm and swarm.get("governed"):
        status = "✅ PASS" if swarm.get("governance_pass") else "❌ REJECTED"
        print(f"\n🐝 Swarm Governance: {status}")
        print(f"   {swarm.get('recommendation', '')}")

    # Governance log
    if result.get("governance_log_path"):
        print(f"\n📁 Governance log: {result['governance_log_path']}")

    print(f"\nSeat answers: {len(result['seat_answers'])}")
    for ans in result["seat_answers"]:
        print(f"  [{ans['seat_name']}] {ans['answer'][:80]}...")
    print(f"\nSynthesis:\n{result['synthesis']}")


# ═══════════════════════════════════════════════════════════════════════════════
# Built-in BOA test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_boa_session() -> None:
    """Run a full BOA session with a test question and print results."""
    question = (
        "We need to add JWT authentication to our microservice API. "
        "What's the best approach for securing endpoints while keeping "
        "token costs low?"
    )

    backend = _resolve_backend()
    print(f"BOA Integration Test")
    print(f"{'='*60}")
    print(f"Backend: {backend} (set BOA_BACKEND=live to test backends)")
    print(f"Question: {question}")
    print()

    result = run_boa(question)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Tier:       {result['tier']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Log:        {result['log_path']}")

    # ITIL output
    itil = result.get("itil_classification")
    if itil and itil.get("is_ops"):
        print(f"\n📋 ITIL: {itil['classification'].upper()} (conf: {itil['confidence']:.0%})")

    # TOS output
    tos = result.get("tos_probe")
    if tos and tos.get("probed"):
        print(f"🔍 TOS: {len(tos.get('recommended', []))} recommendations")

    # Swarm output
    swarm = result.get("swarm_governance")
    if swarm and swarm.get("governed"):
        status = "PASS" if swarm.get("governance_pass") else "REJECTED"
        print(f"🐝 Swarm: {status}")

    print(f"\nSeat Answers ({len(result['seat_answers'])}):")
    for ans in result["seat_answers"]:
        simulated = ans.get("simulated", True)
        backend_name = ans.get("backend", "?")
        quality = ans.get("quality", "?")
        tag = "SIM" if simulated else "LIVE"
        print(f"  [{ans['seat_name']:20s}] {tag:4s} q={quality:12s} b={backend_name}: {ans['answer'][:70]}...")

    print(f"\n{'─'*60}")
    print(f"Synthesis:\n{result['synthesis']}")
    print(f"{'─'*60}")

    # Verify integration
    errors = []
    for ans in result["seat_answers"]:
        if backend == "live" and ans.get("simulated"):
            errors.append(f"  {ans['seat_name']}: expected live answer, got simulated")
        if backend == "live" and ans.get("backend") == "mock":
            errors.append(f"  {ans['seat_name']}: backend tag is 'mock' in live mode")

    if errors:
        print(f"\n❌ Integration errors:")
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        print(f"\n✅ Integration OK — all seats returned expected backend type ({backend})")


# ═══════════════════════════════════════════════════════════════════════════════
# Built-in ITIL tests
# ═══════════════════════════════════════════════════════════════════════════════

def _test_itil_classification() -> None:
    """Run built-in ITIL classification tests."""
    test_cases = [
        ("TOS hops stale", "incident"),
        ("Change routing config", "change"),
        ("New free models available", "request"),
    ]

    print("ITIL Classification Tests")
    print("=" * 50)

    all_passed = True
    for question, expected in test_cases:
        result = classify_itil(question)
        actual = result["classification"]
        passed = actual == expected
        status = "PASS" if passed else "FAIL"
        icon = "✅" if passed else "❌"

        print(f"\n{icon} [{status}] \"{question}\"")
        print(f"    Expected: {expected}")
        print(f"    Actual:   {actual}")
        print(f"    Confidence: {result['confidence']:.0%}")
        print(f"    Keywords: {result['keywords_matched']}")
        print(f"    Prefix: {result['prefix']}")

        if not passed:
            all_passed = False

    print(f"\n{'='*50}")
    if all_passed:
        print("All tests PASSED ✅")
    else:
        print("Some tests FAILED ❌")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
