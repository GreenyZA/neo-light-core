#!/usr/bin/env python3
"""
BOA Runner — Board of Advisors Query Orchestrator with ITIL + Governance.

Accepts a question, classifies it (ops → ITIL), routes through TOS,
collects answers from 6 board seats, synthesises, writes session log,
and logs governance decisions to work/boa/governance/.

Importable but optional — not required by the token_optimiser package.

Usage:
    from token_optimiser.boa import run_boa, classify_itil, decide_governance
    result = run_boa("What's the best way to handle auth in a microservice?")
    gov = decide_governance("add JWT to microservice API")  # fast, for router

Or CLI:
    python -m token_optimiser.boa "your question here"
    python token_optimiser/boa.py "your question here"
    # CLI always uses full board (6 seats + synthesis); programmatic uses decide_governance for speed
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

from token_optimiser.mock_backends import T1MockBackend, T1PrivacyBackend, T2MockBackend, T3MockBackend
from token_optimiser.router import classify_itil

# monica_integration imported lazily inside run_boa to avoid circular imports
# when monica_integration imports lightweight BOA classifiers for governance.

def _ask_prompt(seat: Dict[str, str], question: str, domain: str) -> str:
    """Build a seat prompt that cannot be hijacked by the persona framing.

    WHY THIS EXISTS — a measured failure, not a guess. Six consecutive BOA runs
    on a drone-physics question returned essays about JWT authentication,
    service meshes and the CAP theorem. The control run stated plainly "You are
    six expert advisers on DRONE PROPULSION AND BATTERY PHYSICS, not software
    architects... Do not discuss authentication, APIs, microservices, or
    security", routed to T3 Grok, and still answered with Fowler (2002) on
    microservices. The question arrived intact — every answer quoted it back
    verbatim — and was then answered by something else.

    Diagnosis: the old prompt was `framing + question`, i.e. the persona
    instruction came FIRST and dominated. A small-to-mid model completes a
    system-design framing with a system-design essay regardless of what follows.
    Strength of model does not help; T3 failed identically to T2.

    Fixes, each targeting that mechanism:
      1. The QUESTION comes first, so it is the highest-salience token span.
      2. The domain is stated explicitly and derived from the question.
      3. The persona is scoped to a LENS on that domain, never a replacement
         topic ("as the Architect, examine X" not "you are an architect").
      4. An explicit answer contract with the required first token, so a
         topical answer is structurally cheaper than a drift.
      5. An explicit off-topic prohibition naming the observed drift, since
         stating the ban alone proved insufficient.
    """
    return (
        f"QUESTION (answer this, and only this):\n{question}\n\n"
        f"DOMAIN: {domain}. Do not answer as a generalist; answer as a specialist "
        f"in {domain}.\n\n"
        f"YOUR ROLE: you are the {seat['name']} on an advisory board. "
        f"{seat['focus']}\n"
        f"Apply that role AS A LENS ON THE QUESTION ABOVE. The role shapes HOW you "
        f"answer; it does not change WHAT the question is about.\n\n"
        f"HARD RULES:\n"
        f"- Your FIRST sentence must engage the specific question above. If you "
        f"cannot engage it, say so and state what you would need.\n"
        f"- Do NOT substitute a different topic. In particular do not discuss "
        f"authentication, APIs, microservices, service meshes, or security "
        f"unless the question above is about them.\n"
        f"- Quote or reference the concrete figures, quantities or file names in "
        f"the question. Generic advice that would apply to any question is a "
        f"failed answer.\n"
        f"- Be concise and specific. No preamble.\n"
    )


def _infer_domain(question: str) -> str:
    """Best-effort domain label for the prompt, from the question's own words.

    Deliberately keyword-based and legible. A clever classifier here would be
    its own unreviewed system, and its failure would be invisible.
    """
    q = question.lower()
    probes = (
        (("prop", "rotor", "thrust", "rpm", "hover", "aerodynamic", "ct", "fom",
          "figure of merit", "disc loading", "airfoil", "pitch", "quadrotor",
          "multirotor", "momentum theory"), "multirotor propulsion and propeller aerodynamics"),
        (("battery", "lipo", "li-ion", "peukert", "c-rate", "discharge", "cell",
          "pack", "mah", "wh/kg", "energy density"), "battery and energy physics"),
        (("flight time", "endurance", "payload", "sweet spot", "prop size"), "multirotor performance and flight-time prediction"),
        (("token", "tier", "router", "routing", "optimiser", "t1", "t2", "t3",
          "dispatch", "model selection"), "LLM routing and tier selection"),
        (("itil", "incident", "change record", "service desk", "cab"), "ITIL service management practice"),
        (("auth", "oauth", "token", "credential", "jwt"), "authentication and credential handling"),
    )
    for keys, label in probes:
        if any(k in q for k in keys):
            return label
    return "the specific technical matter described in the question"


def _ask_prompt_legacy(seat: Dict[str, str], question: str) -> str:
    """The old framing-first prompt. Kept for A/B comparison and regression tests."""
    return seat["framing"] + question


SEATS: List[Dict[str, str]] = [
    {
        "id": "architect",
        "name": "Architect",
        # `focus` is what the persona is a LENS on. It must describe HOW to think,
        # never WHAT the topic is — the old `framing` did the latter, and that is
        # what let the persona replace the question's subject entirely.
        "focus": (
            "Examine structural coherence, design trade-offs, and what will break "
            "at scale. Ask what the pieces depend on and what should not be coupled."
        ),
        "framing": (
            "You are the Architect on the Board of Advisors. Focus on system design, "
            "trade-offs, long-term maintainability, and structural coherence. "
            "Question: "
        ),
    },
    {
        "id": "engineer",
        "name": "Engineer",
        "focus": (
            "Examine implementation practicality: what it takes to build, run, test "
            "and maintain. Name concrete steps and call out anything that will not "
            "work as written."
        ),
        "framing": (
            "You are the Engineer on the Board of Advisors. Focus on implementation "
            "details, practicality, performance, and correctness. "
            "Question: "
        ),
    },
    {
        "id": "researcher",
        "name": "Researcher",
        "focus": (
            "Examine evidence, precedents and measured data. Cite specific sources, "
            "figures or standards where you can, and say plainly when evidence is "
            "absent rather than filling the gap."
        ),
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
        "focus": (
            "Interrogate the proposal for risk: blind spots, edge cases, failure modes "
            "and things that were assumed without checking. Be constructively "
            "sceptical and specific."
        ),
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
        "focus": (
            "Judge cost, routing and efficiency. Which tier (T1 local / T2 free / "
            "T3 paid) fits the work, and what is the cheapest viable path?"
        ),
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
        "focus": (
            "Weigh the seats' input and return one recommendation: what to do, in "
            "what order, with what priority, and what must not be started."
        ),
        "framing": (
            "You are the Executive Chair on the Board of Advisors. Synthesise the "
            "board's views into a single recommendation. Decide what to do, in what "
            "order, with what priority. "
            "Question: "
        ),
    },
]

# ITIL classification now lives in router.py (classify_itil) — router assess() always runs it.
# Governance from BOA/ITIL drives tier + swarm_recommendation decisions.
# (imported at module top)

def _on_topic(answer: str, question: str) -> Dict[str, Any]:
    """Cheap drift check: did the seat engage the question it was given?

    Not a judgement of answer quality — only of subject matter. This exists
    because the persona-hijack failure was completely invisible in the output:
    six seats "answered", the session reported success, and four of the answers
    were about a different subject entirely.

    Deliberately simple and legible. It flags for a human; it does not retry,
    and it must never silently rewrite an answer.
    """
    if not answer:
        return {"checked": False, "verdict": "no-answer"}

    a = answer.lower()
    q = question.lower()

    # Terms the question actually uses; overlap implies engagement.
    stop = {
        "the", "a", "an", "of", "and", "or", "to", "in", "is", "it", "for", "on",
        "should", "we", "our", "what", "how", "does", "do", "can", "with", "that",
        "this", "be", "are", "was", "were", "by", "as", "at", "from",
    }

    def _terms(text: str):
        out = set()
        for w in text.replace("/", " ").replace("-", " ").replace(",", " ").split():
            w = w.strip("?.!:;()\"'")
            if len(w) > 3 and w not in stop and not w.isdigit():
                out.add(w)
        return out

    q_terms = _terms(q)
    a_terms = _terms(a)
    if not q_terms:
        return {"checked": False, "verdict": "no-comparable-terms"}

    overlap = len(q_terms & a_terms)
    ratio = overlap / len(q_terms)

    # Known drift signatures observed in six consecutive runs.
    drift_terms = ("microservice", "service mesh", "event bus", "kafka", "nats",
                   "jwt", "oauth", "fido2", "owasp", "cap theorem", "token theft",
                   "clock skew", "dora metric", "bounded context")
    drifted = [t for t in drift_terms if t in a]

    verdict = "on-topic"
    if drifted and ratio < 0.15:
        verdict = "DRIFT"
    elif ratio < 0.08:
        verdict = "suspect-low-overlap"

    return {
        "checked": True,
        "verdict": verdict,
        "overlap": overlap,
        "of_terms": len(q_terms),
        "ratio": round(ratio, 3),
        "drift_terms": drifted,
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
# Programmatic Governance API (for router / auto routing + swarming)
# Lightweight, no full 6-seat synthesis. CLI/run_boa still does full board Q&A.
# ═══════════════════════════════════════════════════════════════════════════════

# Reentry guard (module-level) to prevent infinite recursion when:
#   router.assess -> boa.decide_governance -> optimise/monica -> router.assess
# The guard short-circuits the inner decide call with safe defaults (T2, size=1).
_BOA_REENTRY = {"active": False}


def decide_governance(task: str, context: Optional[Dict[str, Any]] = None, fast: bool = True) -> Dict[str, Any]:
    """
    Programmatic governance decision for routing and swarming.

    Clean internal API the router/integration can call without full verbose
    synthesis or 6-seat execution (optimized for speed on automatic calls).

    - Always runs cheap classify_itil + probe_t2 + validate_swarm_plan.
    - Gets tier via (guarded) optimise.
    - When called from router (context['from_router']=True), influences free-tier
      preference (prefer T2 free cloud over T1 unless privacy/security).
    - Keeps governance logging (via log_governance_decision).
    - Supports BOA_BACKEND=live (noted, but seat exec skipped in fast path).
    - Returns exactly: tier, swarm_size, itil, rationale, seats_used.

    Use fast=True (default) for auto/router calls → lightweight path.
    For full board use run_boa() or CLI.
    """
    context = context or {}
    from_router = bool(context.get("from_router") or context.get("lightweight"))

    if _BOA_REENTRY["active"]:
        # avoid loop: return safe lightweight defaults without further calls
        return {
            "tier": "T2",
            "swarm_size": 1,
            "itil": {
                "is_ops": False,
                "classification": "unknown",
                "confidence": 0.0,
                "prefix": "",
                "keywords_matched": [],
                "requires_chg_number": False,
            },
            "rationale": "reentry guard (avoided router<->boa loop) — default T2 lightweight",
            "seats_used": 1,
        }

    _BOA_REENTRY["active"] = True
    try:
        # ITIL (cheap, always)
        itil = classify_itil(task)

        # TOS probe (cheap)
        tos_probe = probe_t2_gateways(task)

        # Swarm plan validation (cheap)
        swarm_gov = validate_swarm_plan(task)

        # Tier via lazy monica (already guarded at run_boa level too)
        try:
            from token_optimiser.monica_integration import optimise
            decision = optimise(task, context or {})
            tier = decision.get("tier", "T2")
            base_rationale = decision.get("rationale", "")
        except Exception:
            tier = "T2"
            base_rationale = "fallback (optimise unavailable in decide_governance)"

        # Influence free-tier preference when called from router
        if from_router:
            task_l = task.lower()
            is_privacy = any(k in task_l for k in ["privacy", "secret", "local-only", "offline", "sensitive"])
            is_security = any(k in task_l for k in ["security", "auth", "encrypt", "vulnerab", "credential", "key"])
            if tier == "T1" and not is_privacy:
                tier = "T2"
                base_rationale += " | BOA governance: free-tier preference (T2 over T1 for non-private)"
            elif is_security:
                tier = "T3"
                base_rationale += " | BOA governance: security forces T3"
            # note: T2 is already default free preference in router/config

        # Decide swarm_size from governance
        if swarm_gov.get("governed") and swarm_gov.get("should_delegate", False):
            sc = swarm_gov.get("sizing", {}).get("worker_count")
            swarm_size = int(sc) if sc and sc >= 2 else 3
        else:
            swarm_size = 1

        # seats_used: lightweight=1 for auto speed; 6 when full board needed
        # (decide itself never executes seats; it reports what should be used)
        seats_used = 6 if (not fast and not from_router) else 1

        # Build rationale
        rationale = base_rationale
        if itil.get("is_ops"):
            rationale += f" | ITIL:{itil['classification']}@{itil['confidence']}"
        if swarm_gov.get("governed"):
            rationale += f" | swarm_size={swarm_size} (delegate={swarm_gov.get('should_delegate')})"
        if tos_probe.get("probed"):
            rationale += " | TOS-probed"

        # Keep the governance logging (even for fast programmatic calls)
        try:
            # Resolve relative to repo root (supports any clone dir name e.g. neo-light)
            # CHG-2026-032: was parents[2] (= ~/Projects), which wrote session
            # logs and the governance trail into a SHARED parent directory
            # alongside every other project. parents[1] is this repo root.
            _repo_root = Path(__file__).resolve().parents[1]
            work_dir = Path(context.get("work_dir", str(_repo_root / "work/boa")))
            log_governance_decision(
                question=task,
                itil_classification=itil,
                tos_probe=tos_probe,
                swarm_governance=swarm_gov,
                work_dir=work_dir,
            )
        except Exception:
            # logging failure must never break decision path
            pass

        return {
            "tier": tier,
            "swarm_size": swarm_size,
            "itil": itil,
            "rationale": rationale,
            "seats_used": seats_used,
        }
    finally:
        _BOA_REENTRY["active"] = False


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
    domain = _infer_domain(question)
    prompt = _ask_prompt(seat, question, domain)

    if backend == "mock":
        return {
            "seat_id": seat["id"],
            "seat_name": seat["name"],
            "tier": tier,
            "domain": domain,
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
    degraded = False
    try:
        result = backend_cls().generate(prompt, seat_id=seat["id"])
        answer = result.get("response", "")
        model = result.get("model", "unknown")
        quality = result.get("quality", "unknown")
        # A backend that returns no text has failed, even without raising. The
        # old code treated that as success and the seat then contributed an
        # empty vote to the synthesis without anyone noticing.
        if not isinstance(answer, str) or not answer.strip():
            degraded = True
            answer = ""
    except Exception as e:
        degraded = True
        answer = ""
        model = "error"
        quality = "unavailable"

    if degraded:
        # Fall back to a legacy-framing retry before giving up. The new prompt
        # is the fix for persona hijack, but a seat that cannot reach its
        # backend at all is a different failure and deserves one more attempt.
        try:
            result = backend_cls().generate(
                seat["framing"] + question, seat_id=seat["id"]
            )
            answer = result.get("response", "")
            model = result.get("model", "unknown")
            if isinstance(answer, str) and answer.strip():
                degraded = False
                quality = result.get("quality", "unknown")
        except Exception:
            pass

    if degraded:
        # Record the failure as data rather than fabricating an answer. A
        # placeholder that looks like a seat response corrupts the synthesis:
        # six "seats" answered, four were drift, two were placeholders, and
        # nothing in the output said so.
        answer = (
            f"[{seat['name']} — SEAT UNAVAILABLE] Backend did not return an answer "
            f"(tier {tier}). This seat contributed NO input to the synthesis. Do "
            f"not read this as agreement or dissent."
        )
        model = "unavailable"
        quality = "unavailable"

    return {
        "seat_id": seat["id"],
        "seat_name": seat["name"],
        "tier": tier,
        "domain": domain,
        "on_topic_check": _on_topic(answer, question),
        "prompt_preview": prompt[:120],
        "answer": answer,
        "model": model,
        "quality": quality,
        "degraded": degraded,
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


def _governance_statement(
    *,
    question: str,
    tier: str,
    confidence: int,
    itil_classification: Dict[str, Any],
    tos_probe: Dict[str, Any],
    swarm_governance: Dict[str, Any],
    decision: Dict[str, Any],
) -> str:
    """Option B (CHG-2026-032): state what was COMPUTED, never what was ADVISED.

    Replaces `_synthesise`, which built a paragraph from six seat answers. Those
    answers came from `mock_backends.py` — hardcoded strings with no network
    call behind them — so the old synthesis was a confident essay assembled from
    fixtures. Its failure mode was invisible: correct shape, plausible prose,
    exit 0.

    This states only things that were actually computed, and says explicitly that
    no advisory board ran. It is deliberately not styled like a recommendation.
    """
    itil = itil_classification.get("classification", "unknown")
    prefix = f"{itil_classification.get('prefix', '')}{'' if itil_classification.get('prefix') else ''}"
    chg = "a CHG number is required" if itil_classification.get("requires_chg_number") else "no CHG number required"

    parts: List[str] = [
        f"GOVERNANCE DECISION (no advisory board ran; there are no advisory seats). "
        f"Route this to tier {tier} at {confidence}% confidence. "
        f"ITIL classification: {itil} (confidence "
        f"{int(100 * float(itil_classification.get('confidence', 0.0)))}%, {chg})."
    ]

    matched = decision.get("matched_categories") or []
    if matched:
        parts.append(f"Matched categories: {', '.join(str(m) for m in matched)}.")

    if tos_probe:
        recs = tos_probe.get("recommended") or []
        gws = tos_probe.get("gateways") or {}
        live = [g for g, v in gws.items() if isinstance(v, dict) and v.get("has_token")]
        parts.append(
            f"TOS probe: {len(recs)} recommended model(s); "
            f"{len(live)} of {len(gws)} gateway/gateways report a credential."
        )

    sg = swarm_governance or {}
    if sg.get("governed"):
        rec = sg.get("recommendation") or {}
        parts.append(
            f"Swarm governance: governed — recommended {rec.get('recommendation', 'unspecified')} "
            f"at score {rec.get('score', '?')}."
        )
    elif sg:
        parts.append("Swarm governance: not engaged by this question.")

    rationale = decision.get("rationale")
    if rationale:
        parts.append(f"Router reasoning: {rationale}")

    parts.append(
        "This is a routing and classification result computed by the token "
        "optimiser. It is NOT advice, opinion, or review from any model or "
        "persona. The six advisory perspectives survive as a checklist in the "
        "session log for the human to work through."
    )
    return " ".join(parts)


def _governance_statement_placeholder() -> None:  # pragma: no cover
    return None


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

    # Option B (CHG-2026-032): the six perspectives survive as a CHECKLIST for
    # the human to work through. They are NOT answers received from a model, and
    # the heading must make that unmistakable — the original failure was
    # fabricated content that looked like advice received.
    lines.append("## Advisory Checklist — perspectives to consider (NOT advice received)")
    lines.append("")
    lines.append(
        "No advisory seats exist. The board previously reported six seat answers; "
        "they were hardcoded strings from `mock_backends.py` with no model call "
        "behind them (CHG-2026-032). The six perspectives are retained as "
        "structure for the human decision-maker."
    )
    lines.append("")
    for seat in SEATS:
        lines.append(f"- [ ] **{seat['name']}** — {seat['focus']}")
    lines.append("")

    lines.append("## Governance Decision")
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
        Directory for session logs. Defaults to <repo>/work/boa (resolved relative to package for any clone name e.g. neo-light).

    Returns
    -------
    dict
        {
            "tier": str,
            "confidence": int,
            "seat_answers": list[dict],   # always [] since Option B
            "advisory_seats": int,       # always 0 since Option B
            "synthesis": str,            # governance statement, not a board opinion
            "log_path": str,
            "decision": dict,   # raw routing decision
            "itil_classification": dict | None,
            "tos_probe": dict | None,
            "swarm_governance": dict | None,
            "governance_log_path": str | None,
        }
    """
    if work_dir is None:
        # Resolve relative to repo root (supports any clone dir name e.g. neo-light)
        # CHG-2026-032: repo root, not the shared ~/Projects parent.
        _repo_root = Path(__file__).resolve().parents[1]
        work_dir = Path(context.get("work_dir", str(_repo_root / "work/boa"))) if context else (_repo_root / "work/boa")

    # Lazy import to avoid circular dependency with monica_integration
    from token_optimiser.monica_integration import optimise

    # 0. ITIL classification (always run — cheap)
    itil_classification = classify_itil(question)

    # 1. Route via monica_integration.optimise (TokenRouter under the hood).
    decision = optimise(question, context or {})
    tier = decision.get("tier", "T2")
    confidence = decision.get("confidence", 50)

    # For incidents, bump confidence ceiling — these need reliable answers
    if itil_classification.get("classification") == "incident":
        confidence = min(confidence, 80)

    # 2. Option B (CHG-2026-032): NO seat answers are generated.
    #
    # Verified before this change: mock_backends.py held hardcoded prose and
    # BOA_BACKEND=live selected a *different* set of canned strings, not a live
    # endpoint. Seven runs produced zero real intelligence while reporting
    # "Seat answers: 6" and exiting 0. The keys stay for caller compatibility
    # (see test_boa_does_not_fabricate_seat_answers) but are always empty.
    seat_answers: List[Dict[str, Any]] = []

    # 3. TOS probe for routing/quality questions
    tos_probe = probe_t2_gateways(question)

    # 4. Swarm governance for swarm-plan questions
    swarm_governance = validate_swarm_plan(question)

    # 5. Governance statement — an honest account of what was computed, NOT a
    #    board opinion. It must never read as advice received from a model.
    synthesis = _governance_statement(
        question=question,
        tier=tier,
        confidence=confidence,
        itil_classification=itil_classification,
        tos_probe=tos_probe,
        swarm_governance=swarm_governance,
        decision=decision,
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
        # Keys retained for caller compatibility; always empty. Option B.
        "seat_answers": seat_answers,
        "advisory_seats": len(seat_answers),
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

    # Option B (CHG-2026-032): no seat answers exist and none are generated.
    print("\nAdvisory seats: 0 (none configured — see CHG-2026-032)")
    print(f"\nGovernance decision:\n{result['synthesis']}")


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

    # Option B (CHG-2026-032): no seat answers exist or are generated.
    print("\nAdvisory seats: 0 (none configured — CHG-2026-032)")
    if False:
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
    # Option B: the old check here asserted seats returned real (non-mock)
    # content. That check passed on hardcoded strings, so it was a green light
    # on fabrication. It is removed rather than rewritten: with no seats there
    # is nothing to validate, and validate_swarm_plan / the router's own
    # thresholds are covered by the real test suite instead.

    if errors:
        print(f"\n❌ Integration errors:")
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        print("✅ Integration OK — governance layer returned a decision (no seats involved)")


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
