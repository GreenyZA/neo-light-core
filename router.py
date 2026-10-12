#!/usr/bin/env python3
"""
Token Optimiser — Confidence assessment and tier routing engine.

UPDATED 2026-09-19:
- Free tier priority mode: T2 free cloud tried BEFORE T1 when quality is better
- NVIDIA NIM, DeepSeek V4 Flash, Poolside Laguna integrated
- T1 (local 3B) now used for privacy/offline only, not as default cheap path
- Category → preferred free model mapping for intelligent routing

UPDATED 2026-09-19 (v2):
- Parallelization score and delegation routing hint
- Parallelizable keyword detection, file-path counting, word-count scaling
"""

import re
import json
import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

# ── Tier definitions ──────────────────────────────────────────────────────────

@dataclass
class Tier:
    name: str
    label: str
    backend: str
    cost_hint: str
    confidence_min: int
    confidence_max: int

# DEFECT (CHG-2026-034): T1's backend was declared "ollama" while the URL was
    # a llama-server endpoint on :8081. Nothing imports ollama_control for T1,
    # so the label pointed at a service that is not installed or running
    # (`systemctl --user is-active ollama` -> inactive). Any code matching on
    # backend == "ollama" would therefore never reach the real local model, which
    # is why T1 never routed correctly.
TIERS = [
    Tier("T1", "Local free (llama-server, Vulkan/RX480)", "llama_server",
         "$0 local, private, offline",
         confidence_min=80, confidence_max=101),
    Tier("T1b", "Local secondary (Lenovo CPU)", "local_cpu", "$0 local, private, offline, larger model",
         confidence_min=70, confidence_max=80),
    Tier("T2", "Free cloud (OpenRouter/NVIDIA/DeepSeek/Poolside)", "free_cloud", "$0 free tiers, frontier models",
         confidence_min=60, confidence_max=80),
    Tier("T3", "High-tier (Grok/Solar)", "high_tier", "paid plan, security + final judgment",
         confidence_min=0, confidence_max=60),
]


# ── Task category heuristics ──────────────────────────────────────────────────

CATEGORY_RULES = [
    {
        "name": "simple_transform",
        "description": "Simple text transforms, formatting, renaming, listing",
        "keywords": [
            "typo", "fix typo", "formatting", "format", "rename", "renaming",
            "list", "listing", "extract", "extraction", "count", "counting",
            "summarize", "summary", "rephrase", "rewrite", "simple",
            "capitalize", "lowercase", "uppercase", "sort", "filter",
            "trim", "clean", "cleanup", "remove duplicates",
        ],
        "base_confidence": 90,
        "scope_modifiers": {
            "small": 5,
            "medium": 0,
            "large": -10,
        },
    },
    {
        "name": "code_gen_small",
        "description": "Small code generation or modification (< 100 lines)",
        "keywords": [
            "write a function", "write function", "small script", "utility",
            "helper", "snippet", "one-liner", "add a method", "add method",
            "fix bug", "bug fix", "patch", "quick fix", "minor change",
            "add comment", "add docstring", "add type hints",
        ],
        "base_confidence": 75,
        "scope_modifiers": {
            "small": 10,
            "medium": 0,
            "large": -15,
        },
    },
    {
        "name": "code_gen_medium",
        "description": "Medium code generation (100-500 lines)",
        "keywords": [
            "write a class", "write class", "write module", "create endpoint",
            "build feature", "implement", "add endpoint", "add route",
            "generate code", "codegen", "stub", "boilerplate",
        ],
        "base_confidence": 65,
        "scope_modifiers": {
            "small": 10,
            "medium": 0,
            "large": -10,
        },
    },
    {
        "name": "code_gen_large",
        "description": "Large code generation or complex refactor (> 500 lines)",
        "keywords": [
            "refactor", "rewrite module", "migrate", "large change",
            "rewrite", "major change", "rearchitect", "redesign",
            "full rewrite", "complete rewrite",
        ],
        "base_confidence": 50,
        "scope_modifiers": {
            "small": 10,
            "medium": 0,
            "large": -10,
        },
    },
    {
        "name": "documentation",
        "description": "Documentation, explanations, writing",
        "keywords": [
            "document", "documentation", "docs", "readme", "comment",
            "explain", "explanation", "write docs", "write documentation",
            "api docs", "docstring", "guide", "how-to", "tutorial",
            "write a guide", "write guide", "write README",
        ],
        "base_confidence": 78,
        "scope_modifiers": {
            "small": 5,
            "medium": 0,
            "large": -5,
        },
    },
    {
        "name": "security",
        "description": "Security, auth, credentials, sensitive data",
        "keywords": [
            "security", "secure", "auth", "authentication", "authorization",
            "password", "credential", "token", "secret", "api key",
            "encrypt", "decrypt", "hash", "vulnerability", "exploit",
            "injection", "sql injection", "xss", "csrf", "audit",
            "penetration", "security review", "security audit",
        ],
        "base_confidence": 30,
        "scope_modifiers": {
            "small": 10,
            "medium": 0,
            "large": -5,
        },
        "cap_confidence": 55,
    },
    {
        "name": "privacy",
        "description": "Privacy, offline, air-gapped, data-residency, sensitive",
        "keywords": [
            "private", "privacy", "offline", "local-only", "sensitive",
            "gdpr", "hipaa", "phi", "pii", "personal data", "confidential",
            "internal only", "on-prem", "air-gapped", "no internet",
            "disconnected", "local only", "must stay local", "data residency",
            "sovereign", "on-device", "edge", "airgap",
        ],
        "base_confidence": 25,
        "scope_modifiers": {
            "small": 5,
            "medium": 0,
            "large": -5,
        },
        "cap_confidence": 55,
        "force_tier": "T1",
    },
    {
        "name": "architecture",
        "keywords": [
            "architecture", "design", "system design", "architect",
            "pattern", "architectural", "structure", "layout",
            "component", "module design", "api design", "interface design",
            "schema design", "database design", "data model",
        ],
        "base_confidence": 55,
        "scope_modifiers": {
            "small": 10,
            "medium": 0,
            "large": -10,
        },
    },
    {
        "name": "research",
        "description": "Research, web search, analysis, comparison",
        "keywords": [
            "research", "search", "find", "investigate", "analyze", "analysis",
            "compare", "comparison", "evaluate", "review", "survey",
            "look into", "dig into", "explore", "discover",
        ],
        "base_confidence": 62,
        "scope_modifiers": {
            "small": 8,
            "medium": 0,
            "large": -8,
        },
    },
    {
        "name": "testing",
        "description": "Test writing, test framework, test infrastructure",
        "keywords": [
            "test", "testing", "unit test", "integration test", "e2e",
            "pytest", "write tests", "add tests", "test coverage",
            "test suite", "spec", "specification test",
        ],
        "base_confidence": 70,
        "scope_modifiers": {
            "small": 5,
            "medium": 0,
            "large": -10,
        },
    },
    {
        "name": "infrastructure",
        "description": "Docker, CI/CD, deployment, Nix, build systems",
        "keywords": [
            "docker", "dockerfile", "docker-compose", "ci", "cd",
            "github actions", "github workflow", "deploy", "deployment",
            "nix", "flake", "build", "build system", "pipeline",
            "configuration", "config management", "environment",
        ],
        "base_confidence": 68,
        "scope_modifiers": {
            "small": 5,
            "medium": 0,
            "large": -10,
        },
    },
]


# ── Parallelizable keywords ──────────────────────────────────────────────────

PARALLELIZE_KEYWORDS = [
    " and ", " also ", " additionally ", " separate ", " independent ",
    " parallel ", " meanwhile ",
]

# ═══════════════════════════════════════════════════════════════════════════════
# ITIL Classification (moved from boa.py for router-driven governance)
# BOA/ITIL now drives tier and swarm decisions in assess()
# ═══════════════════════════════════════════════════════════════════════════════

_PROJECT_WORK_MARKERS = (
    "first failure wins", "failure rate", "return the", "refactor", "function",
    "test", "tests", "assert", "class ", "def ", "module", "pipeline",
    "optimise", "optimize", "the pipeline", "loop", "variable", "config file",
    "code", "commit", "function returns",
)


def _reads_as_project_work(q: str) -> bool:
    """True when the text is implementation work, not a service event.

    Deliberately a marker list rather than an inference. The ITIL master states
    "Project work is not a ticket", so the safe direction is to decline to
    classify rather than manufacture an incident out of a code comment.
    """
    low = q.lower()
    return any(m in low for m in _PROJECT_WORK_MARKERS)


def _keyword_weight(kw: str) -> float:
    """Specificity weight for an ITIL keyword hit.

    A multi-word phrase is far stronger evidence than a bare noun: "connection
    refused" describes an outage, "refused" is a word. Single words are worth
    less than 1.0 so a lone match cannot out-score a genuine phrase.
    """
    words = kw.split()
    if len(words) >= 3:
        return 1.6
    if len(words) == 2:
        return 1.3
    return 0.8


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
        # DEFECT (CHG-2026-036): the old score was `sum(len(kw.split()) for kw in
        # hits)`, so ONE bare word scored 1.0 and a single-category match reported
        # confidence 1.0 — 100%. Verified reproductions:
        #   "first failure wins - return the first failing task"  -> incident @ 100%
        #   "optimise the failure rate of the pipeline"           -> incident @ 100%
        # Both are project work, and both manufactured an INCIDENT plus a CHG
        # number. Two changes below:
        #   1. `confidence` is the winning category's score SHARED with the
        #      runners-up, so a contested word cannot reach 1.0.
        #   2. A bare single-word hit is discounted — specificity matters more
        #      than raw count, and multi-word phrases like "change request" or
        #      "connection refused" are strong evidence; "slow" alone is not.
        score = sum(_keyword_weight(kw) for kw in hits)
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
    # DEFECT (CHG-2026-036): see above. Competitor mass now suppresses
    # confidence, so one ambiguous bare noun cannot present as certainty.
    competitors = total_score - max_score
    confidence = max_score / (max_score + competitors + 1e-6) if max_score > 0 else 0.0
    confidence = min(confidence, 1.0)
    # A lone bare noun is weak evidence, so it also cannot be a HIGH confidence
    # classification on its own. Two or more weighted hits, or any multi-word
    # phrase, are required before an ops label is asserted with conviction.
    best_hits = matched_keywords.get(best_category, [])
    if max_score < 2.0 and all(len(k.split()) == 1 for k in best_hits):
        # CHG-2026-036: a lone bare noun is weak evidence. The cap is 0.4 —
        # low enough to stop it asserting an INCIDENT, high enough that a real
        # outage described in one word ("down") is still visibly more likely
        # than not. A genuine outage almost always carries a second signal
        # ("down", "outage", "restoring", an INC id) and clears the cap.
        confidence = min(confidence, 0.4)
        # A bare noun inside an engineering sentence is very likely a control
        # word, not an outage: "first failure wins", "failure rate". When the
        # text also reads as implementation work, do not classify as ops at all.
        if _reads_as_project_work(q):
            return {
                "is_ops": False,
                "classification": "unknown",
                "confidence": 0.0,
                "requires_chg_number": False,
                "prefix": "",
                "keywords_matched": [],
                "suppressed_bare_noun": matched_keywords.get(best_category, []),
            }

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

# ── Orchestrator-aware routing ────────────────────────────────────────────────
# Maps task patterns to minimum tier + recommended worker count + review needs.
# Deterministic: pattern order is fixed; first match wins.

ORCHESTRATOR_ROUTING = [
    {
        "pattern": "orchestrate",
        "min_tier": "T2",
        "recommended_workers": (2, 5),
        "needs_review": True,
        "rationale": "Orchestrator tasks need reliable decomposition — T2 minimum.",
    },
    {
        "pattern": "review",
        "min_tier": "T2",
        "recommended_workers": (1, 2),
        "needs_review": False,
        "rationale": "Reviews need critical analysis capability — T2 minimum.",
    },
    {
        "pattern": "simple_worker",
        "min_tier": "T1",
        "recommended_workers": (1, 3),
        "needs_review": False,
        "rationale": "Simple worker tasks can run on T1.",
    },
    {
        "pattern": "complex_worker",
        "min_tier": "T2",
        "recommended_workers": (2, 8),
        "needs_review": True,
        "rationale": "Complex workers need T2 or T3 with post-execution review.",
    },
    {
        "pattern": "multi_worker",
        "min_tier": "T2",
        "recommended_workers": (3, 10),
        "needs_review": True,
        "rationale": "Multi-worker tasks need orchestrator-grade tier and review.",
    },
]

_TIER_ORDER = {"T1": 1, "T1b": 1, "T2": 2, "T3": 3}


def _tier_at_least(tier: str, minimum: str) -> str:
    """Return the higher of *tier* and *minimum* (deterministic, order-based)."""
    return tier if _TIER_ORDER.get(tier, 0) >= _TIER_ORDER.get(minimum, 0) else minimum


def route_orchestrator_task(task_type: str, base_confidence: int) -> "RoutingDecision":
    """
    Route an orchestrator-level task deterministically.

    Given a *task_type* pattern and the base confidence from normal routing,
    return a RoutingDecision that enforces minimum tier, recommended workers,
    and review requirements.  Same task_type + same config ⇒ same decision.
    """
    from token_optimiser.config import config

    task_lower = task_type.lower()
    match = None
    for rule in ORCHESTRATOR_ROUTING:
        if rule["pattern"] in task_lower:
            match = rule
            break

    if match is None:
        # No orchestrator match — return a neutral decision based on confidence only
        tier = _confidence_to_tier(base_confidence)
        return RoutingDecision(
            tier=tier,
            confidence=base_confidence,
            rationale=f"No orchestrator pattern matched for '{task_type}' — using confidence-based tier.",
            task=task_type,
            orchestrator_hints={
                "recommended_workers": None,
                "needs_review": False,
                "min_tier": tier,
            },
        )

    min_tier = match["min_tier"]
    tier = _tier_at_least(_confidence_to_tier(base_confidence), min_tier)

    # If config overrides the minimum (e.g. min_review_tier > rule's min_tier), respect it
    if match["pattern"] == "review" and config.min_review_tier:
        tier = _tier_at_least(tier, config.min_review_tier)
    if match["pattern"] == "orchestrate" and config.min_orchestrator_tier:
        tier = _tier_at_least(tier, config.min_orchestrator_tier)

    workers = match["recommended_workers"]
    needs_review = match["needs_review"]

    # If enforce_review is on, reviews are always flagged
    if config.enforce_review and match["pattern"] in ("review", "complex_worker", "multi_worker", "orchestrate"):
        needs_review = True

    return RoutingDecision(
        tier=tier,
        confidence=base_confidence,
        rationale=match["rationale"],
        task=task_type,
        orchestrator_hints={
            "recommended_workers": workers,
            "needs_review": needs_review,
            "min_tier": min_tier,
        },
    )


def _confidence_to_tier(confidence: int) -> str:
    """Simple confidence → tier mapping (mirrors standard routing bands)."""
    from token_optimiser.config import config
    if confidence >= config.threshold_t1_min:
        return "T1"
    if confidence >= config.threshold_t2_min:
        return "T2"
    return "T3"


# ── Router ─────────────────────────────────────────────────────────────────────

class TokenRouter:
    """
    Main router. Evaluates a task description and returns a routing decision.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.learning_log_path = self.config.get(
            "learning_log_path", "token_optimiser/learning_log.jsonl"
        )
        if "enabled" in self.config:
            self.enabled = self.config["enabled"]
        else:
            try:
                from token_optimiser.config import config as _global
                self.enabled = _global.enabled
            except ImportError:
                self.enabled = True
        self._last_matches: list = []
        self._last_task_text: str = ""  # stored for _select_tier privacy keyword check

    def assess(self, task: str, context: Optional[dict] = None) -> "RoutingDecision":
        """
        Assess a task and return a RoutingDecision with tier, confidence, rationale,
        itil_classification (always), and swarm (with solo/pair/swarm/mega rec).
        ITIL from boa.py logic now drives decisions. Free tier (T2) maximized
        unless security/change forces T3. Non-trivial (>8 swarm score, medium+,
        ops, ambiguous, low-conf) get swarm rec.
        """
        if not self.enabled:
            itil = classify_itil(task)
            return RoutingDecision(
                tier="T2",
                confidence=50,
                rationale="Token optimiser disabled — defaulting to T2",
                task=task,
                context=context or {},
                itil_classification=itil,
                swarm={"recommendation": "solo", "non_trivial": False, "note": "disabled"},
            )

        self._last_task_text = task  # stored for _select_tier privacy keyword check

        words = len(task.split())
        lines_hint = context.get("estimated_lines", None) if context else None

        scope = self._determine_scope(words, lines_hint, task)
        matches = self._match_categories(task)
        confidence = self._compute_confidence(matches, scope, task)
        confidence = max(0, min(99, confidence))

        # Apply category-level confidence caps
        from token_optimiser.config import config
        for m in matches:
            cat_name = m["name"]
            cap = config.category_confidence_caps.get(cat_name)
            if cap is not None:
                confidence = min(confidence, cap)
                break

        # ITIL classification — always run (cheap, drives governance)
        itil_classification = classify_itil(task)

        # Select tier (now passes itil so change/security force T3, else free T2 aggressive)
        unique_categories = len(set(m["name"] for m in matches))
        tier = self._select_tier(confidence, unique_categories, matches, itil_classification=itil_classification)

        rationale = self._build_rationale(matches, scope, confidence, tier, unique_categories)

        # Compute parallelization score
        parallelize_score = self._compute_parallelize_score(task)
        parallelize = parallelize_score >= config.parallelize_threshold

        if parallelize:
            rationale += " | HIGH PARALLELIZATION: Consider using `delegate_task` to distribute across workers."

        # Compute swarm recommendation (swarm-decision dims, >8 = non-trivial)
        # Biased to T2 free usage
        swarm = self._compute_swarm_recommendation(task, scope, itil_classification, matches, confidence)

        # If non-trivial and not forced T3, aggressively use T2 for the main task too
        if swarm.get("non_trivial") and tier not in ("T3", "ASK") and config.free_tier_priority:
            # Don't override privacy-forced local routing (T1/T1b) to cloud T2
            privacy_forced = tier in ("T1", "T1b")
            if not privacy_forced and not (
                    itil_classification.get("classification") == "change" or
                    (matches and any(m.get("name") == "security" for m in matches))):
                tier = "T2"
                rationale += " | ITIL+swarm non-trivial: forced T2 (free tier max)"

        if swarm.get("non_trivial"):
            rationale += f" | SWARM: {swarm['recommendation']} (score={swarm['score']}, workers->{swarm.get('tier_for_workers')})"

        if itil_classification.get("is_ops"):
            rationale += f" | ITIL: {itil_classification['classification']} (conf={itil_classification['confidence']})"

        return RoutingDecision(
            tier=tier,
            confidence=confidence,
            rationale=rationale,
            task=task,
            scope=scope,
            matched_categories=[m["name"] for m in matches],
            parallelize=parallelize,
            parallelize_score=parallelize_score,
            itil_classification=itil_classification,
            swarm=swarm,
        )

    def _compute_parallelize_score(self, task: str) -> float:
        """
        Compute a parallelization score (0.0-1.0) for a task.

        Based on:
        - Number of distinct file paths mentioned in the task (more files → higher)
        - Presence of parallelizable keywords
        - Task word count (longer tasks → more parallelizable)
        """
        from token_optimiser.config import config

        task_lower = task.lower()

        # 1. File path score — count distinct file paths mentioned
        file_patterns = [
            r'[\w./\-_]+\.(py|js|ts|jsx|tsx|go|rs|java|rb|cpp|c|h|hpp|cs|php|swift|kt|scala|sh|bash|zsh|fish|ps1|bat|cmd|yaml|yml|json|toml|xml|html|css|scss|less|md|txt|rst|ini|cfg|env|lock|sql|db|sqlite|csv|tsv|log)',
            r'(?:^|[\s,;])([\w./\-_]{3,}(?:/[\w.\-_]+)+)',
        ]
        file_paths = set()
        for pat in file_patterns:
            for m in re.finditer(pat, task_lower):
                # Use group 1 if it exists, else group 0
                path = m.group(1) if m.lastindex else m.group(0)
                if path and len(path) > 2:
                    file_paths.add(path)

        # Normalize: 0 files → 0.0, 1 → 0.2, 2 → 0.4, 3+ → 0.6
        file_score = min(len(file_paths) * 0.2, 0.6)

        # 2. Keyword score — parallelizable keywords present
        keyword_hits = sum(1 for kw in PARALLELIZE_KEYWORDS if kw in task_lower)
        # Normalize: 0 → 0.0, 1 → 0.15, 2 → 0.3, 3+ → 0.45
        keyword_score = min(keyword_hits * 0.15, 0.45)

        # 3. Word count score — longer tasks are more parallelizable
        word_count = len(task.split())
        # Normalize: 0 words → 0.0, 50 words → 0.2, 100 words → 0.4, 200+ words → 0.6
        word_score = min(word_count / 200.0, 0.6)

        # Combined score: weighted average
        # File paths are the strongest signal (they imply separate units of work)
        # Keywords are moderate signal
        # Word count is a weak baseline signal
        combined = (file_score * 0.5) + (keyword_score * 0.35) + (word_score * 0.15)

        # Clamp to [0.0, 1.0]
        return round(min(max(combined, 0.0), 1.0), 3)

    def _estimate_file_count(self, task: str) -> int:
        """Estimate number of files touched from task text (for swarm scoring)."""
        task_lower = task.lower()
        file_patterns = [
            r'[\w./\-_]+\.(py|js|ts|jsx|tsx|go|rs|java|rb|cpp|c|h|hpp|cs|php|swift|kt|scala|sh|bash|zsh|fish|ps1|bat|cmd|yaml|yml|json|toml|xml|html|css|scss|less|md|txt|rst|ini|cfg|env|lock|sql|db|sqlite|csv|tsv|log)',
            r'(?:^|[\s,;])([\w./\-_]{3,}(?:/[\w.\-_]+)+)',
        ]
        file_paths = set()
        for pat in file_patterns:
            for m in re.finditer(pat, task_lower):
                path = m.group(1) if m.lastindex else m.group(0)
                if path and len(path) > 2:
                    file_paths.add(path)
        return max(1, len(file_paths))

    def _estimate_module_count(self, task: str) -> int:
        """Estimate modules/crates from keywords and path segments."""
        task_lower = task.lower()
        module_keywords = ["module", "package", "crate", "library", "subsystem", "component", "service"]
        kw_count = sum(1 for kw in module_keywords if kw in task_lower)
        # count unique top-level dirs in paths
        dir_matches = re.findall(r'([a-z_][a-z0-9_]*)/', task_lower)
        dirs = set(d for d in dir_matches if len(d) > 1)
        return max(1, kw_count + len(dirs))

    def _estimate_sequential_calls(self, task: str) -> int:
        """Rough count of sequential steps from language in task."""
        task_lower = task.lower()
        seq_indicators = ["then ", "after ", "next ", "followed by", "step ", "sequence", "first ", "second ", "third "]
        count = sum(task_lower.count(ind) for ind in seq_indicators)
        # numbered steps
        numbered = len(re.findall(r'\b\d+[\.\)]\s', task_lower))
        return max(1, count + numbered)

    def _compute_swarm_recommendation(self, task: str, scope: str, itil_classification: Optional[dict], matches: list, confidence: int) -> dict:
        """
        Compute swarm recommendation (solo/pair/swarm/mega) using swarm-decision dimensions.
        Non-trivial = score >8 .
        Strongly biases toward T2 free tier usage for workers unless security/change forces T3.
        """
        from token_optimiser.config import config

        file_count = self._estimate_file_count(task)
        module_count = self._estimate_module_count(task)
        seq_count = self._estimate_sequential_calls(task)

        # map to 1-5 scores per swarm-decision skill
        def dim_score_files(n: int) -> int:
            if n <= 1: return 1
            if n <= 3: return 2
            if n <= 5: return 3
            if n <= 8: return 4
            return 5

        def dim_score_modules(n: int) -> int:
            if n <= 1: return 1
            if n <= 2: return 2
            if n <= 3: return 3
            if n <= 5: return 4
            return 5

        def dim_score_seq(n: int) -> int:
            if n < 5: return 1
            if n < 10: return 2
            if n < 20: return 3
            if n < 40: return 4
            return 5

        files_s = dim_score_files(file_count)
        mods_s = dim_score_modules(module_count)
        seq_s = dim_score_seq(seq_count)

        # cross-cutting: multiple cats, arch, research, infra
        cross = 1
        if len(matches) > 2:
            cross = 4
        elif any(m["name"] in ("architecture", "infrastructure", "research") for m in matches):
            cross = 3
        elif len(matches) > 1:
            cross = 2

        # test/verify: if mentions tests or large scope
        test_c = 1
        task_l = task.lower()
        if "test" in task_l or "verify" in task_l or "coverage" in task_l:
            test_c = 4 if scope == "large" else 3
        elif scope == "large":
            test_c = 3
        elif scope == "medium":
            test_c = 2

        total_score = files_s + mods_s + seq_s + cross + test_c

        if total_score <= 8:
            rec = "solo"
        elif total_score <= 12:
            rec = "pair"
        elif total_score <= 18:
            rec = "swarm"
        else:
            rec = "mega"

        is_non_trivial = total_score > 8

        # ITIL ops or ambiguous/low conf also non-trivial
        is_ops = bool(itil_classification and itil_classification.get("is_ops"))
        unique_cats = len(set(m["name"] for m in matches)) if matches else 0
        low_conf = confidence < config.threshold_t2_min
        ambiguous = unique_cats >= 3
        if is_ops or ambiguous or low_conf or scope in ("medium", "large"):
            is_non_trivial = True
            if rec == "solo":
                rec = "pair" if total_score <= 10 else "swarm"

        # Stronger bias for user's goal: auto-swarm on refactor/integration/packaging/docs work
        # (so user no longer has to manually request swarms)
        task_l = task.lower()
        auto_swarm_triggers = ["refactor", "integration", "packaging", "github", "release", "multiple files", "update several", "docs and", "engine", "add boa", "governance"]
        if any(t in task_l for t in auto_swarm_triggers) and rec in ("solo", "pair"):
            rec = "swarm"
            is_non_trivial = True

        # bias: for free tier max, use T2 workers; only T3 if forced
        forced_t3 = False
        if itil_classification and itil_classification.get("classification") == "change":
            forced_t3 = True
        if matches and any(m.get("name") == "security" for m in matches):
            forced_t3 = True

        tier_for_workers = "T3" if forced_t3 else "T2"

        return {
            "recommendation": rec,
            "score": total_score,
            "dimensions": {
                "files_touched": files_s,
                "modules": mods_s,
                "sequential_tool_calls": seq_s,
                "cross_cutting": cross,
                "test_verify": test_c,
            },
            "non_trivial": is_non_trivial,
            "itil_driven": is_ops,
            "tier_for_workers": tier_for_workers,
            "rationale": f"score={total_score} (>8 non-trivial) → {rec}; workers on {tier_for_workers} (free max unless change/security)",
        }

    def _determine_scope(self, words: int, lines_hint: Optional[int], task_text: str = "") -> str:
        """Determine task scope: small, medium, or large."""
        from token_optimiser.config import config

        if task_text:
            import re
            pattern = config.scope_prefix_pattern
            match = re.search(pattern, task_text, re.IGNORECASE)
            if match:
                raw_value = match.group(1).lower().strip()
                scope_name = config.scope_prefix_values.get(raw_value)
                if scope_name:
                    return scope_name

        if lines_hint is not None:
            if lines_hint < 50:
                return "small"
            elif lines_hint < 200:
                return "medium"
            else:
                return "large"
        if words < config.scope_word_thresholds.get("small", 30):
            return "small"
        elif words < config.scope_word_thresholds.get("medium", 100):
            return "medium"
        else:
            return "large"

    def _match_categories(self, task: str) -> list:
        """Match the task against category rules."""
        task_lower = task.lower()
        matches = []
        for rule in CATEGORY_RULES:
            score = 0
            for kw in rule["keywords"]:
                if kw in task_lower:
                    score += 1
            if score > 0:
                matches.append({**rule, "match_score": score})
        matches.sort(key=lambda m: m["match_score"], reverse=True)
        return matches

    def _compute_confidence(self, matches: list, scope: str, task: str) -> int:
        """Compute confidence from matched categories and scope."""
        if not matches:
            return 55

        total_weight = sum(m["match_score"] for m in matches)
        weighted_conf = 0
        for m in matches:
            weight = m["match_score"] / total_weight
            base = m["base_confidence"]
            modifier = m["scope_modifiers"].get(scope, 0)
            weighted_conf += weight * (base + modifier)

        unique_categories = set(m["name"] for m in matches)
        if len(unique_categories) > 2:
            weighted_conf -= 10

        return int(round(weighted_conf))

    def _select_tier(self, confidence: int, unique_categories: int = 0, matches: list = None, itil_classification: Optional[dict] = None) -> str:
        """Select tier based on confidence score and config thresholds.
        Now driven by ITIL/BOA governance: prioritize free (T2) aggressively unless
        security or change classification forces T3.
        """
        from token_optimiser.config import config

        # ── ITIL / BOA GOVERNANCE OVERRIDE FIRST (forces T3 for change/security even on ambiguous) ─
        if itil_classification and itil_classification.get("is_ops"):
            cls = itil_classification.get("classification")
            if cls == "change":
                return "T3"
            if matches and any(m.get("name") == "security" for m in matches):
                return "T3"
            # incident/problem/request/event: stay on free path (aggressive T2)

        # Ambiguous task check
        if config.ambiguous_task_asks_human and unique_categories >= 3:
            return "ASK"

        # Ask-human override
        if config.ask_human_band and config.threshold_t2_min <= confidence < config.threshold_t1_min:
            return "ASK"

        # ── PRIVACY / OFFLINE SHORT-CIRCUIT ─────────────────────────────────────
        # If privacy_force_t1 is enabled, check task text for privacy/offline keywords.
        # These ALWAYS route to T1 or T1b regardless of free_tier_priority — data never leaves.
        if config.privacy_force_t1 and matches:
            top_match = matches[0]
            privacy_forced = (
                top_match.get("force_tier") == "T1" or
                top_match["name"] == "privacy"
            )
            # Also check raw keyword match from config.privacy_offline_keywords
            task_lower = self._last_task_text.lower() if hasattr(self, '_last_task_text') else ""
            if any(kw in task_lower for kw in config.privacy_offline_keywords):
                privacy_forced = True
            
            if privacy_forced:
                # Check if T1b keywords are also present — route to Lenovo's 7B CPU model
                if hasattr(config, 't1b_keywords') and any(kw in task_lower for kw in config.t1b_keywords):
                    if config.tier_backends.get("T1b", {}).get("enabled", True):
                        return "T1b"
                return "T1"

        # ── ARCHITECTURE MINIMUM TIER ───────────────────────────────────────────
        # Architecture decisions need T2 minimum (free cloud is stronger than local 3B)
        arch_min_tier = None
        if matches and matches[0]["name"] == "architecture":
            arch_min_tier = "T2"

        # ── FREE TIER PRIORITY ──────────────────────────────────────────────────
        # When free_tier_priority is True and the task is NOT security,
        # route to T2 (free cloud) instead of T1 (local 3B).
        # The free cloud models ... are FAR superior ...
        # Prioritize free tiers aggressively unless security/change forces T3.
        if config.free_tier_priority and confidence >= config.threshold_t2_min:
            # Check if this is a security task — security always goes to T3
            if matches and matches[0]["name"] == "security":
                return "T3"
            tier = "T2"
            # Enforce minimum tier if set (e.g., architecture needs T2+)
            if arch_min_tier and _TIER_ORDER.get(tier, 0) < _TIER_ORDER.get(arch_min_tier, 0):
                tier = arch_min_tier
            return tier

        # Standard routing (when free_tier_priority is False)
        if confidence >= config.threshold_t1_min:
            tier = "T1"
            if arch_min_tier and _TIER_ORDER.get(tier, 0) < _TIER_ORDER.get(arch_min_tier, 0):
                tier = arch_min_tier
            return tier

        if confidence >= config.threshold_t2_min:
            if not config.hybrid_band_enabled:
                return "ASK"
            return "T2"

        # Enforce minimum tier even at T3 (e.g., architecture needs T2+)
        if arch_min_tier and _TIER_ORDER.get("T3", 0) < _TIER_ORDER.get(arch_min_tier, 0):
            return arch_min_tier

        return "T3"

    def _build_rationale(self, matches: list, scope: str, confidence: int, tier: str,
                         unique_categories: int = 0) -> str:
        """Build a human-readable rationale for the routing decision."""
        from token_optimiser.config import config

        self._last_matches = matches

        parts = []
        if matches:
            top = matches[0]
            parts.append(f"Matched: {top['name']} (base confidence: {top['base_confidence']})")
            if len(matches) > 1:
                parts.append(f"+{len(matches)-1} more category matches")
        parts.append(f"Scope: {scope}")
        parts.append(f"Confidence: {confidence}%")
        parts.append(f"Thresholds: T1≥{config.threshold_t1_min} | T1b(priv) | T2≥{config.threshold_t2_min} | T3<{config.threshold_t3_always_below}")

        if tier == "ASK":
            reasons = []
            if unique_categories >= 3:
                reasons.append(f"ambiguous ({unique_categories} categories)")
            if config.ask_human_band and config.threshold_t2_min <= confidence < config.threshold_t1_min:
                reasons.append("in hybrid/ask band")
            if reasons:
                parts.append(f"Decision: ASK HUMAN ({' + '.join(reasons)})")
            else:
                parts.append("Decision: ASK HUMAN")
        else:
            tier_obj = next((t for t in TIERS if t.name == tier), None)
            if tier_obj:
                # Add free model info for T2
                if tier == "T2" and config.free_tier_priority and matches:
                    preferred = config.category_free_model_preference.get(matches[0]["name"])
                    if preferred:
                        parts.append(f"Route: {tier} ({tier_obj.backend}) → {preferred}")
                    else:
                        parts.append(f"Route: {tier} ({tier_obj.backend})")
                else:
                    parts.append(f"Route: {tier} ({tier_obj.backend})")
            else:
                parts.append(f"Route: {tier}")
        return " | ".join(parts)


# ── RoutingDecision ────────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    """The router's output for a single task."""
    tier: str
    confidence: int
    rationale: str
    task: str
    scope: str = "unknown"
    matched_categories: list = field(default_factory=list)
    context: Optional[dict] = None
    task_id: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    parallelize: bool = False
    parallelize_score: float = 0.0
    orchestrator_hints: Optional[dict] = None
    itil_classification: Optional[dict] = None
    swarm: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "tier": self.tier,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "task": self.task,
            "scope": self.scope,
            "matched_categories": self.matched_categories,
            "timestamp": datetime.datetime.now().isoformat(),
            "task_id": self.task_id,
            "parallelize": self.parallelize,
            "parallelize_score": self.parallelize_score,
        }
        if self.orchestrator_hints is not None:
            d["orchestrator_hints"] = self.orchestrator_hints
        if self.itil_classification is not None:
            d["itil_classification"] = self.itil_classification
        if self.swarm is not None:
            d["swarm"] = self.swarm
            if isinstance(self.swarm, dict) and "recommendation" in self.swarm:
                d["swarm_recommendation"] = self.swarm["recommendation"]
        return d


# ── Learning Log ──────────────────────────────────────────────────────────────

class LearningLog:
    """Append-only log of routing decisions and their outcomes."""

    def __init__(self, path: str = "token_optimiser/learning_log.jsonl"):
        self.path = path

    def append(self, decision: Optional[RoutingDecision] = None, outcome: Optional[dict] = None, extra: Optional[dict] = None):
        if decision is not None:
            entry = decision.to_dict()
            if decision.context:
                entry["context"] = decision.context
        elif extra is not None:
            entry = dict(extra)
        else:
            raise ValueError("LearningLog.append requires decision or extra")
        if outcome:
            entry["outcome"] = outcome
        entry["feedback_timestamp"] = datetime.datetime.now().isoformat()

        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def append_outcome(self, task_id: str, outcome: dict) -> bool:
        try:
            with open(self.path, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return False

        entries = []
        found = False
        for line in lines:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("task_id") == task_id and "outcome" not in entry:
                entry["outcome"] = outcome
                entry["feedback_timestamp"] = datetime.datetime.now().isoformat()
                found = True
            entries.append(entry)

        if found:
            with open(self.path, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
        return found

    def get_recent(self, limit: int = 50) -> list:
        try:
            with open(self.path, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return []
        entries = []
        for line in lines:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries[-limit:]

    def get_entries(self) -> list:
        """Return ALL entries from the learning log (no limit)."""
        try:
            with open(self.path, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return []
        entries = []
        for line in lines:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def clear(self):
        """Clear all entries from the learning log."""
        try:
            with open(self.path, "w") as f:
                pass
        except FileNotFoundError:
            pass
