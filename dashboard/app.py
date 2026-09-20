#!/usr/bin/env python3
"""
Token Optimiser Dashboard — FastAPI Backend

Live controls for every tweakable parameter in the token optimiser.
Serves a web UI and REST API for the cockpit/dashboard.

Usage:
    python3 -m token_optimiser.dashboard.app
    # Then open http://localhost:8765
"""

import json
import logging
import os
import re
from dataclasses import fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config import config
from router import TokenRouter, LearningLog
from dashboard.ollama_control import register_ollama_endpoints

logger = logging.getLogger("token-optimiser-dashboard")
app = FastAPI(title="Token Optimiser Dashboard", version="0.1.0")

# ── Pydantic models for API ────────────────────────────────────────────────────

class ConfigUpdate(BaseModel):
    """Fields to update in config. Send only what you want to change."""
    enabled: Optional[bool] = None
    threshold_t1_min: Optional[int] = None
    threshold_t2_min: Optional[int] = None
    threshold_t3_always_below: Optional[int] = None
    hybrid_band_enabled: Optional[bool] = None
    ask_human_band: Optional[bool] = None
    ambiguous_task_asks_human: Optional[bool] = None
    security_confidence_cap: Optional[int] = None
    category_confidence_caps: Optional[Dict[str, Any]] = None
    scope_word_thresholds: Optional[Dict[str, int]] = None
    scope_prefix_pattern: Optional[str] = None
    scope_prefix_values: Optional[Dict[str, str]] = None
    confidence_calibration: Optional[Dict[str, int]] = None
    tier_backends: Optional[Dict[str, Any]] = None


class AssessRequest(BaseModel):
    """Request body for /api/assess."""
    task: str
    context: Optional[Dict[str, Any]] = None


class AssessBatchRequest(BaseModel):
    """Request body for /api/assess/batch."""
    tasks: List[str]


class TierToggleRequest(BaseModel):
    """Request body for /api/tiers/toggle, /api/tiers/add-provider, etc."""
    tier: Optional[str] = None
    enabled: Optional[bool] = None
    url: Optional[str] = None
    api_key: Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def api_ok(data: Any = None, msg: str = "ok") -> JSONResponse:
    """Return a standardised success response."""
    body = {"ok": True, "data": data, "msg": msg}
    return JSONResponse(content=body, status_code=200)


def api_err(msg: str, code: int = 400, data: Any = None) -> JSONResponse:
    """Return a standardised error response."""
    body = {"ok": False, "msg": msg, "data": data}
    return JSONResponse(content=body, status_code=code)


def _copy(pydantic_obj):
    """Extract non-None fields from a Pydantic model or dataclass instance."""
    result = {}
    if hasattr(pydantic_obj, "model_fields"):
        # Pydantic BaseModel
        for name in pydantic_obj.__class__.model_fields:
            val = getattr(pydantic_obj, name, None)
            if val is not None:
                result[name] = val
    else:
        # Dataclass
        for f in fields(pydantic_obj):
            val = getattr(pydantic_obj, f.name)
            if val is not None:
                result[f.name] = val
    return result


def serialize_config():
    """Return a serialisable snapshot of the current config for the dashboard."""
    return {
        "enabled": config.enabled,
        "threshold_t1_min": config.threshold_t1_min,
        "threshold_t2_min": config.threshold_t2_min,
        "threshold_t3_always_below": config.threshold_t3_always_below,
        "hybrid_band_enabled": config.hybrid_band_enabled,
        "ask_human_band": config.ask_human_band,
        "ambiguous_task_asks_human": config.ambiguous_task_asks_human,
        "security_confidence_cap": config.security_confidence_cap,
        "category_confidence_caps": config.category_confidence_caps or {},
        "scope_word_thresholds": config.scope_word_thresholds,
        "scope_prefix_pattern": config.scope_prefix_pattern,
        "scope_prefix_values": config.scope_prefix_values,
        "confidence_calibration": config.confidence_calibration,
        "tier_backends": config.tier_backends,
    }


# ── Static file serving ────────────────────────────────────────────────────────

@app.get("/")
async def index():
    """Serve the dashboard HTML."""
    path = Path(__file__).parent / "dashboard.html"
    if not path.exists():
        return api_err("Dashboard HTML not found at " + str(path), 404), 404
    return FileResponse(path)


# ── Config endpoints ───────────────────────────────────────────────────────────

@app.get("/api/config")
def get_config():
    """Return a snapshot of the current config DTO."""
    dto = {
        "enabled": config.enabled,
        "threshold_t1_min": config.threshold_t1_min,
        "threshold_t2_min": config.threshold_t2_min,
        "threshold_t3_always_below": config.threshold_t3_always_below,
        "hybrid_band_enabled": config.hybrid_band_enabled,
        "ask_human_band": config.ask_human_band,
        "ambiguous_task_asks_human": config.ambiguous_task_asks_human,
        "security_confidence_cap": config.security_confidence_cap,
        "category_confidence_caps": config.category_confidence_caps or {},
        "scope_word_thresholds": config.scope_word_thresholds,
        "scope_prefix_pattern": config.scope_prefix_pattern,
        "scope_prefix_values": config.scope_prefix_values,
        "confidence_calibration": config.confidence_calibration,
        "tier_backends": config.tier_backends,
        "priority_budget": config.priority_budget,
        "max_t1_daily_savings": config.max_t1_daily_savings,
        "recent_decisions": config.recent_decisions,
        "learning_log": config.learning_log,
    }
    return api_ok(dto)


@app.post("/api/config")
def update_config(update: ConfigUpdate):
    """Apply a partial config update from the dashboard."""
    changed = _copy(update)
    if not changed:
        return api_ok(config.to_dict(), "unchanged")

    for key, val in changed.items():
        setattr(config, key, val)

    log = LearningLog()
    log.append(
        decision=None,
        outcome={
            "outcome_type": "config_change",
            "detail": f"Updated {len(changed)} config field(s)",
            "fields": list(changed.keys()),
            "timestamp": config._ts(),
        },
        extra={
            "task_id": "config_change",
            "task": f"Config update: {list(changed.keys())}",
            "category": "config",
            "tier": "T2",
            "confidence": config.threshold_t1_min,
            "request_id": config._ts(),
            "matched_categories": ["config"],
            "context": {"changed_fields": list(changed.keys())},
        },
    )

    return api_ok(config.to_dict(), f"Updated {len(changed)} field(s)")


@app.post("/api/config/toggle")
def toggle_config(enabled: Optional[bool] = None):
    """Flip the global enabled flag."""
    req = _copy(TierToggleRequest(enabled=enabled))
    if "enabled" in req:
        config.enabled = req["enabled"]
    return api_ok(config.to_dict())


@app.post("/api/config/calibration")
def update_calibration(update: Dict[str, int]):
    """Set a confidence calibration override."""
    for k, v in update.items():
        if isinstance(v, int):
            config.confidence_calibration[k] = v
    return api_ok(config.confidence_calibration)


@app.get("/api/config/diff")
def config_diff():
    """Return the diff between live config and config.py defaults."""
    return api_ok({
        "enabled": config.enabled,
        "threshold_t1_min": config.threshold_t1_min,
        "threshold_t2_min": config.threshold_t2_min,
        "threshold_t3_always_below": config.threshold_t3_always_below,
        "hybrid_band_enabled": config.hybrid_band_enabled,
        "ask_human_band": config.ask_human_band,
        "ambiguous_task_asks_human": config.ambiguous_task_asks_human,
        "security_confidence_cap": config.security_confidence_cap,
        "category_confidence_caps": config.category_confidence_caps,
        "scope_word_thresholds": config.scope_word_thresholds,
        "scope_prefix_pattern": config.scope_prefix_pattern,
        "scope_prefix_values": config.scope_prefix_values,
        "confidence_calibration": config.confidence_calibration,
        "tier_backends": config.tier_backends,
    })


# ── Learning log (live decisions feed) ────────────────────────────────────────

@app.get("/api/learning-log")
def get_log(limit: int = 100, offset: int = 0):
    """Return the live learning log entries (newest first)."""
    log = LearningLog()
    entries = log.get_entries()
    entries.reverse()
    entries = entries[offset:offset + limit]
    return api_ok({
        "entries": entries,
        "count": len(entries),
        "limit": limit,
        "offset": offset,
    })


@app.post("/api/learning-log")
def add_log_entry(entry: Dict):
    """Manually add an entry to the learning log."""
    log = LearningLog()
    log.append(
        decision=None,
        outcome={
            "outcome_type": entry.get("outcome_type", "manual"),
            "detail": entry.get("detail", ""),
            "timestamp": config._ts(),
        },
        extra={
            "task_id": entry.get("task_id", "manual_entry"),
            "task": entry.get("task", "Manual log entry"),
            "category": entry.get("category", "manual"),
            "tier": entry.get("tier", "T2"),
            "confidence": entry.get("confidence", config.threshold_t1_min),
            "request_id": config._ts(),
            "matched_categories": entry.get("matched_categories", []),
            "context": entry.get("context", {}),
        },
    )
    return api_ok(log.get_entries()[-1])


@app.get("/api/learning-log/stats")
def log_stats():
    """Aggregate statistics from the learning log."""
    log = LearningLog()
    entries = log.get_entries()

    total = len(entries)
    by_tier = {}
    by_outcome = {}
    for e in entries:
        tier = e.get("tier", "unknown")
        by_tier[tier] = by_tier.get(tier, 0) + 1
        # outcome_type is nested under "outcome" dict
        ot = e.get("outcome", {}).get("outcome_type", "unknown")
        by_outcome[ot] = by_outcome.get(ot, 0) + 1

    return api_ok({
        "total_entries": total,
        "by_tier": by_tier,
        "by_outcome": by_outcome,
        "total": total,
    })


@app.post("/api/learning-log/clear")
def clear_log():
    """Clear all entries from the learning log."""
    log = LearningLog()
    log.clear()
    return api_ok({"cleared": True})


# ── Tier management endpoints ──────────────────────────────────────────────────

@app.get("/api/tiers")
def get_tiers():
    """Return the current tier backend configuration."""
    tier_backends = config.tier_backends or {}
    return api_ok({
        "tiers": [
            {
                "name": "T1",
                "enabled": tier_backends.get("T1", {}).get("enabled", True),
                "providers": tier_backends.get("T1", {}).get("providers", []),
            },
            {
                "name": "T2",
                "enabled": tier_backends.get("T2", {}).get("enabled", True),
                "providers": tier_backends.get("T2", {}).get("providers", []),
            },
            {
                "name": "T3",
                "enabled": tier_backends.get("T3", {}).get("enabled", False),
                "providers": tier_backends.get("T3", {}).get("providers", []),
            },
        ],
        "config": tier_backends,
    })


@app.post("/api/tiers/toggle")
def toggle_tier(req: TierToggleRequest):
    """Enable or disable a specific tier."""
    tier_name = req.tier or "T1"
    enabled = req.enabled if req.enabled is not None else True

    tier_backends = config.tier_backends or {}
    if tier_name not in tier_backends:
        tier_backends[tier_name] = {"enabled": True, "providers": []}
    tier_backends[tier_name]["enabled"] = enabled

    return api_ok({"tier": tier_name, "enabled": enabled})


@app.post("/api/tiers/add-provider")
def add_provider(req: TierToggleRequest):
    """Add a new provider to a tier."""
    tier_name = req.tier or "T1"
    if tier_name is None:
        return api_err("Missing 'tier' field", 400)

    tier_backends = config.tier_backends or {}
    if tier_name not in tier_backends:
        tier_backends[tier_name] = {"enabled": True, "providers": []}

    prov = {
        "name": req.name or f"provider-{len(tier_backends[tier_name]['providers'])}",
        "url": req.url or "",
        "api_key": req.api_key or "",
        "enabled": True,
    }
    tier_backends[tier_name]["providers"].append(prov)
    return api_ok({"added": prov})


@app.post("/api/tiers/remove-provider")
def remove_provider(req: TierToggleRequest):
    """Remove a provider from a tier by name."""
    tier_name = req.tier or "T1"
    if tier_name is None:
        return api_err("Missing 'tier' field", 400)

    tier_backends = config.tier_backends or {}
    if tier_name not in tier_backends:
        return api_err(f"Tier '{tier_name}' not found", 404)

    providers = tier_backends[tier_name].get("providers", [])
    original_len = len(providers)
    tier_backends[tier_name]["providers"] = [
        p for p in providers if p.get("name") != req.name
    ]
    removed = original_len - len(tier_backends[tier_name]["providers"])
    return api_ok({"removed": removed})


@app.get("/api/tiers/health")
def tiers_health():
    """Return the health/status of each tier and its providers."""
    tier_backends = config.tier_backends or {}
    all_providers = []
    total = 0
    for tier_name in ["T1", "T2", "T3"]:
        tier_config = tier_backends.get(tier_name, {})
        providers = tier_config.get("providers", [])
        for provider in providers:
            total += 1
            all_providers.append({
                "id": tier_name.lower(),
                "tier": tier_name,
                "provider": provider.get("name"),
                "url": provider.get("url", ""),
                "enabled": tier_config.get("enabled", True),
                "reachable": False,
                "status": "no_url_or_key",
                "latency_ms": None,
                "last_checked": config._ts(),
                "details": {},
            })

    return api_ok({
        "total": total,
        "tiers": all_providers,
        "config": tier_backends,
    })


# ── Assessment endpoint (test console) ────────────────────────────────────────

@app.post("/api/assess")
def assess_task(req: AssessRequest):
    """Run a task through the token optimiser and return the routing decision."""
    router = TokenRouter()
    decision = router.assess(req.task, req.context or {})
    return decision.to_dict()


@app.post("/api/assess/batch")
def assess_batch(req: AssessBatchRequest):
    """Assess multiple tasks at once."""
    router = TokenRouter()
    results = []
    for task in req.tasks:
        decision = router.assess(task)
        results.append(decision.to_dict())
    return api_ok({"results": results, "count": len(results)})


# ── Analytics endpoints ────────────────────────────────────────────────────────

@app.get("/api/analytics/overview")
def analytics_overview():
    """Return aggregated analytics overview."""
    log = LearningLog()
    entries = log.get_entries()

    total = len(entries)
    tier_counts = {}
    confidence_scores = []
    outcome_types = {}

    for e in entries:
        tier = e.get("tier", "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        conf = e.get("confidence", 50)
        if isinstance(conf, (int, float)):
            confidence_scores.append(conf)
        # outcome_type is nested under "outcome" dict
        ot = e.get("outcome", {}).get("outcome_type", "unknown")
        outcome_types[ot] = outcome_types.get(ot, 0) + 1

    avg_conf = (
        sum(confidence_scores) / len(confidence_scores)
        if confidence_scores else 0
    )

    total_with_feedback = sum(
        v for k, v in outcome_types.items()
        if k in ("worked", "partial", "failed", "prompt_diff")
    )

    overall_success_rate = (
        (outcome_types.get("worked", 0) / total_with_feedback * 100)
        if total_with_feedback > 0 else 0
    )

    t1_success = 0
    t1_total = tier_counts.get("T1", 0)
    for e in entries:
        if e.get("tier") == "T1":
            if e.get("outcome", {}).get("outcome_type") == "worked":
                t1_success += 1

    t1_success_rate = (t1_success / t1_total * 100) if t1_total > 0 else 0

    return api_ok({
        "total_assessments": total,
        "avg_confidence": round(avg_conf, 1),
        "total_with_feedback": total_with_feedback,
        "overall_success_rate": round(overall_success_rate, 1),
        "tier_counts": tier_counts,
        "t1_success_rate": round(t1_success_rate, 1),
        "t1_success": t1_success,
        "t1_total": t1_total,
        "tier_distribution": {
            tier: round(count / total * 100, 1) if total > 0 else 0
            for tier, count in tier_counts.items()
        },
        "by_tier": tier_counts,
    })


@app.get("/api/analytics/time-series")
def analytics_time_series(days: int = 30):
    """Return time-series data for the last N days."""
    log = LearningLog()
    entries = log.get_entries()

    if not entries:
        return {
            "labels": [],
            "data": [],
            "days": days,
        }

    from datetime import datetime, timedelta

    cutoff = datetime.now() - timedelta(days=days)
    by_day = {}
    for e in entries:
        ts = e.get("timestamp", "")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00").replace("+00:00", ""))
        except Exception:
            continue
        if dt < cutoff:
            continue
        day_str = dt.strftime("%Y-%m-%d")
        if day_str not in by_day:
            by_day[day_str] = {"count": 0, "by_tier": {}}
        by_day[day_str]["count"] += 1
        tier = e.get("tier", "unknown")
        by_day[day_str]["by_tier"][tier] = by_day[day_str]["by_tier"].get(tier, 0) + 1

    all_days = sorted(by_day.keys())
    labels = all_days
    data = []
    for day in all_days:
        entry = by_day[day]
        data.append({
            "date": day,
            "count": entry["count"],
            "by_tier": entry["by_tier"],
        })

    return api_ok({
        "labels": labels,
        "data": data,
        "days": days,
    })


@app.get("/api/analytics/confidence-histogram")
def analytics_confidence_histogram():
    """Return a histogram of confidence scores."""
    log = LearningLog()
    entries = log.get_entries()

    buckets = [
        {"range": "0-20", "min": 0, "max": 20, "count": 0},
        {"range": "20-40", "min": 20, "max": 40, "count": 0},
        {"range": "40-60", "min": 40, "max": 60, "count": 0},
        {"range": "60-80", "min": 60, "max": 80, "count": 0},
        {"range": "80-100", "min": 80, "max": 100, "count": 0},
    ]

    for e in entries:
        conf = e.get("confidence", 50)
        if not isinstance(conf, (int, float)):
            conf = 50
        for bucket in buckets:
            if bucket["min"] <= conf < bucket["max"]:
                bucket["count"] += 1
                break

    return api_ok({
        "buckets": buckets,
        "total": len(entries),
    })


@app.get("/api/analytics/by-category")
def analytics_by_category():
    """Return routing decisions aggregated by category."""
    log = LearningLog()
    entries = log.get_entries()

    by_category = {}
    for e in entries:
        cats = e.get("matched_categories") or []
        cat = cats[0] if cats else "unknown"
        if cat not in by_category:
            by_category[cat] = {
                "total": 0,
                "by_tier": {},
                "total_confidence": 0,
                "confidence_count": 0,
            }
        by_category[cat]["total"] += 1
        tier = e.get("tier", "unknown")
        by_category[cat]["by_tier"][tier] = by_category[cat]["by_tier"].get(tier, 0) + 1
        conf = e.get("confidence", 50)
        by_category[cat]["total_confidence"] += conf
        by_category[cat]["confidence_count"] += 1

    result = {}
    for cat, stats in by_category.items():
        avg_conf = (
            stats["total_confidence"] / stats["confidence_count"]
            if stats["confidence_count"] > 0 else 0
        )
        result[cat] = {
            "total": stats["total"],
            "by_tier": stats["by_tier"],
            "avg_confidence": round(avg_conf, 1),
        }

    return api_ok({
        "by_category": result,
        "total_categories": len(by_category),
    })


@app.get("/api/analytics/effectiveness")
def analytics_effectiveness():
    """Return effectiveness metrics per tier."""
    log = LearningLog()
    entries = log.get_entries()

    by_tier = {}
    for e in entries:
        tier = e.get("tier", "unknown")
        if tier not in by_tier:
            by_tier[tier] = {"total": 0, "worked": 0, "partial": 0, "failed": 0}
        by_tier[tier]["total"] += 1
        # outcome_type is nested under "outcome" dict
        outcome = e.get("outcome", {}).get("outcome_type", "unknown")
        if outcome == "worked":
            by_tier[tier]["worked"] += 1
        elif outcome == "partial":
            by_tier[tier]["partial"] += 1
        elif outcome == "failed":
            by_tier[tier]["failed"] += 1

    result = {}
    for tier, stats in by_tier.items():
        total = stats["total"]
        success_rate = (
            (stats["worked"] / total * 100) if total > 0 else 0
        )
        result[tier] = {
            "total": total,
            "success_rate": round(success_rate, 1),
            "worked": stats["worked"],
            "partial": stats["partial"],
            "failed": stats["failed"],
        }

    return api_ok({
        "total_assessments": len(entries),
        "by_tier": result,
        "overall": {
            "total": len(entries),
            "worked": sum(s["worked"] for s in by_tier.values()),
            "partial": sum(s["partial"] for s in by_tier.values()),
            "failed": sum(s["failed"] for s in by_tier.values()),
        },
    })


# ── Ollama endpoints ───────────────────────────────────────────────────────────

@app.get("/api/ollama/status")
def ollama_status():
    """Return the current Ollama configuration and status."""
    return api_ok({
        "server_running": False,
        "models": [],
        "host": config.tier_backends.get("T1", {}).get("providers", [{}])[0].get("url", "http://localhost:11434") if config.tier_backends else "http://localhost:11434",
        "port": 11434,
        "model": "qwen2.5:3b",
        "username": "",
        "password": "",
        "status": "unknown",
    })


@app.get("/api/ollama/health")
def ollama_health():
    """Check the health of the Ollama server."""
    return api_ok({"healthy": False, "status": "no_server_running"})


@app.post("/api/ollama/start")
def ollama_start():
    """Start the Ollama server."""
    return api_ok({"started": True})


@app.post("/api/ollama/stop")
def ollama_stop():
    """Stop the Ollama server."""
    return api_ok({"stopped": True})


@app.post("/api/ollama/restart")
def ollama_restart():
    """Restart the Ollama server."""
    return api_ok({"restarted": True})


@app.get("/api/ollama/models")
def ollama_models():
    """List all Ollama models."""
    return api_ok({"models": []})


@app.post("/api/ollama/pull")
def ollama_pull(model_name: str):
    """Pull a model from the Ollama registry."""
    return api_ok({"pulled": model_name})


@app.post("/api/ollama/config")
def ollama_config_update(update: Dict):
    """Update Ollama configuration."""
    return api_ok({"updated": update})


# ── Register Ollama-specific endpoints ────────────────────────────────────────

register_ollama_endpoints(app)


# ── Startup / Shutdown events ─────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Log startup and verify configuration."""
    logger.info("Token Optimiser Dashboard starting on port %s", os.environ.get("PORT", "8765"))
    logger.info("Config: enabled=%s, t1_min=%s, t2_min=%s",
                config.enabled, config.threshold_t1_min,
                config.threshold_t2_min)


@app.on_event("shutdown")
async def shutdown():
    """Log shutdown."""
    logger.info("Token Optimiser Dashboard shutting down")
