"""
Token Optimiser — Combined Runner

When this runs, both the token optimiser engine AND the dashboard start together.
The dashboard web UI is served on port 8765, and the optimiser sits ready to
route tasks through T1/T2/T3.

Usage:
    python3 -m token_optimiser.dashboard.runs

This starts:
    1. The FastAPI dashboard (http://localhost:8765)
    2. The token optimiser in active mode (config.enabled = True)
    3. Ollama control endpoints for managing the T1 server

The dashboard IS the token optimiser interface — every control, toggle, and
analytics view is available through the web UI.

Requirements:
    pip install fastapi uvicorn pydantic httpx
    (or use .venv/bin/python if venv is set up)
"""

import json
import logging
import os
import sys
from pathlib import Path

# Ensure the package is importable
PACKAGE_DIR = Path(__file__).parent.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from config import config
from router import TokenRouter, LearningLog
from token_optimiser.dashboard.app import app, serialize_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("token-optimiser-runs")


def print_startup_banner():
    """Print a clear startup banner showing the current state."""
    print("\n" + "=" * 70)
    print("  TOKEN OPTIMISER + DASHBOARD")
    print("=" * 70)
    print(f"  Dashboard:  http://localhost:{os.environ.get('DASHBOARD_PORT', '8765')}")
    print(f"  Optimiser:  {'ENABLED' if config.enabled else 'DISABLED (T2 baseline)'}")
    print(f"  T1:         {config.tier_backends.get('T1', {}).get('url', 'not configured')} ({config.tier_backends.get('T1', {}).get('model', 'unknown')})")
    print(f"  T2 Gateways: {len(config.tier_backends.get('T2', {}).get('gateways', []))}")
    print(f"  T3 Providers: {len(config.tier_backends.get('T3', {}).get('providers', []))}")
    thresholds = config
    print(f"  Thresholds: T1≥{thresholds.threshold_t1_min} | T2≥{thresholds.threshold_t2_min} | T3<{thresholds.threshold_t3_always_below}")
    print(f"  Security cap: {thresholds.security_confidence_cap}%")
    print("-" * 70)

    # Quick routing sanity check
    router = TokenRouter()
    test_results = []
    for task in [
        "Fix a typo in the README",
        "Add authentication to the API",
        "Refactor the entire state management module",
    ]:
        d = router.assess(task)
        test_results.append(f"    {d.tier:4s} {d.confidence:3d}% — {task[:50]}")

    print("  Quick routing test:")
    for line in test_results:
        print(line)
    print("=" * 70 + "\n")


def main():
    """Start the combined token optimiser + dashboard."""
    port = int(os.environ.get("DASHBOARD_PORT", 8765))
    host = os.environ.get("DASHBOARD_HOST", "0.0.0.0")

    print_startup_banner()

    # Verify Ollama connection if T1 is configured
    try:
        from token_optimiser.dashboard.ollama_control import check_health
        health = check_health()
        if health.get("healthy"):
            print(f"  Ollama:     Connected via {health.get('method')} ({health.get('model_count', 0)} models)")
        else:
            print(f"  Ollama:     Not reachable ({health.get('error', 'unknown')})")
    except Exception as e:
        print(f"  Ollama:     Health check failed: {e}")
    print()

    import uvicorn
    print(f"  Starting FastAPI dashboard on {host}:{port} ...")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
