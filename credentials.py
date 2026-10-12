"""Credential + config loader for the token optimiser.

WHY THIS FILE EXISTS
--------------------
An audit found the router could never reach a free cloud tier. The keys were
valid and present in `~/.hermes/.env` (OpenRouter and NVIDIA both returned
HTTP 200 on /models), but `token_optimiser` only ever called
`os.environ.get(...)`. It never loaded a `.env` file. So from inside the
router's own process, every free tier looked unavailable — and the router,
behaving correctly given its inputs, escalated to T3 (paid).

That is the mechanism behind "45 of 125 decisions went to T3".

It is also the same failure class as the model substitution: the system
believed something about its own environment that was false, and nothing
surfaced the error.

WHAT IT DOES
------------
Loads `~/.hermes/.env` (i.e. `$HERMES_HOME/.env`) into `os.environ` without
overwriting variables that are already set. Existing environment wins, so an
operator can always override a value for one run.

NOUS IS DELIBERATELY NOT LOADED
--------------------------------
Nous is an OAuth device-code credential stored in `~/.hermes/auth.json` with a
refresh token and an ~1 hour expiry — not a static API key. Copying its access
token into an env var would appear to work and then fail silently on expiry.
We report it as present-in-auth.json instead, so the router can reason about
it honestly rather than believing a stale token is a working key.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

__all__ = ["load_env_file", "hermes_home", "nous_credentials", "apply_env"]

# Static-key credentials we are willing to lift into the environment.
_STATIC_KEYS = (
    "OPENROUTER_API_KEY",
    "NVIDIA_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LM_API_KEY",
    "GROK_API_KEY",
)

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def hermes_home() -> Path:
    """Resolve the active Hermes home. Never hardcode ~/.hermes."""
    raw = os.environ.get("HERMES_HOME")
    return Path(raw) if raw else Path.home() / ".hermes"


def load_env_file(path: Optional[Path] = None) -> Dict[str, str]:
    """Parse a .env file into a dict. Tolerant of quotes, comments, blank lines.

    Does not touch os.environ — that is `apply_env`'s job. Kept separate so it
    can be tested without side effects.
    """
    target = path or (hermes_home() / ".env")
    out: Dict[str, str] = {}
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2)
        # Strip trailing inline comment only when clearly outside quotes.
        if "#" in raw and not (raw.startswith(("'", '"'))):
            raw = raw.split("#", 1)[0].strip()
        raw = raw.strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
            raw = raw[1:-1]
        if key:
            out[key] = raw
    return out


def apply_env(path: Optional[Path] = None, *, override: bool = False) -> Dict[str, str]:
    """Load .env into os.environ. Existing values win unless override=True.

    Returns the dict of keys this call actually set — useful for logging
    WITHOUT ever logging the values.
    """
    parsed = load_env_file(path)
    applied: Dict[str, str] = {}
    for key, val in parsed.items():
        if not override and os.environ.get(key):
            continue
        if not val:
            continue
        os.environ[key] = val
        applied[key] = f"<set len={len(val)}>"
    return applied


def nous_credentials(*, refresh: bool = False) -> Dict[str, object]:
    """Report Nous OAuth credential state without importing or exposing it.

    Returns presence/expiry metadata only. Never returns a token value: the
    caller only needs to know whether Nous is usable, and handing a bearer token
    around in a plain dict is how tokens end up in logs.
    """
    result: Dict[str, object] = {
        "kind": "oauth",
        "present": False,
        "expired": None,
        "static_env_key": bool(os.environ.get("NOUS_API_KEY")),
    }
    try:
        data = json.loads((hermes_home() / "auth.json").read_text(encoding="utf-8"))
    except Exception:
        return result

    nous = (data.get("providers") or {}).get("nous") or {}
    if not nous:
        return result

    result["present"] = bool(nous.get("access_token") or nous.get("agent_key"))
    expires = nous.get("expires_at")
    if expires:
        try:
            import time
            # expires_at may be epoch seconds or an ISO string.
            try:
                ts = float(expires)
            except (TypeError, ValueError):
                from datetime import datetime
                ts = datetime.fromisoformat(str(expires).replace("Z", "+00:00")).timestamp()
            result["expired"] = ts < time.time()
            result["expires_in_s"] = int(max(0, ts - time.time()))
        except Exception:
            pass
    result["has_refresh_token"] = bool(nous.get("refresh_token"))
    return result


def credential_report() -> Dict[str, object]:
    """A JSON-safe snapshot of what this process can actually reach.

    Safe to log: booleans and lengths only, never key material.
    """
    return {
        "env": {k: bool(os.environ.get(k)) for k in _STATIC_KEYS},
        "nous": nous_credentials(),
        "env_file_present": (hermes_home() / ".env").is_file(),
    }