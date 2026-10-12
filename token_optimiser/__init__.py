"""
Token Optimiser — Main Package

Budget-aware router T1 local → T2 free cloud → T3 paid.
Plan first, confidence gate, learn after.
"""

from token_optimiser.credentials import apply_env as _apply_env, credential_report

# The router reads credentials via os.environ only. Without this, a free cloud
# tier looks unavailable inside this process and every decision escalates to
# T3 (paid). Existing environment always wins, so an operator can still
# override per-run. Values are never logged.
_CREDENTIALS_LOADED = _apply_env()

from token_optimiser.config import config, set_enabled
from token_optimiser.router import TokenRouter, LearningLog, RoutingDecision, classify_itil
from token_optimiser.monica_integration import optimise, optimiser_enabled, optimise_batch, record_outcome
from token_optimiser.boa import run_boa

__all__ = ["config", "set_enabled", "TokenRouter", "LearningLog", "RoutingDecision",
           "optimise", "optimiser_enabled", "optimise_batch", "record_outcome",
           "run_boa", "classify_itil", "credential_report"]
