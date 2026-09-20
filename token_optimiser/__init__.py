"""
Token Optimiser — Main Package

Budget-aware router T1 local → T2 free cloud → T3 paid.
Plan first, confidence gate, learn after.
"""

from .config import config, set_enabled
from .router import TokenRouter, LearningLog, RoutingDecision, classify_itil
from .monica_integration import optimise, optimiser_enabled, optimise_batch, record_outcome
from .boa import run_boa

__all__ = ["config", "set_enabled", "TokenRouter", "LearningLog", "RoutingDecision",
           "optimise", "optimiser_enabled", "optimise_batch", "record_outcome",
           "run_boa", "classify_itil"]
