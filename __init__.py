"""
Token Optimiser — Main Package

Budget-aware router T1 local → T2 free cloud → T3 paid.
Plan first, confidence gate, learn after.
"""

from token_optimiser.config import config, set_enabled
from token_optimiser.router import TokenRouter, LearningLog, RoutingDecision
from token_optimiser.monica_integration import optimise, optimiser_enabled, optimise_batch, record_outcome

__all__ = ["config", "set_enabled", "TokenRouter", "LearningLog", "RoutingDecision",
           "optimise", "optimiser_enabled", "optimise_batch", "record_outcome"]
