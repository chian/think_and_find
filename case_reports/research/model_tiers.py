"""Declared model tiers for the research agent's operation call sites.

The scientific workflow chooses an operation; this table declares the model
capability that operation needs. Provider configuration maps each tier to an
actual model for one run. Unknown operations use the reasoning tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class ModelTier(str, Enum):
    REASONING = "reasoning"
    FAST = "fast"


CALL_SITE_TIERS = MappingProxyType({
    "propose_guideline_action": ModelTier.REASONING,
    "propose_investigation_strategy": ModelTier.REASONING,
    "propose_strategy_query": ModelTier.REASONING,
    "choose_research_source": ModelTier.FAST,
    "propose_page_child": ModelTier.FAST,
    "propose_table_query": ModelTier.REASONING,
    "extract_research_evidence": ModelTier.REASONING,
    "review_research_evidence": ModelTier.REASONING,
    "assess_assumption_reasonableness": ModelTier.REASONING,
    "review_scenario_scope": ModelTier.REASONING,
    "assess_scenario_model_defaults": ModelTier.REASONING,
    "analyze_scenario_parameter_scope": ModelTier.REASONING,
    "review_research_model": ModelTier.REASONING,
    "assess_source_event_types": ModelTier.REASONING,
    "synthesize_research_findings": ModelTier.REASONING,
})

# Nano-graphrag's table-page extraction is FAST while report and lexical
# reading extraction are REASONING. They share one operation name here, so the
# table reader declares this call-site tier when it invokes the provider.
TABLE_PAGE_EXTRACTION_TIER = ModelTier.FAST


def tier_for_operation(operation: str) -> ModelTier:
    return CALL_SITE_TIERS.get(operation, ModelTier.REASONING)


@dataclass(frozen=True)
class TierPolicy:
    reasoning_model: str
    fast_model: str
    reasoning_effort: str = "medium"
    fast_effort: str = "none"

    def model_for(self, tier: ModelTier) -> str:
        return self.fast_model if tier is ModelTier.FAST else self.reasoning_model

    def effort_for(self, tier: ModelTier) -> str:
        return self.fast_effort if tier is ModelTier.FAST else self.reasoning_effort

    def route(self, operation: str, tier: ModelTier | None = None) -> dict[str, str]:
        tier = tier or tier_for_operation(operation)
        return {"model_tier": tier.value, "model": self.model_for(tier),
                "reasoning_effort": self.effort_for(tier)}
