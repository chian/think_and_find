"""Local TypeSafe Jev boundary for search-candidate relevance measurements."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Mapping

from .._http import ProviderError, http_url, post_json

JEV_API_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-1.13.0"


@dataclass(frozen=True)
class JevDecisionClient:
    api_key: str = field(repr=False)
    endpoint: str = JEV_API_ENDPOINT
    model: str = JEV_MODEL
    timeout: float = 45.0

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValueError(
                "TYPESAFE_API_KEY is required for page relevance assessment"
            )
        object.__setattr__(
            self, "endpoint", http_url(self.endpoint, "Jev endpoint"),
        )
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Jev model must be a non-empty string")

    async def noul(
        self,
        *,
        state: Mapping[str, Any],
        instructions: str,
        criteria: Mapping[str, str],
    ) -> float:
        body = await asyncio.to_thread(
            post_json,
            self.endpoint,
            {
                "state": dict(state),
                "model": self.model,
                "questions": {
                    "decision": {
                        "type": "noul",
                        "instructions": instructions,
                        "criteria": dict(criteria),
                    },
                },
            },
            api_key=self.api_key,
            timeout=self.timeout,
        )
        answers = body.get("answers")
        answer = answers.get("decision") if isinstance(answers, Mapping) else None
        if not isinstance(answer, Mapping) or answer.get("type") != "noul":
            raise ProviderError("Jev response did not contain the requested Noul")
        probability = answer.get("noul")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not 0 <= probability <= 1
        ):
            raise ProviderError(
                "Jev Noul probability must be a number between zero and one"
            )
        return float(probability)

    async def nouls(
        self, *, state: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, float]:
        """Evaluate named Noul questions over one shared comparison state."""
        if not questions:
            raise ValueError("Jev Noul vector requires questions")
        declarations = {}
        for name, question in questions.items():
            instructions = question.get("instructions")
            criteria = question.get("criteria", {})
            if not isinstance(instructions, str) or not instructions.strip() or not isinstance(criteria, Mapping):
                raise ValueError("Jev Noul question requires instructions and criteria")
            declarations[name] = {"type": "noul", "instructions": instructions,
                                  "criteria": dict(criteria)}
        body = await asyncio.to_thread(
            post_json, self.endpoint,
            {"state": dict(state), "model": self.model, "questions": declarations},
            api_key=self.api_key, timeout=self.timeout,
        )
        answers = body.get("answers")
        if not isinstance(answers, Mapping):
            raise ProviderError("Jev response did not contain Noul answers")
        probabilities = {}
        for name in questions:
            answer = answers.get(name)
            probability = answer.get("noul") if isinstance(answer, Mapping) and answer.get("type") == "noul" else None
            if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                    or not 0 <= probability <= 1):
                raise ProviderError(f"Jev response did not contain a valid Noul for {name}")
            probabilities[name] = float(probability)
        return probabilities
