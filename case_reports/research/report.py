"""Direct scientific-sheet report; no report-composer model or extra episode."""
from dataclasses import dataclass
import json
from ...method_loop.episode import ClosedRecord
from .records import Ledger
from .model_tiers import ModelTier


@dataclass(frozen=True)
class ResearchReport(ClosedRecord):
    data: dict
    passed: bool

    @classmethod
    def from_result(cls, result, record, run):
        sheet = Ledger(result).causal_sheet
        rows = {key: {"kind": row["kind"], "data": row["data"], "current": sheet.current(key)}
                for key, row in sheet.rows.items()}
        data = {"research_request": run.research_request, "sheet": rows, "scores": sheet.calculate(),
                "investigation": run.global_goal.preview(),
                "execution": {"ended_by": record.ended_by, "reason": result.stop_reason,
                              "model_calls": run.model_calls, "observed_units": run.total_units},
                "diagnostics": list(result.diagnostics),
                "scientific_completion": "Not inferred from a numerical yield or an episode ending",
                "models": {tier.value: dict(run.providers.model_route("report", tier)) for tier in ModelTier}
                          if run.providers.model_route else {}}
        # passed describes successful execution, never a confirmed causal reversal.
        return cls(data, result.evaluated and not run.provider_unavailable and not any(
            d.get("stage") == "episode_completion" and d.get("status") == "failed" for d in result.diagnostics))

    def as_record(self):
        return {"passed": self.passed, **self.data}

    def markdown(self):
        return "# Causal inference assessment\n\n```json\n" + json.dumps(self.as_record(), indent=2) + "\n```"
