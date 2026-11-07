"""Think and Find's signed scientific-score binding to reusable Bayesian VOC.

Only Discovery and Scientific use this controller. The reusable forecasting
algorithm knows nothing about scientific sheets, episode ownership or stopping.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import math

from ...credit_assignment.causal_progress import compile_episode_points
from ...credit_assignment.research import ResearchStep
from ...credit_assignment.value_of_computation import Prediction, Transition, value_of_computation
from .records import Credit, Ledger, identity


FORMULA = "scientific_score_voc_v1"
ASSUMPTIONS = (
    "Exact-state observations take precedence. Otherwise transitions are exchangeable "
    "within an episode responsibility, available action set, admitted record-kind/currentness "
    "and selected-input-kind features, theory component/gap kinds, exact checked calculation "
    "outputs and nonzero score components. "
    "Transferred signed score changes are additive; a transfer predicting a negative total "
    "is unknown. Record identity, scientific content and candidate identity are not assumed "
    "equivalent scientific evidence. Structural pooling is a forecasting approximation. "
    "Observed-support-only categorical posterior prediction cannot rule out unseen outcomes."
)


@dataclass(frozen=True)
class ScientificState:
    key: str
    score: float
    features: tuple[str, ...]
    actions: tuple[str, ...]


class ScientificTransitionModel:
    """Observed-support categorical prediction, with explicit structural transfer.

    Hypothetical outcomes update only the current forecast's categorical counts.
    Actual state identities retain exact record versions; pooling never adds
    observations or scientific points. Separate responsibilities have separate models.
    """
    def __init__(self):
        self.events = {}

    def observe(self, event):
        old = self.events.get(event["event_id"])
        if old is not None:
            if old != event:
                raise ValueError("A completed VOC observation cannot be silently replaced")
            return False
        self.events[event["event_id"]] = deepcopy(event)
        return True

    def score(self, state):
        return state.score

    def actions(self, state):
        return state.actions

    @staticmethod
    def cell(state, action):
        return state.features, state.actions, action

    def predict(self, state, action, history=()):
        exact = [event for event in self.events.values()
                 if event["before"] == state and event["action"] == action]
        rows = exact or [event for event in self.events.values()
            if self.cell(event["before"], event["action"]) == self.cell(state, action)]
        counts = Counter()
        for event in rows:
            following = event["after"]
            score = state.score + event["score_change"]
            if score < 0 or not math.isfinite(score):
                return None
            # Preserve exact successors when possible, including checked findings.
            target = following if exact else ScientificState(
                "forecast:" + identity([score, following.features, following.actions]),
                score, following.features, following.actions)
            counts[target] += 1
        for imagined in history:
            same = (imagined.state == state if exact else
                    self.cell(imagined.state, imagined.action) == self.cell(state, action))
            if same and imagined.action == action:
                following = imagined.next_state
                score = state.score + following.score - imagined.state.score
                if score < 0 or not math.isfinite(score):
                    return None
                target = following if exact else ScientificState(
                    "forecast:" + identity([score, following.features, following.actions]),
                    score, following.features, following.actions)
                counts[target] += 1
        if not counts:
            return None
        total = sum(counts.values())
        return Prediction(tuple(Transition(count / total, following)
            for following, count in sorted(counts.items(), key=lambda item: repr(item[0]))),
            "observed_support_only_exact" if exact else "observed_support_only_structural_transfer",
            tuple(sorted(event["event_id"] for event in rows)))


def available_actions(source, sheet):
    if source.episode_grain == "discovery":
        revisit = bool(source.starting_questions) or any(row["kind"] != "evidence" for row in sheet.rows.values())
        return ("fresh_start", *(("revisit_existing",) if revisit else ()), "recall")
    from .sheet_tools import _sheet_tool_contract
    schema, _ = _sheet_tool_contract(source, source.record_kinds)
    return ("record", "inspect", "calculate", "reconstruct_theory", "reasoning",
            *(("evaluate",) if schema["properties"]["assessment"] is not False else ()), "recall")


def measured_state(source, result):
    sheet = Ledger(result).causal_sheet
    score = compile_episode_points(sheet)
    total = float(Fraction(score["total"]))
    if not math.isfinite(total):
        raise ValueError("The measured scientific score must be finite")
    records = tuple(sorted((key, row["version"], row["kind"]) for key, row in sheet.rows.items()))
    features = {"responsibility:" + source.episode_grain}
    features.update("record:" + row["kind"] + (":current" if sheet.current(key) else ":stale")
                    for key, row in sheet.rows.items())
    features.update("input:" + sheet.rows[key]["kind"]
                    for key in source.selected_keys if key in sheet.rows)
    for _, row in sheet.of_kind("theory_fragment"):
        fragment = row["data"]
        features.update("theory:" + group for group in ("entities", "quantities", "statements", "gaps")
                        if fragment[group])
        features.update("statement:" + item["kind"] for item in fragment["statements"].values())
    features.update("score:" + name for name, value in score["components"].items() if Fraction(value))
    selected = tuple((key, sheet.rows[key]["version"]) for key in source.selected_keys if key in sheet.rows)
    calculation = deepcopy(source.last_calculation)
    recalled = deepcopy(source.recalled_details)
    if calculation is not None:
        features.add("checked_calculation:" + identity(calculation))
    commission = source.commission.as_record() if source.commission else source.run.research_request
    starts = source.starting_questions if source.episode_grain == "discovery" else ()
    key = identity([source.episode_grain, commission, records, selected, source.scoped_snapshots(),
                    calculation, recalled, source.working_assessment, starts])
    state = ScientificState(key, total, tuple(sorted(features)), available_actions(source, sheet))
    return state, {"records": records, "selected": selected, "calculation": calculation,
                   "recalled": recalled, "components": score["components"]}


@dataclass(frozen=True)
class VOCState:
    stop: bool
    score: float
    history: tuple[float, ...]
    failures: int
    reason: str
    value_of_computation: dict
    rarefaction: None = None


class ScientificVOCController:
    """One actual completed own action per observation; VOC alone controls yield."""
    def __init__(self, source, path, model, *, lookahead=3, epoch="scientific-voc-v1"):
        self.source, self.path, self.model = source, path, model
        self.lookahead, self.epoch = lookahead, epoch
        self._current, self._metadata = measured_state(source, source.ledger.result())
        self._start, self._start_metadata = self._current, self._metadata
        self._observed_total = self._current.score
        self._origin = self._current.score if source.episode_grain == "scientific" else 0.0
        self.history, self.observations, self.failures = [self._current.score - self._origin], [], 0
        self.standing = source.episode_grain == "discovery" and source.run.limits.root_yield_threshold == 0
        self.threshold = ((source.run.limits.root_yield_threshold or 0.0)
                          if source.episode_grain == "discovery" else 0.0)

    def begin(self):
        self._current, self._metadata = measured_state(self.source, self.source.ledger.result())
        self._start, self._start_metadata = self._current, deepcopy(self._metadata)

    def feedback(self):
        report = value_of_computation(self.model, self._current, lookahead=self.lookahead)
        known = not report.unmodeled_decisions
        best = max((item.expected_gain for item in report.estimates
                    if item.expected_gain is not None), default=None)
        should_yield = known and best is not None and best <= self.threshold and not self.standing
        evidence = {event for estimate in report.estimates for event in estimate.evidence_ids}
        own = {event["event_id"] for event in self.observations}
        transferred_history = [deepcopy(self.model.events[event].get("history", {
            "event_id": event, "action": self.model.events[event]["action"],
            "score_change": self.model.events[event]["score_change"]}))
            for event in sorted(evidence - own)]
        return {"formula_version": FORMULA, "forecast": report.as_record(),
            "score_binding": "sum_of_five_scientific_components", "score_units": "scientific points",
            "scope": self.source.episode_grain, "standing_investigation": self.standing,
            "should_yield": should_yield, "yield_gain_threshold": self.threshold, "assumptions": ASSUMPTIONS,
            "credit_history": deepcopy(self.observations),
            "forecast_evidence": transferred_history,
            "instruction": "Use these conditional forecasts and measured signed outcomes to choose work. "
                "A zero-credit finding may enable a valuable follow-up. Unknown forecasts are not zero. "
                "Keep scientific scope and evidence qualifications; forecasts award no scientific credit. "
                "Lookahead describes the forecast, not a limit on episode execution."}

    def state(self):
        feedback = self.feedback()
        reason = ("standing_investigation" if self.standing else
                  "voc_no_gain_within_lookahead" if feedback["should_yield"] else
                  "voc_unknown" if feedback["forecast"]["unmodeled_decisions"] else "voc_improvement")
        return VOCState(feedback["should_yield"], self._current.score - self._origin,
                        tuple(self.history), self.failures, reason, feedback)

    def observe(self, unit_label, value, *, is_root):
        if not isinstance(value, Credit) or self.source.pending is None:
            raise TypeError("Scientific VOC requires a completed research unit and its pending sheet")
        following, metadata = measured_state(self.source, self.source.pending)
        change = following.score - self._observed_total
        if not math.isclose(self._start.score, self._observed_total, abs_tol=1e-12):
            raise ValueError("Shared refresh must not replace the parent's pre-action observation baseline")
        proposal = self.source.last_proposal if isinstance(self.source.last_proposal, dict) else {}
        action = proposal.get("action")
        if (self.source.episode_grain == "discovery" and action != "recall"
                and isinstance(proposal.get("question"), str)):
            revisited = proposal.get("inputs") or proposal.get("snapshots") or (
                proposal["question"] in self.source.starting_questions[:-1])
            action = "revisit_existing" if revisited else "fresh_start"
        action = action if isinstance(action, str) and action in self._start.actions else "proposal_failure"
        produced = sorted(set(metadata["records"]) - set(self._start_metadata["records"]))
        exposed = sorted(set(metadata["selected"]) - set(self._start_metadata["selected"]))
        checked = (["calculation:" + identity(metadata["calculation"])]
                   if metadata["calculation"] is not None and
                   metadata["calculation"] != self._start_metadata["calculation"] else [])
        provided_checks = (["calculation:" + identity(self._start_metadata["calculation"])]
                           if self._start_metadata["calculation"] is not None else [])
        consumed = set(self._start_metadata["selected"])
        sheet = Ledger(self.source.pending).causal_sheet
        def items(value):
            return value if isinstance(value, (tuple, list)) else ()
        for record in items(proposal.get("records")):
            if not isinstance(record, dict):
                continue
            consumed.update((key, sheet.rows[key]["version"]) for key in items(record.get("dependencies"))
                            if isinstance(key, str) and key in sheet.rows)
        for key in items(proposal.get("keys")):
            if isinstance(key, str) and key in sheet.rows:
                consumed.add((key, sheet.rows[key]["version"]))
        inputs = list(items(proposal.get("inputs")))
        for name in ("commission", "theory_commission"):
            commission = proposal.get(name)
            if isinstance(commission, dict):
                inputs.extend(items(commission.get("inputs")))
        for key in inputs:
            if isinstance(key, str) and key in sheet.rows:
                consumed.add((key, sheet.rows[key]["version"]))
        event_id = self.source.controller_key + ":" + str(len(self.observations) + 1)
        history = {"event_id": event_id, "action": action, "unit": unit_label,
            "score_before": self._observed_total, "score_after": following.score,
            "score_change": change, "evaluated": value.evaluated, "yield_observed": True,
            "components": metadata["components"], "produced_records": produced,
            "exposed_records": exposed, "checked_findings": checked,
            "provided_checked_findings": provided_checks,
            "consumed_records": sorted(consumed),
            "depends_on": [old["event_id"] for old in self.observations
                if (consumed & ({(key, version) for key, version, _ in old["produced_records"]}
                                | set(map(tuple, old["exposed_records"]))))
                or set(provided_checks) & set(old["checked_findings"])],
            "diagnostics": [{name: diagnostic.get(name) for name in ("stage", "category", "status", "why")}
                            for diagnostic in self.source.pending.diagnostics
                            if diagnostic not in self.source.ledger.diagnostics]}
        self.model.observe({"event_id": event_id, "before": self._start, "after": following,
            "action": action, "score_change": change, "history": history})
        self.observations.append(history)
        self.failures += int(not value.evaluated)
        self._current, self._metadata, self._observed_total = following, metadata, following.score
        self.history.append(following.score - self._origin)
        state = self.state()
        return ResearchStep(state.stop, state.score, change, value.evaluated, state.reason)

    def refresh(self, value):
        before = self._current.score
        self._current, self._metadata = measured_state(self.source, self.source.run.global_goal.ledger.result())
        return self._current.score - before

    def transitioned(self, epoch):
        return ScientificVOCController(self.source, self.path, self.model,
                                       lookahead=self.lookahead, epoch=epoch)
