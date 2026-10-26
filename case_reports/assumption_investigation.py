"""An additive justification branch for the existing evidence-admission parents.

This binding does not select scenarios, calibrate infectiousness, or calculate
risk. Its parent owns the fields and context to be justified.
"""

from dataclasses import dataclass, replace
import json
import math
from collections.abc import Mapping

from ..credit_assignment.research import ResearchController
from ..method_loop import (
    ChildEpisodeUnit, Episode, EpisodeGoal, EpisodeRequest, Grain, Leaf,
    ReportContract, SourceEnd,
)
from ..method_loop.communication import ParentModelInputError
from .research.assumptions import RUBRIC, register_candidate
from .research.progress import ADDITIONAL_SUPPORT, parameter_support
from .research.records import Credit, Ledger, ResearchMessage, ResearchResult
from ..task_executors.scenario_metrics.evaluate import CONDITIONS, UNITS


@dataclass(frozen=True)
class AssumptionLimits:
    yield_window: int = 4
    text_window_chars: int = 12000

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.yield_window < 2 or self.text_window_chars < 100:
            raise ValueError("Invalid assumption investigation limits")


@dataclass(frozen=True)
class AssumptionServices:
    search: object = None
    retrieve: object = None
    search_limit: int = 8
    limits: AssumptionLimits | None = None


def latest_assessments(result):
    """Use the current judgment per field, including withdrawals of support."""
    return {a["field"]: a for a in result.assumptions}


def assumption_credit(result, fields):
    points = {row["field"]: row["points"]
              for row in parameter_support(result, fields, hypotheses_only=True)}
    return Credit(evaluated=result.evaluated, qualities=tuple(
        (field, "parameter_learning", points.get(field, 0))
        for field in fields
    ), quality_resolution=ADDITIONAL_SUPPORT)


def assumption_summary(result, completion, request, *, evidence_options=None):
    """Project existing records into the closed legacy child-return contract."""
    def number(value, *, nullable=False):
        if value is None and nullable:
            return None
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Assumption return requires a finite numeric value")
        return value

    latest = latest_assessments(result)
    fields = set(UNITS) | set(CONDITIONS)
    verdicts = {"unassessed", "supported", "provisional", "unsupported", "contradicted", "fixed_condition"}
    assumptions = []
    for field, assessment in latest.items():
        if field not in fields:
            continue
        review = assessment.get("review", {})
        verdict = review.get("verdict", "unassessed")
        if not isinstance(verdict, str) or verdict not in verdicts:
            raise ValueError("Assumption return requires a declared review verdict")
        completed = review.get("completed", verdict != "unassessed")
        if type(completed) is not bool:
            raise ValueError("Assumption return review completion must be boolean")
        criteria = {}
        for name in RUBRIC:
            if name not in review.get("criteria", {}):
                continue
            grade = review["criteria"][name]["grade"]
            if type(grade) is not int or grade not in (0, 1, 2):
                raise ValueError("Assumption return criterion grades must be 0, 1 or 2")
            criteria[name] = grade
        assumptions.append({
            "field": field, "value": number(assessment["value"], nullable=True), "unit": UNITS.get(field),
            "verdict": verdict, "reasonableness": number(assessment["reasonableness"]),
            "review_completed": completed, "criteria": criteria,
        })
    failures = []
    for diagnostic in result.diagnostics:
        if diagnostic.get("status") != "failed":
            continue
        category = diagnostic.get("category")
        failures.append({
            "operation": ("parent_model_input" if diagnostic.get("stage") == "parent_model_input"
                          else "proposal" if diagnostic.get("stage") == "proposal" else "execution"),
            "status": "failed",
            "category": category if isinstance(category, str) and category in {
                "invalid_action_contract", "provider_failure", "malformed_json", "duplicate_json_key",
                "child_identity_in_parent_input",
            } else "operation_failed",
        })
    ended_by = completion.ended_by if completion is not None else "in_progress"
    if not isinstance(ended_by, str) or ended_by not in {
        "exhausted", "yield_stop", "incomplete", "source_failed", "in_progress",
    }:
        raise ValueError("Assumption return requires a declared completion code")
    if type(result.evaluated) is not bool:
        raise ValueError("Assumption return operation completion must be boolean")
    report = {
        "assumptions": assumptions,
        "numeric_evidence": [{"field": fact.field, "value": number(fact.value),
            "unit": UNITS[fact.field], "kind": fact.kind}
            for fact in result.facts if fact.field in UNITS
            and fact.kind in {"source_observation", "source_model_parameter"}],
        "evidence_fields": sorted({field for evidence in result.evidence
                                   for field in evidence.get("fields", ()) if field in fields}),
        "operation_status": "completed" if result.evaluated else "failed",
        "failed_operations": failures,
        "completion": ended_by,
    }
    if evidence_options is not None:
        from .research.communication import decision_records
        catalog = decision_records(Ledger(result), target_subjects=(
            row["subject"] for row in result.scenarios if row.get("role", "target") == "target"))["evidence"]
        if list(catalog) != list(evidence_options):
            raise ValueError("Evidence catalog must preserve the ledger's declared references")
        report["evidence_options"] = catalog
    return report


def branch_contract(fields, context):
    return ReportContract(
        decision=f"Decide whether {', '.join(fields)} can be admitted for this parent's defined context: {context}",
        measurements={"credit": "Reviewed justification, not populated fields or source counts."},
        information={
            "assumptions": "Rows: canonical field, numeric value or null, canonical unit or null, verdict enum (unassessed/supported/provisional/unsupported/contradicted/fixed_condition), numeric reasonableness, review_completed boolean, and numeric rubric criterion grades. No claim or explanatory prose.",
            "numeric_evidence": "Rows: canonical field, numeric value, canonical unit, and source kind enum (source_observation/source_model_parameter). Source text and provenance remain in scientific records.",
            "evidence_fields": "Canonical field names addressed by recorded qualitative evidence; presence does not establish support.",
            "operation_status": "Enum completed/failed for the returned operation; scientific support is separate.",
            "failed_operations": "Rows: operation enum parent_model_input/proposal/execution, status failed, and category enum child_identity_in_parent_input/invalid_action_contract/provider_failure/malformed_json/duplicate_json_key/operation_failed. child_identity_in_parent_input means request construction failed before model dispatch. Detailed failures remain in audit records.",
            "completion": "Method completion enum exhausted/yield_stop/incomplete/source_failed; in_progress only for the local state projection.",
        },
    )


class AssumptionInvestigation:
    """Compose just the branch; the existing admission episode remains its parent."""

    def __init__(self, *, fields, context, llm, sources=(), search=None,
                 retrieve=None, search_limit=8, limits=None):
        self.fields = tuple(fields)
        self.context = dict(context)
        self.llm, self.search, self.retrieve = llm, search, retrieve
        self.sources, self.search_limit = tuple(sources), search_limit
        self.limits = limits or AssumptionLimits()
        self.calls = self.sequence = 0
        self.record = None
        self.documents = {}
        self.controllers = {}
        self.subject = "admission_target"
        self.grain = Grain("assumption_investigation", "evidence approach or assumption candidate",
                           "justified assumptions for the admission parent", self.controller)
        self.evidence_grain = Grain("assumption_evidence", "focused source reading",
                                    "admitted supporting or conflicting evidence", self.controller)
        self.assessment_grain = Grain("assumption_assessment", "candidate review",
                                      "reasonableness assessment", self.controller)

    @property
    def children(self):
        return {self.grain: (self.evidence_grain, self.assessment_grain)}

    def agent(self, request):
        if self.llm is None:
            raise ValueError("Assumption investigation requires an LLM provider")
        self.calls += 1
        result = self.llm(request)
        if isinstance(result, str):
            result = json.loads(result)
        if not isinstance(result, Mapping):
            raise ValueError("Assumption model response must be an object")
        return result

    def controller(self, path):
        scientific, initial = self.controllers[path[-1][1]]
        return ResearchController(self.fields, initial=initial, capacity=2,
                                  window=self.limits.yield_window,
                                  epoch="assumption_justification_v1", point_scores=scientific)

    def episode(self, parent, source, grain, objective, contract):
        self.sequence += 1
        key = f"{grain.name}:{self.sequence}"
        self.controllers[key] = (source.scientific, source.credit(source.ledger.result()))
        goal = EpisodeGoal.child(parent.goal, objective=objective,
                                 result_contract={"fields": list(self.fields)})
        return Episode(grain, key, source,
                       EpisodeRequest(goal, ResearchMessage(objective, (self.subject,)), contract),
                       source.build_result, on_unit=source.after_verdict)

    def open(self, parent, contract):
        initial = ResearchResult(scenarios=({
            "subject": self.subject, "description": str(self.context),
            "basis": "Scope supplied by the existing admission parent; not a new scenario.",
        },))
        source = InvestigationSource(self, initial)
        episode = self.episode(parent, source, self.grain,
                               f"Justify {', '.join(self.fields)} for the admission parent's context.", contract)
        return replace(episode, on_close=lambda record: setattr(self, "record", record))


class AssumptionSource:
    """Shared method_loop plumbing, without shared agent capabilities."""

    scientific = True

    def __init__(self, branch, inherited):
        self.branch, self.ledger = branch, Ledger(inherited)
        self.pending = None
        self.failure = ""
        self.evaluated = 0

    def credit(self, result):
        if self.scientific:
            return assumption_credit(result, self.branch.fields)
        return result.credit(self.branch.fields)

    def receive(self, result, completion=None, request=None):
        self.pending = result
        accumulated = Ledger(self.ledger.result())
        for assessment in result.assumptions:
            accumulated.assumptions.pop(assessment["key"], None)
        accumulated.merge(result)
        return self.credit(accumulated.result(evaluated=result.evaluated))

    def after_verdict(self, item, contribution, view):
        if self.pending is not None:
            # Reassessment order matters, even when a previously seen claim returns.
            for assessment in self.pending.assumptions:
                self.ledger.assumptions.pop(assessment["key"], None)
            self.ledger.merge(self.pending)
            self.evaluated += int(self.pending.evaluated)
            self.pending = None

    def build_result(self, record):
        reason = self.failure or (record.controller_state.reason if record.ended_by == "yield_stop" else record.ended_by)
        return self.ledger.result(evaluated=bool(self.evaluated), stop_reason=reason)

    def inputs(self, view, **kwargs):
        scientific_state = assumption_summary(self.ledger.result(), None, None,
                                             evidence_options=self.ledger.evidence_options())
        return view.model_inputs({
            "objective": view.goal.as_record()["objective"],
            "context": self.branch.context, "fields": list(self.branch.fields),
            "subject": self.branch.subject,
            "evidence_options": scientific_state.pop("evidence_options"),
            "evidence_options_contract": {
                "references": "evidence:N keys identify the full records supplied to the independent reviewer when selected.",
                "fields": "Canonical quantities addressed by the record; a relevance tag does not establish a value or support.",
                "value_and_unit": "Finite scalar in canonical units, or null for qualitative evidence.",
                "kind_and_role": "Declared source provenance and evidence role; observed_fields names actual observed quantities.",
                "source_and_subject_references": "Local references distinguish sources and their event or population subjects.",
                "metadata_presence": "has_quote and has_context indicate retained metadata; presence does not establish support.",
                "numeric_constraints": "Typed intervals and bounds in canonical units; a range does not automatically supply a selected scalar.",
            },
            "scientific_state": scientific_state,
            "credit_history": list(view.controller_state.history), **kwargs,
        }, reports=view.child_reports)

    def leaf(self, function, label):
        def execute(_):
            try:
                return function()
            except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
                return ResearchResult(evaluated=False, diagnostics=({
                    "stage": label, "status": "failed", "category": "operation_failed",
                    "why": f"{type(exc).__name__}: {exc}",
                },))
        return Leaf(None, execute, lambda _, result: self.receive(result), label)

    def next(self, view):
        if self.failure:
            return SourceEnd("source_failed")
        before_calls = self.branch.calls
        try:
            return self.propose(view)
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            if (isinstance(exc, ParentModelInputError) or
                    (isinstance(exc, (ValueError, TypeError, KeyError)) and self.branch.calls == before_calls)):
                self.failure = "invalid_parent_model_input"
                self.ledger.merge(ResearchResult(evaluated=False, diagnostics=({
                    "stage": "parent_model_input", "status": "failed",
                    "category": ("child_identity_in_parent_input" if isinstance(exc, ParentModelInputError)
                                 else "invalid_action_contract"), "why": str(exc),
                },)))
                return SourceEnd("source_failed")
            result = ResearchResult(evaluated=False, diagnostics=({
                "stage": "proposal", "status": "failed", "category": "invalid_action_contract",
                "why": f"{type(exc).__name__}: {exc}",
            },))
            return self.leaf(lambda: result, "invalid_proposal")


class InvestigationSource(AssumptionSource):
    """Choose evidence approaches and candidate assumptions, not calculations."""

    def propose(self, view):
        from .assumption_evidence import EvidenceSource

        proposal = self.branch.agent(self.inputs(
            view, operation="propose_assumption_investigation", rubric=RUBRIC,
            response_contract={
                "action": "gather_evidence or assess_assumption; no stop action",
                "objective": "specific uncertainty to resolve within the parent's requested fields",
                "query": "for gather_evidence: a new search query, or empty to examine supplied sources",
                "field": "for assess_assumption: one requested field",
                "value": "finite proposed number", "unit": "canonical unit",
                "statement": "precise conditional claim", "basis": "reasoned inference",
                "evidence": ["admitted evidence option keys, including contrary evidence"],
            },
            instruction=(
                "Resolve the parent's assumption question, using structured child reports, scientific state and numerical yield history. "
                "You may revise the evidence strategy or propose and reassess a candidate. "
                "An inferred value need not be printed in a source, but requires evidence and a bounded inference. "
                "Do not invent evidence, redefine the scenario, calculate risk or prescribe an acceptable risk budget. "
                "Do not override explicit target constraints. Only the numerical controller decides stopping."
            ),
        ))
        objective = proposal.get("objective")
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("An assumption child needs a scoped objective")
        if proposal.get("action") == "gather_evidence":
            query = proposal.get("query", "")
            if not isinstance(query, str):
                raise ValueError("Evidence query must be text")
            source = EvidenceSource(self.branch, self.ledger.result(), query=query, question=objective)
            grain = self.branch.evidence_grain
            contract = branch_contract(self.branch.fields, self.branch.context)
        elif proposal.get("action") == "assess_assumption":
            if proposal.get("field") not in self.branch.fields:
                raise ValueError("Candidate field is outside the admission parent's scope")
            constraints = self.branch.context.get("target_scenario", {})
            if proposal["field"] in constraints and proposal.get("value") != constraints[proposal["field"]]:
                raise ValueError("An assumption cannot override an explicit target constraint")
            proposal = {**proposal, "subject": self.branch.subject}
            source = AssessmentSource(self.branch, self.ledger.result(), proposal)
            grain = self.branch.assessment_grain
            contract = branch_contract((proposal["field"],), self.branch.context)
        else:
            raise ValueError("Unknown assumption action; the agent cannot stop the episode")
        child = self.branch.episode(view, source, grain, objective, contract)
        return ChildEpisodeUnit(child, self.receive, assumption_summary)


class AssessmentSource(AssumptionSource):
    """One independently reviewed candidate; no searching or proposal capability."""

    def __init__(self, branch, inherited, proposal):
        super().__init__(branch, inherited)
        self.proposal = proposal

    def propose(self, view):
        if view.units_consumed:
            return None
        return self.leaf(lambda: register_candidate(
            self.proposal, self.ledger,
        ), "register_parameter")
