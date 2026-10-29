"""Existing Episode composition with causal-sheet scientific state."""
from dataclasses import dataclass, replace
import hashlib
import math
import json
from pathlib import Path
from typing import Callable, Mapping

from ...credit_assignment.research import ResearchController
from ...method_loop import ChildEpisodeUnit, Episode, EpisodeGoal, EpisodeRequest, EpisodeTree, Grain, Leaf, SourceEnd
from ...method_loop.communication import ParentModelInputError
from ...task_executors._http import ProviderError
from ...task_executors.evidence_extraction.argo import ModelOutputError
from ...task_executors.outbreak_search.queries import search_results_from_provider, source_locator, source_text
from .communication import compact_reading_coverage
from .goal_state import GLOBAL_GOAL_PROGRESS, ScientificGoalState
from .model_tiers import ModelTier, tier_for_operation
from .records import Credit, Ledger, ResearchMessage, ResearchResult, diagnostic_value, identity, scenario_relation_key
from .causal_commission import SheetCommission
from .grading import GRADING_FIELDS

TREE = {
    "scientific": ("search_strategy", "reasoning"),
    "reasoning": (),
    "search_strategy": ("search",),
    "search": ("page",),
    "page": ("source_table", "report", "lexical_probe"),
    "source_table": (), "report": (), "lexical_probe": (),
}

@dataclass(frozen=True)
class ResearchLimits:
    search_limit: int = 8
    text_window_chars: int = 12000
    yield_window: int = 4
    root_yield_window: int = 12
    root_yield_threshold: float | None = None

    def __post_init__(self):
        if self.root_yield_threshold is not None and (isinstance(self.root_yield_threshold, bool) or
                not isinstance(self.root_yield_threshold, (int, float)) or
                not math.isfinite(self.root_yield_threshold) or self.root_yield_threshold < 0):
            raise ValueError("root_yield_threshold must be finite and nonnegative")
        for name in ("search_limit", "text_window_chars", "yield_window", "root_yield_window"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if min(self.yield_window, self.root_yield_window) < 2 or self.search_limit > 100 or self.text_window_chars < 100:
            raise ValueError("Invalid research control limits")

@dataclass(frozen=True)
class ResearchProviders:
    agent: Callable[[Mapping], Mapping]
    search: Callable
    retrieve: Callable
    event: Callable | None = None
    trace_event: Callable | None = None
    page_assessor: object | None = None
    agent_for_tier: Callable[[Mapping, ModelTier], Mapping] | None = None
    model_route: Callable[[str, ModelTier], Mapping[str, str]] | None = None
    relation_event: Callable | None = None
    source_target_assessor: Callable[[Mapping], Mapping[str, float]] | None = None
    scenario_parameter_assessor: Callable[[Mapping], Mapping] | None = None
    preflight_attempts: list[Mapping] | None = None
    checkpoint: object | None = None

def selected_choice(value, options, field):
    if not isinstance(value, str) or value not in options:
        raise ValueError(f"{field} must be one exact declared key {list(options)!r}; received {value!r}")
    return value

def changed_records(result, inherited):
    """Commit this episode's changes without replaying stale inherited claims."""
    keyed = {"assumptions": lambda row: row["key"],
             "scenarios": lambda row: row["subject"],
             "source_event_types": lambda row: row["type_id"],
             "source_target_relationships": lambda row: (
                 row["source"], row["source_subject"], row["target_subject"], row["parameter"]),
             "scenario_parameter_relations": scenario_relation_key,
             "facts": lambda fact: fact.key}
    changes = {}
    for name in ResearchResult.__dataclass_fields__:
        records = getattr(result, name)
        if not isinstance(records, tuple):
            continue
        key = keyed.get(name, identity)
        previous = {key(row): row for row in getattr(inherited, name)}
        changes[name] = tuple(row for row in records
                              if key(row) not in previous or previous[key(row)] != row)
    return replace(result, **changes)

class ResearchRun:
    def __init__(self, providers, limits, research_request, *, initial=None):
        self.providers, self.limits = providers, limits
        self.research_request = research_request
        self.initial = initial or ResearchResult()
        self.global_goal = ScientificGoalState(self.initial)
        self.documents = {}
        self.model_attempts = (providers.checkpoint.archive.sequence("model_attempts", providers.preflight_attempts or ())
                              if providers.checkpoint else list(providers.preflight_attempts or ()))
        self.provider_events = []
        self.provider_unavailable = None
        self.model_calls = len(self.model_attempts)
        self.total_units = self.sequence = self.diagnostic_sequence = 0
        self.controller_specs, self.live_controllers, self.sources = {}, {}, {}
        self.root_source = self.root_key = None
        root = Path(__file__).resolve().parents[2]
        paths = list(root.glob("*.py"))
        for folder in ("case_reports", "credit_assignment", "method_loop", "task_executors"):
            paths.extend((root / folder).rglob("*.py"))
        self.implementation = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(paths)}
        if providers.checkpoint:
            self.implementation = providers.checkpoint.run_start_implementation(self.implementation)
        self.context = {"research_request": research_request}
        units = {
            "scientific": "a completed search strategy, reasoning commission, or scientific sheet action",
            "reasoning": "a completed derivation, check, comparison or calculation for the assigned logical question",
            "search_strategy": "a completed search for one proposed query",
            "search": "a completed source investigation from this query's results",
            "page": "a completed selected reading episode",
            "source_table": "one table query and source-grounded extraction",
            "report": "one document-order window and source-grounded extraction",
            "lexical_probe": "one ranked source window and source-grounded extraction",
        }
        self.grains = {name: Grain(name, unit, "scoped findings and causal-sheet state", self.controller)
                       for name, unit in units.items()}
        self.tree = EpisodeTree(self.grains["scientific"], {
            self.grains[parent]: tuple(self.grains[name] for name in children)
            for parent, children in TREE.items() if children})
        self.emit_trace("run_started", research_request=research_request,
                        implementation_sha256=self.implementation, episode_tree=TREE)

    def controller(self, path):
        source = self.controller_specs[path[-1][1]]
        controller = ResearchController(source.credit_columns, window=self.limits.yield_window,
            root_window=self.limits.root_yield_window, initial=source.credit(source.ledger.result()), point_scores=True,
            yield_threshold=self.limits.root_yield_threshold if len(path) == 1 else None)
        self.live_controllers[path] = controller
        return controller

    def root_credit_projection(self, status=None):
        return (status or self.global_goal.preview())["investigation_score"]

    def refresh_controllers(self, source, status):
        current_path = next((path for path in self.live_controllers if path[-1][1] == source.controller_key), None)
        if current_path is None:
            return
        for path, controller in self.live_controllers.items():
            if current_path[:len(path)] == path:
                owner = self.controller_specs[path[-1][1]]
                gain = controller.refresh(owner.credit(self.global_goal.ledger.result()))
                if gain:
                    self.emit_trace("shared_credit_refreshed", episode_path=[list(item) for item in path],
                                    gain=gain, score=controller.state().score,
                                    meaning="Shared state refresh; no completed-unit observation added")

    def episode(self, grain, source, *, goal, focus=(), contract=None, proposal=None):
        self.sequence += 1
        key = f"{grain}:{self.sequence}"
        source.controller_key, source.episode_grain = key, grain
        source.focus = tuple(focus)
        source.goal, source.parent_contract = goal, contract
        source.actor = "reasoner:" + identity(goal.goal_id)[:16]
        if contract is not None:
            source.commission = source.commission_type(**contract.as_record()["measurements"]["scope"])
            source.selected_keys = list(source.commission.inputs)
        self.sources[goal.goal_id] = source
        self.controller_specs[key] = source
        if grain == "scientific":
            self.root_source, self.root_key = source, key
        episode = Episode(self.grains[grain], key, source, EpisodeRequest(
            goal, ResearchMessage(str(goal.objective), (), diagnostic_value(proposal or {})), contract),
            source.build_result, on_unit=source.after_verdict)
        if self.providers.checkpoint:
            episode = self.providers.checkpoint.instrument(episode, lambda: self.global_goal.ledger.result().as_record())
        return episode

    def child(self, parent, source, grain, objective, *, contract, proposal):
        goal = EpisodeGoal.child(parent.goal, objective=objective, result_contract={"information": contract.as_record()})
        return self.episode(grain, source, goal=goal, contract=contract, proposal=proposal)

    def emit_trace(self, kind, **payload):
        if self.providers.trace_event:
            self.providers.trace_event(diagnostic_value({"event": kind, **payload}))

    def search(self, query, limit):
        event = {"operation": "search", "query": query, "limit": limit, "status": "started"}
        self.provider_events.append(event)
        try:
            results = search_results_from_provider(self.providers.search(query, limit))
        except (ProviderError, OSError, ValueError) as exc:
            event.update(status="failed", why=str(exc))
            self.emit_trace("provider_event", provider_event=event)
            raise
        event.update(status="completed", sources=[source_locator(row) for row in results])
        self.emit_trace("provider_event", provider_event=event)
        return results

    def retrieve(self, candidate):
        event = {"operation": "retrieve", "source": source_locator(candidate), "status": "started"}
        self.provider_events.append(event)
        try:
            result = self.providers.retrieve(candidate)
            body = result if isinstance(result, str) else source_text(result or {})
            if not isinstance(body, str) or not body.strip():
                raise ValueError("Retrieval returned no source text")
        except (ProviderError, OSError, ValueError) as exc:
            event.update(status="failed", why=str(exc))
            self.emit_trace("provider_event", provider_event=event)
            raise
        event.update(status="completed", characters=len(body), sha256_utf8=hashlib.sha256(body.encode()).hexdigest())
        self.emit_trace("provider_event", provider_event=event)
        return result

    def agent(self, request, *, completing=False, tier=None):
        if self.provider_unavailable is not None:
            exc = ProviderError(self.provider_unavailable["why"])
            exc.model_call_diagnostic = dict(self.provider_unavailable)
            raise exc
        self.model_calls += 1
        operation = request.get("operation", "")
        tier = tier or tier_for_operation(operation)
        route = dict(self.providers.model_route(operation, tier)) if self.providers.model_route else {}
        attempt = {"call": self.model_calls, "operation": request.get("operation", ""),
                   "phase": "mandatory_completion" if completing else "work",
                   "model_tier": tier.value, **route,
                   "request": diagnostic_value(request)}
        try:
            result = (self.providers.agent_for_tier(request, tier)
                      if self.providers.agent_for_tier else self.providers.agent(request))
            attempt["response"] = diagnostic_value(result)
            if not isinstance(result, Mapping):
                raise ValueError("Research agent must return a JSON object")
        except (ProviderError, OSError, ValueError) as exc:
            diagnostic = {**self.model_diagnostic(exc), "model_call": self.model_calls,
                          "operation": attempt["operation"]}
            exc.model_call_diagnostic = diagnostic
            attempt.update(status="response_failed", diagnostic=diagnostic)
            if diagnostic.get("category") == "provider_usage_limit" or any(
                    term in str(exc).lower() for term in ("http 401", "http 403", "unauthorized", "token has expired")):
                self.provider_unavailable = dict(diagnostic)
            if isinstance(exc, ModelOutputError):
                exc.diagnostic.update(model_call=self.model_calls, operation=attempt["operation"])
                attempt["raw_response"] = diagnostic_value(exc.raw_response)
            self.emit_trace("model_attempt", attempt=attempt)
            raise
        else:
            attempt["status"] = "json_object_received_not_yet_admitted"
        finally:
            # Archive only the completed attempt: the provider response and
            # failure status are filled in after the request is dispatched.
            self.model_attempts.append(attempt)
        self.emit_trace("model_attempt", attempt=attempt)
        return result

    def complete(self, request, *, tier=None):
        """Record mandatory processing as a model attempt."""
        return self.agent(request, completing=True, tier=tier)

    def model_diagnostic(self, exc):
        if hasattr(exc, "model_call_diagnostic"):
            return dict(exc.model_call_diagnostic)
        details = dict(exc.diagnostic) if isinstance(exc, ModelOutputError) else {
            "category": "provider_failure" if isinstance(exc, (ProviderError, OSError)) else "invalid_action_contract",
            "why": str(exc)}
        return details

    def identify_diagnostic(self, diagnostic):
        if "diagnostic_id" in diagnostic:
            self.diagnostic_sequence = max(self.diagnostic_sequence, diagnostic["diagnostic_id"])
            return diagnostic
        self.diagnostic_sequence += 1
        return {**diagnostic, "diagnostic_id": self.diagnostic_sequence}

class ResearchSource:
    def __init__(self, run, inherited=()):
        self.run, self.ledger = run, Ledger(inherited)
        self.initial = self.ledger.result()
        self.focus = ()
        self.goal = self.parent_contract = self.commission = None
        self.requested_subjects = ()
        self.pending = self.pending_global = None
        self.controller_key = self.episode_grain = None
        self.failure = ""
        self.completion_diagnostic = None
        self.evaluated_units = 0
        self.last_proposal = None
        self.selected_keys = []
        self.selected_snapshots = []
        self.catalog_offset = 0
        self.last_calculation = None

    def scoped_snapshots(self):
        refs = [*(self.commission.snapshots if self.commission else ()), *self.selected_snapshots]
        return list({(ref["key"], ref["version"]): ref for ref in refs}.values())

    def after_verdict(self, unit, contribution, view):
        self.run.total_units += 1
        if self.pending is not None:
            changes = changed_records(self.pending, self.ledger.result())
            self.ledger.merge(self.pending)
            if changes.causal_sheet_updates:
                self.selected_keys = list(dict.fromkeys(r["key"] for r in changes.causal_sheet_updates))[-2:]
            status = self.run.global_goal.commit(self.pending_global or self.pending)
            self.run.refresh_controllers(self, status)
            self.evaluated_units += int(self.pending.evaluated)
            event = dict(unit=unit.label, evaluated=self.pending.evaluated,
                operation_status="completed" if self.pending.evaluated else "failed",
                gain=view.controller_step.gain, score=view.controller_step.score,
                stop=view.controller_step.stop, model_calls=self.run.model_calls,
                global_goal_score=status["investigation_score"], scientific_calculations=status["scientific_calculations"],
                root_progress_binding=status["root_progress_binding"],
                diagnostics=list(self.pending.diagnostics[-3:]))
            self.run.emit_trace("credit_verdict", episode_id=self.goal.goal_id,
                episode_type=type(self).__name__, question=str(self.goal.objective),
                contract=self.parent_contract.as_record() if self.parent_contract else None, **event)
            if self.run.providers.event:
                self.run.providers.event(event)
            self.pending = self.pending_global = None
            update = contribution.episode_update
            if update is not None:
                closed = self.run.sources.pop(update.goal.goal_id, None)
                if closed is not None:
                    self.run.controller_specs.pop(closed.controller_key, None)
                    for path in tuple(self.run.live_controllers):
                        if path[-1][1] == closed.controller_key:
                            del self.run.live_controllers[path]

    def assess(self, result):
        return result

    def project(self, result):
        result = self.assess(result)
        committed = self.run.global_goal.ledger.result()
        working = Ledger(committed)
        working.merge(changed_records(result, self.ledger.result()))
        self.pending = working.result(evaluated=result.evaluated, stop_reason=result.stop_reason)
        self.pending_global = changed_records(self.pending, committed)
        return self.credit(self.pending)

    def credit(self, result):
        raise NotImplementedError("Each episode owns its credit projection")

    def parent_report(self, result, completion=None, request=None):
        baseline = self.ledger.result() if request is not None else self.initial
        delta = changed_records(result, baseline)
        sheet = Ledger(baseline)
        sheet.merge(delta)
        return {"completion": {"ended_by": getattr(completion, "ended_by", result.stop_reason) or "in_progress",
                               "evaluated": result.evaluated},
                "causal_sheet_changes": sheet.causal_sheet.return_page(delta.causal_sheet_updates,
                    calculation_context=getattr(self.commission, "calculation_context", None)),
                "reading_coverage": compact_reading_coverage(delta.reading_attempts),
                "operations": [{k: row.get(k) for k in ("stage", "category", "status", "why")}
                               for row in delta.diagnostics[-3:]]}

    def context(self):
        sheet = self.ledger.causal_sheet
        return {"question": self.commission.question if self.commission else self.run.research_request,
                "scientific_inputs": sheet.view(self.commission.inputs) if self.commission else [],
                "historical_inputs": sheet.snapshot_view(self.scoped_snapshots())}

    def inputs(self, view, **kwargs):
        sheet = self.ledger.causal_sheet
        keys = list(dict.fromkeys([*(self.commission.inputs if self.commission else ()),
            *([subject for subject, _ in getattr(self.commission, "assignments", ())] if self.commission else []), *self.selected_keys]))
        if self.commission and getattr(self.commission, "calculation_context", ""):
            keys = list(dict.fromkeys([*keys, getattr(self.commission, "calculation_context", "")]))
        keys = list(dict.fromkeys([*keys, *sheet.inferences_for(keys)]))
        selected = sheet.view(keys)
        historical = sheet.snapshot_view(self.scoped_snapshots())
        rows = self.catalog_rows()
        page = rows[self.catalog_offset:self.catalog_offset + 20]
        catalog = [{"key": key, "kind": row["kind"], "current": sheet.current(key),
                    "label": str(row["data"].get("name", row["data"].get("statement", row["data"].get("finding",
                        row["data"].get("bounds", row["data"].get("conflict", row["data"].get("argument", "")))))))[:160]}
                   for key, row in page]
        status = self.run.global_goal.preview()
        assessment_keys = []
        for key in keys:
            row = sheet.rows[key]
            assessment = key if row["kind"] == "importance" else row["data"].get("assessment") if row["kind"] == "importance_result" else None
            if row["kind"] in {"expectation", "expectation_support", "support_contribution", "causal_change", *GRADING_FIELDS}:
                assessment = key
            if assessment and assessment not in assessment_keys:
                assessment_keys.append(assessment)
        selected_calculations = sheet.calculation_view(assessment_keys[:8],
            context=getattr(self.commission, "calculation_context", None)) if assessment_keys else {}
        declared = {"task": self.instructions, "objective": str(view.goal.objective),
            "commission": self.commission.as_record() if self.commission else None,
            "selected_sheet_inputs": selected, "historical_sheet_inputs": historical, "catalog": catalog,
            "catalog_page": {"offset": self.catalog_offset, "total": len(rows)},
            "scores": {"selected_calculations": selected_calculations,
                       "assessment_count": len(status["scientific_calculations"]["assessments"]),
                       "root_progress_binding": status["root_progress_binding"],
                       "global_credit": status["scientific_calculations"]["support_credit"]["total"],
                       "policy_details": status["scientific_calculations"]["support_credit"]["policy_details"],
                       "controller_limitation": status["controller_limitation"],
                       "meaning": status["meaning"]},
            "last_calculation": self.last_calculation,
            "own_results": self.parent_report(self.ledger.result()), **kwargs}
        return self.bounded_inputs(view, declared)

    def bounded_inputs(self, view, declared):
        # Only the latest direct-child report is steering.
        inputs = view.model_inputs(declared, reports=view.child_reports[-1:])
        size = len(json.dumps(inputs, ensure_ascii=False).encode())
        if size > 65000:
            raise ValueError(f"Scoped model input is {size} bytes; narrow selected sheet records or reading scope")
        return inputs

    def build_result(self, record):
        reason = self.failure or (record.controller_state.reason if record.ended_by == "yield_stop" else record.ended_by)
        if self.failure and self.completion_diagnostic is None:
            self.completion_diagnostic = self.run.identify_diagnostic({
                "stage": "episode_completion", "status": "failed",
                "category": "provider_usage_limit" if "provider_usage_limit" in reason else "source_termination",
                "why": reason,
                "requested_fields": list(self.focus),
                "question": str(self.goal.objective),
                "completion": {"ended_by": record.ended_by, "reason": reason,
                               "units_consumed": record.units_consumed,
                               "evaluated_units": self.evaluated_units},
            })
            self.ledger.diagnostics.append(self.completion_diagnostic)
        evaluated = bool(self.evaluated_units) or (
            record.units_consumed == 0 and not self.failure and record.ended_by == "exhausted")
        return self.ledger.result(evaluated=evaluated, stop_reason=reason)

    def receive(self, result, completion, request):
        if not isinstance(result, ResearchResult):
            raise TypeError("Unexpected scientific child result")
        return self.project(result)

    def edge(self, child):
        return ChildEpisodeUnit(child, self.receive, child.source.parent_report)

    def failure_result(self, exc, stage):
        return ResearchResult(evaluated=False, diagnostics=({
            "stage": stage, "status": "failed", **self.run.model_diagnostic(exc),
            "requested_fields": list(self.focus),
            "question": self.parent_contract.decision if self.parent_contract else str(self.goal.objective) if self.goal else "",
            "proposal": diagnostic_value(self.last_proposal),
            **({"extraction_state": exc.extraction_state,
                "unreviewed_candidates": exc.unreviewed_candidates}
               if hasattr(exc, "extraction_state") else {})},))

    def leaf(self, function, label):
        def execute(_):
            try:
                return function()
            except (ProviderError, ValueError, TypeError, KeyError,
                    ZeroDivisionError, OverflowError, OSError) as exc:
                return self.failure_result(exc, label)
        return Leaf(None, execute, lambda _, result: self.project(result), label)

    def next(self, view):
        if self.run.provider_unavailable is not None:
            self.failure = "provider_usage_limit_not_scientific_convergence"
            return SourceEnd("source_failed")
        if self.failure:
            return SourceEnd("source_failed")
        before_calls = self.run.model_calls
        try:
            # A parent may have been suspended while a child revised a premise.
            # Its next calculation must use the current main-table inputs.
            # Distinct hypothesis keys remain
            # alternatives; only matching record identities receive revisions.
            self.ledger.merge(self.run.global_goal.ledger.result())
            self.last_proposal = None
            return self.propose(view)
        except (ProviderError, ValueError, TypeError, KeyError, OSError) as exc:
            if (isinstance(exc, ParentModelInputError) or
                    (isinstance(exc, (ValueError, TypeError, KeyError))
                     and not isinstance(exc, (ProviderError, OSError)) and self.run.model_calls == before_calls)):
                # A malformed model response can be revised on a later choice.
                # A request-construction failure has no new model response and
                # cannot be repaired by recording another zero-gain unit.
                self.failure = "invalid_parent_model_input"
                failure = self.failure_result(exc, "parent_model_input")
                self.ledger.merge(replace(failure, diagnostics=tuple(
                    self.run.identify_diagnostic(d) for d in failure.diagnostics)))
                return SourceEnd("source_failed")
            result = self.failure_result(exc, "proposal")
            return self.leaf(lambda: result, f"proposal_failure:{view.units_consumed}")
