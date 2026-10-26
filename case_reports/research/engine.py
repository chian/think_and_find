"""Composition and mandatory lifecycle shared by the approved episode bindings."""

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Callable, Mapping

from ...credit_assignment.research import ResearchController
from ...method_loop import (
    ChildEpisodeUnit, Episode, EpisodeGoal, EpisodeRequest, EpisodeTree,
    Grain, Leaf, ReportContract, SourceEnd,
)
from ...method_loop.communication import ParentModelInputError
from ...task_executors._http import ProviderError
from ...task_executors.evidence_extraction.argo import ModelOutputError
from ...task_executors.outbreak_search.queries import search_results_from_provider, source_locator, source_text
from ...task_executors.scenario_metrics.catalog import METRICS, PAPER, metric_catalog_record, metric_routes
from ...task_executors.scenario_metrics.evaluate import metric_conditions
from ...task_executors.scenario_metrics.dependencies import PARAMETER_DERIVATIONS
from ...task_executors.scenario_metrics.conditional import CONDITIONAL_INPUTS, conditional_inputs
from .communication import calculation_deferral_contract, decision_records, input_error_view, reading_coverage
from .goal_state import GLOBAL_GOAL_PROGRESS, ScientificGoalState
from .model_tiers import ModelTier, tier_for_operation
from .model_defaults import retained_defaults
from .progress import PARAMETER_EVIDENCE, parameter_requirements, parameter_support, progress_credit
from .records import (Credit, Ledger, ResearchMessage, ResearchResult, diagnostic_value,
                      identity, scenario_relation_key, target_value_model_view as target_value_decision_view)
from .tools import TOOLS, tool_options


TREE = {
    "evaluate_airborne_safety_guideline": ("investigate",),
    "investigate": ("strategy",),
    "strategy": ("search",),
    "search": ("page",),
    "page": ("source_table", "report", "lexical_probe"),
    "source_table": (), "report": (), "lexical_probe": (),
}
SCIENTIFIC = {"evaluate_airborne_safety_guideline", "investigate"}
CONSISTENCY = "evidence_consistency"


@dataclass(frozen=True)
class ResearchLimits:
    search_limit: int = 8
    text_window_chars: int = 12000
    yield_window: int = 4
    root_yield_window: int = 12

    def __post_init__(self):
        for name, value in vars(self).items():
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


def research_parameter_requirements(parameter, scenarios, subjects=()):
    """Retained baselines are available premises, not research requirements."""
    selected = [scenarios[name] for name in subjects if name in scenarios] if subjects else [
        row for row in scenarios.values() if row.get("role", "target") == "target"]
    retained = set.intersection(*(set(retained_defaults(row)) for row in selected)) if selected else set()
    return parameter_requirements(parameter) - retained


def calculation_decision_view(status, *, detailed, target_subjects=()):
    """Current values, credit and dependency codes without child-authored prose."""
    import math
    from ...task_executors.scenario_metrics.evaluate import UNITS, CONDITIONS

    statuses = {"not_calculated", "calculated_up_to_date", "calculated_needs_updating"}
    units = {**{name: metric.unit for name, metric in METRICS.items()}, **UNITS}
    fields = set(units) | set(CONDITIONS)
    targets = set(target_subjects)
    regimes = {route.regime for name in METRICS for route in metric_routes(name)}

    def number(value):
        return value if type(value) in (int, float) and math.isfinite(value) else None

    def enum(value, choices):
        return value if isinstance(value, str) and value in choices else None

    def subject(value):
        if not isinstance(value, str) or not value:
            return None
        return value if value in targets else "source_subject:" + identity(value)[:12]

    def unit(value, field):
        return value if isinstance(value, str) and value == units.get(field) else None

    view = {name: (status.get(name, 0) if type(status.get(name, 0)) is int
                   and status.get(name, 0) >= 0 else None) for name in sorted(statuses)}
    view["points"] = number(status.get("points", 0))
    # The scoring projection supplies these rates for both full and compact views.
    view["credit_policy"] = dict(status.get("credit_policy", {}))
    if not detailed:
        return view
    view.update(columns=["subject", "field", "kind", "status", "computed_value", "unit", "points",
                         "ready_to_calculate", "ready_to_recalculate", "matches_main_selection",
                         "changed_inputs", "note_ids", "computed_limit", "review_status",
                         "method_credit", "current_result_credit", "support_credit",
                          "accepted_current", "available_as_input", "parameterization_ref", "routes",
                          "input_status", "error_refs"], rows=[], notes=[])
    note_ids = {}
    for step in status.get("steps", ()):
        changed, notes = set(), set()

        def add_note(note):
            key = identity(note)
            if key not in note_ids:
                note_ids[key] = len(view["notes"])
                view["notes"].append(note)
            notes.add(note_ids[key])

        if step.get("matches_main_selection") is False:
            differences = []
            for change in step.get("main_input_differences", ()):
                field = enum(change.get("field"), fields)
                if field is not None:
                    changed.add(field)
                differences.append({"subject": subject(change.get("subject")), "field": field,
                    "used_value": number(change.get("used_value")),
                    "used_unit": unit(change.get("used_unit"), field),
                    "current_selected_value": number(change.get("current_selected_value")),
                    "current_selected_unit": unit(change.get("current_selected_unit"), field)})
            add_note({"code": "main_input_changed", "inputs": differences})
        review = enum(step.get("review_status"), {"accepted", "rejected", "incomplete"})
        if step.get("status") == "calculated_needs_updating":
            for alternative in step.get("result_alternatives", ()):
                for change in alternative.get("changed_dependencies", ()):
                    field = enum(change.get("field"), fields)
                    if field is not None:
                        changed.add(field)
                    kind = enum(change.get("kind"), {
                        "selected_target_value", "assumption", "model_default", "scenario_constraint",
                        "request_constraint", "condition", "input", "output_contract", "provenance"})
                    note = {"code": "dependency_changed", "kind": kind, "field": field}
                    for key in ("previous_value", "selected_value"):
                        if key in change:
                            note[key] = number(change[key])
                    add_note(note)
        limit = enum(step.get("computed_limit"), {"finite", "unbounded", "function"})
        value = number(step.get("computed_value"))
        if limit == "unbounded":
            value = "unbounded"
        elif limit == "function" and isinstance(step.get("computed_value"), Mapping):
            relation = step["computed_value"]
            kernel = enum(relation.get("kernel"), set(CONDITIONAL_INPUTS))
            coefficients = {"initial_risk_rate_s", "risk_acceleration_s2", "removal_rate_s",
                            "risk_rate_s", "risk_tolerance", "distance_rate_m_s"}
            value = {"kernel": kernel, "variable": enum(relation.get("variable"), {"exposure_time_s"}),
                "variable_unit": enum(relation.get("variable_unit"), {"s"}),
                "regime": enum(relation.get("regime"), {"well_mixed_distance"}),
                "inputs": {name: number(item) for name, item in relation.get("inputs", {}).items()
                           if name in units},
                "coefficients": {name: number(item) for name, item in relation.get("coefficients", {}).items()
                                 if name in coefficients},
                "domain": {"minimum": number(relation.get("domain", {}).get("minimum")),
                           "minimum_inclusive": relation.get("domain", {}).get("minimum_inclusive") is True},
                "boundaries": [{"time_s": number(item.get("time_s"))}
                               for item in relation.get("boundaries", ()) if isinstance(item, Mapping)]}
        field = enum(step.get("field"), fields)
        label = step.get("parameterization")
        routes = []
        for index, route in enumerate(step.get("routes", ()), 1):
            routes.append({"route_ref": f"route:{index}",
                "regime": enum(route.get("regime"), regimes),
                "missing_inputs": sorted(name for name in route.get("missing_inputs", ()) if name in units),
                "missing_conditions": sorted(name for name in route.get("missing_conditions", ()) if name in CONDITIONS),
                "constraint_failures": [{"field": enum(item.get("field"), fields),
                    "operator": enum(item.get("operator"), {"equal", "minimum"}),
                    "required_value": number(item.get("required_value")),
                    "actual_values": [number(value) for value in item.get("actual_values", ())]}
                    for item in route.get("constraint_failures", ())],
                "ready": route.get("ready") is True})
        view["rows"].append([subject(step.get("subject")), field,
            enum(step.get("kind"), {"requested_output", "intermediate"}), enum(step.get("status"), statuses),
            value, unit(step.get("unit"), field), number(step.get("points")),
            step.get("ready_to_calculate") is True, step.get("ready_to_recalculate") is True,
            step.get("matches_main_selection") if type(step.get("matches_main_selection")) is bool else None,
            sorted(changed), sorted(notes), limit, review,
            number(step.get("method_credit")), number(step.get("current_result_credit")),
            number(step.get("support_credit")), step.get("accepted_current") is True,
            step.get("available_as_input") is True,
            "parameterization:" + identity(label)[:12] if isinstance(label, str) and label else None, routes,
            enum(step.get("input_status"), {"error", "corrected"}),
            [ref for ref in step.get("error_refs", ()) if isinstance(ref, str)
             and ref.startswith("error:") and len(ref) == 18
             and all(character in "0123456789abcdef" for character in ref[6:])]])
    return view


class ResearchRun:
    def __init__(self, providers, limits, research_request, metrics, *, initial=None, scenario_selection=None):
        self.providers, self.limits = providers, limits
        self.research_request, self.metrics = research_request, metrics
        self.initial = initial
        self.global_goal = ScientificGoalState(metrics, initial)
        self.fixed_scenarios = initial is not None
        self.tools = {k: v for k, v in TOOLS.items()
                      if not self.fixed_scenarios or k != "compare_predictions_with_observations"}
        self.documents = {}
        self.model_attempts = (providers.checkpoint.archive.sequence(
            "model_attempts", providers.preflight_attempts or ())
            if providers.checkpoint is not None else list(providers.preflight_attempts or ()))
        self.provider_events = []
        self.provider_unavailable = None
        self.model_calls = len(self.model_attempts)
        self.total_units = self.sequence = 0
        self.diagnostic_sequence = 0
        self.controller_specs = {}
        self.live_controllers = {}
        self.sources = {}
        self.root_source = self.root_key = None
        root = Path(__file__).resolve().parents[2]
        paths = list(root.glob("*.py"))
        for folder in ("case_reports", "credit_assignment", "method_loop", "task_executors"):
            paths.extend((root / folder).rglob("*.py"))
        self.implementation = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(paths)}
        if providers.checkpoint is not None:
            self.implementation = providers.checkpoint.run_start_implementation(self.implementation)
        self.context = {"research_request": research_request, "reference_model": PAPER,
                        "scenario_selection": scenario_selection,
                        "requested_metrics": list(metrics), "fixed_scenarios": self.fixed_scenarios}
        units = {
            "evaluate_airborne_safety_guideline": "a selected calculation tool or completed investigation",
            "investigate": "a completed evidence strategy with mandatory candidate assessment",
            "strategy": "a completed search for a proposed query within this approach",
            "search": "a completed source episode selected from this query's results",
            "page": "a completed reading episode selected from physically available material",
            "source_table": "a table query execution with mandatory evidence admission",
            "report": "a document-order reading window with retained context",
            "lexical_probe": "a ranked unprocessed source chunk",
        }
        self.grains = {name: Grain(name, unit, "structured parent results and scientific state", self.controller)
                       for name, unit in units.items()}
        self.tree = EpisodeTree(self.grains["evaluate_airborne_safety_guideline"], {
            self.grains[parent]: tuple(self.grains[name] for name in children)
            for parent, children in TREE.items() if children
        })
        self.emit_trace("run_started", research_request=research_request, metrics=list(metrics),
                        scenario_selection=scenario_selection,
                        target_set=[row["subject"] for row in initial.scenarios
                                    if row.get("role", "target") == "target"] if initial else [],
                        target_fixed_inputs={row["subject"]: {
                            "fixed_parameters": row.get("fixed_parameters", {}),
                            "fixed_conditions": row.get("fixed_conditions", {})}
                            for row in initial.scenarios if row.get("role", "target") == "target"}
                            if initial else {},
                        implementation_sha256=dict(self.implementation))
        if initial is not None:
            scope = self.global_goal.preview()
            self.emit_trace("scenario_parameter_scope_initialized", model_calls=self.model_calls,
                independent_values_needed=scope["recorded_target_inputs"]["required"],
                scenario_parameter_relations=list(initial.scenario_parameter_relations),
                calculation_steps=scope.get("calculation_steps", {}),
                score_weights=scope["score_weights"])

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

    def controller(self, path):
        focus, initial, scientific, outputs = self.controller_specs[path[-1][1]]
        controller = ResearchController((GLOBAL_GOAL_PROGRESS,), window=self.limits.yield_window,
                                        root_window=self.limits.root_yield_window,
                                        initial=initial, point_scores=True)
        self.live_controllers[path] = controller
        return controller

    def root_credit_projection(self, status=None):
        """Read-only root score from the shared admitted ledger, including nested child work."""
        if self.root_key is None:
            return None
        return (self.global_goal.preview() if status is None else status)["weighted_score"]

    def refresh_controllers(self, source, status):
        """Update active ancestor scores without adding yield observations."""
        current_path = next((path for path in self.live_controllers
                             if path[-1][1] == source.controller_key), None)
        if current_path is None:
            return
        credit = self.global_goal.credit_from_status(status)
        for path, controller in self.live_controllers.items():
            if current_path[:len(path)] != path:
                continue
            gain = controller.refresh(credit)
            if gain:
                self.emit_trace("shared_credit_refreshed", episode_path=[list(item) for item in path],
                    source_episode_id=source.goal.goal_id, model_calls=self.model_calls,
                    gain=gain, score=controller.state().score,
                    meaning="Current shared score changed; completed-unit yield history is unchanged.")

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
            if diagnostic.get("category") == "provider_usage_limit":
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

    def episode(self, grain, source, *, goal, focus, contract=None, proposal=None):
        self.sequence += 1
        key = f"{grain}:{self.sequence}"
        source.controller_key = key
        source.episode_grain = grain
        source.focus = tuple(focus) if contract is None else (PARAMETER_EVIDENCE, GLOBAL_GOAL_PROGRESS)
        source.scientific = grain in SCIENTIFIC
        source.goal = goal
        self.sources[goal.goal_id] = source
        source.parent_contract = contract
        source.requested_subjects = tuple(contract.measurements.get("subjects", ())) if contract else ()
        if contract is not None:
            source.parameter = contract.measurements.get("parameter")
            source.parameter_requirements = research_parameter_requirements(
                source.parameter, source.ledger.scenarios, source.requested_subjects)
        self.controller_specs[key] = (source.focus, source.credit(source.ledger.result()), source.uses_scientific_credit(), source.output_fields)
        # Capture before descendants run, including inherited operands. A
        # suspended ancestor must notice work committed inside its child.
        source.return_baseline = source.return_progress()
        if grain == "evaluate_airborne_safety_guideline":
            self.root_source, self.root_key = source, key
        episode = Episode(self.grains[grain], key, source, EpisodeRequest(
            goal, ResearchMessage(str(goal.objective), source.requested_subjects,
                                  diagnostic_value(proposal or {})), contract),
            source.build_result, on_unit=source.after_verdict)
        if self.providers.checkpoint is not None:
            episode = self.providers.checkpoint.instrument(episode,
                lambda: self.global_goal.ledger.result().as_record())
        return episode

    def child(self, parent, source, grain, objective, *, contract, proposal):
        parent_source = self.sources[parent.goal.goal_id]
        if (parent_source.parameter is not None
                and contract.measurements.get("parameter") != parent_source.parameter):
            raise ValueError("A search branch cannot change its assigned calculation parameter")
        goal = EpisodeGoal.child(parent.goal, objective=objective,
                                 result_contract={"information": contract.as_record()})
        return self.episode(grain, source, goal=goal, focus=(),
                            contract=contract, proposal=proposal)


class ResearchSource:
    def __init__(self, run, inherited=()):
        if isinstance(inherited, ResearchResult):
            inherited = replace(inherited, diagnostics=tuple(run.identify_diagnostic(d) for d in inherited.diagnostics))
        self.run, self.ledger = run, Ledger(inherited)
        self.initial = self.ledger.result()
        self.focus, self.scientific = (), False
        self.goal = None
        self.parent_contract = None
        self.requested_subjects = ()
        self.parameter = None
        self.parameter_requirements = frozenset()
        self.credit_subjects = None
        self.output_fields = None
        self.pending = None
        self.pending_global = None
        self.controller_key = None
        self.episode_grain = None
        self.return_baseline = None
        self.cooperative_return_reason = ""
        self.failure = ""
        self.completion_diagnostic = None
        self.evaluated_units = 0
        self.last_proposal = None
        self.information_assessment = []
        self.evidence_handoff_baseline = None

    def contract(self, decision, *, subjects=(), meaning=None, deliverable="source_findings", parameter=None,
                 target_scenarios=None):
        """The invoking binding supplies a decision and its required measurements."""
        measurements = {"question": meaning or decision,
                        "deliverable": deliverable, "subjects": list(subjects),
                        "parameter": self.parameter if parameter is None else parameter}
        if target_scenarios is not None:
            measurements["target_scenarios"] = target_scenarios
        return ReportContract(decision=decision,
            measurements=measurements,
            information={
                "completion": "Object: ended_by is a declared episode-end code; evaluated says whether any work was evaluated. A source_failed or yield_stop return does not establish scientific completion.",
                "parameter_results": "Rows: parameter is a declared quantity or condition; value_state is numeric_value_returned, numeric_constraint_returned, qualitative_only, or no_value_returned. Numeric values use canonical units. numeric_constraints gives typed intervals or bounds; a range does not automatically supply a chosen scalar. scope distinguishes target, source and assigned; target is a declared scenario, source_subject and sources are deterministic references to retained records. origin and evidence_role are declared provenance codes. relevance links each value to target scenarios with numeric scores. No_value_returned does not mean evidence is absent. The shared main table remains authoritative for calculations.",
                "reviews": "Rows: parameter, subject references, numeric value/unit, completed, verdict, and numeric criterion grades. Verdict is unassessed, supported, provisional, unsupported, contradicted, fixed_condition or unknown; it describes review support, not whether a number is found. Supporting, contradicting and invalid evidence counts refer to that judgment.",
                "input_errors": "Typed correction footnotes: stable error_ref, recorded field/value/unit, error_code, source references, and affected target groups. error_code is quantity_mismatch, value_mismatch, unit_mismatch or source_attribution_mismatch. Each target reports the actual current value and resolution: error means no valid replacement is selected or the errored record remains selected; replaced means the former main record was removed and another recorded value selected; invalid_alternative_excluded means the bad record was not the main value. These are exact record corrections, not grades for uncertain applicability. No reviewer prose is returned.",
                "calculations": "Rows: declared subject and tool, output parameter/status/value/unit/limit, typed missing_inputs, symbolic function metadata when present, and review completion/admission booleans. Comparison records retain predicted/observed/residual/absolute_error and a comparison category. Execution results and scientific review are distinct. Current versus stale calculation credit remains in the shared calculation_steps table.",
                "reading_coverage": "Rows: deterministic source reference, document_characters, examined_characters, reviewed_characters, unexamined_characters, and completed/failed/not_started attempt counts. Examined means text reached extraction; reviewed means processing completed. Neither establishes that a quantity is absent.",
                "operations": "Rows: declared stage/category/status, affected parameter and subject references, value_retained, recovery, and count. These describe recorded attempts, not unresolved issues. child_identity_in_parent_input means request construction failed before model dispatch. Recovery is succeeded_later, retry_failed, not_retried or unknown; only a matching later read establishes reading recovery. No error prose or child instructions are returned.",
            })

    def build_result(self, record):
        reason = self.cooperative_return_reason or self.failure or (record.controller_state.reason if record.ended_by == "yield_stop" else record.ended_by)
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

    def after_verdict(self, unit, contribution, view):
        self.run.total_units += 1
        if self.pending is not None:
            new_diagnostics = [d for d in self.pending.diagnostics if d not in self.ledger.diagnostics]
            self.ledger.merge(self.pending)
            before_goal = self.run.global_goal.preview()
            previous_relations = dict(self.run.global_goal.ledger.source_target_relationships)
            previous_parameter_relations = dict(self.run.global_goal.ledger.scenario_parameter_relations)
            previous_assumptions = dict(self.run.global_goal.ledger.assumptions)
            previous_estimates = set(self.run.global_goal.ledger.estimates)
            global_status = self.run.global_goal.commit(self.pending_global or self.pending)
            self.run.refresh_controllers(self, global_status)
            changed_relations = [row for key, row in self.run.global_goal.ledger.source_target_relationships.items()
                                 if previous_relations.get(key) != row]
            changed_parameter_relations = [row for key, row in self.run.global_goal.ledger.scenario_parameter_relations.items()
                                           if previous_parameter_relations.get(key) != row]
            for row in changed_parameter_relations:
                self.run.emit_trace("scenario_parameter_compared", episode_type=type(self).__name__,
                    model_calls=self.run.model_calls, relationship=row)
            for row in changed_relations:
                self.run.emit_trace("source_target_relationship_assessed", status="assessed",
                    episode_type=type(self).__name__, model_calls=self.run.model_calls,
                    investigated_target_subjects=list(self.requested_subjects),
                    source_event={"locator": row["source"], "subject": row["source_subject"],
                                  "roles": row["source_role"]},
                    target_scenario=row["target_subject"], assigned_parameter=row["parameter"],
                    relationship=row)
                if self.run.providers.relation_event:
                    self.run.providers.relation_event({
                        "investigated_target_subjects": list(self.requested_subjects),
                        "source": row["source"], "source_subject": row["source_subject"],
                        "source_roles": row["source_role"], "target_subject": row["target_subject"],
                        "parameter": row["parameter"], "score": row["score"],
                        "known": row["known"], "considered": row["considered"]})
            for key, row in self.run.global_goal.ledger.assumptions.items():
                if (previous_assumptions.get(key) == row
                        or row.get("role") in {"scenario_constraint", "request_constraint"}
                        or row.get("subject") not in global_status["target_inputs"]):
                    continue
                verdict = row.get("review", {}).get("verdict")
                self.run.emit_trace("target_input_status",
                    status=("reviewed_usable" if verdict in {"supported", "provisional"} else
                            "recorded_unassessed" if verdict == "unassessed" else
                            "recorded_unsupported" if verdict == "unsupported" else "contradicted"),
                    episode_type=type(self).__name__, model_calls=self.run.model_calls,
                    subject=row.get("subject"), field=row.get("field"), value=row.get("value"),
                    unit=row.get("unit"), review_verdict=verdict, assumption_key=key)
            for key, row in self.run.global_goal.ledger.estimates.items():
                if key in previous_estimates:
                    continue
                scenario = self.run.global_goal.ledger.scenarios.get(row.get("subject"), {})
                self.run.emit_trace("calculation_inputs_selected", episode_type=type(self).__name__,
                    model_calls=self.run.model_calls, scenario_subject=row.get("subject"),
                    scenario_role=scenario.get("role", "unregistered_source_context"),
                    input_source="main_table" if scenario.get("role", "target") == "target" else "source_records",
                    parameterization=row.get("label"), parameters=row.get("parameters"),
                    assumption_versions=row.get("assumption_versions", {}),
                    calculated_outputs=[name for name, output in row.get("metrics", {}).items()
                                        if output.get("status") == "calculated"])
            root_projection = self.run.root_credit_projection(global_status)
            if (global_status["scores"] != before_goal["scores"]
                    or global_status["target_inputs"] != before_goal["target_inputs"]
                    or global_status.get("calculation_steps") != before_goal.get("calculation_steps")
                    or global_status["input_errors"] != before_goal["input_errors"]
                    or changed_relations or changed_parameter_relations):
                self.run.emit_trace("global_goal_progress", episode_type=type(self).__name__,
                    unit=unit.label, model_calls=self.run.model_calls,
                    scores=global_status["scores"],
                    score_weights=global_status["score_weights"],
                    weighted_score=global_status["weighted_score"],
                    root_credit_projection=root_projection,
                    values_found=global_status["recorded_target_inputs"]["recorded"],
                    values_needed=global_status["recorded_target_inputs"]["required"],
                    target_routes={name: row["best_route"] for name, row in global_status["target_routes"].items()},
                    target_assumptions=global_status["target_assumptions"],
                    recorded_target_inputs=global_status["recorded_target_inputs"],
                    independent_target_values=global_status["independent_target_values"],
                    calculation_steps=global_status.get("calculation_steps", {}),
                    input_errors=global_status["input_errors"],
                    target_inputs=global_status["target_inputs"],
                    source_target_learning=global_status["source_target_learning"],
                    calibration=global_status["calibration"],
                    requested_outputs=global_status["requested_outputs"])
            self.evaluated_units += int(self.pending.evaluated)
            if self.run.providers.event:
                self.run.providers.event({"unit": unit.label, "evaluated": self.pending.evaluated,
                    "admitted_facts": len(self.pending.facts), "estimates": len(self.pending.estimates),
                    "operation_status": "completed" if self.pending.evaluated else "failed",
                    "gain": view.controller_step.gain,
                    "stop": view.controller_step.stop, "model_calls": self.run.model_calls,
                    "global_goal_score": global_status["weighted_score"],
                    "root_credit_projection": root_projection,
                    "target_assumptions": global_status["target_assumptions"],
                    "recorded_target_inputs": global_status["recorded_target_inputs"],
                    "source_target_learning": global_status["source_target_learning"],
                    "calculation_steps": global_status.get("calculation_steps", {}),
                    "input_errors": global_status["input_errors"],
                    "requested_outputs": global_status["requested_outputs"]})
            self.run.emit_trace("credit_verdict", episode_id=self.goal.goal_id,
                episode_type=type(self).__name__, question=str(self.goal.objective), focus=list(self.focus),
                contract=self.parent_contract.as_record() if self.parent_contract else None,
                unit=unit.label, evaluated=self.pending.evaluated, gain=view.controller_step.gain,
                score=view.controller_step.score, stop=view.controller_step.stop,
                parameter=self.parameter, parameter_evidence=self.information_assessment,
                model_calls=self.run.model_calls,
                operation_status="completed" if self.pending.evaluated else "failed",
                diagnostics=new_diagnostics)
            self.pending = None
            self.pending_global = None
            update = contribution.episode_update
            if update is not None:
                # The typed child report already exists and its scientific
                # changes are now committed. Keep only active episode sources.
                closed_source = self.run.sources.pop(update.goal.goal_id, None)
                if closed_source is not None:
                    self.run.controller_specs.pop(closed_source.controller_key, None)
                    for path in tuple(self.run.live_controllers):
                        if path[-1][1] == closed_source.controller_key:
                            del self.run.live_controllers[path]

    def assess(self, result):
        """Validate/register records before the shared candidate pipeline."""
        return result


    def project(self, result):
        self.pending = self.assess(result)
        inherited = self.ledger.result()
        committed = self.run.global_goal.ledger.result()
        working = Ledger(committed)
        working.merge(changed_records(self.pending, inherited))
        from .relevance import assess_numeric_target_relationships
        from .target_values import project_source_target_values, target_value_candidates
        candidates = target_value_candidates(working)
        try:
            working.merge(assess_numeric_target_relationships(working,
                self.run.providers.source_target_assessor,
                disease=self.run.context.get("disease_of_interest"), candidates=candidates))
        except (ProviderError, ValueError, TypeError, KeyError, OSError) as exc:
            # An unavailable relevance grade leaves a found value ungraded;
            # it does not discard it or postpone its target-value projection.
            working.merge(self.failure_result(exc, "source_value_relevance"))
        working.merge(project_source_target_values(working, candidates=candidates))
        self.pending = working.result(evaluated=self.pending.evaluated,
                                      stop_reason=self.pending.stop_reason)
        self.pending = replace(self.pending, diagnostics=tuple(
            self.run.identify_diagnostic(d) for d in self.pending.diagnostics))
        self.pending_global = changed_records(self.pending, committed)
        # Local evidence assessment still sees the full branch; shared credit
        # receives only its new records, so old inherited claims cannot replace
        # a more recent judgment already committed by another episode.
        self.assess_information(self.pending)
        return self.run.global_goal.credit(self.pending_global)

    def uses_scientific_credit(self):
        return self.scientific

    def credit(self, result):
        self.assess_information(result)
        return self.run.global_goal.credit(changed_records(result, self.ledger.result()))

    def assess_information(self, result):
        """Keep the local contract assessment separate from shared projection."""
        if self.uses_scientific_credit():
            return
        if (self.parent_contract is not None
                and self.parent_contract.measurements.get("deliverable") == "source_findings"
                and self.parent_contract.measurements.get("target_scenarios") is not None):
            from .relevance import target_relevance_credit
            all_targets = self.parent_contract.measurements["target_scenarios"]
            targets = {name: all_targets[name] for name in self.requested_subjects if name in all_targets} or all_targets
            groups = {row["type_id"]: row for row in result.source_event_types}
            self.information_assessment, _ = target_relevance_credit(
                result, self.parameter_requirements, groups, targets, parameter=self.parameter)
            return
        self.information_assessment = parameter_support(result, self.parameter_requirements,
            hypotheses_only=self.parent_contract.measurements.get("deliverable") == "assumption_hypotheses",
            subjects=self.requested_subjects)

    def evidence_handoff_ready(self):
        """Return new parameter evidence promptly for a higher-level scientific decision."""
        points = sum(float(row.get("points", 0)) for row in self.information_assessment)
        if self.evidence_handoff_baseline is None:
            self.evidence_handoff_baseline = points
            return False
        return self.evaluated_units > 0 and points > self.evidence_handoff_baseline + 1e-9

    def return_progress(self):
        """Decision changes, independent of evidence grades and credit amounts."""
        from .target_values import eligible_source_values

        status = self.run.global_goal.preview()
        targets = identity([
            [(row["group_id"], row["found"], row.get("value"), row.get("unit"))
             for row in status["independent_target_values"]],
            [(row["subject"], row["field"], row["status"], row.get("computed_value"),
              row.get("unit"), row.get("ready_to_calculate"), row.get("ready_to_recalculate"))
             for row in status.get("calculation_steps", {}).get("steps", ())],
        ])
        operands = frozenset(identity([fact.subject, fact.field, fact.value, fact.unit])
            for fact in eligible_source_values(self.run.global_goal.ledger)
            if fact.field in self.parameter_requirements)
        return targets, operands

    def cooperative_return_ready(self, view):
        """Unwind at an ordinary unit boundary; never preempt running work."""
        if (self.episode_grain in {None, "evaluate_airborne_safety_guideline"}
                or not view.units_consumed or self.return_baseline is None):
            return False
        targets, operands = self.return_progress()
        old_targets, old_operands = self.return_baseline
        reason = ("target_progress_changed" if targets != old_targets else
                  "new_calculation_operands" if self.episode_grain != "investigate"
                  and operands - old_operands else "")
        if not reason:
            return False
        self.cooperative_return_reason = reason
        self.run.emit_trace("cooperative_return", episode_id=self.goal.goal_id,
            episode_type=type(self).__name__, reason=reason,
            completed_units=view.units_consumed, model_calls=self.run.model_calls,
            meaning="Return committed results after a completed own unit so the parent can choose its next action; controller observations are unchanged.")
        return True

    def receive(self, result, completion, request):
        if not isinstance(result, ResearchResult):
            raise TypeError("Unexpected scientific child result")
        return self.project(result)

    def edge(self, child):
        return ChildEpisodeUnit(child, self.receive, self.parent_report)

    def parent_report(self, result, completion=None, request=None):
        """Project existing validated work records into a closed parent JSON return."""
        import math
        from ...task_executors.scenario_metrics.evaluate import CONDITIONS, UNITS
        from ...task_executors.scenario_metrics.conditional import CONDITIONAL_INPUTS, conditional_inputs

        baseline = self.ledger.result() if request is not None else self.initial
        delta = changed_records(result, baseline)
        source = self.run.sources[request.goal.goal_id] if request is not None else self
        targets = {name for name, row in self.run.global_goal.ledger.scenarios.items()
                   if row.get("role", "target") == "target"}
        units = {**{name: metric_catalog_record(name).get("unit") for name in METRICS}, **UNITS}
        quantities = set(units) | set(CONDITIONS)
        all_facts = {fact.key: fact for fact in (*baseline.facts, *result.facts)}
        all_assumptions = {row["key"]: row for row in (*baseline.assumptions, *result.assumptions)}
        source_subjects, locators = {}, {}
        for row in (*[fact.semantic_record() for fact in all_facts.values()],
                    *baseline.evidence, *result.evidence, *all_assumptions.values(),
                    *baseline.calculations, *result.calculations):
            subject = row.get("subject")
            if subject is not None and subject not in targets:
                source_subjects.setdefault(subject, "source_subject:" + identity(subject)[:12])
            if subject is not None and row.get("source"):
                locators.setdefault(subject, set()).add(row["source"])
        known_sources = {locator for values in locators.values() for locator in values}
        known_sources.update(row["source"] for row in (*baseline.reading_attempts, *result.reading_attempts))

        def enum(value, allowed, default="unknown"):
            return value if isinstance(value, str) and value in allowed else default

        def number(value):
            return value if type(value) in (int, float) and math.isfinite(value) else None

        def source_reference(locator):
            return "source:" + identity(locator)[:12] if locator in known_sources else None

        def subject_refs(subject):
            return {"scope": "target" if subject in targets else "source",
                    "target": subject if subject in targets else None,
                    "source_subject": source_subjects.get(subject),
                    "sources": sorted(source_reference(locator) for locator in locators.get(subject, ()))}

        def unit(field, value):
            expected = units.get(field)
            if value is not None and expected is not None and value != expected:
                raise ValueError("Parent report requires canonical units")
            return expected

        def relevance(subject, field, value_key):
            rows = []
            for relation in result.source_target_relationships:
                if (relation.get("source_subject") != subject or relation.get("parameter") != field
                        or relation.get("target_subject") not in targets):
                    continue
                score = number(relation.get("source_value_scores", {}).get(value_key))
                if score is not None:
                    rows.append({"target": relation["target_subject"], "score": score})
            return rows

        parameters = []
        for fact in delta.facts:
            if fact.field not in quantities:
                continue
            parameters.append({**subject_refs(fact.subject), "parameter": fact.field,
                "value_state": "numeric_value_returned", "value": number(fact.value),
                "unit": unit(fact.field, fact.unit),
                "origin": enum(fact.kind, {"source_observation", "source_model_parameter", "derived",
                    "model_assumption", "user_assumption", "model_default", "scenario_constraint", "request_constraint"}),
                "relevance": relevance(fact.subject, fact.field, fact.key)})
        for row in delta.assumptions:
            field = row.get("field")
            if field not in quantities or number(row.get("value")) is None:
                continue
            if row.get("role") in {"scenario_constraint", "request_constraint", "model_default"}:
                continue
            parameters.append({**subject_refs(row["subject"]), "parameter": field,
                "value_state": "numeric_value_returned", "value": number(row["value"]),
                "unit": unit(field, row.get("unit")), "origin": "model_assumption",
                "available": not row.get("source_candidate_unavailable", False),
                "relevance": [{"target": relation["target_subject"],
                    "source_subject": source_subjects.get(relation.get("source_subject")),
                    "source": source_reference(relation.get("source")),
                    "score": number(relation.get("score"))}
                    for relation in row.get("source_applicability", {}).get("relationships", ())
                    if relation.get("target_subject") in targets and relation.get("parameter") == field
                    and number(relation.get("score")) is not None]})
        evidence_catalog = decision_records(Ledger(ResearchResult(evidence=delta.evidence)),
                                            target_subjects=targets)["evidence"]
        for index, row in enumerate(delta.evidence, 1):
            typed = evidence_catalog[f"evidence:{index}"]
            for field in sorted(set(row.get("fields", ())) & quantities):
                constraints = [bound for bound in typed["numeric_constraints"] if bound["field"] == field]
                parameters.append({**subject_refs(row["subject"]), "parameter": field,
                    "value_state": "numeric_constraint_returned" if constraints else "qualitative_only",
                    "value": None, "unit": units.get(field), "numeric_constraints": constraints,
                    "evidence_role": enum(row.get("evidence_role"),
                        {"observation", "model_statement", "context", "request_constraint"}),
                    "observed": field in row.get("observed_fields", ())})
        if source.parameter in quantities and not any(row["parameter"] == source.parameter for row in parameters):
            parameters.append({"scope": "assigned", "targets": sorted(set(source.requested_subjects) & targets),
                "parameter": source.parameter, "value_state": "no_value_returned", "value": None,
                "unit": units.get(source.parameter)})

        reviews = []
        for row in delta.assumptions:
            if row.get("field") not in quantities:
                continue
            review = row.get("review", {})
            reviews.append({**subject_refs(row["subject"]), "parameter": row["field"],
                "value": number(row.get("value")), "unit": unit(row["field"], row.get("unit")),
                "completed": bool(review.get("completed", review.get("verdict") not in (None, "unassessed"))),
                "verdict": enum(review.get("verdict"),
                    {"unassessed", "supported", "provisional", "unsupported", "contradicted", "fixed_condition"}),
                "grades": {key: item["grade"] for key, item in review.get("criteria", {}).items()
                    if key in {"support", "applicability", "physical_consistency", "uncertainty"}
                    and isinstance(item, dict) and type(item.get("grade")) is int and item["grade"] in (0, 1, 2)},
                "supporting_evidence_count": len(review.get("supporting_evidence", ())),
                "contradicting_evidence_count": len(review.get("contradicting_evidence", ())),
                "invalid_evidence_count": len(review.get("invalid_evidence", ()))})

        calculations = []
        for row in delta.calculations:
            outputs = []
            for field, output in {**row.get("output", {}), **row.get("intermediate_outputs", {}),
                                  **row.get("derived_parameters", {})}.items():
                if field not in quantities or not isinstance(output, dict):
                    continue
                value = number(output.get("value"))
                missing = []
                for item in output.get("missing", ()):
                    prefix, separator, name = item.partition(":") if isinstance(item, str) else (None, "", None)
                    if isinstance(item, str) and item in quantities:
                        requirement = {"kind": "condition" if item in CONDITIONS else "parameter", "parameter": item}
                    elif separator and prefix in {"condition", "conditional_dependency"} and name in quantities:
                        requirement = {"kind": prefix, "parameter": name}
                    else:
                        requirement = {"kind": "execution_error" if output.get("status") == "undefined" else "unknown",
                                       "parameter": None}
                    if requirement not in missing:
                        missing.append(requirement)
                projected = {"parameter": field,
                    "status": enum(output.get("status", "calculated" if value is not None else "not_calculated"),
                        {"calculated", "not_calculated", "unsupported", "inapplicable", "missing_inputs",
                         "unsupported_variable", "undefined"}),
                    "value": value, "unit": unit(field, output.get("unit")),
                    "limit": enum(output.get("limit"), {"finite", "unbounded", "empty", "function"}, None),
                    "missing_inputs": missing}
                function = output.get("function")
                if output.get("limit") == "function" and isinstance(function, dict):
                    kernel = enum(function.get("kernel"), set(CONDITIONAL_INPUTS), None)
                    regime = enum(function.get("regime"), {"well_mixed_distance"}, None)
                    coefficients = {"initial_risk_rate_s", "risk_acceleration_s2", "removal_rate_s",
                                    "risk_rate_s", "risk_tolerance", "distance_rate_m_s"}
                    domain = function.get("domain", {})
                    projected["function"] = {
                        "version": function.get("version") if type(function.get("version")) is int
                            and function["version"] in {1, 2} else None,
                        "kernel": kernel, "regime": regime,
                        "variable": enum(function.get("variable"), {"exposure_time_s"}, None),
                        "variable_unit": "s" if function.get("variable_unit") == "s" else None,
                        "domain": {"minimum": number(domain.get("minimum")),
                                   "minimum_inclusive": domain.get("minimum_inclusive")
                                   if type(domain.get("minimum_inclusive")) is bool else None},
                        "inputs": {name: number(value) for name, value in function.get("inputs", {}).items()
                                   if kernel is not None and name in conditional_inputs(kernel, regime=regime)},
                        "coefficients": {name: number(value) for name, value in function.get("coefficients", {}).items()
                                         if name in coefficients},
                        "boundaries": [{"time_s": number(boundary.get("time_s")),
                                        "kind": "background_risk_equals_criterion"}
                                       for boundary in function.get("boundaries", ())
                                       if isinstance(boundary, dict) and number(boundary.get("time_s")) is not None],
                    }
                outputs.append(projected)
            review = row.get("model_review", {})
            calculation = {**subject_refs(row["subject"]), "tool": enum(row.get("tool"), set(TOOLS)),
                "outputs": outputs, "review_completed": bool(review.get("completed", "admit" in review)),
                "review_admitted": review.get("admit") if type(review.get("admit")) is bool else None,
                "support_complete": bool(row.get("support_complete", True))}
            if row.get("tool") == "compare_predictions_with_observations":
                comparison = row.get("output", {})
                metric = enum(comparison.get("metric"), set(METRICS), None)
                calculation["comparison"] = {
                    "metric": metric, "unit": units.get(metric),
                    **{name: number(comparison.get(name))
                       for name in ("predicted", "observed", "residual", "absolute_error")},
                    "category": enum(comparison.get("category"),
                        {"calibration_consistency", "out_of_calibration_comparison"}),
                    "independent_validation": comparison.get("independent_validation")
                        if type(comparison.get("independent_validation")) is bool else None,
                }
            calculations.append(calculation)

        coverage = []
        for row in reading_coverage(delta.reading_attempts):
            coverage.append({"source": source_reference(row["source"]),
                "document_characters": row["document_characters"],
                **{name + "_characters": sum(end - start for start, end in row[name + "_ranges"])
                   for name in ("examined", "reviewed", "unexamined")},
                "attempts": {name: row["outcomes"].get(name, 0)
                             for name in ("completed", "failed", "not_started")}})

        stages = {"proposal", "parent_model_input", "source_reading", "source_retrieval", "source_retrieval_failure",
            "extraction", "applicability_review", "evidence_admission", "independent_evidence_review", "reading_memory",
            "source_value_relevance", "source_event_relevance", "source_event_assessment", "source_event_classification",
            "mandatory_reasonableness_assessment", "candidate_reasonableness_assessment",
            "selected_source_reasonableness_assessment", "selected_input_reasonableness_assessment",
            "model_support_assessment", "model_review", "input_record_correction", "synthesis", "completion", "episode_completion", *TOOLS}
        categories = {"malformed_json", "duplicate_json_key", "nonfinite_json_number", "non_text_model_output",
            "non_object_json", "incomplete_model_output", "invalid_action_contract", "provider_usage_limit",
            "provider_error", "provider_failure", "transport_failure", "invalid_candidate", "rejected_evidence",
            "assessment_incomplete", "invalid_memory", "no_relevant_evidence", "source_termination", "child_identity_in_parent_input",
            "quantity_mismatch", "value_mismatch", "unit_mismatch", "source_attribution_mismatch"}
        operations = []
        for diagnostic in delta.diagnostics:
            if diagnostic.get("stage") == "assumption_assessment":
                continue
            category = ("child_identity_in_parent_input" if diagnostic.get("why") ==
                        "parent report contains a child audit identity" else enum(diagnostic.get("category"), categories))
            operation = {"stage": enum(diagnostic.get("stage"), stages), "category": category,
                "status": enum(diagnostic.get("status"), {"failed", "rejected", "not_applied", "completed", "applied"}),
                "parameter": diagnostic.get("field") if diagnostic.get("field") in quantities else None,
                "target": diagnostic.get("subject") if diagnostic.get("subject") in targets else None,
                "source_subject": source_subjects.get(diagnostic.get("subject")),
                "source": source_reference(diagnostic.get("source")),
                "value_retained": diagnostic.get("hypothesis_available") is True,
                "recovery": "unknown"}
            failed_reading = next((attempt for attempt in result.reading_attempts
                if diagnostic.get("stage") == "source_reading" and diagnostic.get("model_call") is not None
                and attempt.get("first_model_call") == diagnostic["model_call"] and attempt["status"] == "failed"), None)
            if failed_reading is not None:
                later = [attempt for attempt in result.reading_attempts
                    if attempt.get("first_model_call", 0) > failed_reading.get("first_model_call", 0)
                    and all(attempt.get(key) == failed_reading.get(key)
                            for key in ("source", "source_version", "ranges"))]
                operation["recovery"] = ("succeeded_later" if any(a["status"] == "completed" for a in later)
                    else "retry_failed" if any(a["status"] == "failed" for a in later) else "not_retried")
            existing = next((row for row in operations if {k: v for k, v in row.items() if k != "count"} == operation), None)
            if existing is None:
                operations.append({**operation, "count": 1})
            else:
                existing["count"] += 1
        ended_by = getattr(completion, "ended_by", result.stop_reason)
        prior_errors = {identity(note) for note in input_error_view(Ledger(baseline))}
        input_errors = [note for note in input_error_view(Ledger(result)) if identity(note) not in prior_errors]
        return {"completion": {"ended_by": enum(ended_by,
                    {"exhausted", "yield_stop", "incomplete", "saturated", "source_failed", "interrupted", "completed", "stopped"},
                    "in_progress" if completion is None and not result.stop_reason else "unknown"),
                    "evaluated": bool(result.evaluated)},
                "parameter_results": list({identity(row): row for row in parameters}.values()),
                "reviews": reviews, "calculations": calculations,
                "input_errors": input_errors,
                "reading_coverage": coverage, "operations": operations}

    def context(self):
        if self.parent_contract is not None:
            context = {"disease_of_interest": self.run.context.get("disease_of_interest"),
                       "parent_information_contract": self.parent_contract.as_record()}
            if "target_scenarios" in self.parent_contract.measurements:
                context["target_scenario_use"] = (
                    "The contract's target_scenarios describe the caller's intended application, "
                    "not observed source events or evidence. Use their activity, population and "
                    "conditions to guide evidence relevance for the assigned parameter, including "
                    "Cq applicability. Related settings and indirect evidence remain useful when "
                    "their transfer assumptions and differences are identified. Retain each "
                    "source event's actual conditions; never substitute target-case inputs for "
                    "outbreak observations or add scoring requirements from these descriptions.")
            if self.scientific:
                context.update(fixed_scenarios=self.run.fixed_scenarios,
                    defined_scenarios={name: self.ledger.scenarios[name] for name in self.requested_subjects
                                       if name in self.ledger.scenarios})
            return context
        return {**self.run.context, "defined_scenarios": dict(self.ledger.scenarios),
                "known_source_subjects": sorted(
                    {f.subject for f in self.ledger.facts.values() if f.kind.startswith("source_")}
                    | {e["subject"] for e in self.ledger.evidence.values() if e.get("kind") == "source_evidence"}),
                "input_roles": {
                    "fixed_case_conditions": "Caller-owned givens; preserve them, do not rediscover or replace them.",
                    "calculation_variables": "Retain or solve the variables declared by the case. Do not assign arbitrary point values merely to fill an output.",
                    "missing_physical_inputs": "The investigation may propose named, qualified hypotheses using bounds, indirect evidence, physical reasoning or explicit guesses. Lack of direct evidence lowers support; it is not a prohibition on proposing or calculating a hypothesis. These are not new scenarios or caller givens.",
                },
                "calculated_model_behavior": tool_options(self.ledger),
                "requested_subjects": list(self.requested_subjects),
                "implemented_model": {
                    "tools": self.run.tools,
                    "parameter_derivations": [d.as_record() for d in PARAMETER_DERIVATIONS],
                    "duration_relationships": {name: {"variable": "exposure_time_s",
                        "routes": [{"regime": route.regime,
                                    "bound_inputs": list(conditional_inputs(name, regime=route.regime)),
                                    "conditions": sorted(metric_conditions(name, regime=route.regime))}
                                   for route in metric_routes(name)]}
                        for name in CONDITIONAL_INPUTS},
                    "quantities": {name: metric_catalog_record(name) for name in METRICS},
                    "interpretation": "Executable repository kernels, not retrieved evidence or independent validation. "
                         "Target calculations use the main table, including its current estimates; evidence support affects credit and "
                        "conclusions, not execution. Calculated behavior is not empirical evidence. Missing source equations "
                        "do not imply a missing kernel; investigate fidelity when consequential. "
                        "An existing kernel may still have no applicable route for a selected case. Parameter constraints "
                        "are enforced by execution, not evidence-quality gates. Condition prose cannot override them or "
                        "introduce a new kernel. Unsupported hypotheses remain usable within the declared model domain. "
                        "This catalog grants no child permission to call the root's tools.",
                },
                "parent_information_contract": self.parent_contract.as_record() if self.parent_contract else None}

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
            if self.cooperative_return_ready(view):
                return None
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

    def resolve_decision_references(self, proposal):
        """Resolve only declared action references to the original stored records."""
        validate_deferrals = getattr(self, "_validate_calculation_deferrals", None)
        if validate_deferrals is not None:
            validate_deferrals(proposal)
        catalog = self._decision_records
        subjects = {ref: name for name, ref in catalog["subjects"].items()}
        models = {ref: name for name, ref in catalog["parameterizations"].items()}

        def resolve(row):
            if not isinstance(row, Mapping):
                return row
            resolved = dict(row)
            if isinstance(row.get("subject"), str):
                resolved["subject"] = subjects.get(row["subject"], row["subject"])
            if isinstance(row.get("subjects"), list):
                resolved["subjects"] = [subjects.get(name, name) if isinstance(name, str) else name
                                        for name in row["subjects"]]
            if isinstance(row.get("parameterization"), str):
                resolved["parameterization"] = models.get(row["parameterization"], row["parameterization"])
            if isinstance(row.get("conditions"), Mapping):
                conditions = {}
                for field, value in row["conditions"].items():
                    stored = catalog["conditions"].get(value) if isinstance(value, str) else None
                    if stored is not None:
                        if stored["field"] != field or stored["subject"] != resolved.get("subject"):
                            raise ValueError("A condition reference must belong to the selected subject and field")
                        value = stored["statement"]
                    elif isinstance(value, str) and value.startswith("condition:"):
                        raise ValueError("Unknown condition reference")
                    conditions[field] = value
                resolved["conditions"] = conditions
            for name in ("scenarios", "candidates", "calculation_deferrals"):
                if isinstance(row.get(name), list):
                    resolved[name] = [resolve(item) for item in row[name]]
            return resolved

        return resolve(proposal)

    def inputs(self, view, **kwargs):
        own_results = self.parent_report(self.ledger.result())
        targets = {name for name, row in self.run.global_goal.ledger.scenarios.items()
                   if row.get("role", "target") == "target"}
        catalog = decision_records(self.ledger, target_subjects=targets)
        self._decision_records = catalog
        context = self.context()
        if "defined_scenarios" in context:
            context["defined_scenarios"] = {catalog["subjects"][name]: catalog["scenarios"][catalog["subjects"][name]]
                for name in context["defined_scenarios"] if name in catalog["subjects"]}
        if "known_source_subjects" in context:
            context["known_source_subjects"] = [catalog["subjects"][name] for name in context["known_source_subjects"]
                                               if name in catalog["subjects"]]
        if "calculated_model_behavior" in context:
            context["calculated_model_behavior"] = catalog["calculations"]
        global_status = self.run.global_goal.preview()
        calculation_table = calculation_decision_view(
            global_status.get("calculation_steps", {}), detailed=self.scientific, target_subjects=targets)
        self._validate_calculation_deferrals = None
        if kwargs.get("operation") in {"propose_guideline_action", "propose_investigation_strategy"}:
            deferral_schema, self._validate_calculation_deferrals = calculation_deferral_contract(calculation_table)
            contract = kwargs["response_contract"]
            contract["properties"]["calculation_deferrals"] = deferral_schema
            contract.setdefault("allOf", []).append({
                "if": {"required": ["action"], "properties": {
                    "action": {"enum": [action for action in kwargs["allowed_actions"]
                                         if action in {"investigate", "evidence_strategy"}]}}},
                "then": {"required": ["calculation_deferrals"]},
                "else": {"properties": {"calculation_deferrals": False}}})
        def schema_references(schema, *, subject_property=False):
            if not isinstance(schema, Mapping):
                return schema
            mapped = {}
            for key, value in schema.items():
                if key == "properties":
                    mapped[key] = {name: schema_references(item, subject_property=name in {"subject", "subjects"})
                                   for name, item in value.items()}
                elif subject_property and key == "const" and isinstance(value, str):
                    mapped[key] = catalog["subjects"].get(value, value)
                elif subject_property and key == "enum":
                    mapped[key] = [catalog["subjects"].get(item, item) for item in value]
                elif isinstance(value, Mapping):
                    mapped[key] = schema_references(value, subject_property=subject_property)
                elif key in {"allOf", "anyOf", "oneOf"}:
                    mapped[key] = [schema_references(item, subject_property=subject_property) for item in value]
                else:
                    mapped[key] = value
            return mapped

        # Only declared reference fields are translated. A scientific subject
        # named "value" must never rewrite a JSON Schema property named value.
        def reference_view(value, field=None):
            if field == "response_contract":
                return schema_references(value)
            if isinstance(value, str):
                references = (catalog["subjects"] if field in {
                    "subject", "subjects", "target", "targets", "source_subject", "target_subject", "origin_subject",
                    "requested_subjects", "known_source_subjects"} else
                    catalog["parameterizations"] if field == "parameterization" else
                    catalog["sources"] if field == "source" else {})
                return references.get(value, value)
            if isinstance(value, Mapping):
                return {key: reference_view(item, key) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [reference_view(item, field) for item in value]
            return value

        return view.model_inputs(reference_view({
            "objective": str(view.goal.objective), "context": context,
            "global_scientific_goal": {
                "scores": global_status["scores"],
                "score_weights": global_status["score_weights"],
                "weighted_score": global_status["weighted_score"],
                "target_routes": {name: row["best_route"] for name, row in global_status["target_routes"].items()},
                "target_assumptions": global_status["target_assumptions"],
                "recorded_target_inputs": global_status["recorded_target_inputs"],
                "independent_target_values": [target_value_decision_view(row, decision_view=True, target_subjects=targets)
                                              for row in global_status["independent_target_values"]],
                "calculation_steps": calculation_table,
                "input_errors": global_status["input_errors"],
                "target_inputs": {subject: {field: target_value_decision_view(row, decision_view=True, target_subjects=targets) for field, row in fields.items()}
                                  for subject, fields in global_status["target_inputs"].items()},
                "source_target_learning": global_status["source_target_learning"],
                "calibration": global_status["calibration"],
                "requested_outputs": global_status["requested_outputs"],
                "meaning": "Current shared Goal projection, updated after each child verdict before its return. One calculation_steps table contains requested outputs and intermediates, their computed values and units, status, credit, and readiness. Rows use calculation_steps.columns; note_ids indexes calculation_steps.notes. Each row's routes separates missing_inputs, missing_conditions and constraint_failures; a present value incompatible with a route is not a missing value. Root and investigation receive the table; acquisition children receive its counts and credit. The three statuses are not_calculated, calculated_up_to_date, and calculated_needs_updating. Target calculations automatically use the current main-table values, fixed inputs, retained defaults and current computed inputs. Calculation requests do not select numeric alternatives. Status describes numerical execution and whether the used inputs match the current main table. Requested numerical outputs are complete when every requested row is calculated_up_to_date; no support approval or calibration is required for final reporting. Notes identify changed dependencies. error_refs point to input_errors footnotes containing exact record errors and their current replacement state. input_status=error is supplementary to the three computation statuses; corrected means an eligible replacement is already selected. The rejected record and replacement remain visible in the same footnote. Weak applicability alone is not an input error. A row's ready_to_recalculate flag describes availability of main-table inputs and model conditions. The episode determines which operation to run. A previously successful calculation retains only its method credit while its numerical result is obsolete. Recorded target values include conditional estimates with their provenance; no separate parameter-support review follows registration. Historical review fields are not a calculation or reporting prerequisite. Route counts describe input availability, not completed calculations."},
            "own_results": own_results,
            "parameter_progress": [{"field": row["field"], "points": row["points"],
                                    "alternative_values": len(row["alternatives"])}
                                   for row in self.information_assessment if "field" in row],
            "numeric_yield": {"score": view.controller_state.score,
                "history": list(view.controller_state.history), "reason": view.controller_state.reason,
                "history_meaning": "History records this episode's completed-unit observations. Current score includes committed child changes before their return; a refresh is not another yield observation.",
                "score_meaning": "The same global scientific score drives every episode: found independent target values plus calculation-step credit, using global_scientific_goal.score_weights. Extracted, calculated and inferred values enter the same candidate pipeline; the highest field-specific relevance selects the current value, without an admission threshold or priority by origin. Provenance remains explicit. A replacement does not create another independent value. Local parameter_progress describes this branch's evidence and is not its controller score. Scores refresh after committed child work before return; repetition earns no gain. Calculation credit can decrease when its inputs change. This score is not confidence or a completion fraction."},
            **kwargs,
        }), reports=view.child_reports)
