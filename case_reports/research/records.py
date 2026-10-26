"""Closed scientific records; source evidence is distinct from model choices."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import hashlib
import json
import math
from typing import Any, Mapping

from ...method_loop import ClosedRecord
from ...task_executors.scenario_metrics.conditional import is_duration_relation
from ...task_executors.scenario_metrics.evaluate import output_meets_contract, exposure_outputs_agree


def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def scenario_relation_key(row):
    if row.get("kind") == "feature":
        return ("feature", row["name"])
    if row.get("kind") == "dependency":
        return ("dependency", row["parameter"])
    return ("group", row["group_id"])


def finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def diagnostic_value(value):
    """Retain malformed provider proposals without emitting invalid audit JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, Mapping):
        return {str(k): diagnostic_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [diagnostic_value(v) for v in value]
    return value


def condensed_diagnostics(diagnostics):
    """Describe failure patterns without replaying rejected prose as context."""
    groups = {}
    for diagnostic in diagnostics:
        summary = {k: v for k, v in diagnostic.items()
                   if k not in {"proposal", "diagnostic_id", "model_call", "unreviewed_candidates"}}
        proposal = diagnostic.get("proposal")
        if isinstance(proposal, Mapping):
            summary["received_fields"] = sorted(proposal)
            summary["candidate"] = {k: proposal[k] for k in ("subject", "field", "value", "unit", "kind")
                                    if k in proposal}
        key = identity(summary)
        if key not in groups:
            groups[key] = {**summary, "retained_records": 0, "model_calls": []}
        groups[key]["retained_records"] += 1
        if "model_call" in diagnostic and diagnostic["model_call"] not in groups[key]["model_calls"]:
            groups[key]["model_calls"].append(diagnostic["model_call"])
    return list(groups.values())


def cited_assessment_evidence(assessment):
    """Retain explicit inference provenance and any historical review citations."""
    review = assessment.get("review", {})
    references = {key for criterion in review.get("criteria", {}).values()
                  for key in criterion.get("evidence", ())}
    references.update(review.get("supporting_evidence", ()))
    references.update(review.get("contradicting_evidence", ()))
    references.update(row["evidence_ref"] if isinstance(row, dict) else row
                      for row in review.get("invalid_evidence", ()))
    constraint = review.get("request_constraint")
    if constraint is not None:
        references.update(constraint["evidence"])
    source_ids = set(assessment.get("derivation", {}).get("source_evidence_ids", ()))
    return {key: item for key, item in assessment.get("evidence", {}).items()
            if key in references or identity(item) in source_ids}


def target_value_model_view(row, *, decision_view=False, target_subjects=()):
    """Project typed decision data; scientific reviewers retain their full records."""
    if decision_view:
        from ...task_executors.scenario_metrics.catalog import METRICS
        from ...task_executors.scenario_metrics.evaluate import UNITS, CONDITIONS

        targets = set(target_subjects)
        units = {**{name: metric.unit for name, metric in METRICS.items()}, **UNITS}
        fields = set(units) | set(CONDITIONS)

        def number(value):
            return value if type(value) in (int, float) and math.isfinite(value) else None

        def enum(value, choices):
            return value if isinstance(value, str) and value in choices else None

        def subject(value):
            if not isinstance(value, str) or not value:
                return None
            return value if value in targets else "source_subject:" + identity(value)[:12]

        def score(value):
            value = number(value)
            return value if value is not None and 0 <= value <= 1 else None

        view = {}
        for key in ("field", "parameter"):
            value = enum(row.get(key), fields)
            if value is not None:
                view[key] = value
        field = view.get("field", view.get("parameter"))
        for key in ("value", "points", "discovery_points", "learning_bonus", "reasonableness"):
            if key in row:
                view[key] = number(row[key])
        if "unit" in row:
            expected = units.get(field)
            view["unit"] = (row["unit"] if isinstance(row["unit"], str)
                            and row["unit"] in set(units.values())
                            and (expected is None or row["unit"] == expected) else None)
        for key in ("found", "reviewed", "source_candidate_unavailable"):
            if key in row:
                view[key] = row[key] if type(row[key]) is bool else None
        for key in ("subject", "origin_subject"):
            if key in row:
                view[key] = subject(row[key])
        if "subjects" in row:
            subjects = row["subjects"] if isinstance(row["subjects"], (list, tuple)) else ()
            view["subjects"] = [subject(value) for value in subjects
                                if isinstance(value, str) and value]
        group = row.get("group_id", row.get("scope_group_id"))
        if isinstance(group, str) and group:
            view["group_ref"] = "target_group:" + identity(group)[:12]
        if "status" in row:
            view["status"] = enum(row["status"], {
                "missing", "reviewed_usable", "recorded_unassessed", "recorded_unsupported", "contradicted"})
        if "review_verdict" in row:
            view["review_verdict"] = enum(row["review_verdict"], {
                "unassessed", "supported", "provisional", "unsupported", "contradicted", "fixed_condition"})
        if "input_status" in row:
            view["input_status"] = enum(row["input_status"], {"found", "missing", "error", "corrected"})
        if "error_refs" in row:
            view["error_refs"] = [ref for ref in row["error_refs"] if isinstance(ref, str)
                and ref.startswith("error:") and len(ref) == 18
                and all(character in "0123456789abcdef" for character in ref[6:])]
        applicability = row.get("source_applicability")
        if isinstance(applicability, Mapping):
            relationships = []
            for relation in applicability.get("relationships", ()):
                if not isinstance(relation, Mapping):
                    continue
                locator = relation.get("source")
                compact = {
                    "source_ref": "source:" + identity(locator)[:12]
                        if isinstance(locator, str) and locator else None,
                    "source_subject": subject(relation.get("source_subject")),
                    "target_subject": subject(relation.get("target_subject")),
                    "parameter": enum(relation.get("parameter"), fields),
                    "score": score(relation.get("score")),
                }
                for key in ("known", "considered"):
                    if key in relation:
                        value = relation[key]
                        compact[key] = value if type(value) is int and value >= 0 else None
                relationships.append(compact)
            view["source_applicability"] = {
                "status": enum(applicability.get("status"), {"assessed", "unknown"}),
                "score": score(applicability.get("score")), "relationships": relationships}
        elif "source_applicability" in row:
            view["source_applicability"] = None
        return view

    applicability = row.get("source_applicability")
    if not isinstance(applicability, dict):
        return row
    relationships = []
    for relation in applicability.get("relationships", ()):
        compact = {key: relation[key] for key in (
            "source", "source_subject", "target_subject", "parameter", "score", "basis", "source_contexts")
            if key in relation}
        value_context = relation.get("source_value_contexts", {}).get(relation.get("source_fact_key"))
        if value_context is not None:
            compact["source_contexts"] = [value_context]
        relationships.append(compact)
    return {**row, "source_applicability": {
        **{key: applicability[key] for key in ("status", "score", "meaning") if key in applicability},
        "relationships": relationships}}


def assumption_model_view(assumption, *, evidence_options=None, numeric_options=None):
    """Present inference meaning without exposing protected child audit identities."""
    view = dict(target_value_model_view(assumption))
    # Native quantities, references and contexts are presented below through
    # derivation inputs. Their internal fact identities are audit-only indexes.
    view.pop("source_fact_keys", None)
    derivation = assumption.get("derivation", {})
    identities = ("source_evidence_ids", "input_fact_keys",
                  "input_assumption_versions", "input_assumption_claims")
    if not any(derivation.get(name) for name in identities):
        return view
    evidence = assumption.get("evidence", {}) if evidence_options is None else evidence_options
    numeric = {} if numeric_options is None else numeric_options
    source_ids, input_keys = set(derivation.get("source_evidence_ids", ())), set(derivation.get("input_fact_keys", ()))
    inputs = {}
    for ref, fact in numeric.items():
        if fact.key in input_keys and fact.key not in inputs:
            inputs[fact.key] = {"reference": ref, **fact.semantic_record()}
    view["derivation"] = {
        "source_refs": [ref for ref, record in evidence.items() if identity(record) in source_ids],
        "inputs": list(inputs.values()),
        "unavailable_input_count": len(input_keys - inputs.keys()),
        "added_assumptions": list(derivation.get("added_assumptions", ())),
    }
    return view


def assessment_summary(assessment, *, numeric_options=None):
    evidence = cited_assessment_evidence(assessment)
    view = assumption_model_view(assessment, evidence_options=evidence, numeric_options=numeric_options)
    return {**{k: v for k, v in view.items() if k not in {"key", "evidence", "assessment_basis_version"}},
            "evidence": evidence,
            "considered_evidence_count": len(assessment["evidence"])}


def numeric_provenance(ledger):
    """Index recorded numeric premises by identity for inference and source lineage.

    Target calculations obtain their inputs from the main table in tools.py.
    This index preserves the records cited by an inference, including alternatives.
    """
    from .model_defaults import retained_defaults
    from .scenario_sharing import shared_assumptions

    def retained(subject, field, value):
        return retained_defaults(ledger.scenarios.get(subject, {})).get(field, {}).get("value") == value

    records = {key: fact for key, fact in ledger.facts.items()
        if not (fact.kind == "model_assumption" and fact.parents
                and ledger.assumptions.get(fact.parents[0], {}).get("source_candidate_unavailable"))
        and (fact.kind != "model_default" or retained(fact.subject, fact.field, fact.value))}
    for assumption in ledger.assumptions.values():
        if assumption["value"] is None or assumption.get("source_candidate_unavailable"):
            continue
        default = assumption.get("role") == "model_default"
        if default and not retained(assumption["subject"], assumption["field"], assumption["value"]):
            continue
        fact = Fact(assumption["subject"], assumption["field"], assumption["value"],
            assumption["unit"], "model_default" if default else "model_assumption", assumption["basis"],
            context=assumption["statement"], parents=(assumption["key"],))
        records[fact.key] = fact
    for subject in ledger.scenarios:
        for assumption, relation in shared_assumptions(ledger, subject):
            fact = Fact(subject, assumption["field"], assumption["value"],
                assumption["unit"], "model_assumption", assumption["basis"],
                context=f"Shared from {assumption['subject']}: {assumption['statement']}",
                parents=(assumption["key"], identity(relation)))
            records[fact.key] = fact
    return records



def numeric_references(ledger):
    """Parent-local citations for inference and native source inputs only.

    Canonical identities remain in numeric_provenance and committed lineage;
    these labels can cross the parent communication boundary. Target arithmetic
    never consumes this reference view.
    """
    return {f"input:{index}": fact for index, fact in
            enumerate(numeric_provenance(ledger).values(), 1)}


def assumption_claim(assessment):
    """Identify the computation's claim separately from its evidence review."""
    return identity({k: assessment[k] for k in ("subject", "field", "value", "unit", "statement")})


def assumption_matches(record, key, assessment):
    if not assessment:
        return False
    version = record.get("assumption_versions", {}).get(key)
    if version is None or version == assessment.get("version"):
        return True
    claim = record.get("assumption_claims", {}).get(key)
    return claim is not None and claim == assumption_claim(assessment)


def current_support(record, assessments, *, metric=None, ledger=None):
    """Evidence quality is a credit/report property, not permission to compute."""
    if (not record.get("model_review", {}).get("admit", True)
            or not record.get("model_review", {}).get("completed", True)):
        return 0.0
    if metric is not None and ("metric_support" in record or "intermediate_support" in record):
        dependency = record.get("metric_support", {}).get(metric,
                     record.get("intermediate_support", {}).get(metric, {}))
        return current_support(dependency, assessments, ledger=ledger)
    if not current_scenario_links(record, ledger):
        return 0.0
    if not record.get("support_complete", True):
        return 0.0
    versions = record.get("assumption_versions", {})
    keys = record.get("assumption_keys", versions)
    return min((assessments.get(k, {}).get("reasonableness", 0)
                if assumption_matches(record, k, assessments.get(k)) else 0
                for k in keys), default=0)


def current_computation(record, assessments, metric, *, ledger=None):
    """A computed conditional consequence is distinct from its evidential support."""
    row = record.get("metrics", record.get("output", {})).get(metric,
          record.get("intermediate_outputs", {}).get(metric, {}))
    if not isinstance(row, Mapping) or row.get("status") != "calculated":
        return False
    value = row.get("value")
    if not (is_duration_relation(row) or (row.get("limit") == "unbounded" and value is None) or
            (row.get("limit") == "finite" and type(value) in (int, float) and math.isfinite(value))):
        return False
    if not output_meets_contract(row, record.get("output_contracts", {}).get(metric)):
        return False
    dependency = record.get("metric_support", {}).get(metric,
                 record.get("intermediate_support", {}).get(metric, record))
    return current_premises(dependency, assessments, ledger=ledger)


def current_scenario_links(record, ledger):
    links = record.get("scenario_parameter_links", ())
    bindings = record.get("scenario_input_bindings", ())
    condition_bindings = record.get("scenario_condition_bindings", ())
    if not links and not bindings and not condition_bindings:
        return True
    if ledger is None:
        return False
    for binding in bindings:
        scenario = ledger.scenarios.get(binding["subject"], {})
        if binding["kind"] == "model_default":
            current = scenario.get("model_defaults", {}).get(binding["field"], {})
            if current.get("decision") != "retain":
                return False
        else:
            current = scenario.get("fixed_parameters", {}).get(binding["field"], {})
        if any(current.get(name) != binding[name] for name in ("value", "unit")):
            return False
    for binding in condition_bindings:
        scenario = ledger.scenarios.get(binding["subject"], {})
        if scenario.get("fixed_conditions", {}).get(binding["field"]) != binding["statement"]:
            return False
    from .scenario_sharing import shared_assumptions
    for link in links:
        if not any(a["key"] == link["assumption_key"] and (
                   identity(row) == link["relation_version"] or
                   (link.get("scope_version") == sharing_scope_version(row)
                    and link.get("assumption_claim") == assumption_claim(a)))
                   for a, row in shared_assumptions(ledger, link["target_subject"])):
            return False
    return True


def sharing_scope_version(relation):
    """Sharing scope is unchanged when only the evidence review is renewed."""
    return identity({key: value for key, value in relation.items() if key != "assumption_version"})


def current_premises(record, assessments, *, ledger=None):
    """Actual premise and scope changes govern numerical freshness, not review verdicts."""
    if record.get("invalidated_dependencies"):
        return False
    if not current_scenario_links(record, ledger):
        return False
    for key in record.get("assumption_keys", record.get("assumption_versions", {})):
        current = assessments.get(key)
        if not assumption_matches(record, key, current):
            return False
    return True


def admitted_calculation(record, assessments, *, ledger, metric=None, field=None):
    """Credit eligibility for reviewed calculations, never permission to execute."""
    from .progress import evidence_key

    if metric is not None and field is not None:
        raise ValueError("Select an output metric or a derived field, not both")
    invalid = {evidence_key(item) for assessment in assessments.values()
               for item in assessment.get("invalidated_evidence", ())}
    calculations = tuple(ledger.calculations.values())
    same_model = ("tool", "subject", "input_fact_keys", "conditions", "assumption_versions",
                  "operator", "basis", "parameterization")

    def reviewed(row):
        review = row.get("model_review", {})
        return review.get("admit") is True and review.get("completed", True) is True

    def producing_calculation(key):
        producers = [row for row in calculations if key in row.get("output_fact_keys", ())]
        if not producers:
            return None
        selected = producers[-1]
        if selected.get("model_review", {}).get("completed", True) is False:
            completed = [row for row in producers
                         if row.get("model_review", {}).get("completed", True) is True
                         and all(row.get(name) == selected.get(name) for name in same_model)]
            if completed:
                selected = completed[-1]
        return selected

    def visit(row, selected_metric, selected_field, visiting):
        key = (row.get("version") or identity(row), selected_metric, selected_field)
        if key in visiting or not reviewed(row):
            return False
        dependency = (row.get("metric_support", {}).get(selected_metric,
                      row.get("intermediate_support", {}).get(selected_metric, row))
                      if selected_metric is not None else
                      row.get("derived_support", {}).get(selected_field, row)
                      if selected_field is not None else row)
        if (not reviewed(dependency)
                or not current_premises(dependency, assessments, ledger=ledger)):
            return False
        records = list(row.get("input_records", ()))
        for version in row.get("calculation_versions", ()):
            calculation = ledger.calculations.get(version)
            if calculation is not None:
                records.extend(calculation.get("input_records", ()))
        inputs = {identity(item): item for item in records}
        inputs.update((fact.key, fact.as_record()) for fact in ledger.facts.values())
        visiting = {*visiting, key}
        for fact_key in dependency.get("input_fact_keys", row.get("input_fact_keys", ())):
            fact = inputs.get(fact_key)
            if fact is None or evidence_key(fact) in invalid:
                return False
            if fact.get("kind") == "derived":
                producer = producing_calculation(fact_key)
                if producer is None or not visit(producer, None, fact["field"], visiting):
                    return False
        return True

    return visit(record, metric, field, set())


def evidence_summary(evidence):
    """A decision view, not a replacement for the provenance-bearing record."""
    return {key: value for key, value in evidence.items() if key not in {"quote", "source_version"}}


def evidence_inventory(numeric, qualitative):
    """Index changed subjects/fields/sources without resending their whole bodies."""
    groups = {}
    for form, records in (("numeric", numeric), ("qualitative", qualitative)):
        for record in records:
            key = (record["subject"], record.get("source", ""), record.get("kind", "source_evidence"), form)
            group = groups.setdefault(key, {"subject": key[0], "source": key[1], "kind": key[2],
                                           "form": form, "fields": set(), "records": 0})
            group["fields"].update([record["field"]] if "field" in record else record.get("fields", ()))
            group["records"] += 1
    return [{**group, "fields": sorted(group["fields"])} for group in groups.values()]


@dataclass(frozen=True)
class Fact(ClosedRecord):
    subject: str
    field: str
    value: float
    unit: str
    kind: str
    basis: str
    source: str = ""
    quote: str = ""
    context: str = ""
    parents: tuple[str, ...] = ()
    source_version: str = ""

    def __post_init__(self):
        for name in ("subject", "field", "unit", "basis"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"Fact.{name} is required")
        object.__setattr__(self, "value", finite(self.value, self.field))
        if self.kind not in {"user_assumption", "request_constraint", "scenario_constraint", "model_default", "source_observation", "source_model_parameter", "model_assumption", "derived"}:
            raise ValueError("Unknown fact kind")
        if any(not isinstance(getattr(self, name), str) for name in ("source", "quote", "context")):
            raise ValueError("Fact source, quote and context must be text")
        if self.kind.startswith("source_") and not (self.source and self.quote and self.context):
            raise ValueError("Source facts require an exact quote and applicability context")

    @property
    def key(self):
        return identity(self.as_record())

    @property
    def credit_key(self):
        # Mirrors, paraphrases and repeated quotations do not manufacture yield.
        return identity([self.subject, self.field, self.value, self.unit])

    def as_record(self):
        return {"subject": self.subject, "field": self.field, "value": self.value,
                "unit": self.unit, "kind": self.kind, "basis": self.basis,
                "source": self.source, "quote": self.quote, "context": self.context,
                "parents": list(self.parents), "source_version": self.source_version}

    def semantic_record(self):
        return {k: v for k, v in self.as_record().items() if k != "parents"}


@dataclass(frozen=True)
class ResearchMessage(ClosedRecord):
    objective: str
    subjects: tuple[str, ...]
    proposal: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def as_record(self):
        return {"objective": self.objective, "subjects": list(self.subjects),
                "proposal": dict(self.proposal)}


@dataclass(frozen=True)
class Credit(ClosedRecord):
    identities: tuple[tuple[str, str], ...] = ()
    evaluated: bool = True
    qualities: tuple[tuple[str, str, float], ...] = ()
    quality_resolution: float | None = None

    def as_record(self):
        return {"identities": [list(pair) for pair in self.identities], "evaluated": self.evaluated,
                "qualities": [list(row) for row in self.qualities], "quality_resolution": self.quality_resolution}


@dataclass(frozen=True)
class ResearchResult(ClosedRecord):
    facts: tuple[Fact, ...] = ()
    estimates: tuple[Mapping[str, Any], ...] = ()
    findings: tuple[str, ...] = ()
    evaluated: bool = True
    stop_reason: str = ""
    evidence: tuple[Mapping[str, Any], ...] = ()
    assumptions: tuple[Mapping[str, Any], ...] = ()
    scenarios: tuple[Mapping[str, Any], ...] = ()
    diagnostics: tuple[Mapping[str, Any], ...] = ()
    calculations: tuple[Mapping[str, Any], ...] = ()
    reading_attempts: tuple[Mapping[str, Any], ...] = ()
    source_event_types: tuple[Mapping[str, Any], ...] = ()
    source_target_relationships: tuple[Mapping[str, Any], ...] = ()
    scenario_parameter_relations: tuple[Mapping[str, Any], ...] = ()

    def as_record(self):
        return {"facts": [f.as_record() for f in self.facts],
                "estimates": list(self.estimates), "findings": list(self.findings),
                "evaluated": self.evaluated,
                "stop_reason": self.stop_reason, "evidence": list(self.evidence),
                "assumptions": list(self.assumptions), "scenarios": list(self.scenarios),
                "diagnostics": list(self.diagnostics), "calculations": list(self.calculations),
                "reading_attempts": list(self.reading_attempts),
                "source_event_types": list(self.source_event_types),
                "source_target_relationships": list(self.source_target_relationships),
                "scenario_parameter_relations": list(self.scenario_parameter_relations)}

    def audit_identifiers(self):
        return tuple(f.key for f in self.facts)

    def credit(self, focus, *, scientific=False, subjects=None, outputs=None):
        if scientific:
            from .assumptions import scientific_credit
            return scientific_credit(self, focus, subjects=subjects, outputs=outputs)
        # A record cannot judge its contribution without the invoking contract.
        # Acquisition bindings perform that mandatory assessment before projection.
        return Credit(evaluated=self.evaluated)

    def synthesis(self):
        return {"evidence": [f.semantic_record() for f in self.facts],
                "estimates": [{"subject": e["subject"], "label": e["label"],
                               "metrics": e["metrics"]} for e in self.estimates],
                "findings": list(self.findings),
                "completion": self.stop_reason, "assumptions": [{k: v for k, v in a.items() if k != "key"} for a in self.assumptions],
                "qualitative_evidence": list(self.evidence), "scenarios": list(self.scenarios),
                "diagnostics": list(self.diagnostics), "reading_attempts": list(self.reading_attempts),
                "source_event_types": list(self.source_event_types),
                "source_target_relationships": list(self.source_target_relationships),
                "scenario_parameter_relations": list(self.scenario_parameter_relations)}


class Ledger:
    """Episode-owned admitted knowledge, updated only after a numeric verdict."""

    def __init__(self, facts=()):
        inherited = facts if isinstance(facts, ResearchResult) else ResearchResult(tuple(facts))
        self.facts = {}
        self.estimates = {}
        self.findings = []
        self.evidence = {}
        self.assumptions = {}
        self.scenarios = {}
        self.diagnostics = []
        self.calculations = {}
        self.reading_attempts = {}
        self.source_event_types = {}
        self.source_target_relationships = {}
        self.scenario_parameter_relations = {}
        self.merge(inherited)

    def merge(self, result):
        self.facts.update((f.key, f) for f in result.facts)
        for estimate in result.estimates:
            model = estimate.get("parameterization")
            if model:
                for previous_key, previous in tuple(self.estimates.items()):
                    if previous.get("parameterization") == model and previous["subject"] == estimate["subject"]:
                        combined = compatible_estimates(previous, estimate)
                        if combined is not None:
                            estimate = combined
                            del self.estimates[previous_key]
                            break
            key = identity([estimate["subject"], estimate["parameters"], estimate["conditions"],
                            estimate.get("assumption_versions", {}), model,
                            estimate.get("model_review", {}).get("completed", True)])
            previous = self.estimates.get(key)
            if previous:
                estimate = {**previous, **estimate, "metrics": {**previous["metrics"], **estimate["metrics"]},
                    "scenario_parameter_links": list({identity(row): row for row in
                        (*previous.get("scenario_parameter_links", ()), *estimate.get("scenario_parameter_links", ()))}.values()),
                    "metric_support": {**previous.get("metric_support", {}), **estimate.get("metric_support", {})},
                    "metric_dependencies": {**previous.get("metric_dependencies", {}), **estimate.get("metric_dependencies", {})},
                    "intermediate_outputs": {**previous.get("intermediate_outputs", {}), **estimate.get("intermediate_outputs", {})},
                    "intermediate_support": {**previous.get("intermediate_support", {}), **estimate.get("intermediate_support", {})},
                    "intermediate_dependencies": {**previous.get("intermediate_dependencies", {}), **estimate.get("intermediate_dependencies", {})},
                    "scenario_input_bindings": list({identity(row): row for row in
                        (*previous.get("scenario_input_bindings", ()), *estimate.get("scenario_input_bindings", ()))}.values()),
                    "scenario_condition_bindings": list({identity(row): row for row in
                        (*previous.get("scenario_condition_bindings", ()), *estimate.get("scenario_condition_bindings", ()))}.values()),
                    "output_contracts": {**previous.get("output_contracts", {}), **estimate.get("output_contracts", {})},
                    "calculation_versions": list(dict.fromkeys([*previous.get("calculation_versions", ()),
                                                                *estimate.get("calculation_versions", ())]))}
            self.estimates[key] = estimate
        self.findings.extend(v for v in result.findings if v not in self.findings)
        self.evidence.update((identity(e), e) for e in result.evidence)
        self.assumptions.update((a["key"], a) for a in result.assumptions)
        self.scenarios.update((s["subject"], s) for s in result.scenarios)
        self.diagnostics.extend(d for d in result.diagnostics if d not in self.diagnostics)
        self.calculations.update((c["version"], c) for c in result.calculations)
        self.reading_attempts.update((identity(a), a) for a in result.reading_attempts)
        self.source_event_types.update((row["type_id"], row) for row in result.source_event_types)
        self.source_target_relationships.update(
            ((row["source"], row["source_subject"], row["target_subject"], row["parameter"]), row)
            for row in result.source_target_relationships)
        self.scenario_parameter_relations.update(
            (scenario_relation_key(row), row)
            for row in result.scenario_parameter_relations)

    def options(self):
        return {f"fact:{i}": f for i, f in enumerate(self.facts.values(), 1)}

    def evidence_options(self):
        records = [f.semantic_record() for f in self.facts.values()
                   if f.kind in {"source_observation", "source_model_parameter", "request_constraint"}]
        return {f"evidence:{i}": e for i, e in enumerate([*records, *self.evidence.values()], 1)}

    def assumption_options(self):
        numeric = numeric_references(self)
        return {f"assumption:{i}": assessment_summary(a, numeric_options=numeric)
                for i, a in enumerate(self.assumptions.values(), 1)}

    def calculation_options(self):
        return {f"calculation:{i}": c for i, c in enumerate(self.calculations.values(), 1)}

    def result(self, **kwargs):
        return ResearchResult(tuple(self.facts.values()), tuple(self.estimates.values()),
                              tuple(self.findings), evidence=tuple(self.evidence.values()),
                              assumptions=tuple(self.assumptions.values()), scenarios=tuple(self.scenarios.values()),
                              diagnostics=tuple(self.diagnostics), calculations=tuple(self.calculations.values()),
                              reading_attempts=tuple(self.reading_attempts.values()),
                              source_event_types=tuple(self.source_event_types.values()),
                              source_target_relationships=tuple(self.source_target_relationships.values()),
                              scenario_parameter_relations=tuple(self.scenario_parameter_relations.values()), **kwargs)


def compatible_estimates(left, right):
    """Join tool outputs only within an explicitly named, compatible model."""
    # Check both records, including duplicate output names that a dict merge hides.
    model_ids = {}
    for estimate in (left, right):
        if not exposure_outputs_agree(estimate["metrics"]):
            return None
        for row in estimate["metrics"].values():
            if row.get("status") == "calculated" and row.get("model_id"):
                background = row["background"]
                if model_ids.setdefault(background, row["model_id"]) != row["model_id"]:
                    return None
    if (left.get("support_complete", True) != right.get("support_complete", True) or
            left.get("model_review", {}).get("admit", True) != right.get("model_review", {}).get("admit", True) or
            left.get("model_review", {}).get("completed", True) != right.get("model_review", {}).get("completed", True)):
        return None
    for name in ("parameters", "conditions"):
        if any(left.get(name, {})[k] != value for k, value in right.get(name, {}).items()
               if k in left.get(name, {})):
            return None
    for key, version in right.get("assumption_versions", {}).items():
        if key in left.get("assumption_versions", {}) and left["assumption_versions"][key] != version:
            claim = left.get("assumption_claims", {}).get(key)
            if claim is None or claim != right.get("assumption_claims", {}).get(key):
                return None
    parameters = {**left["parameters"], **right["parameters"]}
    conditions = {**left["conditions"], **right["conditions"]}
    from ...task_executors.scenario_metrics.evaluate import evaluate_parameters
    try:
        evaluate_parameters(parameters, conditions, (), subject=left["subject"], label=left["label"])
    except ValueError:
        return None
    support = {a["key"]: a for a in (*left.get("assumption_support", ()), *right.get("assumption_support", ()))}
    return {**left, **right, "parameters": parameters, "conditions": conditions,
            "scenario_parameter_links": list({identity(row): row for row in
                (*left.get("scenario_parameter_links", ()), *right.get("scenario_parameter_links", ()))}.values()),
            "metrics": {**left["metrics"], **right["metrics"]},
            "metric_support": {**left.get("metric_support", {}), **right.get("metric_support", {})},
            "metric_dependencies": {**left.get("metric_dependencies", {}), **right.get("metric_dependencies", {})},
            "intermediate_outputs": {**left.get("intermediate_outputs", {}), **right.get("intermediate_outputs", {})},
            "intermediate_support": {**left.get("intermediate_support", {}), **right.get("intermediate_support", {})},
            "intermediate_dependencies": {**left.get("intermediate_dependencies", {}), **right.get("intermediate_dependencies", {})},
            "scenario_input_bindings": list({identity(row): row for row in
                (*left.get("scenario_input_bindings", ()), *right.get("scenario_input_bindings", ()))}.values()),
            "scenario_condition_bindings": list({identity(row): row for row in
                (*left.get("scenario_condition_bindings", ()), *right.get("scenario_condition_bindings", ()))}.values()),
            "output_contracts": {**left.get("output_contracts", {}), **right.get("output_contracts", {})},
            "derived_dependencies": {**left.get("derived_dependencies", {}), **right.get("derived_dependencies", {})},
            "input_provenance": {**left.get("input_provenance", {}), **right.get("input_provenance", {})},
            "derived_parameters": {**left.get("derived_parameters", {}), **right.get("derived_parameters", {})},
            "assumption_versions": {**left.get("assumption_versions", {}), **right.get("assumption_versions", {})},
            "assumption_claims": {**left.get("assumption_claims", {}), **right.get("assumption_claims", {})},
            "assumption_keys": list(dict.fromkeys([*left.get("assumption_keys", ()), *right.get("assumption_keys", ())])),
            "invalidated_dependencies": list(dict.fromkeys([*left.get("invalidated_dependencies", ()),
                                                             *right.get("invalidated_dependencies", ())])),
            "assumption_support": list(support.values()),
            "reasonableness": min(left.get("reasonableness", 0), right.get("reasonableness", 0)),
            "calculation_versions": list(dict.fromkeys([*left.get("calculation_versions", ()), *right.get("calculation_versions", ())]))}
