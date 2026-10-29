"""Versioned scientific sheet shared through ordinary ResearchResult deltas.

This module stores public scientific arguments and their premises, never private
agent histories. It contains no agent loop, task selector, or episode itinerary.
"""

from copy import deepcopy
from hashlib import sha256
import json

from ...credit_assignment.causal_sheet import compile_sheet
from ...credit_assignment import causal_support
from .grading import GRADING_FIELDS, module_for, grade_record, compact_calculation
from .importance import (PROTOCOL_FIELDS, RESULT_FIELDS, validate_protocol, validate_result,
                         result_key, score_importance, protocol_identity)
from . import source_formalization


# Deliberately closed records. References below are scientific object identities,
# not child execution identities. Every referenced input is version-bound.
FIELDS = {
    "inference": ("statement", "variables", "intervention", "outcome", "grain", "domain"),
    "premise": ("statement", "scope", "basis"),
    "theory": ("name", "mechanism", "predictions", "admission", "same_as", "distinction", "compared_with"),
    "application": ("inference", "inference_version", "theory", "context_key", "context", "context_distinction", "direction",
                    "inference_use", "consequence_without_inference"),
    "evidence": ("source", "locator", "study_id", "finding", "measurements"),
    "source_formalization": source_formalization.FIELDS,
    "reasoning": ("author", "premises", "steps", "conclusion", "limitations"),
    "verification": ("reasoning", "reviewer", "verdict", "checks", "counterexample"),
    "comparison": ("left", "right", "mapping", "conflict", "repair_search_bounds"),
    "repair": ("comparison", "adjustment", "construction_evidence", "predictions", "test_evidence", "costs"),
    "expectation": ("inference", "bounds", "basis", "rationale", "grounds", "revision"),
    "expectation_break": ("expectation", "triggers", "expected", "observed", "conflict", "limitations"),
    "expectation_support": ("inference", "expectation", "grounds", "argument", "limitations"),
    "support_contribution": causal_support.SUPPORT_FIELDS,
    "causal_change": causal_support.CHANGE_FIELDS,
    "importance": PROTOCOL_FIELDS,
    "importance_result": RESULT_FIELDS,
}
FIELDS.update(GRADING_FIELDS)

LIST_FIELDS = {"variables", "predictions", "compared_with", "premises", "steps", "checks",
               "construction_evidence", "test_evidence", "sources"}
ENUMS = {
    "admission": {"candidate", "distinct", "same_theory"},
    "direction": {"supports", "possible_reversal", "unresolved"},
    "verdict": {"valid", "invalid", "unresolved"},
}

BOUND_BASES = {"literature", "cross_literature_reasoning", "provisional_judgment"}
BOUND_CHANGES = {"expand_bounds", "shrink_bounds", "change_inference", "repair_theory", "reconsider"}
REPAIR_KINDS = {"not_a_repair", "better_definition", "missing_dependency", "general_rule", "exceptional_patch", "unresolved"}
REVISION_FIELDS = {"prior", "triggers", "changes", "change", "repair_kind", "proposed_rule",
                   "general_rules", "rule_reasoning", "retained", "lost", "new_predictions", "limitations"}
TRIGGER_KINDS = {"evidence", "source_formalization", "premise", "theory", "application", "reasoning", "verification",
                 "comparison", "repair", "expectation_break", "importance_result", *GRADING_FIELDS}


def _object(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{name} requires exactly {sorted(fields)}")


def _items(value, name, *, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{name} must be a {'nonempty ' if nonempty else ''}list")
    return value


def _text(value, field, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{field} must be {'possibly empty' if empty else 'nonempty'} text")


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode()).hexdigest()


class CausalSheet:
    """Candidate inferences with scoped theory applications and assessments.

    Updates use optimistic version checks: a stale child cannot overwrite a
    changed premise or judgment. Old dependent judgments stay in the audit but
    cease contributing until reconsidered against current inputs.
    """

    def __init__(self, updates=()):
        self.rows = {}
        self.history = []
        self._versions = set()
        self._snapshots = {}
        self.merge(updates)

    def of_kind(self, kind):
        return [(key, row) for key, row in self.rows.items() if row["kind"] == kind]

    def propose(self, key, kind, data, *, dependencies=()):
        """Construct a validated update without mutating the admitted sheet."""
        _text(key, "record key")
        data = deepcopy(data)
        if kind == "application" and not data.get("inference_version"):
            data["inference_version"] = self.rows[data["inference"]]["version"]
        refs = set(dependencies) | self._references(kind, data)
        missing = refs - self.rows.keys()
        if missing:
            raise ValueError(f"Unknown scientific dependencies: {sorted(missing)}")
        previous = self.rows.get(key)
        update = {"key": key, "kind": kind, "data": data,
                  "previous": previous["version"] if previous else None,
                  "dependencies": {ref: self.rows[ref]["version"] for ref in sorted(refs)}}
        if kind == "application":
            selected = self.snapshot({"key": data["inference"], "version": data["inference_version"]})
            if selected["kind"] != "inference":
                raise ValueError("Application must identify a version of G")
            update["dependencies"][data["inference"]] = data["inference_version"]
        if previous and all(previous[name] == update[name] for name in ("kind", "data", "dependencies")):
            return deepcopy(previous)
        update["version"] = _digest(update)
        self._validate(update)
        return update

    def put(self, key, kind, data, *, dependencies=()):
        update = self.propose(key, kind, data, dependencies=dependencies)
        self.merge((update,))
        return update

    def merge(self, updates):
        """Validate the complete delta before committing any of it."""
        staged = object.__new__(CausalSheet)
        staged.rows = dict(self.rows)
        staged.history = list(self.history)
        staged._versions = set(self._versions)
        staged._snapshots = dict(self._snapshots)
        for update in updates:
            update = deepcopy(update)
            if not isinstance(update, dict) or set(update) != {
                    "key", "kind", "data", "previous", "dependencies", "version"}:
                raise ValueError("Malformed causal sheet update")
            if update["version"] != _digest({k: v for k, v in update.items() if k != "version"}):
                raise ValueError("Causal sheet update digest mismatch")
            if update["version"] in staged._versions:
                continue
            staged._validate(update)
            staged.rows[update["key"]] = update
            staged.history.append(update)
            staged._versions.add(update["version"])
            staged._snapshots[update["version"]] = update
        self.rows, self.history, self._versions = staged.rows, staged.history, staged._versions
        self._snapshots = staged._snapshots

    def _references(self, kind, data):
        expected = set(FIELDS.get(kind, ()))
        if kind in {"support_contribution", "causal_change"} and isinstance(data, dict):
            accepted = getattr(causal_support, "accepted_fields", None)
            if accepted is not None:
                expected = set(accepted(kind, data))
            elif "verification_required" in data:
                expected.add("verification_required")  # Retired audit field, no scoring meaning.
        if kind == "support_assessment" and isinstance(data, dict) and "source_formalization" not in data:
            expected.discard("source_formalization")  # Historical assessments remain readable, without hypothesis credit.
        if kind == "application" and isinstance(data, dict) and "inference_version" not in data:
            expected.discard("inference_version")
        if kind not in FIELDS or not isinstance(data, dict) or set(data) != expected:
            raise ValueError(f"{kind!r} requires exactly its declared sheet fields")
        refs = []
        if kind in GRADING_FIELDS:
            refs = list(module_for(kind).references(data))
        elif kind == "theory":
            # Identity comparisons are metadata, not inferential premises. Making
            # them causal dependencies would prevent later duplicate consolidation.
            refs = [data["same_as"]] if data["same_as"] else []
        elif kind == "application":
            refs = [data["inference"], data["theory"]]
        elif kind == "reasoning":
            refs = data["premises"]
        elif kind == "verification":
            refs = [data["reasoning"]]
        elif kind == "comparison":
            refs = [data["left"], data["right"]]
        elif kind == "repair":
            refs = [data["comparison"], *data["construction_evidence"], *data["test_evidence"]]
        elif kind == "importance":
            refs = [data["application"]]
        elif kind == "source_formalization":
            refs = source_formalization.references(data)
        elif kind == "importance_result":
            refs = [data["assessment"]]
        elif kind in {"support_contribution", "causal_change"}:
            refs = causal_support.references(kind, data)
        elif kind in {"expectation", "expectation_support"}:
            refs = [*_items(data["grounds"], "grounds")]
            if kind == "expectation":
                refs.append(data["inference"])
        if not isinstance(refs, list) or any(not isinstance(ref, str) or not ref for ref in refs):
            raise ValueError("Scientific references must be nonempty record keys")
        return set(refs)

    def snapshot(self, reference):
        """Resolve an immutable audit reference, never a live dependency edge."""
        _object(reference, {"key", "version"}, "snapshot reference")
        for name in ("key", "version"):
            _text(reference[name], name)
        row = self._snapshots.get(reference["version"])
        if row is None or row["key"] != reference["key"]:
            raise ValueError("Unknown scientific snapshot key/version")
        return deepcopy(row)

    def anchor(self, key):
        return {"key": key, "version": self.rows[key]["version"]}

    def snapshot_references(self, kind, data):
        """Direct audit references only; no recursive expansion of public arguments."""
        if kind in GRADING_FIELDS:
            function = getattr(module_for(kind), "snapshot_references", None)
            return list(function(data)) if function else []
        if kind == "application" and data.get("inference_version"):
            return [{"key": data["inference"], "version": data["inference_version"]}]
        if kind == "causal_change":
            return [data["source"], data["target"]]
        if kind == "expectation_break":
            return [data["expectation"], *data["triggers"]]
        if kind == "expectation_support":
            return [data["inference"], data["expectation"]]
        if kind == "expectation" and data["revision"] is not None:
            revision = data["revision"]
            return [revision["prior"], *revision["triggers"],
                    *(item["application"] for name in ("retained", "lost") for item in revision[name])]
        return []

    def _validate_bounds(self, key, kind, data):
        def typed_snapshot(reference, kinds):
            row = self.snapshot(reference)
            if row["kind"] not in kinds:
                raise ValueError(f"Snapshot must reference one of {sorted(kinds)}")
            return row

        def triggers(items):
            for item in _items(items, "triggers", nonempty=True):
                typed_snapshot(item, TRIGGER_KINDS)

        if kind == "expectation_support":
            inference = typed_snapshot(data["inference"], {"inference"})
            expectation = typed_snapshot(data["expectation"], {"expectation"})
            if (expectation["data"]["inference"] != inference["key"] or
                    expectation["dependencies"][inference["key"]] != inference["version"]):
                raise ValueError("Support must address the exact G version bound by the expected generalizability bounds")
            for ref in _items(data["grounds"], "grounds", nonempty=True):
                if ref not in self.rows or self.rows[ref]["kind"] not in TRIGGER_KINDS:
                    raise ValueError("Support requires scientific evidence, theory or argument inputs")
                row = self.rows[ref]
                if row["kind"] == "application" and (row["data"]["inference"] != inference["key"] or
                        row["dependencies"][inference["key"]] != inference["version"]):
                    raise ValueError("A supporting application must address the selected historical G version")
            for name in ("argument", "limitations"):
                _text(data[name], name)
            previous = self.rows.get(key)
            if previous and any(previous["data"][name] != data[name] for name in ("inference", "expectation")):
                raise ValueError("A support identity retains its exact G and expected-bounds versions")
            return
        if kind == "expectation_break":
            previous = self.rows.get(key)
            if previous and previous["data"]["expectation"] != data["expectation"]:
                raise ValueError("A break retains the exact expectation it challenged")
            signature = sorted({ref["key"] for ref in data["triggers"]})
            if any(other != key and row["data"]["expectation"] == data["expectation"] and
                   sorted({ref["key"] for ref in row["data"]["triggers"]}) == signature
                   for other, row in self.of_kind("expectation_break")):
                raise ValueError("Update the existing break for these findings; rewording creates no new break")
            typed_snapshot(data["expectation"], {"expectation"})
            triggers(data["triggers"])
            for name in ("expected", "observed", "conflict", "limitations"):
                _text(data[name], name)
            return
        self._require_kind(data["inference"], "inference")
        if data["basis"] not in BOUND_BASES:
            raise ValueError("Expected generalizability bounds need a declared basis")
        for name in ("bounds", "rationale"):
            _text(data[name], name)
        if data["basis"] != "provisional_judgment" and not data["grounds"]:
            raise ValueError("Grounded expectations need scientific inputs; provisional judgment may omit them")
        for ref in data["grounds"]:
            if self.rows[ref]["kind"] not in TRIGGER_KINDS:
                raise ValueError("Expectation grounds must be scientific evidence, theory or argument inputs")
        revision, previous = data["revision"], self.rows.get(key)
        if revision is None:
            if previous:
                raise ValueError("Revising expected generalizability bounds requires the prior expectation and an explanation")
            return
        _object(revision, REVISION_FIELDS, "expectation revision")
        prior = typed_snapshot(revision["prior"], {"expectation"})
        if previous and revision["prior"] != self.anchor(key):
            raise ValueError("Revision must preserve the immediately preceding expectation snapshot")
        if prior["data"]["inference"] != data["inference"]:
            raise ValueError("Revise G under its existing inference identity; do not rename it through a bounds revision")
        triggers(revision["triggers"])
        changes = _items(revision["changes"], "changes", nonempty=True)
        if any(not isinstance(item, str) or item not in BOUND_CHANGES for item in changes):
            raise ValueError("Unknown expected generalizability bounds revision operation")
        if prior["dependencies"][data["inference"]] != self.rows[data["inference"]]["version"] and "change_inference" not in changes:
            raise ValueError("Changed G must be acknowledged explicitly in the expectation revision")
        if revision["repair_kind"] not in REPAIR_KINDS:
            raise ValueError("Unknown proposed repair interpretation")
        for name in ("change", "proposed_rule", "general_rules", "rule_reasoning", "limitations"):
            _text(revision[name], name)
        for name in ("retained", "lost"):
            for item in _items(revision[name], name):
                _object(item, {"application", "effect"}, name + " consequence")
                application = typed_snapshot(item["application"], {"application"})
                if application["data"]["inference"] != data["inference"]:
                    raise ValueError("Consequence must refer to an application of this G")
                _text(item["effect"], "consequence effect")
        for prediction in _items(revision["new_predictions"], "new_predictions"):
            _text(prediction, "new prediction")

    def _require_kind(self, key, kind):
        if key not in self.rows or self.rows[key]["kind"] != kind:
            raise ValueError(f"{key!r} must reference a {kind} record")

    def _validate(self, update):
        key, kind, data = update["key"], update["kind"], update["data"]
        _text(key, "record key")
        if len(key) > 160:
            raise ValueError("Scientific keys are identifiers, not prose; maximum 160 characters")
        refs = self._references(kind, data)
        if kind in GRADING_FIELDS:
            module_for(kind).validate(self, key, data)
        elif kind == "importance":
            validate_protocol(data)
        elif kind == "source_formalization":
            source_formalization.validate(self, key, data)
        elif kind == "importance_result":
            self._require_kind(data["assessment"], "importance")
            validate_result(data, self.rows[data["assessment"]]["data"])
        elif kind in {"expectation", "expectation_break", "expectation_support"}:
            self._validate_bounds(key, kind, data)
        elif kind in {"support_contribution", "causal_change"}:
            causal_support.validate(self, key, kind, data)
        else:
            for name, value in data.items():
                if name in LIST_FIELDS:
                    if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
                        raise ValueError(f"{name} must be a list of nonempty strings")
                elif name == "measurements":
                    if not isinstance(value, dict):
                        raise ValueError("measurements must retain native named values/units/uncertainties")
                else:
                    _text(value, name, empty=name in {"same_as", "reasoning", "counterexample"})
                if name in ENUMS and value not in ENUMS[name]:
                    raise ValueError(f"Invalid {name}: {value!r}")
        prior = self.rows.get(key)
        if update["previous"] != (prior["version"] if prior else None):
            raise ValueError(f"Stale update for {key}; reconsider against the current sheet")
        if prior and prior["kind"] != kind:
            raise ValueError("A scientific identity cannot change record kind")
        dependencies = update["dependencies"]
        if not isinstance(dependencies, dict) or not refs <= dependencies.keys() or key in dependencies:
            raise ValueError("All inputs must be version-bound; self-dependency is invalid")
        for ref, version in dependencies.items():
            historical_claim = (kind == "application" and ref == data["inference"] and
                                version == data["inference_version"] and version in self._snapshots)
            if ref not in self.rows or (self.rows[ref]["version"] != version and not historical_claim):
                raise ValueError(f"Changed or missing dependency: {ref}")
            if self._depends_on(ref, key):
                raise ValueError("Inferential dependency cycle; keep competing arguments separate")
        if kind == "theory":
            for ref in data["compared_with"]:
                self._require_kind(ref, "theory")
            if data["admission"] == "same_theory":
                self._require_kind(data["same_as"], "theory")
            elif data["same_as"]:
                raise ValueError("Only same_theory admissions may name same_as")
            if data["admission"] == "distinct":
                existing = {k for k, r in self.of_kind("theory")
                            if k != key and r["data"]["admission"] == "distinct"}
                if not existing <= set(data["compared_with"]):
                    raise ValueError("Distinct theory admission must compare existing distinct theories; otherwise retain candidate")
        elif kind == "application":
            self._require_kind(data["inference"], "inference")
            selected = self.snapshot({"key": data["inference"], "version": data["inference_version"]})
            if selected["kind"] != "inference" or dependencies[data["inference"]] != data["inference_version"]:
                raise ValueError("Application must bind exactly its declared G version")
            self._require_kind(data["theory"], "theory")
            canonical = self.canonical_theory(data["theory"])
            if prior is None and any(self.canonical_theory(row["data"]["theory"]) == canonical
                    and row["data"]["inference"] == data["inference"]
                    and row["data"]["inference_version"] == data["inference_version"]
                    and row["data"]["context_key"] == data["context_key"]
                    for _, row in self.of_kind("application")):
                raise ValueError("This inference/theory/context already has an application; attach evidence or revise its assessment")
            if prior and tuple(prior["data"][k] for k in ("inference", "inference_version", "theory", "context_key")) != tuple(data[k] for k in ("inference", "inference_version", "theory", "context_key")):
                raise ValueError("An application identity retains its inference, theory and context")
        elif kind == "reasoning":
            if not data["steps"] or not data["premises"]:
                raise ValueError("A public derivation needs explicit premises and checkable steps")
        elif kind == "verification":
            self._require_kind(data["reasoning"], "reasoning")
            if data["reviewer"] == self.rows[data["reasoning"]]["data"]["author"]:
                raise ValueError("Independent verification cannot be self-certification")
            if not data["checks"]:
                raise ValueError("Verification must identify the checked derivation steps or premises")
        elif kind == "comparison":
            for ref in (data["left"], data["right"]):
                self._require_kind(ref, "application")
            if self.rows[data["left"]]["data"]["inference"] != self.rows[data["right"]]["data"]["inference"]:
                raise ValueError("Compare uses of the same candidate inference; state the mapped opposing commitment in the reversal application")
            if data["left"] == data["right"]:
                raise ValueError("A comparison needs different applications")
        elif kind == "repair":
            self._require_kind(data["comparison"], "comparison")
            for ref in (*data["construction_evidence"], *data["test_evidence"]):
                self._require_kind(ref, "evidence")
        elif kind == "importance":
            self._require_kind(data["application"], "application")
            if any(other != key and protocol_identity(row["data"]) == protocol_identity(data) for other, row in self.of_kind("importance")):
                raise ValueError("Reuse the same assessment for an identical comparison protocol")
            if prior and prior["data"] != data:
                raise ValueError("Comparison protocols are immutable; record a separate explicit contrast or domain revision")
        elif kind == "importance_result":
            if key != result_key(data["assessment"], data["circumstance"]):
                raise ValueError("One canonical result per assessment and circumstance")

    def _depends_on(self, start, target):
        todo, seen = [start], set()
        while todo:
            key = todo.pop()
            if key == target:
                return True
            if key not in seen:
                seen.add(key)
                todo.extend(self.rows[key]["dependencies"])
        return False

    def inferences_for(self, keys):
        """Find only inference identities relevant to explicitly selected records."""
        todo, seen, found = list(keys), set(), set()
        while todo:
            key = todo.pop()
            if key in seen:
                continue
            seen.add(key)
            row = self.rows[key]
            if row["kind"] == "inference":
                found.add(key)
            if row["kind"] == "expectation_break":
                found.add(self.snapshot(row["data"]["expectation"])["data"]["inference"])
            elif row["kind"] == "expectation_support":
                found.add(row["data"]["inference"]["key"])
            todo.extend(row["dependencies"])
        return sorted(found)

    def _context_row(self, key, version, pins):
        if version is not None and version in pins.get(key, set()):
            return self._snapshots[version]
        row = self.rows[key]
        return row if version is None or row["version"] == version else None

    def _claim_pins(self, row, pins):
        additional = {}
        if row["kind"] in GRADING_FIELDS:
            additional = module_for(row["kind"]).context_pins(self, row["data"])
        elif row["kind"] == "expectation_support":
            for name in ("inference", "expectation"):
                ref = row["data"][name]
                additional.setdefault(ref["key"], set()).add(ref["version"])
        elif row["kind"] in {"support_contribution", "causal_change"}:
            additional = causal_support.context_pins(self, row["kind"], row["data"])
        merged = {key: set(versions) for key, versions in pins.items()}
        for key, versions in additional.items():
            merged.setdefault(key, set()).update(versions)
        return merged

    def current(self, key, *, _version=None, _pins=None):
        pins = _pins or {}
        row = self._context_row(key, _version, pins)
        if row is None:
            return False
        pins = self._claim_pins(row, pins)
        return all(self.current(ref, _version=version, _pins=pins)
                   for ref, version in row["dependencies"].items())

    def canonical_theory(self, key):
        self._require_kind(key, "theory")
        while self.rows[key]["data"]["admission"] == "same_theory":
            key = self.rows[key]["data"]["same_as"]
        return key

    def usable(self, key, *, _version=None, _pins=None):
        """Version consistency only; model opinions cannot admit or refute operands."""
        pins = _pins or {}
        row = self._context_row(key, _version, pins)
        if row is None:
            return False
        pins = self._claim_pins(row, pins)
        if not self.current(key, _version=_version, _pins=pins):
            return False
        return all(self.usable(ref, _version=version, _pins=pins) for ref, version in row["dependencies"].items())

    def calculate(self):
        return compile_sheet(self)

    def calculation_pins(self, context):
        if not context:
            return {}
        if context not in self.rows or self.rows[context]["kind"] not in {"expectation_support", "support_contribution", "causal_change", *GRADING_FIELDS}:
            raise ValueError("Calculation context must identify a scientific support or change record")
        return self._claim_pins(self.rows[context], {})

    def calculation_view(self, keys, *, context=None, offset=None, limit=8):
        """Selected calculated results, never an unbounded global ledger in a prompt."""
        if not isinstance(keys, list) or not 1 <= len(keys) <= 8 or len(set(keys)) != len(keys):
            raise ValueError("Select one to eight distinct assessment keys")
        detail_requested = offset is not None
        offset = 0 if offset is None else offset
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 8:
            raise ValueError("Calculation offset must be nonnegative and limit within 1..8")
        pins = self.calculation_pins(context)
        result = {}
        for key in keys:
            kind = self.rows[key]["kind"]
            if kind == "importance":
                from .dependence import dependence_result
                cases = self.rows[key]["data"]["domain"]["cases"]
                dependence = dependence_result(self, key, _pins=pins,
                    case_ids=[case["id"] for case in cases[offset:offset + limit]] if detail_requested else [])
                dependence_summary = {name: dependence[name] for name in
                    ("application", "inference", "formal_relationship", "dependence")}
                if detail_requested:
                    dependence_summary["cases"] = dependence["cases"]
                dependence_summary["case_page"] = {"offset": offset, "total": len(cases),
                    "next_offset": offset + limit if offset + limit < len(cases) else None}
                value = {**score_importance(self, key, _pins=pins), "calculation_context": context,
                         "dependence_sheet": dependence_summary}
            elif kind in GRADING_FIELDS:
                value = grade_record(self, key)
            else:
                value = causal_support.selected_credit(self, key)
            result[key] = compact_calculation(value, offset=0 if kind == "importance" else offset, limit=limit, summary=not detail_requested)
        return result

    def view(self, keys):
        """Explicitly scoped child inputs; no recursive expansion of the ledger."""
        keys = list(dict.fromkeys(keys))
        result = [{"key": key, "kind": self.rows[key]["kind"], "data": self.rows[key]["data"],
                   "version": self.rows[key]["version"], "dependencies": self.rows[key]["dependencies"],
                   "previous": self.rows[key]["previous"],
                   "current": self.current(key)} for key in keys]
        return deepcopy(result)

    def snapshot_view(self, references):
        """At most eight exact historical records, with their original input pins."""
        if not isinstance(references, (list, tuple)) or len(references) > 8:
            raise ValueError("Select at most eight scientific snapshots")
        result = []
        for reference in references:
            row = self.snapshot(reference)
            # 'current' has no scientific meaning for an immutable audit snapshot.
            result.append({**row, "audit_snapshot": True,
                           "is_latest_version": self.rows[row["key"]]["version"] == row["version"]})
        return result

    def return_page(self, updates, *, offset=0, limit=8, calculation_context=None):
        """Small deterministic change report; large derivations remain addressable."""
        keys = list(dict.fromkeys(row["key"] for row in updates))
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 8:
            raise ValueError("Return pagination requires offset >= 0 and limit in 1..8")
        rows = []
        for key in keys[offset:offset + limit]:
            row, status = self.rows[key], "recorded"
            data = row["data"]
            if row["kind"] in {"importance", "importance_result"}:
                assessment = key if row["kind"] == "importance" else data["assessment"]
                score = score_importance(self, assessment, _pins=self.calculation_pins(calculation_context))
                status = "calculated" if score["value"] is not None else "unresolved"
                detail = {"assessment": assessment, "calculation_context": calculation_context,
                          "calculation_current": score["current"], **{name: score[name] for name in
                    ("quantity", "lower", "upper", "value", "resolved_fraction")}}
            elif row["kind"] in GRADING_FIELDS:
                graded = grade_record(self, key)
                detail = {"calculation": compact_calculation(graded, limit=2, summary=True)}
                status = graded.get("status", "calculated")
            elif row["kind"] in {"support_contribution", "causal_change"}:
                detail = {"calculated_credit": compact_calculation(causal_support.selected_credit(self, key), limit=2, summary=True)}
            else:
                detail = {name: data[name] for name in ("admission", "direction") if name in data}
                if row["kind"] == "verification":
                    detail.update(legacy_model_opinion=data["verdict"], verification_established=False, scoring_effect="none")
                if row["kind"] == "evidence":
                    measurements = data["measurements"]
                    detail.update(source=data["source"][:500], locator=data["locator"][:240],
                                  study_id=data["study_id"][:160],
                                  source_version=measurements.get("source_version"),
                                  bearing=measurements.get("relevance_basis", "")[:800],
                                  reasoning_question=measurements.get("reasoning_question", "")[:800])
                elif row["kind"] == "source_formalization":
                    detail.update(evidence=data["evidence"], assessment=data["assessment"],
                                  method=data["method"], quantity=data["quantity"], units=data["units"])
                elif row["kind"] == "expectation":
                    revision = data["revision"]
                    detail.update(basis=data["basis"], prior=revision["prior"] if revision else None,
                                  changes=revision["changes"] if revision else [])
                elif row["kind"] == "expectation_break":
                    detail["expectation"] = data["expectation"]
                elif row["kind"] == "expectation_support":
                    detail.update(inference=data["inference"], expectation=data["expectation"])
            meaning = " ".join(str(data[name]) for name in (
                "statement", "name", "distinction", "context", "inference_use", "finding",
                "conclusion", "argument", "derivation", "limitations", "conflict", "adjustment", "costs", "basis", "bounds", "rationale", "question", "contrast", "error") if name in data)
            if row["kind"] == "verification":
                meaning = " ".join(data["checks"]) + " " + data["counterexample"]
            rows.append({"key": key, "kind": row["kind"], "version": row["version"], "status": status,
                         "current": self.current(key), "finding": meaning[:360],
                         "finding_truncated": len(meaning) > 360, **detail})
        return {"changes": rows, "total_changes": len(keys),
                "next_offset": offset + limit if offset + limit < len(keys) else None}
