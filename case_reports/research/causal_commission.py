"""Scoped jobs and readers for the causal sheet, used by existing episodes.

No model is called here. A reasoning episode may repeatedly use these functions,
inspect public arguments and calculate their consequences. Literature has separate contracts.
"""

from dataclasses import dataclass

from ...credit_assignment.causal_sheet import CRITERIA, investigation_credit
from ...credit_assignment import causal_support
from .causal_sheet import (CausalSheet, FIELDS, LIST_FIELDS, ENUMS, BOUND_BASES,
                           BOUND_CHANGES, REPAIR_KINDS)
from .importance import protocol_schema, score_importance
from .grading import GRADING_FIELDS, module_for


TASK = """Investigate the parent's scoped discovery question about candidate causal inferences or examples.
Choose useful reasoning, evidence acquisition, or executable calculation within your commission.
Explore new candidates when the commission calls for discovery; use explicit existing
inference identities when examining their applications or bounds. Do not rename the
same claim to create another candidate. Each application names the inference it assesses.
Your scientific arguments may be first-principles deductions, model comparisons,
counterexamples, or derivations of predictions. State the premises, public checkable
steps, conclusion, and limits. Logical validity alone does not establish empirical premises.

When exploring theories, distinguish a substantive new mechanism or predictive structure
from another application of the same theory. Explain how the focal inference contributes:
which prediction or explanation changes if the inference is removed or reversed?
Attach essentially identical applications to the same theory and group genuinely identical
contexts under the same context_key. A different context can add generalizability evidence
without adding another theory. Compare proposed new theories with the existing theory
catalogue. Keep uncertain identities as candidates. Retain possible causal reversals even
before their support, comparability or resistance to repair is established.

Keep G's meanings, grain, intervention and expected generalizability bounds explicit
(expectation records hold those bounds; inference.domain is only the initial scope hint).
The purpose of these bounds is to expose and
explain broken expectations, not establish an absolutely correct domain. Ground them in
literature or reasoning across literature where possible. When that is unavailable,
explicitly provisional judgment is allowed for proposing and revising these bounds only;
it does not authorize numerical ratings. Missing literature must not prevent exploration.
Revise expected generalizability bounds as more hypotheses are investigated. Record the
previous expectation, triggering finding/theory, proposed change and its reasoning. A
revision can expand expected bounds, repair a theory, or expose a broken causal inference.
Retain earlier failed expectations instead of erasing them with the revised wording.
Distinguish better definitions or a missing dependency following a general rule set from
a magic fudge factor or a special rule introduced outside that set. Show the actual rule,
what motivates it, and which earlier applications it preserves or loses. A discontinuity
alone is not an exception when it follows from the stated general rule set. Keep unresolved
repair arguments open; merely labeling a patch trivial or non-trivial establishes nothing.
The importance protocol domain instead fixes the circumstance denominator for one
reproducible measurement; its immutability does not freeze G's expected generalizability bounds.
Record an initial expectation with revision=null, bounds, basis, rationale and grounds.
Provisional judgment can have grounds=[]. Revising the same expectation requires revision.prior
to identify its exact preceding key/version. A separately keyed expectation with revision.prior
can explore a competing revision. Preserve a failure in an expectation_break, referencing the
exact expectation and triggering scientific snapshots; a revision need not be decided yet.
Revision triggers and retained/lost applications are immutable snapshot references, not live
dependencies. Select old records through snapshots in inspect or a child commission. Current
grounds and the inference remain live inputs: changed G makes an expectation stale until an
explicit revision acknowledges change_inference. Do not use an old snapshot as if it were
current support. Audit links preserve historical claims; they do not certify the argument.
For each revision supply the proposed rule, general rules and reasoning, retained/lost
application snapshots with consequences, and new predictions (or [] when none). These fields
record an argument, not a proof or a score. An expectation revision may expand, shrink, change G,
repair a theory or reconsider the expectation. No fixed sequence of records is required.
Record later support for G as it is in expectation_support, pinned to exact inference and
expectation snapshots with current scientific grounds and an argument. Support is not
certainty or a probability of universal truth. Later aligned hypotheses or facts can support
the original G after a break and revision; retain that attribution to the original versions.
Revise an existing support argument when its evidence changes; do not rediscover the break
under a new identity. The host compiles support and proportional change credit from the explicit operands below.
Model-specific realizations and mappings may differ; an unexplained label match is not
an established correspondence. A contradiction caused only by new contrary evidence is
different from incompatible causal commitments arising from the chosen conceptualization.

For repairs, separate facts used to construct or fit the adjustment from independent tests
of its additional consequences. Record retained successes, scope losses, information costs,
and whether one mechanism generates further applications. No search proves all repairs
impossible. No independent test means unknown corroboration, not zero corroboration.

For importance, define the question, G's realization, its alteration, fixed circumstances,
positive weights, and answer expression. The host computes changed circumstance mass /
total declared circumstance mass with exact typed-answer equality. Choose equations with
a focal relation or parameter, a linked pair in an acyclic SCM, or discrete simulation /
master equations over a declared parameter domain as appropriate; none comes first.
The runtime model supplies scientific definitions, never importance scores or result rows.
Use evaluate to calculate selected predeclared case IDs. Finite exhaustive evaluation,
sampled frequency, and a full continuous-domain measure are different claims: a finite
grid is not a continuous-space proof. This host supports explicit acyclic equations,
acyclic deterministic SCMs and deterministic discrete dynamics with rational operands;
unsupported or failed calculations remain unresolved. Do not change the denominator,
weights, alteration, or comparator after observing results. Record a separate explicit
protocol revision when scientifically necessary; it is not duplicate progress.
Do not turn a possible reversal into an established result, missing information into zero,
or papers into independent experiments. Return the requested findings and their provenance.
Support is counted from explicit support_contribution operands linked to an expectation_support
argument for the exact G and expected generalizability bounds. Select applications whose
hypotheses meaningfully depend on G and additional aligned observations from native source
measurements. The public argument must address each selected operand. Reuse canonical theory,
study and source observation identities across copies and paraphrases. Contexts of one theory
are not new hypotheses; an additional observation can contribute separately. A citation or
an arbitrary new measurement label is not an additional observation. Use typed native.observations
with source-established observation IDs or precise source locators; units and uncertainty
are attributes, not additional facts. Keep unresolved hypothesis identities pending. These counts are support
for G as it is, not certainty or a probability. Provisional bounds alone create no support.
For a necessary useful change, record one causal_change for the existing expectation_break,
exact source/target expectation snapshots, its importance protocol and an argument explaining
necessity, useful consequences, and why the measured alteration represents this change.
Model opinions never certify these operands; executable checks must establish their particular formal claims. Reuse the same change case as support grows.
The host compiles executed support operands and a scoped restored-prediction fraction,
including support added after the break. These explicit implementation conventions are
reported in policy_details; raw changed-answer importance remains a separate quantity. Alternative descriptions, repairs and adapters do not create more breaks.
Never enter a numerical support grade or change score. Retain repair_kind without inventing
a multiplier for its label. Parent transcripts and recursive child traces are not inputs or deliverables.
"""


@dataclass(frozen=True)
class SheetCommission:
    question: str
    inputs: tuple[str, ...]
    assignments: tuple[tuple[str, str], ...] = ()
    snapshots: tuple[dict, ...] = ()
    calculation_context: str = ""

    def __post_init__(self):
        if not isinstance(self.inputs, (list, tuple)) or any(not isinstance(key, str) for key in self.inputs):
            raise ValueError("Commission inputs must be scientific record keys")
        if not isinstance(self.assignments, (list, tuple)) or any(
                not isinstance(pair, (list, tuple)) or len(pair) != 2 or
                any(not isinstance(value, str) for value in pair) for pair in self.assignments):
            raise ValueError("Assignments must be (subject, criterion) pairs")
        object.__setattr__(self, "inputs", tuple(dict.fromkeys(self.inputs)))
        object.__setattr__(self, "assignments", tuple(dict.fromkeys(tuple(pair) for pair in self.assignments)))
        if not isinstance(self.snapshots, (list, tuple)) or len(self.snapshots) > 8 or any(
                not isinstance(ref, dict) or set(ref) != {"key", "version"} or
                any(not isinstance(value, str) or not value for value in ref.values()) for ref in self.snapshots):
            raise ValueError("Commission snapshots must be at most eight exact key/version references")
        object.__setattr__(self, "snapshots", tuple(dict(ref) for ref in self.snapshots))

    def as_record(self):
        return {"question": self.question, "inputs": list(self.inputs),
                "assignments": [list(pair) for pair in self.assignments],
                "snapshots": [dict(ref) for ref in self.snapshots],
                "calculation_context": self.calculation_context}

    def report_contract(self):
        from ...method_loop.communication import ReportContract
        return ReportContract(decision=self.question, measurements={"scope": self.as_record()},
            information={
                "completion": "Actual execution status; yielding is not scientific completion.",
                "causal_sheet_changes": "At most eight changed scientific records with identity, kind, "
                    "currentness, admission/direction or score. next_offset permits further scoped retrieval. "
                    "Full evidence and public derivations are retained in the sheet, not repeated in the return.",
                "reasoning_results": "At most eight public arguments: premises, conclusion, limitations, and calculation results; full steps remain addressable by key.",
                "unresolved_premises": "Referenced premises without source grounding or independent validation; these are requests for parent assessment, not refutations.",
                "operations": "At most three recent failures or limitations relevant to the next decision."})

    def request(self, sheet):
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("A commission needs a substantive scientific question")
        if not isinstance(self.calculation_context, str):
            raise ValueError("Calculation context must be an explicit record key or empty")
        investigation_credit(sheet, self.assignments, context=self.calculation_context)  # Validate assigned numerical operands.
        keys = set(self.inputs) | {subject for subject, _ in self.assignments}
        for key in self.inputs:
            if sheet.rows[key]["kind"] == "importance_result":
                assessment = sheet.rows[key]["data"]["assessment"]
                if assessment not in keys:
                    raise ValueError(f"Reasoning commission includes failed or completed result {key!r} "
                                     f"but omits its importance protocol {assessment!r}")
        if self.calculation_context:
            keys.add(self.calculation_context)
        keys.update(sheet.inferences_for(keys))
        return {"task": TASK, "question": self.question,
                "inputs": sheet.view(sorted(keys)),
                "historical_inputs": sheet.snapshot_view(self.snapshots),
                "assigned_calculations": [{"subject": subject, "criterion": name,
                    "meaning": CRITERIA[name], "current_calculation": sheet.calculation_view([subject], context=self.calculation_context)[subject]}
                    for subject, name in self.assignments]}



SNAPSHOT_SCHEMA = {"type": "object", "additionalProperties": False,
    "required": ["key", "version"], "properties": {
        "key": {"type": "string", "minLength": 1},
        "version": {"type": "string", "minLength": 1}}}


def bounds_schema(kind):
    text = {"type": "string", "minLength": 1}
    def obj(properties):
        return {"type": "object", "additionalProperties": False,
                "required": list(properties), "properties": properties}
    def array(items, minimum=0):
        return {"type": "array", "items": items, "minItems": minimum}
    if kind == "expectation_break":
        return obj({"expectation": SNAPSHOT_SCHEMA, "triggers": array(SNAPSHOT_SCHEMA, 1),
                    **{name: text for name in ("expected", "observed", "conflict", "limitations")}})
    if kind == "expectation_support":
        return obj({"inference": SNAPSHOT_SCHEMA, "expectation": SNAPSHOT_SCHEMA,
                    "grounds": array(text, 1), "argument": text, "limitations": text})
    consequence = obj({"application": SNAPSHOT_SCHEMA, "effect": text})
    revision = obj({"prior": SNAPSHOT_SCHEMA, "triggers": array(SNAPSHOT_SCHEMA, 1),
        "changes": array({"enum": sorted(BOUND_CHANGES)}, 1), "change": text,
        "repair_kind": {"enum": sorted(REPAIR_KINDS)},
        **{name: text for name in ("proposed_rule", "general_rules", "rule_reasoning", "limitations")},
        "retained": array(consequence), "lost": array(consequence), "new_predictions": array(text)})
    return obj({"inference": text, "bounds": text, "basis": {"enum": sorted(BOUND_BASES)},
                "rationale": text,
                "grounds": {**array(text), "description": "Scientific grounds: evidence, source formalization, premise, theory, application, reasoning, comparison, repair, expectation break, importance result or a grading-sheet record. An importance protocol itself is not evidence for expected bounds."},
                "revision": {"anyOf": [{"type": "null"}, revision]}})


def record_schema(kind):
    """The reader's concrete output shape, derived from the sheet it populates."""
    if kind in GRADING_FIELDS:
        return module_for(kind).record_schema(SNAPSHOT_SCHEMA)
    if kind == "verification":
        raise ValueError("Model-authored verification is not an output type")
    if kind in {"support_contribution", "causal_change"}:
        return causal_support.record_schema(kind, SNAPSHOT_SCHEMA)
    if kind == "importance":
        return protocol_schema()
    if kind == "source_formalization":
        from .source_formalization import record_schema as formalization_schema
        return formalization_schema()
    if kind == "importance_result":
        raise ValueError("Calculation receipts are host-generated, never model-authored records")
    if kind in {"expectation", "expectation_break", "expectation_support"}:
        return bounds_schema(kind)
    reference_descriptions = {
        "theory": {"same_as": "Exact theory record key only for same_theory admission; otherwise empty string.",
                   "compared_with": "Exact theory record keys, never inference or application keys."},
        "application": {"inference": "Exact inference record key for G.",
                        "theory": "Exact theory record key for this application."},
        "comparison": {"left": "Exact application record key. Both sides must be different applications of the same G; never use an inference, evidence, theory or reasoning key.",
                       "right": "Exact application record key. Both sides must be different applications of the same G; never use an inference, evidence, theory or reasoning key."},
        "repair": {"comparison": "Exact comparison record key.",
                   "construction_evidence": "Exact evidence record keys used to construct or fit the adjustment; reasoning records belong in a public argument, not here.",
                   "test_evidence": "Exact evidence record keys from independent tests; reasoning records belong in a public argument, not here."},
    }
    properties = {}
    for name in FIELDS[kind]:
        if name in ENUMS:
            schema = {"type": "string", "enum": sorted(ENUMS[name])}
        elif kind == "reasoning" and name == "premises":
            schema = {"type": "array", "minItems": 1,
                      "description": "Exact scientific record keys, not premise prose. Create a premise record first for a new assumption, then use its key here; use evidence keys for source-grounded premises. Records earlier in this batch may also be referenced.",
                      "items": {"type": "string", "minLength": 1}}
        elif name in LIST_FIELDS:
            schema = {"type": "array", "items": {"type": "string", "minLength": 1}}
        elif name == "measurements":
            schema = {"type": "object", "description": "Native values, units, uncertainty and scope; not invented confidence"}
        else:
            schema = {"type": "string", "minLength": 0 if name in {"same_as", "reasoning", "counterexample"} else 1}
        if name == "inference_version":
            schema = {"type": "string", "description": "Exact version of G, including a commissioned historical version; empty uses current G"}
        if name in reference_descriptions.get(kind, {}):
            schema["description"] = reference_descriptions[kind][name]
        properties[name] = schema
    return {"type": "object", "additionalProperties": False,
            "required": list(FIELDS[kind]), "properties": properties}


def _reference_locations(value, reference, path="data"):
    """Identify the fields that supplied one rejected scientific record key."""
    if isinstance(value, dict):
        return [location for name, item in value.items()
                for location in _reference_locations(item, reference, f"{path}.{name}")]
    if isinstance(value, (list, tuple)):
        return [location for index, item in enumerate(value)
                for location in _reference_locations(item, reference, f"{path}[{index}]")]
    return [path] if value == reference else []


_TYPED_REFERENCE_FIELDS = {
    "theory": {"same_as": "theory", "compared_with": "theory"},
    "application": {"inference": "inference", "theory": "theory"},
    "comparison": {"left": "application", "right": "application"},
    "repair": {"comparison": "comparison", "construction_evidence": "evidence",
               "test_evidence": "evidence"},
    "importance": {"application": "application"},
    "source_formalization": {"evidence": "evidence", "assessment": "importance"},
}


def read_sheet_result(sheet, commission, records, *, actor, owned_subjects=()):
    """Admit individually valid records against the commissioned view.

    The caller supplies the exact sheet snapshot used for the commission. A later
    global merge detects conflicting revisions. A malformed record cannot discard
    earlier valid records in the same action. These are proposed factual/logic
    inputs, not an LLM's choice of total score or continuation verdict.
    """
    commission.request(sheet)
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("The caller supplies the actual author/reviewer identity")
    staged = CausalSheet(sheet.history)
    updates = []
    diagnostics = []
    owned_subjects = set(owned_subjects)
    visible = set(commission.inputs) | {subject for subject, _ in commission.assignments}
    # An iterative child can build on records it admitted in earlier own units.
    # Owned records remain scoped even when they are outside the small selected
    # input page shown for the current decision.
    visible.update(owned_subjects)
    if commission.calculation_context:
        visible.add(commission.calculation_context)
    visible.update(sheet.inferences_for(visible))
    for record in records:
        kind = record.get("kind", "<unknown>") if isinstance(record, dict) else "<unknown>"
        key = record.get("key", "<unknown>") if isinstance(record, dict) else "<unknown>"
        try:
            if not isinstance(record, dict) or set(record) != {"key", "kind", "data", "dependencies"}:
                raise ValueError("Reader records require key, kind, data, dependencies")
            kind, data = record["kind"], dict(record["data"])
            if kind == "reasoning":
                data["author"] = actor
            elif kind == "verification":
                raise ValueError("Model verdicts are not verification; record a public reasoning critique instead")
            if kind == "importance_result":
                raise ValueError("Only the host evaluate action can produce importance results")
            if kind in {"support_contribution", "causal_change"} and "verification_required" in data:
                raise ValueError("The model-verdict credit gate has been removed")
            if kind == "importance" and data["application"] not in visible:
                raise ValueError("Child cannot define a protocol outside its scientific commission")
            if key in sheet.rows and key not in visible:
                raise ValueError("Child cannot revise an uncommissioned scientific record")
            references = staged._references(kind, data) | set(record["dependencies"])
            missing = references - visible
            if missing:
                locations = []
                for reference in sorted(missing):
                    fields = _reference_locations(data, reference)
                    if reference in record["dependencies"]:
                        fields.extend(f"dependencies[{index}]" for index, item in enumerate(record["dependencies"])
                                      if item == reference)
                    locations.append(f"{reference!r} at {', '.join(fields) if fields else 'scientific reference'}")
                raise ValueError(f"Unscoped keys: {'; '.join(locations)}. Use commissioned or own admitted "
                                 "records, create new records earlier in the batch, or ask the parent for inputs")
            for field, expected_kind in _TYPED_REFERENCE_FIELDS.get(kind, {}).items():
                value = data[field]
                for reference in value if isinstance(value, list) else [value]:
                    if reference and reference in staged.rows and staged.rows[reference]["kind"] != expected_kind:
                        raise ValueError(f"{field} requires a {expected_kind} record key; "
                                         f"{reference!r} is a {staged.rows[reference]['kind']} record")
            update = staged.propose(key, kind, data, dependencies=record["dependencies"])
            allowed_snapshots = {(ref["key"], ref["version"]) for ref in commission.snapshots}
            allowed_snapshots.update((ref, sheet.rows[ref]["version"]) for ref in visible if ref in sheet.rows)
            allowed_snapshots.update((ref, staged.rows[ref]["version"]) for ref in visible)
            if any((ref["key"], ref["version"]) not in allowed_snapshots
                   for ref in staged.snapshot_references(kind, data)):
                raise ValueError("Uncommissioned historical snapshot; select its key/version explicitly")
            staged.merge((update,))
        except (ValueError, KeyError, TypeError) as exc:
            diagnostics.append({"stage": "scientific_record", "status": "failed",
                                "category": "invalid_action_contract",
                                "why": f"{kind} record {key!r}: {exc}"})
            continue
        updates.append(update)
        visible.add(key)
        if key not in sheet.rows:
            owned_subjects.add(key)
    from .records import ResearchResult
    return ResearchResult(causal_sheet_updates=tuple(updates), diagnostics=tuple(diagnostics),
                          evaluated=bool(updates))
