"""Episode-owned assessments and factual, nonrecursive job returns.

This module supplies transport primitives, not a decision policy or a grader.
Each concrete source declares the questions its assessment must answer.
"""
from copy import deepcopy
from fractions import Fraction

from .records import Ledger, identity


RETURN_FIELDS = {
    "job": "The commissioned question and this episode's responsibility; permits interpreting the answer without recovering a prior prompt.",
    "completion": "Actual execution status; yielding does not establish scientific completion.",
    "assessment": "This episode's current answer to its commissioned job, qualifications and unresolved questions; not a transcript or a validity verdict.",
    "selected_results": "Exact scientific references and actual grading receipts selected to substantiate that answer.",
    "unreviewed_results": "Factual results received after the last assessment update, including the final unit when the controller yields. Flat results, never nested child histories.",
    "details_ref": "Recall this reference for the job's complete record/receipt index; then recall selected exact versions. Details stay out of ordinary handoffs.",
    "score_component_changes": "Host-calculated changes in the explicit investigation components since this episode began. Child scores are not added to parent credit.",
}

ASSESSMENT_INSTRUCTION = (
    " In working_assessment, update your own current answer to this job using completed results. "
    "Replace superseded interpretations rather than appending a running history. Retain material qualifications, "
    "broken expectations and unresolved questions. Select exact record_refs, receipt_refs and detail_refs "
    "needed to substantiate the answer or revisit an unresolved issue. Do not copy the child's report or "
    "list every finding. Do not describe the action you are about to take as completed. This assessment "
    "does not assign credit or certify scientific validity. The host retains actual results, including "
    "results received immediately before yielding. action=recall retrieves referenced records, receipts "
    "or job detail indexes within this same episode; it earns no points and imposes no action order."
    " Each operation's diagnostics describe that completed action and retain their originating responsibility "
    "and repair owner. A successful inspection does not repair a prior rejected record or grant new premise "
    "permissions. Preserve still-relevant prior problems in your assessment and report missing commissioned "
    "inputs as unresolved premises for the parent."
)


def reference(row):
    return {"key": row["key"], "version": row["version"], "kind": row["kind"]}


def public_receipt(receipt):
    return {key: value for key, value in receipt.items() if key != "episode"}


def public_diagnostic(diagnostic):
    return {key: deepcopy(diagnostic[key]) for key in (
        "stage", "category", "status", "why", "origin", "repair_owner", "repair_action")
        if key in diagnostic}


def action_diagnostic(source, diagnostic, unit, choice):
    """Attribute a new failure once; receiving parents retain its origin."""
    diagnostic = deepcopy(diagnostic)
    diagnostic.setdefault("origin", {
        "responsibility": type(source).__name__, "action": unit, "choice": choice,
        "own_action": source.own_action_count + 1,
        "model_call": diagnostic.get("model_call", source.run.model_calls),
    })
    if diagnostic.get("stage") == "scientific_record" and "Unscoped keys:" in diagnostic.get("why", ""):
        owner = "commissioning_parent"
        repair = "commission_required_premise_inputs; inspection alone cannot grant dependency permission"
    elif diagnostic.get("stage") in {"reading_memory", "reading_memory_reference"}:
        owner, repair = "originating_reader", "select_supplied_reference_or_correct_reading_memory"
    else:
        owner, repair = "originating_episode", "correct_the_reported_action_or_return_its_unresolved_requirement"
    diagnostic.setdefault("repair_owner", owner)
    diagnostic.setdefault("repair_action", repair)
    return diagnostic


def assessment_schema(fields):
    properties = {key: {"type": "string", "description": meaning} for key, meaning in fields.items()}
    properties.update(
        record_refs={"type": "array", "items": {"type": "object", "additionalProperties": False,
            "required": ["key", "version"], "properties": {
                "key": {"type": "string"}, "version": {"type": "string"}}}},
        receipt_refs={"type": "array", "uniqueItems": True, "items": {"type": "string"}},
        detail_refs={"type": "array", "uniqueItems": True, "items": {"type": "string"}})
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def decision_schema(schema, fields):
    schema = deepcopy(schema)
    branches = schema.get("oneOf", [schema])
    for branch in branches:
        branch.setdefault("properties", {})["working_assessment"] = assessment_schema(fields)
        branch.setdefault("required", []).append("working_assessment")
    recall = {"type": "object", "additionalProperties": False,
        "required": ["action", "references", "working_assessment"], "properties": {
            "action": {"const": "recall"},
            "references": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                "required": ["kind", "key"], "properties": {"kind": {"enum": ["record", "receipt", "detail"]},
                    "key": {"type": "string"}, "version": {"type": "string"}}}},
            "working_assessment": assessment_schema(fields)}}
    return {"oneOf": [schema, recall]}


def initialize(source):
    source.working_assessment = {**{key: "" for key in source.assessment_fields},
                                 "record_refs": [], "receipt_refs": [], "detail_refs": []}
    source.unreviewed_results = []
    source.unreviewed_children = []
    source.reviewed_child_count = 0
    source.assessment_error = ""
    source.known_details = set()
    source.known_receipts = set()
    source.known_records = set()
    source.detail_refs = []
    source.recalled_details = None


def learn_references(source, value):
    """Remember accessible references without putting their full index in prompts."""
    if isinstance(value, dict):
        if isinstance(value.get("details_ref"), str):
            source.known_details.add(value["details_ref"])
        for ref in value.get("detail_refs", ()) if isinstance(value.get("detail_refs"), (list, tuple)) else ():
            source.known_details.add(ref)
        if isinstance(value.get("key"), str) and isinstance(value.get("version"), str):
            source.known_records.add((value["key"], value["version"]))
        if isinstance(value.get("id"), str) and value["id"].startswith("grading-action:"):
            source.known_receipts.add(value["id"])
        for ref in value.get("receipt_refs", ()) if isinstance(value.get("receipt_refs"), (list, tuple)) else ():
            source.known_receipts.add(ref)
        for child in value.values():
            learn_references(source, child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            learn_references(source, child)


def accept_assessment(source, assessment, child_count):
    # A malformed memory update never invalidates scientific records or gates
    # credit. Preserve prior state and unreviewed results for the next choice.
    from jsonschema import Draft202012Validator
    errors = list(Draft202012Validator(assessment_schema(source.assessment_fields)).iter_errors(assessment))
    if errors:
        source.assessment_error = errors[0].message
        return
    try:
        for ref in assessment["record_refs"]:
            source.ledger.causal_sheet.snapshot_view([ref])
            if (ref["key"], ref["version"]) not in source.known_records:
                raise ValueError("Assessment references a record not supplied to this episode")
        if not set(assessment["receipt_refs"]) <= source.known_receipts:
            raise ValueError("Assessment references a receipt not supplied to this episode")
        if not set(assessment["detail_refs"]) <= source.known_details:
            raise ValueError("Assessment references a detail index not supplied to this episode")
    except (ValueError, KeyError) as exc:
        source.assessment_error = str(exc)
        return
    source.working_assessment = deepcopy(assessment)
    source.run.emit_trace("assessment_updated", episode_type=type(source).__name__,
        incorporated_child_results=child_count - source.reviewed_child_count,
        incorporated_own_results=len(source.unreviewed_results),
        retained_records=len(assessment["record_refs"]), retained_receipts=len(assessment["receipt_refs"]))
    source.reviewed_child_count = child_count
    source.unreviewed_results = []
    source.unreviewed_children = []
    source.assessment_error = ""


def score_changes(source, result):
    from ...credit_assignment.causal_progress import compile_episode_points
    before = compile_episode_points(Ledger(source.initial).causal_sheet)["components"]
    after = compile_episode_points(Ledger(result).causal_sheet)["components"]
    return {name: float(Fraction(value) - Fraction(before[name])) for name, value in after.items()}


def unit_result(source, changes):
    """Only this completed primitive's facts; never a projection of its ledger."""
    from .grading_handoff import evidence_needs
    rows = list(changes.causal_sheet_updates)
    evidence = [row for row in rows if row["kind"] == "evidence"]
    packet = {
        "question": source.commission.question if source.commission else source.run.research_request,
        "operation": deepcopy(source.last_own_action),
        "record_refs": [reference(row) for row in rows],
        "findings": [{"record": reference(row), "source": row["data"]["source"],
            "locator": row["data"]["locator"], "finding": row["data"]["finding"],
            "bearing": row["data"]["measurements"].get("relevance_basis", ""),
            "reasoning_question": row["data"]["measurements"].get("reasoning_question", "")}
            for row in evidence],
        "grading_needs": evidence_needs(source.ledger.causal_sheet, [row["key"] for row in evidence]),
        "grading_outcomes": [public_receipt(row) for row in changes.grading_actions],
        "operation_findings": list(changes.findings),
        "diagnostics": [public_diagnostic(row) for row in changes.diagnostics],
    }
    if source.last_proposal and source.last_proposal.get("action") == "calculate" and not changes.grading_actions:
        packet["calculation"] = deepcopy(source.last_calculation)
    if source.recalled_details is not None:
        packet["recalled_details"] = source.recalled_details
        source.recalled_details = None
    learn_references(source, packet)
    return packet


def flatten_child(report):
    """Carry a final, unreviewed child outcome without recursively nesting it."""
    answer = {key: value for key, value in report.items() if key != "unreviewed_results"}
    return [answer, *deepcopy(report["unreviewed_results"])]


def selected_results(source, result):
    from .grading_handoff import evidence_needs
    sheet = Ledger(result).causal_sheet
    # The pending packets already carry these facts. Do not send a second copy
    # under the current assessment merely because it selected the same record.
    pending_records, pending_receipts = set(), set()
    def collect(value):
        if isinstance(value, dict):
            for ref in value.get("record_refs", ()):
                pending_records.add((ref["key"], ref["version"]))
            for receipt in value.get("grading_outcomes", ()):
                pending_receipts.add(receipt["id"])
            for child in value.values():
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)
    collect([*source.unreviewed_results, *source.unreviewed_children])
    refs = [ref for ref in source.working_assessment["record_refs"]
            if (ref["key"], ref["version"]) not in pending_records]
    # Use the selected versions, not silently revised current evidence.
    from types import SimpleNamespace
    selected = [sheet.snapshot(ref) for ref in refs]
    receipts = {row["id"]: row for row in result.grading_actions}
    return {"grading_needs": [need for row in selected if row["kind"] == "evidence"
                             for need in evidence_needs(SimpleNamespace(rows={row["key"]: row}), [row["key"]])],
        "grading_outcomes": [public_receipt(receipts[key]) for key in source.working_assessment["receipt_refs"]
                             if key not in pending_receipts]}


def job_report(source, result, completion=None):
    from .engine import changed_records
    from .episode_contracts import completion_fields
    delta = changed_records(result, source.initial)
    index = {"question": source.commission.question if source.commission else source.run.research_request,
        "record_refs": [reference(row) for row in delta.causal_sheet_updates],
        "receipt_refs": [row["id"] for row in delta.grading_actions],
        "detail_refs": list(dict.fromkeys(source.detail_refs)),
        "own_actions": deepcopy(source.own_action_outcomes)}
    ref = "job-details:" + identity(index)
    if ref not in source.run.handoff_details:
        source.run.handoff_details[ref] = index
        source.run.emit_trace("job_details", details_ref=ref, details=index)
    ended = completion_fields(result, completion)
    if source.completion_diagnostic:
        ended["failure"] = {key: source.completion_diagnostic.get(key)
                            for key in ("stage", "category", "status", "why")}
    report = {"job": {"question": index["question"], "responsibility": type(source).__name__},
        "completion": ended,
        "assessment": deepcopy(source.working_assessment),
        "selected_results": selected_results(source, result),
        "unreviewed_results": deepcopy([*source.unreviewed_children, *source.unreviewed_results]),
        "details_ref": ref}
    if source.parent_contract is None or "score_component_changes" in source.parent_contract.information:
        report["score_component_changes"] = score_changes(source, result)
    return report


def recall(source, proposal):
    """Read-only retrieval, chosen as one ordinary own unit."""
    from .records import ResearchResult
    def execute():
        items = []
        for ref in proposal["references"]:
            kind, key = ref["kind"], ref["key"]
            if kind == "detail":
                if key not in source.known_details:
                    raise ValueError("Recall needs a job reference supplied to this episode")
                value = deepcopy(source.run.handoff_details[key])
            elif kind == "receipt":
                if key not in source.known_receipts:
                    raise ValueError("Recall needs a grading receipt supplied to this episode")
                value = public_receipt(source.ledger.grading_actions[key])
            else:
                version = ref.get("version", source.ledger.causal_sheet.rows[key]["version"])
                if (key, version) not in source.known_records:
                    raise ValueError("Recall needs an exact record supplied to this episode")
                value = source.ledger.causal_sheet.snapshot_view([{"key": key, "version": version}])[0]
            learn_references(source, value)
            items.append({"reference": ref, "value": value})
        source.recalled_details = items
        return ResearchResult()
    return source.leaf(execute, "recall")
