"""Source-linked grading needs and factual Page tool outcomes, without verdicts."""
from fractions import Fraction

from .records import identity, Ledger
from .grading import GRADING_FIELDS
from ...credit_assignment.causal_progress import compile_episode_points


GRADING_NEED_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["kind", "question", "qualifications", "sheet_keys"],
    "properties": {
        "kind": {"enum": ["G_registration", "expected_generalizability_bounds",
                          "theory_context_application", "support", "causal_change"]},
        "question": {"type": "string", "minLength": 1},
        "qualifications": {"type": "string"},
        "sheet_keys": {"type": "array", "uniqueItems": True,
                       "items": {"type": "string"}}}}


def validate_needs(needs, visible):
    if not isinstance(needs, list):
        raise ValueError("grading_needs must be a list; [] means no proposed assessment")
    for need in needs:
        if not isinstance(need, dict) or set(need) != set(GRADING_NEED_SCHEMA["required"]):
            raise ValueError("A grading need requires kind, question, qualifications and sheet_keys")
        if need["kind"] not in GRADING_NEED_SCHEMA["properties"]["kind"]["enum"]:
            raise ValueError("Unknown grading need kind")
        if not isinstance(need["question"], str) or not need["question"].strip():
            raise ValueError("A grading need must state the scientific question")
        if not isinstance(need["qualifications"], str):
            raise ValueError("Grading need qualifications must be text")
        if (not isinstance(need["sheet_keys"], list)
                or any(not isinstance(key, str) or key not in visible for key in need["sheet_keys"])):
            raise ValueError("A grading need may reference only supplied sheet keys")
    return needs


def evidence_needs(sheet, keys):
    needs = []
    for key in dict.fromkeys(keys):
        row = sheet.rows[key]
        if row["kind"] != "evidence":
            continue
        data = row["data"]
        for need in data["measurements"].get("grading_needs", []):
            needs.append({"id": "grading-need:" + identity([key, need])[:24],
                          "evidence": key, "evidence_version": row["version"],
                          "source": data["source"], "locator": data["locator"], **need})
    return needs


def grading_report(delta, sheet):
    """Project job-specific needs/outcomes separately from generic finding pagination.

    All need IDs survive the handoff. Repeated actions on the same need expose
    their latest factual result, not a transcript. Full actions remain in audit.
    A returned calculation is not a declaration that the question is resolved.
    """
    keys = [row["key"] for row in delta.causal_sheet_updates]
    needs = evidence_needs(sheet, keys)
    actions = list(delta.grading_actions)
    # An existing source's evidence may be inherited while Page assesses it now.
    extra = [key for action in actions for key in action["evidence"]]
    needs = list({need["id"]: need for need in [*needs, *evidence_needs(sheet, extra)]}.values())
    latest = {}
    for action in actions:
        for need in action["needs"] or ["unflagged:" + action["source"]]:
            latest[(need, action["action"])] = action
    retained = {action["id"]: action for action in actions
                if any(Fraction(value) for value in action["score_component_changes"].values())}
    retained.update((action["id"], action) for action in latest.values())
    for need in needs:
        need["outcomes"] = [action["id"] for action in retained.values() if need["id"] in action["needs"]]
    return {"grading_needs": needs,
            "grading_outcomes": list(retained.values()),
            "grading_action_count": len(actions),
            "grading_status_meaning": "Needs remain questions; outcomes are actual tools and computed results, not model validity verdicts."}


def page_action_outcome(source, result):
    """Record what the selected Page tool actually returned; award nothing here."""
    from .engine import changed_records
    proposal = source.last_proposal or {}
    action = proposal.get("action")
    if action not in {"record", "inspect", "evaluate", "calculate", "compare_observations"}:
        return result
    from dataclasses import replace
    before = source.ledger.result()
    staged = Ledger(before)
    staged.merge(result)
    sheet = staged.causal_sheet
    delta = changed_records(staged.result(), before)
    before_score = compile_episode_points(source.ledger.causal_sheet)["components"]
    after_score = compile_episode_points(sheet)["components"]
    needs = evidence_needs(sheet, (key for key, row in sheet.of_kind("evidence")
                                  if row["data"]["source"] == source.document.source))
    selected = set(proposal.get("grading_needs", []))
    selected_needs = [need for need in needs if need["id"] in selected]
    references = set(proposal.get("keys", []))
    references.update(proposal.get("evidence", []))
    references.update(proposal.get("templates", []))
    references.update(row["key"] for row in delta.causal_sheet_updates)
    if proposal.get("assessment"):
        references.add(proposal["assessment"])
    evidence, seen = {need["evidence"] for need in selected_needs}, set()
    todo = list(references)
    while todo:
        key = todo.pop()
        if key in seen or key not in sheet.rows:
            continue
        seen.add(key)
        row = sheet.rows[key]
        if row["kind"] == "evidence":
            evidence.add(key)
        todo.extend(row["dependencies"])
    selected.update(need["id"] for need in needs if need["evidence"] in evidence)
    # Include actual grader results even when record/evaluate, rather than a
    # separate calculate, caused the score to update.
    changed = sorted(delta.causal_sheet_updates, key=lambda row: row["kind"] == "evidence")
    calculation_keys = list(dict.fromkeys([
        *([proposal["evaluation_context"]] if proposal.get("evaluation_context") else []),
        *(key for key in sorted(references) if key in sheet.rows and sheet.rows[key]["kind"] in
          {"importance", "support_contribution", "causal_change", *GRADING_FIELDS})]))
    calculations = ({} if not result.evaluated else source.last_calculation if action == "calculate" else
                    sheet.calculation_view(calculation_keys[:8]) if calculation_keys else {})
    receipt = {
        "source": source.document.source, "episode": source.goal.goal_id,
        "unit": source.own_action_count + 1, "action": action,
        "needs": sorted(selected), "evidence": sorted(evidence),
        "rationale": proposal.get("rationale", ""),
        "records": sheet.return_page(changed),
        "calculation": calculations,
        "score_component_changes": {name: str(Fraction(value) - Fraction(before_score[name]))
                                    for name, value in after_score.items()},
        "diagnostics": [{name: diagnostic.get(name) for name in
                         ("stage", "category", "status", "why", "evidence")}
                        for diagnostic in result.diagnostics],
        "execution": "completed" if result.evaluated and not result.diagnostics else
                     "partial" if result.evaluated else "failed"}
    receipt["id"] = "grading-action:" + identity(receipt)[:24]
    return replace(result, grading_actions=(*result.grading_actions, receipt))
