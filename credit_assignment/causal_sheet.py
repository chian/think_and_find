"""Calculated influence of G on answers; no ordinal grades or merit multipliers.

Importance is a per-application, per-contrast quantity. Unrelated domains and
questions are not pooled into a purported global scientific merit score.
"""

from fractions import Fraction


CRITERIA = {
    "importance": "Compute changed circumstance mass / total declared mass; calculation progress measures resolved mass",
    "expected_bounds": "Circumstance mass with resolved scope, expected behavior and G relation / declared mass",
    "support": "Resolved source-formalization/G and observation/G comparisons / selected comparison count",
    "repair": "Circumstance mass with resolved repair and applicable scoped-necessity comparisons / declared mass",
}
CRITERION_KINDS = {"importance": "importance", "expected_bounds": "expectation_test",
                   "support": "support_assessment", "repair": "repair_assessment"}


def compile_sheet(sheet):
    from ..case_reports.research.importance import score_importance
    from .causal_support import compile_support
    assessments = {key: score_importance(sheet, key) for key, _ in sheet.of_kind("importance")}
    inferences = {}
    for key, value in assessments.items():
        application = sheet.rows[value["application"]]["data"]
        inferences.setdefault(application["inference"], {})[key] = value
    from ..case_reports.research.grading import GRADING_FIELDS, grade_record
    subsheets = {kind: {key: grade_record(sheet, key) for key, _ in sheet.of_kind(kind)} for kind in GRADING_FIELDS}
    return {"calculation": "causal-support-and-change-v3", "assessments": assessments, "subsheets": subsheets,
            "support_credit": compile_support(sheet),
            "inferences": inferences,
            "meaning": "Each fraction measures G's influence for its recorded application, contrast, question and measure. "
                       "Support and proportional change credit are compiled from explicit identified contributions; "
                       "importance fractions retain their measurement scopes."}


def _calculation_coverage(sheet, subject, criterion, pins):
    from ..case_reports.research.grading import grade_record
    if criterion == "importance":
        from ..case_reports.research.importance import score_importance, protocol_identity
        score = score_importance(sheet, subject, _pins=pins)
        return Fraction(score["resolved_fraction"]), Fraction(score["resolution"]), protocol_identity(sheet.rows[subject]["data"])
    grade = grade_record(sheet, subject)
    # Scientific operands, not an assessment alias, identify a completed job.
    from hashlib import sha256
    import json
    operands = dict(sheet.rows[subject]["data"])
    for field in ("rationale", "limitations"):
        operands.pop(field, None)
    identity = sha256(json.dumps(operands, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if criterion == "support":
        hypothesis = grade["hypothesis"]["status"]
        resolved = int(hypothesis in {"demonstrated", "not_demonstrated_in_expected_bounds", "source_statement_conflict"})
        known_fact_states = {"aligned", "not_aligned", "quantity_or_units_mismatch", "source_conditions_mismatch",
                             "outside_expected_bounds", "G_relation_not_satisfied"}
        resolved += sum(fact["status"] in known_fact_states for fact in grade["facts"])
        total = 1 + len(grade["facts"])
        return Fraction(resolved, total), Fraction(1, total), identity
    total = sum((Fraction(case["weight"]) for case in grade["cases"]), Fraction())
    resolved = Fraction()
    for case in grade["cases"]:
        if criterion == "expected_bounds":
            complete = case["in_scope"] is False or (case["in_scope"] is True and
                case["baseline_fits"] is not None and case.get("original_G_relation") is not None)
        else:
            complete = case["classification"] == "outside_scope" or (
                case["classification"] not in {"scope_unresolved", "unresolved"} and
                case.get("original_G_relation") is not None and case.get("candidate_original_G_relation") is not None)
            if case["classification"] == "repaired" and case.get("candidate_original_G_relation") is False:
                complete = complete and case["scoped_necessity"] is not None
        if complete:
            resolved += Fraction(case["weight"])
    resolution = min(Fraction(case["weight"]) for case in grade["cases"]) / total
    return resolved / total, resolution, identity


def investigation_credit(sheet, assignments, *, context=None):
    """Coverage of explicit assigned calculations, including negative results.

    This measures completed calculations rather than support for a conclusion.
    Numerical values and finite domains come from the actual subscoring sheet.
    """
    from ..case_reports.research.records import Credit
    pins = sheet.calculation_pins(context)
    values, resolutions, seen = {}, [], set()
    for subject, criterion in sorted(set(tuple(pair) for pair in assignments)):
        if criterion not in CRITERIA:
            raise ValueError("Select a declared executable calculation criterion")
        sheet._require_kind(subject, CRITERION_KINDS[criterion])
        coverage, resolution, identity = _calculation_coverage(sheet, subject, criterion, pins)
        if (criterion, identity) in seen:
            continue
        seen.add((criterion, identity))
        values[subject] = coverage
        resolutions.append(resolution)
    from .causal_support import controller_projection
    projection = controller_projection(values, min(resolutions) if resolutions else 1)
    if not projection["supported"]:
        return Credit(evaluated=False)
    return Credit(qualities=tuple(("causal_assessment", key, value) for key, value in projection["values"].items()),
                  quality_resolution=projection["resolution"] if resolutions else None)
