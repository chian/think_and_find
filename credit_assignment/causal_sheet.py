"""Calculated influence of G on answers; no ordinal grades or merit multipliers.

Importance is a per-application, per-contrast quantity. Unrelated domains and
questions are not pooled into a purported global scientific merit score.
"""


CRITERIA = {
    "importance": "Compute changed circumstance mass / total declared mass",
    "expected_bounds": "Determine where expected behavior and the original G relation hold or fail within expected generalizability bounds",
    "support": "Compare formalized source statements and observations against G to identify support or conflict",
    "repair": "Determine whether changing G restores failed predictions and whether compared G-preserving alternatives do so",
}
CRITERION_KINDS = {"importance": "importance", "expected_bounds": "expectation_test",
                   "support": "support_assessment", "repair": "repair_assessment"}


def compile_sheet(sheet):
    from ..case_reports.research.importance import score_importance
    from .causal_support import compile_support
    from .causal_progress import compile_progress
    from ..case_reports.research.discovery_goals import compile_discovery_points
    assessments = {key: score_importance(sheet, key) for key, _ in sheet.of_kind("importance")}
    inferences = {}
    for key, value in assessments.items():
        application = sheet.rows[value["application"]]["data"]
        inferences.setdefault(application["inference"], {})[key] = value
    from ..case_reports.research.grading import GRADING_FIELDS, grade_record
    subsheets = {kind: {key: grade_record(sheet, key) for key, _ in sheet.of_kind(kind)} for kind in GRADING_FIELDS}
    return {"calculation": "causal-support-and-task-alterations-v5", "assessments": assessments, "subsheets": subsheets,
            "support_credit": compile_support(sheet), "registrations": compile_progress(sheet),
            "discovery": compile_discovery_points(sheet),
            "inferences": inferences,
            "meaning": "Each fraction measures G's influence for its recorded application, contrast, question and measure. "
                       "Executing calculations earns zero points; their findings inform support and change credit. "
                       "Support, applicability differences and gradual task improvements are compiled from explicit executed contributions; "
                       "importance fractions retain their measurement scopes."}


def validate_assignments(sheet, assignments, *, context=None):
    """Validate a calculation commission without assigning execution credit."""
    sheet.calculation_pins(context)
    for subject, criterion in assignments:
        if criterion not in CRITERIA:
            raise ValueError("Select a declared executable calculation criterion")
        sheet._require_kind(subject, CRITERION_KINDS[criterion])
