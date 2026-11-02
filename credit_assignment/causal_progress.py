"""Observable investigation progress, separate from support for a conclusion.

Registration units identify scientific objects, not their truth or merit. A
calculation unit is the resolved mass of its declared operands. Neither is a
model-authored numerical grade.
"""

from fractions import Fraction
from hashlib import sha256
import json

from .causal_sheet import CRITERION_KINDS, _calculation_coverage


def _identity(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _inference_identity(row):
    data = row["data"]
    # The statement and initial domain hint are prose; expected generalizability
    # bounds are recorded separately. Exact typed operands form the dedup key.
    return _identity({name: data[name] for name in
                      ("variables", "intervention", "outcome", "grain")})


def _referenced_inference_identity(sheet, key, version):
    return _inference_identity(sheet.snapshot({"key": key, "version": version}))


def compile_progress(sheet):
    """Return deduplicated, inspectable progress operands for the shared sheet.

    A registered candidate, an articulated expectation for it, each mapped
    theory/context relationship, and an explicit support argument are distinct
    work products. They earn no support-for-G. Executable calculations earn
    their resolved fraction even when their scientific answer is negative.
    Reworded records and revisions
    of the same object do not manufacture a second registration unit.
    """
    registrations, calculations = {}, {}
    inference_ids = {key: _inference_identity(row) for key, row in sheet.of_kind("inference")}
    for ident in inference_ids.values():
        registrations["progress:inference:" + ident] = Fraction(1)
    for _, row in sheet.of_kind("expectation"):
        key = row["data"]["inference"]
        ident = _referenced_inference_identity(sheet, key, row["dependencies"][key])
        registrations["progress:expected_bounds:" + ident] = Fraction(1)
    for _, row in sheet.of_kind("application"):
        data = row["data"]
        ident = _identity([_referenced_inference_identity(sheet, data["inference"], data["inference_version"]),
                           sheet.canonical_theory(data["theory"]), data["context_key"]])
        registrations["progress:application:" + ident] = Fraction(1)
    for _, row in sheet.of_kind("expectation_support"):
        data = row["data"]
        ident = _identity([_referenced_inference_identity(sheet, data["inference"]["key"],
                                                         data["inference"]["version"]),
                           data["expectation"]])
        registrations["progress:support_argument:" + ident] = Fraction(1)

    # Extraction alone earns no credit. Source claims and observations remain
    # evidence operands for the explicit scientific assessments.

    resolutions = [Fraction(1)]
    for criterion, kind in CRITERION_KINDS.items():
        for subject, _ in sheet.of_kind(kind):
            coverage, resolution, ident = _calculation_coverage(sheet, subject, criterion, {})
            key = "progress:calculation:" + criterion + ":" + ident
            calculations[key] = max(calculations.get(key, Fraction()), coverage)
            resolutions.append(resolution)
    points = {**registrations, **calculations}
    return {"points": {key: str(value) for key, value in sorted(points.items())},
            "point_dependencies": {},
            "registration_points": {key: str(value) for key, value in sorted(registrations.items())},
            "calculation_points": {key: str(value) for key, value in sorted(calculations.items())},
            "total": str(sum(points.values(), Fraction())),
            "resolution": str(min(resolutions)),
            "meaning": "Registration counts candidate G, its articulated expected generalizability bounds, "
                       "mapped theory/context relationships and an explicit support argument once per scored identity. "
                       "Extracting source claims or observations earns zero points. "
                       "Calculation progress is resolved declared operand mass, including negative results. "
                       "These are investigation units, not support for G or scientific merit grades. "
                       "Exact structural identity does not establish semantic equivalence of differently worded claims."}


def combine_episode_points(support, progress):
    """Expose the two meanings while supplying one controller point map."""
    return {"support": support, "progress": progress,
            "points": {**progress["points"], **support["points"]},
            "resolution": str(min(Fraction(support["resolution"]), Fraction(progress["resolution"])))}


def compile_episode_points(sheet):
    from .causal_support import compile_support
    return combine_episode_points(compile_support(sheet), compile_progress(sheet))
