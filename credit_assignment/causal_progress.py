"""Credit for registered G, expected generalizability bounds and applications."""

from fractions import Fraction
from hashlib import sha256
import json


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
    """Count the three user-confirmed registration rewards, once per identity."""
    registrations, dependencies = {}, {}

    def register(point, record):
        registrations[point] = Fraction(1)
        dependencies.setdefault(point, []).append(record)
    inference_ids = {key: _inference_identity(row) for key, row in sheet.of_kind("inference")}
    for key, ident in inference_ids.items():
        register("progress:inference:" + ident, key)
    for record, row in sheet.of_kind("expectation"):
        key = row["data"]["inference"]
        ident = _referenced_inference_identity(sheet, key, row["dependencies"][key])
        register("progress:expected_bounds:" + ident, record)
    for record, row in sheet.of_kind("application"):
        data = row["data"]
        ident = _identity([_referenced_inference_identity(sheet, data["inference"], data["inference_version"]),
                           sheet.canonical_theory(data["theory"]), data["context_key"]])
        register("progress:application:" + ident, record)
    return {"points": {key: str(value) for key, value in sorted(registrations.items())},
            "point_dependencies": dependencies,
            "total": str(sum(registrations.values(), Fraction())),
            "resolution": "1",
            "meaning": "One point for registering G, one for its expected generalizability bounds, "
                       "and one per registered theory/context application. Extraction, writing a support argument "
                       "and completing calculations earn zero points. Repeated scored identities earn zero new points."}


def combine_episode_points(support, registrations):
    """Expose every scored component and the exact points summed by controllers."""
    points = {**registrations["points"], **support["points"]}
    prefixes = {"G_registration": "progress:inference:",
                "expected_generalizability_bounds_registration": "progress:expected_bounds:",
                "theory_context_application_registration": "progress:application:",
                "support_for_G": "support:", "causal_change": "change:"}
    components = {name: str(sum((Fraction(value) for key, value in points.items()
                                if key.startswith(prefix)), Fraction()))
                  for name, prefix in prefixes.items()}
    return {"support": support, "registrations": registrations, "points": points,
            "components": components, "total": str(sum(map(Fraction, components.values()), Fraction())),
            "resolution": str(min(Fraction(support["resolution"]), Fraction(registrations["resolution"])))}


def compile_episode_points(sheet):
    from .causal_support import compile_support
    return combine_episode_points(compile_support(sheet), compile_progress(sheet))
