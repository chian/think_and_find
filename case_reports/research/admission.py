"""Check source fidelity and units; target relevance is assessed downstream."""

from __future__ import annotations

import hashlib

from ...task_executors._http import ProviderError
from ...task_executors.scenario_metrics.evaluate import CONDITIONS, UNITS, validate_parameter
from ...task_executors.scenario_metrics.catalog import METRICS
from ...task_executors.unit_conversion.units import METERS_PER_FOOT, SECONDS_PER_DAY
from .records import Fact, ResearchResult, diagnostic_value, finite
from .assumptions import text as required_text


CONVERSIONS = {
    ("m3/h", "m3/s"): 1/3600, ("m3/min", "m3/s"): 1/60,
    ("L/s", "m3/s"): 0.001,
    ("L/min", "m3/s"): 1/60000, ("cm3/min", "m3/s"): 1e-6/60,
    ("h", "s"): 3600, ("min", "s"): 60,
    ("1/h", "1/s"): 1/3600, ("ACH", "1/s"): 1/3600,
    ("um", "m"): 1e-6, ("cm2", "m2"): 1e-4, ("mL", "m3"): 1e-6,
    ("m/h", "m/s"): 1/3600, ("cm/s", "m/s"): 0.01, ("%", "fraction"): 0.01,
    ("quanta/h", "quanta/s"): 1/3600, ("count", "person"): 1,
    ("uPa*s", "Pa*s"): 1e-6,
    ("ft", "m"): METERS_PER_FOOT, ("ft2", "m2"): METERS_PER_FOOT**2,
    ("ft3", "m3"): METERS_PER_FOOT**3, ("cfm", "m3/s"): METERS_PER_FOOT**3 / 60,
    ("cm", "m"): .01, ("mm", "m"): .001, ("cm3", "m3"): 1e-6, ("L", "m3"): .001,
    ("day", "s"): SECONDS_PER_DAY, ("1/day", "1/s"): 1 / SECONDS_PER_DAY,
    ("dimensionless", "scalar"): 1, ("dimensionless", "fraction"): 1,
    ("%", "scalar"): .01, ("um3/cm3", "fraction"): (1e-6 / .01)**3,
}


EVIDENCE_FIELDS = frozenset(UNITS) | frozenset(CONDITIONS) | frozenset(METRICS)
SOURCE_UNITS = {
    field: sorted({unit} | {printed for printed, target in CONVERSIONS if target == unit})
    for field, unit in UNITS.items()
}
NUMERIC_CONSTRAINT_FIELDS = {
    "interval": ("minimum", "maximum", "minimum_inclusive", "maximum_inclusive"),
    "lower_bound": ("minimum", "minimum_inclusive"),
    "upper_bound": ("maximum", "maximum_inclusive"),
}


def extraction_response_schema(source):
    """Use the same field vocabulary as admission, with explicit JSON types."""
    text = {"type": "string", "minLength": 1}
    shared = {name: dict(text) for name in ("subject", "quote", "context", "basis")}
    shared["subject"]["description"] = "Stable actual event or population identity, reusing known_source_subjects when applicable. Do not create a new event identity for each measured field, table, or attendee subgroup; preserve those distinctions in context. Do not substitute a target scenario."
    shared["quote"]["description"] = "One exact contiguous original source span, preserving qualifiers."
    shared["context"]["description"] = "Disease, strain, population, activity and event identity."
    shared["basis"]["description"] = "What this supports, constrains or contradicts, and what it does not establish."
    facts = {**shared, "field": {"type": "string", "enum": sorted(UNITS)},
             "value": {"type": "number", "description":
                       "Finite scalar number explicitly printed in the source, not a quoted numeric string. "
                       "Do not put a range, bound, expression or null here. Preserve those in evidence instead."},
             "unit": {"type": "string", "enum": sorted({unit for units in SOURCE_UNITS.values() for unit in units}),
                      "description": "Select an equivalent notation from accepted_source_units for this field, preserving the source's printed numeric value and scale. For example, hour and h are equivalent labels, but h and s have different scales. Do not convert the number yourself. If no unit represents the reported quantity, preserve it as qualitative evidence rather than renaming the quantity or dropping its dimensions."},
             "kind": {"type": "string", "enum": ["request_constraint"] if source == "request" else
                      ["source_observation", "source_model_parameter"]}}
    constraint = {"oneOf": [{"type": "object", "additionalProperties": False,
        "required": ["field", "relation", "unit", *names], "properties": {
            "field": {"type": "string", "enum": sorted(UNITS)},
            "relation": {"const": relation}, "unit": facts["unit"],
            **{name: {"type": "boolean" if name.endswith("_inclusive") else "number"} for name in names},
        }} for relation, names in NUMERIC_CONSTRAINT_FIELDS.items()]}
    evidence = {**shared, "claim": {**text, "description":
               "Source-supported qualitative claim, range, bound, equation or applicability condition."},
                "numeric_constraints": {"type": "array", "items": constraint,
                    "description": "Typed ranges and bounds in this evidence's exact quote. Preserve printed endpoints and accepted source units; deterministic conversion stores canonical bounds. Exact scalar values belong in facts, not this array. Each entry inherits this evidence record's source, quote and context. Use [] when no declared quantity and supported unit faithfully represent the measurement; do not invent a mapping for an unrelated native quantity."},
                "evidence_role": {"type": "string", "enum": ["request_constraint"] if source == "request" else
                    ["observation", "model_statement", "context"],
                    "description": "Observation reports an empirical event, measurement or outcome, including ranges. A model statement is a formula, fitted parameter, assumption or hypothetical case, not an observation."},
                "observed_fields": {"type": "array", "uniqueItems": True,
                    "items": {"type": "string", "enum": sorted(UNITS)},
                    "description": "Only quantities actually observed or measured in this quotation, including reported ranges or bounds. Related inferred quantities belong in fields instead. Use [] for model statements, context, requester constraints, or observations with no matching declared quantity."},
                "fields": {"type": "array", "minItems": 1,
                           "description": "Declared quantities or conditions this evidence informs, not necessarily quantities measured directly in the source",
                           "items": {"type": "string", "enum": sorted(EVIDENCE_FIELDS)}}}
    return {"type": "object", "additionalProperties": False,
            "required": ["facts", "evidence"], "properties": {
                "facts": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                          "required": list(facts), "properties": facts}},
                "evidence": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                             "required": list(evidence), "properties": evidence,
                             "allOf": [{"if": {"properties": {"evidence_role": {
                                 "enum": ["model_statement", "context", "request_constraint"]}}},
                                  "then": {"properties": {"observed_fields": {"maxItems": 0}}}}]}}}}


def canonical_notation(unit):
    if isinstance(unit, str):
        unit = unit.strip().replace("\u2212", "-")
        parts = unit.split("/")
        if len(parts) == 2 and all(part.strip() for part in parts):
            unit = "/".join("1" if part.strip() == "1" else canonical_notation(part) for part in parts)
    aliases = {"quanta/m^3": "quanta/m3", "quanta/m\u00b3": "quanta/m3",
               "infection quanta/m3 exhaled air": "quanta/m3",
               "q/m3": "quanta/m3", "q/m^3": "quanta/m3", "q/m\u00b3": "quanta/m3",
               "q/h": "quanta/h", "q/s": "quanta/s",
               "people": "person", "persons": "person",
               "unitless": "dimensionless", "1": "dimensionless",
               "hour": "h", "hours": "h", "hr": "h", "hr.": "h",
               "minute": "min", "minutes": "min", "mins": "min",
               "second": "s", "seconds": "s", "sec": "s", "sec.": "s",
               "foot": "ft", "feet": "ft", "ft^2": "ft2", "ft\u00b2": "ft2",
               "ft^3": "ft3", "ft\u00b3": "ft3", "ft3/min": "cfm", "ft^3/min": "cfm",
               "m^3": "m3", "m\u00b3": "m3", "cm^3": "cm3", "cm\u00b3": "cm3",
               "cm^2": "cm2", "m\u00b2/s": "m2/s", "m^2/s": "m2/s",
               "days": "day", "d": "day", "/day": "1/day", "day-1": "1/day", "day^-1": "1/day",
               "quanta m^-3": "quanta/m3", "quanta m\u207b\u00b3": "quanta/m3",
               "m\u00b3/s": "m3/s", "m\u00b3/h": "m3/h", "m^3/s": "m3/s", "m^3/h": "m3/h",
               "m\u00b2": "m2", "cm\u00b2": "cm2", "\u03bcm": "um", "\u00b5m": "um",
               "h^-1": "1/h", "h-1": "1/h", "h\u207b\u00b9": "1/h", "/h": "1/h",
               "per h": "1/h", "ac/h": "ACH",
               "quanta per h": "quanta/h", "quanta h-1": "quanta/h", "quanta h^-1": "quanta/h",
               "q h-1": "quanta/h", "q h^-1": "quanta/h", "q h\u207b\u00b9": "quanta/h",
               "m3 h-1": "m3/h", "m3 h^-1": "m3/h", "m^3 h^-1": "m3/h",
               "m\u00b3 h\u207b\u00b9": "m3/h", "m3 min-1": "m3/min", "m3 min^-1": "m3/min",
               "kg m-3": "kg/m3", "kg m^-3": "kg/m3", "kg m\u207b\u00b3": "kg/m3",
               "s^-1": "1/s", "s-1": "1/s",
               "\u03bcPa*s": "uPa*s", "\u00b5Pa*s": "uPa*s", "\u03bcPa\u00b7s": "uPa*s",
               "um^3/cm^3": "um3/cm3", "\u03bcm3/cm3": "um3/cm3", "\u03bcm\u00b3/cm\u00b3": "um3/cm3"}
    return aliases.get(unit, unit)


def exact_span(quote, source):
    if not isinstance(quote, str) or not quote.strip() or quote not in source:
        raise ValueError("Evidence quote is not an exact source span")
    return quote


def review_candidates(agent, request, source_version):
    candidates = request["candidates"]
    try:
        review = agent(request)
        decisions = review.get("decisions")
        if not isinstance(decisions, list) or len(decisions) != len(candidates):
            raise ValueError("Applicability review must decide every candidate")
        seen = set()
        for decision in decisions:
            if not isinstance(decision, dict):
                raise ValueError("Each evidence review decision must be an object")
            index = decision.get("option")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(candidates) or index in seen:
                raise ValueError("Invalid or repeated review option")
            seen.add(index)
            if type(decision.get("admit")) is not bool or not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
                raise ValueError("Review requires boolean admit and a reason")
            constraint_count = len(candidates[index]["normalized"].get("numeric_constraints", ()))
            if constraint_count and "accepted_numeric_constraints" not in decision:
                raise ValueError("Review must explicitly identify accepted numeric constraints")
            selected = decision.get("accepted_numeric_constraints", [])
            if (not isinstance(selected, list)
                    or any(type(item) is not int or not 0 <= item < constraint_count for item in selected)
                    or len(set(selected)) != len(selected)
                    or (not decision["admit"] and selected)):
                raise ValueError("Accepted numeric constraints must be unique valid indices on an admitted record")
        return decisions
    except (ProviderError, OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        # Keep discovery distinct from independent admission, even if the latter cannot finish.
        index = []
        for candidate in candidates:
            record = candidate["normalized"]
            quote = record["quote"]
            key, span = min(((k, s) for k, s in request["available_source_spans"].items() if quote in s),
                            key=lambda item: len(item[1]))
            start = span.index(quote)
            index.append({"option": candidate["option"],
                **{k: record[k] for k in ("subject", "field", "fields", "value", "unit", "claim",
                                        "evidence_role", "observed_fields") if k in record},
                "location": {"span": key, "start_in_span": start, "end_in_span": start + len(quote)}})
        exc.extraction_state = {
            "status": "independent_review_incomplete", "source": request["source"],
            "source_version": source_version, "candidate_count": len(candidates), "candidates": index,
            "checks_completed": "Exact source-span match; declared fields; numeric representation, conversion and domain checks where applicable",
            "not_established": "Independent verification of meaning, units, entailment and applicability; these candidates are not admitted evidence or a completed rejection",
        }
        exc.unreviewed_candidates = diagnostic_value(candidates)
        raise


def extract_and_admit(agent, *, text, source, objective, context, question, feedback=(), source_spans=None,
                      source_version=None):
    spans = source_spans or {"current": text}
    source_version = source_version or hashlib.sha256(text.encode()).hexdigest()

    def quote_span(quote):
        for span in spans.values():
            try:
                return exact_span(quote, span)
            except ValueError:
                pass
        raise ValueError("Evidence quote is not contained in any one available original source span; "
                         "do not join separate rows or reformat source text")
    response = agent({
        "operation": "extract_research_evidence", "objective": objective,
        "question": question, "context": context,
        "previous_attempt_results": list(feedback),
        "canonical_units": UNITS, "accepted_source_units": SOURCE_UNITS,
        "model_conditions": CONDITIONS, "source": source, "source_text": text,
        "response_contract": extraction_response_schema(source),
        "instruction": (
            "Return an instance of response_contract, which is a JSON Schema, not an output template. "
            "Numeric facts require finite JSON numbers, never strings. Put reported ranges, inequalities "
            "and equations in evidence with their exact quotes and qualifiers. A unit is not a variable "
            "name or an inferred quantity. Select an equivalent notation from accepted_source_units "
            "without changing the printed numeric value. The deterministic converter handles scale "
            "changes after extraction. If the printed quantity or scale has no supported representation, "
            "retain it as qualitative evidence with its original numbers, units and meaning. "
            "For each qualitative evidence record, also fill numeric_constraints for any faithfully "
            "represented declared quantity: interval uses minimum, maximum and their inclusive flags; "
            "lower_bound uses minimum and minimum_inclusive; upper_bound uses maximum and "
            "maximum_inclusive. Supply printed finite endpoints and accepted "
            "source units, without converting the numbers yourself. A strict inequality has its "
            "inclusive flag false. Never choose an endpoint or midpoint as a scalar fact. These "
            "constraints supplement the source record; a faithful scalar value still belongs in facts. "
            "Use [] for equations, ambiguous OCR, unsupported native quantities or unsupported units. "
            "Do not combine different activity-specific bounds into one interval. "
            "Filtration efficiency is not penetration, per-person flow is not total flow, and RNA "
            "concentration is not a virion or quanta concentration. Related measurements can support "
            "later explicit inferences but must not be relabeled as those inferred quantities. "
            "Capture source-supported scalar values for the requested target cases' parameters "
            "and their derivation ingredients, including fields outside this branch's assigned "
            "parameter. The local question guides reading, not which faithful values may be retained. "
            "Keep a native numeric value even when its activity, population, variant or setting "
            "differs from the current target, or its use needs other inputs or a transfer assumption. "
            "For example, a recipient inhalation rate remains a breathing-flow value with that "
            "recipient role in context; it need not independently determine source infectiousness. "
            "Different activity-specific emission rates remain separate source model parameters. "
            "Do not downgrade a supported scalar to qualitative context because it cannot answer "
            "the current question alone. Target relevance and applicability are graded downstream. "
            "Retain the actual source subject and its conditions; extraction does not assign the "
            "value to a target case. Explain possible uses and limitations in basis. "
            "Do not put population labels in unit: record human counts in person or count, unitless values in "
            "dimensionless, and percentages in %. Retain the counted population in context. "
            "If OCR or flattened "
            "typesetting makes a number ambiguous, retain the expression as qualitative evidence and "
            "explain the ambiguity instead of silently repairing it. Never choose a midpoint "
            "or endpoint to manufacture a scalar. Evidence can concern declared parameters, conditions "
            "or calculated outputs, but a quoted equation or range is not itself a computed output or "
            "supported target assumption. Measurements of ingredients or related systems can inform an "
            "assumption even without a dedicated numeric field. Preserve such measurements in qualitative "
            "evidence with their numbers, units and context, and tag the declared quantities they inform. "
            "Do not rename an ingredient's measurement as the final parameter or perform the inference "
            "during extraction. Label empirical ranges or bounds as evidence_role=observation, and "
            "identify the actual observed quantities in observed_fields. A modeling assumption, fitted "
            "parameter or equation is model_statement, even if reported by a source; it has no observed_fields. "
            "Keep the same event subject for its measurements and modeling choices; context preserves "
            "case definitions, populations, activity, and exposure segments. Reuse known_source_subjects "
            "only when they identify the same event or population, not merely a related topic. "
            "Do not merge different events or erase subgroup and time-window distinctions. "
            "Use empty arrays when nothing qualifies. For source=request, "
            "only explicit requester-defined conditions are evidence; requested unknown answers and "
            "workflow requirements are not scientific evidence."
        ),
    })
    candidates, diagnostics = [], []
    rows = response.get("facts", [])
    if not isinstance(rows, list):
        raise ValueError("Extraction facts must be a list")
    for raw in rows:
        try:
            field = raw["field"]
            if field not in UNITS:
                raise ValueError("Unknown field")
            quote = quote_span(raw["quote"])
            unit = UNITS[field]
            if not isinstance(raw["value"], (int, float)) or isinstance(raw["value"], bool):
                raise ValueError(f"{field} must be a finite JSON number, not {type(raw['value']).__name__}; "
                                 "preserve ranges, bounds and expressions in qualitative evidence")
            value = finite(raw["value"], field)
            printed_unit = canonical_notation(raw["unit"])
            factor = 1 if printed_unit == unit else CONVERSIONS.get((printed_unit, unit))
            if factor is None:
                raise ValueError(f"Unsupported unit conversion from {raw['unit']!r} to {unit!r}")
            value *= factor
            validate_parameter(field, value, unit)
            kinds = {"request_constraint"} if source == "request" else {"source_observation", "source_model_parameter"}
            if raw["kind"] not in kinds:
                raise ValueError("An extraction cannot create an assumed or derived fact")
            fact = Fact(raw["subject"], field, value, unit, raw["kind"], raw["basis"],
                        source, quote, raw["context"], source_version=source_version)
            candidates.append((raw, fact))
        except (KeyError, TypeError, ValueError) as exc:
            field = raw.get("field", "unknown") if isinstance(raw, dict) else "unknown"
            diagnostics.append({"stage": "extraction", "category": "invalid_candidate", "status": "rejected", "source": source,
                                "field": field, "why": str(exc), "proposal": raw})
    qualitative = response.get("evidence", [])
    if not isinstance(qualitative, list):
        raise ValueError("Qualitative evidence must be a list")
    for raw in qualitative:
        try:
            item = {name: required_text(raw[name], name) for name in ("subject", "claim", "context", "basis")}
            fields = raw.get("fields")
            if not isinstance(fields, list) or not fields:
                raise ValueError("Qualitative evidence requires a nonempty list of declared parameter, condition or output names")
            invalid = [f for f in fields if not isinstance(f, str) or f not in EVIDENCE_FIELDS]
            if invalid:
                raise ValueError(f"Unknown evidence fields {invalid!r}; use names declared in response_contract")
            item["fields"] = list(dict.fromkeys(fields))
            role = raw.get("evidence_role")
            roles = {"request_constraint"} if source == "request" else {"observation", "model_statement", "context"}
            if role not in roles:
                raise ValueError("Qualitative evidence requires a declared evidence_role")
            observed = raw.get("observed_fields")
            if (not isinstance(observed, list) or any(not isinstance(field, str) or field not in UNITS for field in observed)
                    or len(set(observed)) != len(observed) or (role != "observation" and observed)):
                raise ValueError("observed_fields must name actual observed quantities, not inferred/model quantities or requester constraints")
            item.update(evidence_role=role, observed_fields=observed)
            item.update(source=source, quote=quote_span(raw["quote"]),
                        source_version=source_version,
                        kind="request_constraint" if source == "request" else "source_evidence")
            constraints = raw.get("numeric_constraints", [])
            if not isinstance(constraints, list):
                diagnostics.append({"stage": "extraction", "category": "invalid_candidate", "status": "rejected",
                    "source": source, "why": "numeric_constraints must be a list", "proposal": constraints})
                constraints = []
            item["numeric_constraints"], accepted_constraints = [], []
            for constraint in constraints:
                try:
                    if not isinstance(constraint, dict):
                        raise ValueError("Each numeric constraint must be an object")
                    relation = constraint.get("relation")
                    if (relation not in NUMERIC_CONSTRAINT_FIELDS
                            or set(constraint) != {"field", "relation", "unit", *NUMERIC_CONSTRAINT_FIELDS[relation]}):
                        raise ValueError("Numeric constraint fields must match its declared relation")
                    field = constraint["field"]
                    if field not in UNITS:
                        raise ValueError("Numeric constraint requires a declared quantity")
                    target_unit = UNITS[field]
                    printed_unit = canonical_notation(constraint["unit"])
                    factor = 1 if printed_unit == target_unit else CONVERSIONS.get((printed_unit, target_unit))
                    if factor is None:
                        raise ValueError("Numeric constraint has no supported canonical unit conversion")
                    normalized = {"field": field, "relation": relation, "unit": target_unit}
                    for name in NUMERIC_CONSTRAINT_FIELDS[relation]:
                        value = constraint[name]
                        if name.endswith("_inclusive"):
                            if type(value) is not bool:
                                raise ValueError("Numeric constraint inclusivity must be boolean")
                            normalized[name] = value
                        else:
                            if type(value) not in (int, float):
                                raise ValueError("Numeric constraint values must be finite JSON numbers")
                            normalized[name] = finite(finite(value, name) * factor, name)
                    if relation == "interval" and (normalized["minimum"] > normalized["maximum"]
                            or (normalized["minimum"] == normalized["maximum"] and not (
                                normalized["minimum_inclusive"] and normalized["maximum_inclusive"]))):
                        raise ValueError("Numeric constraint interval is empty or reversed")
                    item["numeric_constraints"].append(normalized)
                    accepted_constraints.append(constraint)
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    diagnostics.append({"stage": "extraction", "category": "invalid_candidate", "status": "rejected",
                        "source": source, "field": constraint.get("field") if isinstance(constraint, dict) else None,
                        "why": str(exc), "proposal": constraint})
            candidates.append(({**raw, "numeric_constraints": accepted_constraints}, item))
        except (KeyError, TypeError, ValueError) as exc:
            diagnostics.append({"stage": "extraction", "category": "invalid_candidate", "status": "rejected", "source": source,
                                "why": str(exc), "proposal": raw})
    admitted, evidence = [], []
    if candidates:
        decisions = review_candidates(agent, {
            "operation": "review_research_evidence",
            "objective": "Verify each candidate's faithful representation of its native source quantity and context.",
            "context": {"review_scope": "source_faithfulness_only",
                        "target_relevance": "Graded separately; not a reason to reject a faithful source record."},
            "source": source, "source_text": text,
            "available_source_spans": spans,
            "candidates": [{"option": i, "raw": raw, "normalized": fact.semantic_record() if isinstance(fact, Fact) else fact,
                            **({"unit_conversion": {"printed_value": raw["value"], "printed_unit": raw["unit"],
                                "canonical_unit": fact.unit,
                                "factor": 1 if canonical_notation(raw["unit"]) == fact.unit else
                                          CONVERSIONS[(canonical_notation(raw["unit"]), fact.unit)]}}
                               if isinstance(fact, Fact) else {
                                   "numeric_constraint_conversions": [{"index": index,
                                       "printed": printed, "canonical": normalized,
                                       "factor": 1 if canonical_notation(printed["unit"]) == normalized["unit"] else
                                           CONVERSIONS[(canonical_notation(printed["unit"]), normalized["unit"])]}
                                       for index, (printed, normalized) in enumerate(
                                           zip(raw["numeric_constraints"], fact["numeric_constraints"]))]})}
                           for i, (raw, fact) in enumerate(candidates)],
            "instruction": (
                "Independently verify each numerical value and unit against its quote and source. "
                "The raw candidate records the printed number and unit; normalized records use canonical "
                "units after the explicit unit_conversion. Check that conversion rather than requiring "
                "the canonical number to occur verbatim in the source. Raw basis text refers to the "
                "printed representation, not a claim that normalization did not happen. "
                "Check that event identity, disease/strain/population/activity and the distinction "
                "between observations, fitted parameters and hypothetical scenarios faithfully "
                "describe the SOURCE. Admit every faithfully represented native value. A different "
                "activity, population, variant, setting, weak target relevance, missing companion "
                "input or need for transfer is never a reason to reject it. This review does not "
                "decide whether a source value should be used for a target. Existing source-target "
                "relevance and target-input assessments make that decision separately. "
                "Reject errors of source representation: ventilation relabeled as total loss, RNA "
                "copies relabeled as quanta, a modeled value called a measurement, or an invented "
                "value or source attribution. Preserve native roles in context: recipient inhalation "
                "is a breathing flow even when it is not a matched source exhalation rate. "
                "A source's hypothetical room inputs remain source model parameters; they need "
                "not estimate the parameter of the current investigation. Do not require a "
                "numerical contribution to that local question. "
                " Qualitative claims must be entailed by the quoted passage. "
                "Verify each normalized numeric_constraint against its original quote, "
                "including the measured quantity, interval or inequality, endpoint inclusivity and "
                "activity/population qualifiers. numeric_constraint_conversions gives its printed and "
                "canonical forms and deterministic factor. Do not require a converted bound to appear "
                "verbatim in the source. Constraints inherit this record's source and context; they "
                "are reported bounds, not selected target inputs. For qualitative records, "
                "admit judges the original claim, attribution and evidence fields independently of "
                "the supplemental constraints. accepted_numeric_constraints lists only the zero-based "
                "indices of normalized constraints faithfully represented by the quote; omit an "
                "unsupported constraint without rejecting otherwise faithful qualitative evidence. "
                "Use [] for numeric facts, records with no constraints, rejected records, or when no "
                "constraint is supported. Explain any omitted constraints in reason. The qualitative record's fields are relevance "
                "tags, not assertions that the source directly measures those quantities. A related "
                "measurement or model relation can inform a downstream parameter. Do not reject a "
                "source-supported claim merely because a relevance tag names an inferred quantity. "
                "Reject unjustified tags or a claim that equates distinct physical quantities. "
                "This does not relax numeric fact field/value/unit identity or license inventing "
                "an inferred value during extraction. Independently verify "
                "evidence_role and observed_fields: a reported measurement or event-outcome range can "
                "be observation, but a fitted model parameter or assumed range cannot. observed_fields "
                "must identify quantities actually observed in the quote, not merely quantities the "
                "evidence might help infer. For source=request, require evidence_role=request_constraint, "
                "no observed_fields, and admit "
                "only conditions the requester explicitly defines, not scientific answers they ask you to discover."
            ),
            "response_contract": {"type": "object", "additionalProperties": False,
                "required": ["decisions"], "properties": {
                    "decisions": {"type": "array", "minItems": len(candidates), "maxItems": len(candidates),
                        "description": "Exactly one independent decision for each candidate option, without repeats",
                        "items": {"type": "object", "additionalProperties": False,
                            "required": ["option", "admit", "reason", "accepted_numeric_constraints"], "properties": {
                                "option": {"type": "integer", "enum": list(range(len(candidates)))},
                                "admit": {"type": "boolean", "description": "True for a faithful native source record, regardless of target relevance. For qualitative evidence, decide its original claim and fields independently of supplemental numeric constraints. False only for an error in source representation, quantity, attribution or entailment."},
                                "reason": {"type": "string", "minLength": 1},
                                "accepted_numeric_constraints": {"type": "array", "uniqueItems": True,
                                    "items": {"type": "integer", "minimum": 0},
                                    "description": "Zero-based indices of faithful normalized numeric constraints; [] for numeric facts, no constraints, rejected records, or no faithful constraints."}},
                            "allOf": [*({"if": {"properties": {"option": {"const": index}}},
                                "then": {"properties": {"accepted_numeric_constraints": {
                                    "items": {"type": "integer", "enum": list(range(len(item["numeric_constraints"])))}
                                        if not isinstance(item, Fact) and item["numeric_constraints"] else False}}}}
                                for index, (_raw, item) in enumerate(candidates)),
                                {"if": {"properties": {"admit": {"const": False}}},
                                 "then": {"properties": {"accepted_numeric_constraints": {"maxItems": 0}}}}]}}}},
        }, source_version)
        for decision in decisions:
            index = decision["option"]
            if decision["admit"]:
                item = candidates[index][1]
                if isinstance(item, Fact):
                    admitted.append(item)
                else:
                    selected = set(decision.get("accepted_numeric_constraints", ()))
                    evidence.append({**item, "numeric_constraints": [constraint
                        for index, constraint in enumerate(item["numeric_constraints"]) if index in selected]})
                    for constraint_index, constraint in enumerate(item["numeric_constraints"]):
                        if constraint_index not in selected:
                            diagnostics.append({"stage": "applicability_review", "category": "rejected_evidence",
                                "status": "rejected", "source": source, "field": constraint["field"],
                                "why": decision["reason"], "proposal": constraint})
            else:
                item = candidates[index][1]
                field = item.field if isinstance(item, Fact) else item["claim"]
                diagnostics.append({"stage": "applicability_review", "category": "rejected_evidence", "status": "rejected",
                                    "source": source, "claim": field, "why": decision["reason"],
                                    "proposal": candidates[index][0]})
    if not rows and not qualitative:
        diagnostics.append({"stage": "extraction", "category": "no_relevant_evidence", "status": "completed", "source": source,
                            "why": "The reading returned no relevant evidence"})
    return ResearchResult(tuple(admitted),
                          evidence=tuple(evidence), diagnostics=tuple(diagnostic_value(d) for d in diagnostics),
                          evaluated=not (rows or qualitative) or bool(candidates))
