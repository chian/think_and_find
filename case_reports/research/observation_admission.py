"""Calculate returned observations against explicitly commissioned support templates.

No agents or search loops run here. An existing support_assessment fixes the
application, G/B_G snapshots, prediction, and finite protocol. The source supplies
native observations; the host matches conditions and executes their fixed cases.
"""
from .causal_sheet import CausalSheet
from .importance import evaluate_cases, rational
from .records import ResearchResult, identity
from .support_observations import observation_identity


def assess_returned_observations(source, result):
    templates = [key for key in source.commission.inputs
                 if source.ledger.causal_sheet.rows[key]["kind"] == "support_assessment"]
    # Child results may carry their inherited host ledger. Only their actual
    # evidence changes may trigger this commissioned source calculation.
    incoming = {row["key"]: row for row in result.causal_sheet_updates if row["kind"] == "evidence"}
    previous = source.ledger.causal_sheet.rows
    evidence_keys = [key for key, row in incoming.items()
                     if key not in previous or previous[key]["version"] != row["version"]]
    if not templates or not evidence_keys or not result.evaluated:
        return result
    staged = CausalSheet(source.ledger.causal_sheet.history)
    staged.merge(result.causal_sheet_updates)
    start = len(staged.history)
    diagnostics = []
    for template_key in templates:
        template = staged.rows[template_key]["data"]
        support = staged.rows[template["support"]]["data"]
        pins = staged.calculation_pins(template_key)
        if not staged.usable(template_key, _pins=pins):
            continue
        protocol = staged.rows[template["importance"]]["data"]
        for evidence_key in evidence_keys:
            evidence = staged.rows[evidence_key]["data"]
            native = evidence["measurements"].get("native")
            observations = native.get("observations", {}) if isinstance(native, dict) else {}
            if not isinstance(observations, dict):
                diagnostics.append({"stage": "observation_calculation", "status": "unresolved",
                    "category": "unsupported_source_operand", "why": "Native observations are not a named map",
                    "evidence": evidence_key})
                continue
            for observation_key, observation in observations.items():
                try:
                    observation_id, _ = observation_identity(evidence, observation_key)
                except (ValueError, KeyError, TypeError, AttributeError) as exc:
                    diagnostics.append({"stage": "observation_calculation", "status": "unresolved",
                        "category": "unsupported_source_operand", "why": str(exc), "evidence": evidence_key})
                    continue
                conditions = observation["conditions"]
                if (not isinstance(conditions, dict) or
                    any(observation[field] != template["prediction"][field] for field in ("quantity", "units"))):
                    continue
                matches = [case["id"] for case in protocol["domain"]["cases"]
                           if set(conditions) == set(case["inputs"]) and
                           all(rational(str(conditions[name]) if type(conditions[name]) is float else conditions[name]) == rational(value)
                               for name, value in case["inputs"].items())]
                if not matches:
                    continue
                token = identity([support["inference"], support["expectation"], template["application"],
                                  template["importance"], observation_id,
                                  template_key, staged.rows[template_key]["version"],
                                  evidence_key, staged.rows[evidence_key]["version"], observation_key])[:24]
                support_key, argument_key = "observed-support:" + token, "observation-comparison:" + token
                check_key, contribution_key = "observed-assessment:" + token, "observed-contribution:" + token
                staged.put(support_key, "expectation_support", {
                    "inference": support["inference"], "expectation": support["expectation"],
                    "grounds": [template["application"], evidence_key],
                    "argument": "Compare this source observation within the explicitly commissioned G/B_G realization",
                    "limitations": "Source extraction and the supplied model/quantity mapping remain explicit scientific inputs"})
                staged.put(argument_key, "reasoning", {
                    "author": "host:observation-comparison", "premises": [template["application"], evidence_key, template["importance"]],
                    "steps": ["Match source-native quantity, units and numeric conditions to the commissioned prediction and fixed circumstances",
                              "Execute the declared paired model comparison; derive alignment and G dependence with the support calculator"],
                    "conclusion": "The attached assessment reports computed alignment, dependence or unresolved operands",
                    "limitations": "This public calculation description is not a model verdict or a proof of empirical adequacy"})
                data = {**template, "support": support_key,
                        "facts": [{"evidence": evidence_key, "observation": observation_key, "case_id": matches[0]}]}
                staged.put(check_key, "support_assessment", data)
                staged.merge(evaluate_cases(staged, template["importance"], [matches[0]], context=check_key))
                staged.put(contribution_key, "support_contribution", {
                    "support": support_key, "applications": [template["application"]],
                    "facts": [{"evidence": evidence_key, "observation": observation_key}],
                    "argument": argument_key, "checks": [check_key]})
    from dataclasses import replace
    return replace(result, causal_sheet_updates=(*result.causal_sheet_updates, *staged.history[start:]),
                   diagnostics=(*result.diagnostics, *diagnostics))
