"""Calculate returned observations against explicitly commissioned support templates.

No agents or search loops run here. An existing support_assessment fixes the
application, G/B_G snapshots, prediction, and finite protocol. The source supplies
native observations; the host matches conditions and executes their fixed cases.
"""
from .causal_sheet import CausalSheet
from .importance import evaluate_cases, rational
from .records import ResearchResult, identity
from .support_observations import observation_identity


def compare_observations(sheet, templates, evidence_keys):
    """Page-selected fixed observation comparison; never called by readers."""
    for key in templates:
        sheet._require_kind(key, "support_assessment")
    for key in evidence_keys:
        sheet._require_kind(key, "evidence")
    staged = CausalSheet(sheet.history)
    start = len(staged.history)
    diagnostics = []
    matched = 0
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
                matched += 1
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
                # This new contribution assesses the returned observation. A
                # theoretical claim in the template retains its own assessment.
                data = {**template, "support": support_key, "source_formalization": "",
                        "facts": [{"evidence": evidence_key, "observation": observation_key, "case_id": matches[0]}]}
                staged.put(check_key, "support_assessment", data)
                staged.merge(evaluate_cases(staged, template["importance"], [matches[0]], context=check_key))
                staged.put(contribution_key, "support_contribution", {
                    "support": support_key, "applications": [],
                    "facts": [{"evidence": evidence_key, "observation": observation_key}],
                    "reasoning_key": argument_key, "checks": [check_key]})
    if not matched and not diagnostics:
        diagnostics.append({"stage": "observation_calculation", "status": "unresolved",
            "category": "no_matching_observation",
            "why": "No usable template matched source-native quantity, units and exact numeric circumstances"})
    return ResearchResult(causal_sheet_updates=tuple(staged.history[start:]), diagnostics=tuple(diagnostics))
