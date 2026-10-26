"""Calculations retain method credit while their selected premises change."""

import copy
import unittest

from episodic_workflow.case_reports.research.assumptions import (
    RUBRIC, assess_assumption, assumption_candidate, candidate_response_schema, unassessed_assumption,
)
from episodic_workflow.case_reports.research.calculation_steps import calculation_step_status
from episodic_workflow.case_reports.research.goal_state import scientific_goal_status
from episodic_workflow.case_reports.research.records import Fact, Ledger, ResearchResult, identity
from episodic_workflow.case_reports.research.tools import execute_tool, main_calculation_inputs, tool_options
from episodic_workflow.task_executors.scenario_metrics.evaluate import UNITS


OUTPUT = "concentration_relaxation_time_s"


def fixture():
    ledger = Ledger(ResearchResult(scenarios=({
        "subject": "target", "description": "Synthetic air volume",
        "output_contracts": {OUTPUT: {}},
        "fixed_conditions": {"well_mixed": "Synthetic uniform background"},
    },)))
    for field, value in (("room_volume_m3", 100.0), ("removal_capacity_m3_s", .2)):
        add_assumption(ledger, field, value)
    ledger.scenarios["target"]["fixed_parameters"] = {"removal_capacity_m3_s": {"value": .2, "unit": "m3/s"}}
    return ledger


def add_assumption(ledger, field, value):
    candidate = {"subject": "target", "field": field, "value": value, "unit": UNITS[field],
                 "statement": "Synthetic numerical hypothesis", "basis": "Offline lifecycle test"}
    row = unassessed_assumption(candidate, ledger)
    ledger.merge(ResearchResult(assumptions=(row,)))
    return row


def calculate(ledger, *, review=None, label="one case", tool="calculate_geometry_and_removal"):
    result = execute_tool({"action": tool, "subject": "target",
        "metrics": [OUTPUT], "basis": "Examine conditional relaxation time", "parameterization": label},
        ledger, (OUTPUT,), review=review or {"admit": True, "reason": "Compatible synthetic calculation"})
    ledger.merge(result)
    return result


def rows(ledger):
    return {row["field"]: row for row in calculation_step_status(ledger.result(), (OUTPUT,))["steps"]}


class CalculationStepTests(unittest.TestCase):
    def test_mask_alternative_reference_cannot_override_main_one(self):
        ledger = fixture()
        metrics = ("quanta_emission_rate_s", "emission_rate_quanta_s")
        ledger.scenarios["target"]["output_contracts"] = {name: {} for name in metrics}
        ledger.scenarios["target"]["assumption_needs"] = [
            "mask_penetration", "breathing_flow_m3_s", "quanta_concentration_exhaled_m3"]
        main = add_assumption(ledger, "mask_penetration", 1.)
        main["source_applicability"] = {"score": .12}
        ledger.merge(ResearchResult(assumptions=(main,)))
        alternative = add_assumption(ledger, "mask_penetration", .7)
        add_assumption(ledger, "breathing_flow_m3_s", .00025)
        add_assumption(ledger, "quanta_concentration_exhaled_m3", 1.)
        action = {"action": "calculate_transmission_metrics", "subject": "target",
            "facts": [alternative["key"]], "metrics": ["emission_rate_quanta_s"],
            "basis": "An obsolete proposal cannot select a different mask value",
            "parameterization": "main inputs", "conditions": {
                "single_infectious_source": "One source", "uniform_breathing_and_masks": "Shared inputs"}}
        result = execute_tool(action, ledger, metrics,
            review={"admit": True, "reason": "Compatible conditional arithmetic"})
        ledger.merge(result)
        self.assertEqual(result.calculations[0]["input_values"]["mask_penetration"], 1.)
        steps = {r["field"]: r for r in calculation_step_status(ledger.result(), metrics)["steps"]}
        self.assertAlmostEqual(steps["emission_rate_quanta_s"]["computed_value"], .00025)
        self.assertTrue(steps["emission_rate_quanta_s"]["matches_main_selection"])
        self.assertEqual(steps["emission_rate_quanta_s"]["main_input_differences"], [])
        alternative["source_applicability"] = {"score": .9}
        ledger.merge(ResearchResult(assumptions=(alternative,)))
        stale = {r["field"]: r for r in calculation_step_status(ledger.result(), metrics)["steps"]}
        self.assertEqual(stale["emission_rate_quanta_s"]["status"], "calculated_needs_updating")
        self.assertEqual(stale["emission_rate_quanta_s"]["points"], .025)
        self.assertEqual(stale["quanta_emission_rate_s"]["status"], "calculated_up_to_date")
        result = execute_tool(action, ledger, metrics,
            review={"admit": True, "reason": "Compatible conditional arithmetic"})
        self.assertEqual(result.calculations[0]["input_values"]["mask_penetration"], .7)
        self.assertAlmostEqual(result.estimates[0]["metrics"]["emission_rate_quanta_s"]["value"], .000175)

    def test_obsolete_alternative_reference_is_replaced_by_current_main_input(self):
        ledger = fixture()
        main = next(a for a in ledger.assumptions.values() if a["field"] == "room_volume_m3")
        main["source_applicability"] = {"score": .8}
        alternative = add_assumption(ledger, "room_volume_m3", 200.)
        result = execute_tool({"action": "calculate_geometry_and_removal", "subject": "target",
            "facts": [alternative["key"]], "metrics": [OUTPUT], "basis": "Obsolete alternative reference",
            "parameterization": "automatic main"}, ledger, (OUTPUT,),
            review={"admit": True, "reason": "Compatible arithmetic"})
        ledger.merge(result)
        row = rows(ledger)[OUTPUT]
        self.assertEqual(row["status"], "calculated_up_to_date")
        self.assertEqual(row["computed_value"], 500.)
        self.assertTrue(row["matches_main_selection"])
        self.assertEqual(row["main_input_differences"], [])
        self.assertEqual(result.calculations[0]["input_values"]["room_volume_m3"], 100.)
        replacement = add_assumption(ledger, "room_volume_m3", 300.)
        replacement["source_applicability"] = {"score": .9}
        ledger.merge(ResearchResult(assumptions=(replacement,)))
        self.assertEqual(rows(ledger)[OUTPUT]["status"], "calculated_needs_updating")
        self.assertEqual(rows(ledger)[OUTPUT]["points"], .025)
        calculate(ledger, label="updated main")
        self.assertAlmostEqual(rows(ledger)[OUTPUT]["computed_value"], 1500.)

    def test_legacy_alternative_cannot_retain_current_credit(self):
        for legacy in (False, True):
            with self.subTest(no_snapshot=legacy):
                ledger = fixture()
                original = next(a for a in ledger.assumptions.values() if a["field"] == "room_volume_m3")
                original["source_applicability"] = {"score": .5}
                result = calculate(ledger)
                replacement = add_assumption(ledger, "room_volume_m3", 200.)
                replacement["source_applicability"] = {"score": .9}
                ledger.merge(ResearchResult(assumptions=(replacement,)))
                record = copy.deepcopy(result.calculations[0])
                for dependency in (record, *record["metric_support"].values(), *record["derived_support"].values()):
                    if legacy:
                        dependency.pop("selected_target_bindings", None)
                    else:
                        for binding in dependency.get("selected_target_bindings", ()):
                            binding.update(follows_main_selection=False, selected_value=200.)
                ledger.calculations = {record["version"]: record}
                ledger.estimates = {}
                row = rows(ledger)[OUTPUT]
                self.assertEqual(row["status"], "calculated_needs_updating")
                self.assertEqual(row["points"], .025)
                self.assertEqual(row["computed_value"], 500.)
                self.assertEqual(row["main_input_differences"][0]["used_value"], 100.)
                self.assertEqual(row["main_input_differences"][0]["current_selected_value"], 200.)

    def test_computed_input_binding_uses_the_same_result_as_calculation_table(self):
        ledger = fixture()
        results = []
        for admit, basis in ((False, "First unsupported result"), (True, "Second accepted result")):
            results.append(execute_tool({"action": "calculate_geometry_and_removal", "subject": "target",
                "metrics": [OUTPUT], "basis": basis, "parameterization": basis}, ledger, (OUTPUT,),
                review={"admit": admit, "reason": basis}))
        for result in results:
            ledger.merge(result)
        row = rows(ledger)["removal_rate_s"]
        self.assertEqual(row["calculation_version"], results[1].calculations[0]["version"])
        fact = next(f for f in main_calculation_inputs(ledger, "target") if f.field == "removal_rate_s")
        self.assertIn(fact.key, results[1].calculations[0]["output_fact_keys"])
        self.assertEqual((fact.value, fact.unit), (row["computed_value"], row["unit"]))

    def test_execution_snapshot_tracks_numeric_main_changes_for_any_candidate_origin(self):
        ledger = fixture()
        main = next(a for a in ledger.assumptions.values() if a["field"] == "room_volume_m3")
        main["source_applicability"] = {"score": .4}
        result = calculate(ledger)
        self.assertTrue(rows(ledger)[OUTPUT]["matches_main_selection"])
        self.assertEqual(next(iter(tool_options(ledger).values()))["current_computed_metrics"], [OUTPUT])
        main["source_applicability"] = {"score": .8}
        self.assertEqual(rows(ledger)[OUTPUT]["status"], "calculated_up_to_date")
        replacement = add_assumption(ledger, "room_volume_m3", 200.)
        replacement["source_applicability"] = {"score": .9}
        ledger.merge(ResearchResult(assumptions=(replacement,)))
        row = rows(ledger)[OUTPUT]
        self.assertEqual(row["status"], "calculated_needs_updating")
        self.assertEqual(row["points"], .025)
        self.assertFalse(row["matches_main_selection"])
        self.assertEqual(row["main_input_differences"][0]["used_value"], 100.)
        self.assertEqual(row["main_input_differences"][0]["current_selected_value"], 200.)
        self.assertEqual(next(iter(tool_options(ledger).values()))["current_computed_metrics"], [])
        change = row["result_alternatives"][0]["changed_dependencies"][0]
        self.assertEqual((change["previous_value"], change["selected_value"]), (100., 200.))
        self.assertIn("selected_target_bindings", result.estimates[0]["metric_support"][OUTPUT])

    def test_reusing_stale_derived_input_cannot_rebase_its_selection_snapshot(self):
        ledger = fixture()
        main = next(a for a in ledger.assumptions.values() if a["field"] == "room_volume_m3")
        main["source_applicability"] = {"score": .4}
        calculate(ledger)
        replacement = add_assumption(ledger, "room_volume_m3", 200.)
        replacement["source_applicability"] = {"score": .9}
        ledger.merge(ResearchResult(assumptions=(replacement,)))
        rate = next(key for key, fact in ledger.facts.items()
                    if fact.field == "removal_rate_s" and fact.kind == "derived")
        reused = execute_tool({"action": "calculate_geometry_and_removal", "subject": "target",
            "facts": [rate], "metrics": [OUTPUT], "basis": "Reuse old derived input after main changed",
            "parameterization": "reuse stale producer"}, ledger, (OUTPUT,),
            review={"admit": True, "reason": "Arithmetic can execute with an old input"})
        ledger.merge(reused)
        row = next(a for a in rows(ledger)[OUTPUT]["result_alternatives"]
                   if a["parameterization"] == "reuse stale producer")
        self.assertEqual(row["status"], "calculated_up_to_date")
        self.assertEqual(row["computed_value"], 1000.)
        self.assertTrue(row["matches_main_selection"])
        self.assertEqual(row["main_input_differences"], [])
        self.assertNotIn(rate, reused.calculations[0]["input_fact_keys"])
        self.assertEqual(reused.calculations[0]["input_values"]["room_volume_m3"], 200.)

    def test_upfront_steps_have_zero_credit_and_report_recipe_readiness(self):
        ledger = fixture()
        steps = rows(ledger)
        self.assertTrue({OUTPUT, "removal_rate_s"} <= set(steps))
        self.assertNotIn("infectious_dose_virions", steps, "Unrelated executable metrics are not useful steps")
        self.assertTrue(all(row["status"] == "not_calculated" and row["points"] == 0 for row in steps.values()))
        self.assertTrue(steps[OUTPUT]["ready_to_calculate"])
        self.assertTrue(steps["removal_rate_s"]["ready_to_calculate"])
        self.assertNotIn("removal_capacity_m3_s", steps, "Fixed input is not a required calculation")

    def test_computed_intermediate_and_output_are_separate_from_values_found(self):
        ledger = fixture()
        before = scientific_goal_status(ledger.result(), (OUTPUT,))
        calculate(ledger)
        after = scientific_goal_status(ledger.result(), (OUTPUT,))
        self.assertEqual(after["recorded_target_inputs"], before["recorded_target_inputs"])
        self.assertEqual(after["scores"]["found_parameters"], 1,
                         "The recorded volume hypothesis earns discovery; computed quantities do not")
        self.assertEqual(after["scores"]["calculation_steps"], .25)
        self.assertEqual(rows(ledger)["removal_rate_s"]["status"], "calculated_up_to_date")
        self.assertNotIn("removal_capacity_m3_s", rows(ledger),
                         "Passing through a fixed supplied quantity is not a calculation")

    def test_repetition_and_relabelling_do_not_multiply_credit(self):
        ledger = fixture()
        calculate(ledger)
        before = calculation_step_status(ledger.result(), (OUTPUT,))["points"]
        calculate(ledger)
        calculate(ledger, label="different label")
        self.assertEqual(calculation_step_status(ledger.result(), (OUTPUT,))["points"], before)

    def test_values_follow_selected_calculation_through_staleness_and_recalculation(self):
        ledger = fixture()
        self.assertIsNone(rows(ledger)[OUTPUT]["computed_value"])
        calculate(ledger)
        first = rows(ledger)[OUTPUT]
        self.assertEqual(first["computed_value"], 500.)
        self.assertEqual(first["unit"], "s")
        key = next(key for key, a in ledger.assumptions.items() if a["field"] == "room_volume_m3")
        changed = {**ledger.assumptions[key], "statement": "Updated volume hypothesis", "value": 200.}
        changed["version"] = identity(changed)
        ledger.assumptions[key] = changed
        stale = rows(ledger)[OUTPUT]
        self.assertEqual(stale["status"], "calculated_needs_updating")
        self.assertEqual(stale["computed_value"], 500.)
        self.assertEqual(stale["calculation_version"], first["calculation_version"])
        self.assertEqual(stale["points"], .025)
        calculate(ledger, label="updated case")
        updated = rows(ledger)[OUTPUT]
        self.assertEqual(updated["status"], "calculated_up_to_date")
        self.assertEqual(updated["computed_value"], 1000.)
        self.assertEqual(updated["parameterization"], "updated case")
        self.assertEqual(updated["points"], .125)

    def test_fresh_uncredited_alternative_preserves_old_reusable_credit(self):
        ledger = fixture()
        calculate(ledger)
        key = next(key for key, a in ledger.assumptions.items() if a["field"] == "room_volume_m3")
        changed = {**ledger.assumptions[key], "statement": "Revised physical interpretation"}
        changed["version"] = identity(changed)
        ledger.assumptions[key] = changed
        calculate(ledger, label="fresh unreviewed case",
                  review={"admit": False, "completed": False, "reason": "Review unavailable"})
        row = rows(ledger)[OUTPUT]
        self.assertEqual(row["status"], "calculated_up_to_date")
        self.assertEqual(row["parameterization"], "fresh unreviewed case")
        self.assertEqual(row["points"], .025)
        self.assertFalse(row["accepted_current"])
        self.assertNotEqual(row["credit_calculation_version"], row["calculation_version"])
        self.assertEqual(row["credit_basis"], "0.025 credit is retained from previously accepted work "
            "in 'one case'; it was not earned by the displayed calculation.")

    def test_zero_and_unbounded_values_remain_explicit(self):
        ledger = fixture()
        ledger.scenarios["target"]["fixed_parameters"]["removal_capacity_m3_s"]["value"] = 0.
        for assumption in ledger.assumptions.values():
            if assumption["field"] == "removal_capacity_m3_s":
                assumption["value"] = 0.
                assumption["version"] = identity(assumption)
        calculate(ledger)
        selected = rows(ledger)
        self.assertEqual(selected["removal_rate_s"]["computed_value"], 0.)
        self.assertEqual(selected[OUTPUT]["computed_value"], "unbounded")
        self.assertEqual(selected[OUTPUT]["computed_limit"], "unbounded")

    def test_changed_claim_retains_only_method_credit_and_recalculation_restores_result(self):
        ledger = fixture()
        calculate(ledger)
        key = next(key for key, a in ledger.assumptions.items() if a["field"] == "room_volume_m3")
        changed = {**ledger.assumptions[key], "statement": "Revised physical interpretation"}
        changed["version"] = identity(changed)
        ledger.assumptions[key] = changed
        stale = rows(ledger)
        for field in (OUTPUT, "removal_rate_s"):
            self.assertEqual(stale[field]["status"], "calculated_needs_updating")
            self.assertEqual(stale[field]["points"], .025)
            self.assertTrue(stale[field]["ready_to_recalculate"])
            self.assertTrue(stale[field]["result_alternatives"][0]["changed_dependencies"])
        calculate(ledger)
        self.assertEqual(rows(ledger)[OUTPUT]["status"], "calculated_up_to_date")
        self.assertEqual(rows(ledger)[OUTPUT]["points"], .125)

    def test_new_alternative_and_unchanged_claim_review_do_not_stale_previous_result(self):
        ledger = fixture()
        calculate(ledger)
        add_assumption(ledger, "room_volume_m3", 200)
        for key, old in tuple(ledger.assumptions.items()):
            revised = {**old, "basis": "Further review without changing the selected claim"}
            revised["version"] = identity(revised)
            ledger.assumptions[key] = revised
        self.assertEqual(rows(ledger)[OUTPUT]["status"], "calculated_up_to_date")

    def test_rejected_or_incomplete_execution_review_has_no_method_credit(self):
        for review in ({"admit": False, "reason": "Incompatible model"},
                       {"admit": False, "completed": False, "reason": "Review unavailable"}):
            with self.subTest(review=review):
                ledger = fixture()
                calculate(ledger, review=review)
                row = rows(ledger)[OUTPUT]
                self.assertEqual(row["status"], "calculated_up_to_date")
                self.assertEqual(row["review_status"], "incomplete" if review.get("completed") is False else "rejected")
                self.assertEqual(row["uncredited_reason"], review["reason"])
                self.assertEqual(row["computed_value"], 500.)
                self.assertFalse(row["accepted_current"])
                self.assertGreater(calculation_step_status(ledger.result(), (OUTPUT,))["calculated_up_to_date"], 0)
                self.assertEqual(calculation_step_status(ledger.result(), (OUTPUT,))["points"], 0)

    def test_missing_condition_failure_is_not_prior_completion(self):
        ledger = fixture()
        ledger.scenarios["target"]["fixed_conditions"] = {}
        calculate(ledger)
        self.assertEqual(rows(ledger)[OUTPUT]["status"], "not_calculated")
        self.assertEqual(rows(ledger)[OUTPUT]["points"], 0)
        self.assertEqual(rows(ledger)["removal_rate_s"]["points"], .125,
                         "A downstream missing condition does not reject a valid intermediate")

    def test_executed_output_that_misses_contract_is_visible_without_credit(self):
        ledger = fixture()
        ledger.scenarios["target"]["output_contracts"][OUTPUT] = {"result_form": "scalar",
            "operation": "different_requested_operation"}
        calculate(ledger)
        row = rows(ledger)[OUTPUT]
        self.assertEqual(row["status"], "calculated_needs_updating")
        self.assertEqual(row["points"], 0)
        self.assertEqual(row["uncredited_reason"], "Arithmetic completed but does not meet the requested output contract")

    def test_inconsistent_parameterization_has_no_intermediate_credit(self):
        ledger = fixture()
        add_assumption(ledger, "floor_area_m2", 10)
        add_assumption(ledger, "room_height_m", 20)
        with self.assertRaises(ValueError):
            execute_tool({"action": "calculate_geometry_and_removal", "subject": "target",
                "metrics": [OUTPUT],
                "basis": "Intentionally inconsistent geometry", "parameterization": "inconsistent"},
                ledger, (OUTPUT,), review={"admit": True, "reason": "Fixture accepted before numerical check"})
        self.assertFalse(ledger.calculations)
        self.assertFalse(ledger.estimates)
        self.assertFalse([fact for fact in ledger.facts.values() if fact.kind == "derived"])
        self.assertEqual(calculation_step_status(ledger.result(), (OUTPUT,))["points"], 0)

    def test_declared_duration_variable_is_not_a_missing_scalar_input(self):
        ledger = fixture()
        metric = "indoor_reproductive_number"
        scenario = ledger.scenarios["target"]
        scenario["output_contracts"] = {metric: {}}
        scenario["calculation_variables"] = ["exposure_time_s"]
        scenario["fixed_conditions"].update(single_infectious_source="One source",
            uniform_breathing_and_masks="Uniform breathing and masks")
        for field, value in (("susceptible_count", 10), ("breathing_flow_m3_s", .001),
                ("quanta_concentration_exhaled_m3", 20), ("mask_penetration", 1), ("relative_transmissibility", 1)):
            add_assumption(ledger, field, value)
        steps = calculation_step_status(ledger.result(), (metric,))["steps"]
        row = next(row for row in steps if row["field"] == metric)
        self.assertTrue(row["ready_to_calculate"])
        result = execute_tool({"action": "calculate_transmission_metrics", "subject": "target",
            "metrics": [metric], "basis": "Duration relation",
            "parameterization": "symbolic time"}, ledger, (metric,),
            review={"admit": True, "reason": "Compatible synthetic duration relation"})
        ledger.merge(result)
        calculated = next(row for row in calculation_step_status(ledger.result(), (metric,))["steps"]
                          if row["field"] == metric)
        self.assertEqual(calculated["status"], "calculated_up_to_date")
        self.assertEqual(calculated["computed_limit"], "function")
        self.assertEqual(calculated["computed_value"], result.estimates[0]["metrics"][metric]["function"])

    def test_requested_input_echo_is_not_a_completed_calculation(self):
        ledger = fixture()
        field = "removal_rate_s"
        ledger.scenarios["target"]["fixed_parameters"][field] = {"value": .002, "unit": UNITS[field]}
        ref = "obsolete-reference"
        result = execute_tool({"action": "calculate_geometry_and_removal", "subject": "target",
            "facts": [ref], "metrics": [field], "basis": "Direct input echo", "parameterization": "echo"},
            ledger, (field,), review={"admit": True, "reason": "Compatible input"})
        ledger.merge(result)
        status = calculation_step_status(ledger.result(), (field,))
        self.assertEqual(status["points"], 0)

    def test_stale_support_bonus_is_withdrawn(self):
        ledger = fixture()
        for key, old in tuple(ledger.assumptions.items()):
            ledger.assumptions[key] = {**old, "reasonableness": 1.0,
                "review": {"verdict": "supported", "completed": True}}
        # The fixed condition is a real premise record, not evidence discovered by this calculation.
        condition = {"subject": "target", "field": "well_mixed", "value": None, "unit": "condition",
            "statement": "Synthetic uniform background", "basis": "Fixed case", "key": "fixed-condition",
            "version": "fixed-v1", "reasonableness": 1.0, "role": "scenario_constraint",
            "review": {"verdict": "fixed_condition", "completed": True}, "evidence": {}}
        ledger.assumptions[condition["key"]] = condition
        calculate(ledger)
        self.assertEqual(rows(ledger)[OUTPUT]["points"], 1.0)
        key = next(key for key, a in ledger.assumptions.items() if a["field"] == "room_volume_m3")
        ledger.assumptions[key]["review"] = {"verdict": "contradicted", "completed": True}
        self.assertEqual(rows(ledger)[OUTPUT]["points"], .025)

    def test_actual_assessment_refutes_selected_value_without_discarding_alternative(self):
        ledger = fixture()
        calculate(ledger)
        old = next(a for a in ledger.assumptions.values() if a["field"] == "room_volume_m3")
        alternative = add_assumption(ledger, "room_volume_m3", 200)
        self.assertNotEqual(old["key"], alternative["key"])
        self.assertEqual(rows(ledger)[OUTPUT]["status"], "calculated_up_to_date")
        review = {"verdict": "contradicted", "criteria": {
            name: {"grade": 0, "reason": "Offline contradiction lifecycle fixture", "evidence": []}
            for name in RUBRIC}, "limitations": [], "alternatives": [], "missing_evidence": [],
            "explanation": "The selected numerical hypothesis is rejected in this fixture",
            "request_constraint": None, "supporting_evidence": [], "contradicting_evidence": [],
            "invalid_evidence": []}
        assessed = assess_assumption(lambda request: review, {**old, "evidence": []}, ledger, {})
        ledger.merge(assessed)
        self.assertEqual(assessed.assumptions[0]["key"], old["key"])
        self.assertEqual(rows(ledger)[OUTPUT]["points"], .025)
        self.assertTrue(rows(ledger)[OUTPUT]["ready_to_recalculate"])
        self.assertEqual(ledger.assumptions[alternative["key"]]["review"]["verdict"], "unassessed")

    def test_retained_default_is_not_a_candidate_until_promoted(self):
        ledger = fixture()
        field = "gravity_m_s2"
        default = {"decision": "retain", "value": 9.81, "unit": UNITS[field]}
        ledger.scenarios["target"]["model_defaults"] = {field: default}
        action = {"subject": "target", "field": field, "value": 9.7, "unit": UNITS[field],
                  "statement": "Synthetic alternative gravity", "basis": "Offline test"}
        with self.assertRaises(ValueError):
            assumption_candidate(action, ledger)
        choices = candidate_response_schema(scenarios=ledger.scenarios)["oneOf"][0]["allOf"][0]["then"]["properties"]["field"]["enum"]
        self.assertNotIn(field, choices)
        default["decision"] = "revisit"
        self.assertEqual(assumption_candidate(action, ledger)["value"], 9.7)

    def test_default_withdrawal_marks_selected_computation_stale(self):
        ledger = fixture()
        volume = next(key for key, a in ledger.assumptions.items() if a["field"] == "room_volume_m3")
        del ledger.assumptions[volume]
        default = {"decision": "retain", "value": 100., "unit": UNITS["room_volume_m3"],
                   "basis": "Synthetic default binding", "provenance": "Offline fixture"}
        ledger.scenarios["target"]["model_defaults"] = {"room_volume_m3": default}
        ledger.merge(ResearchResult(facts=(Fact("target", "room_volume_m3", 100., UNITS["room_volume_m3"],
            "model_default", "Synthetic default binding"),)))
        refs = []
        result = execute_tool({"action": "calculate_geometry_and_removal", "subject": "target",
            "facts": refs, "metrics": [OUTPUT], "basis": "Default binding lifecycle",
            "parameterization": "default case"}, ledger, (OUTPUT,),
            review={"admit": True, "reason": "Compatible synthetic default"})
        ledger.merge(result)
        self.assertEqual(rows(ledger)[OUTPUT]["points"], .125)
        rate = next(key for key, fact in ledger.facts.items()
                    if fact.field == "removal_rate_s" and fact.kind == "derived")
        downstream = execute_tool({"action": "calculate_geometry_and_removal", "subject": "target",
            "facts": [rate], "metrics": [OUTPUT], "basis": "Reuse previously calculated rate",
            "parameterization": "downstream reuse"}, ledger, (OUTPUT,),
            review={"admit": True, "reason": "Compatible synthetic downstream calculation"})
        ledger.merge(downstream)
        default["decision"] = "revisit"
        row = rows(ledger)[OUTPUT]
        self.assertEqual(row["points"], .025)
        self.assertFalse(row["ready_to_recalculate"])
        self.assertTrue(any(change["kind"] == "model_default" for alternative in row["result_alternatives"]
                            for change in alternative["changed_dependencies"]))
        reused = next(alternative for alternative in row["result_alternatives"]
                      if alternative["parameterization"] == "downstream reuse")
        self.assertEqual(reused["status"], "calculated_needs_updating")
        self.assertEqual(reused["computed_value"], 500.)
        self.assertTrue(any(change["kind"] == "model_default" and change["field"] == "room_volume_m3"
                            for change in reused["changed_dependencies"]))


if __name__ == "__main__":
    unittest.main()
