"""Shared calculation revisions reach controllers before a child returns."""

from dataclasses import replace
from episodic_workflow.case_reports.research.records import numeric_references
import json
import unittest

from episodic_workflow.case_reports.research.assumptions import unassessed_assumption
from episodic_workflow.case_reports.research.engine import (
    ResearchLimits, ResearchProviders, ResearchRun, ResearchSource, changed_records,
    research_parameter_requirements,
    calculation_decision_view,
)
from episodic_workflow.case_reports.research.goal_state import GLOBAL_GOAL_PROGRESS, scientific_goal_status
from episodic_workflow.case_reports.research.model_defaults import initialize_model_defaults
from episodic_workflow.case_reports.research.scenarios import BAZANT_BUSH_CASES, initial_scenarios
from episodic_workflow.task_executors.scenario_metrics.evaluate import DEFAULT_METRICS
from episodic_workflow.case_reports.research.records import Credit, Fact, Ledger, ResearchResult, identity
from episodic_workflow.case_reports.research.tools import execute_tool
from episodic_workflow.credit_assignment.research import ResearchController
from episodic_workflow.method_loop import Context, EpisodeGoal


def completed_removal():
    ledger = Ledger(ResearchResult(scenarios=({
        "subject": "case", "description": "Synthetic room", "role": "target",
        "assumption_needs": ["room_volume_m3"], "output_contracts": {},
        "fixed_parameters": {"removal_capacity_m3_s": {"value": .2, "unit": "m3/s"}},
    },)))
    for field, value, unit in (("room_volume_m3", 100, "m3"),):
        claim = unassessed_assumption({"subject": "case", "field": field,
            "value": value, "unit": unit, "statement": "Synthetic " + field,
            "basis": "Explicit conditional test value"}, ledger)
        ledger.merge(ResearchResult(assumptions=(claim,)))
    calculation = execute_tool({"action": "calculate_geometry_and_removal", "subject": "case",
        "basis": "Synthetic removal",
        "parameterization": "test", "metrics": ["removal_rate_s"]}, ledger,
        ("removal_rate_s",), review={"admit": True, "reason": "Compatible synthetic inputs"})
    ledger.merge(calculation)
    original = next(row for row in ledger.assumptions.values() if row["field"] == "room_volume_m3")
    revised = {**original, "statement": "The same number now describes a different physical volume"}
    revised["version"] = identity({k: v for k, v in revised.items() if k != "version"})
    return ledger.result(), revised


class SharedCalculationUpdateTests(unittest.TestCase):
    def test_default_calculation_decision_table_has_bounded_prompt_size(self):
        initial = initialize_model_defaults(initial_scenarios(BAZANT_BUSH_CASES))
        full = scientific_goal_status(initial, DEFAULT_METRICS)["calculation_steps"]
        compact = calculation_decision_view(full, detailed=True)
        self.assertGreaterEqual(len(compact["rows"]), 96)
        self.assertEqual(len(compact["rows"]), len(full["steps"]))
        self.assertLess(len(json.dumps(compact)), 15000)
        self.assertLess(len(json.dumps(compact)), len(json.dumps(full)) / 5)
        self.assertEqual(compact["points"], full["points"])
        self.assertEqual(set(row[3] for row in compact["rows"]), {"not_calculated"})
        self.assertNotIn("rows", calculation_decision_view(full, detailed=False))

    def test_transitive_default_change_reaches_single_calculation_table(self):
        output = "concentration_relaxation_time_s"
        ledger = Ledger(ResearchResult(scenarios=({
            "subject": "case", "description": "Synthetic default dependency",
            "fixed_conditions": {"well_mixed": "Synthetic uniform background"},
            "fixed_parameters": {"removal_capacity_m3_s": {"value": .2, "unit": "m3/s"}},
            "model_defaults": {"room_volume_m3": {"decision": "retain", "value": 100., "unit": "m3"}},
        },), facts=(
            Fact("case", "room_volume_m3", 100, "m3", "model_default", "Synthetic baseline"),
            Fact("case", "removal_capacity_m3_s", .2, "m3/s", "user_assumption", "Synthetic capacity"),
        )))
        for metrics, fields in (([], {"removal_capacity_m3_s"}), ([output], {"removal_rate_s"})):
            result = execute_tool({"action": "calculate_geometry_and_removal", "subject": "case",
                "metrics": metrics, "basis": "upstream" if not metrics else "downstream"},
                ledger, (output,), review={"admit": True, "reason": "Synthetic arithmetic"})
            ledger.merge(result)
        before = scientific_goal_status(ledger.result(), (output,))
        before_row = next(row for row in before["calculation_steps"]["steps"] if row["field"] == output)
        self.assertEqual(before_row["status"], "calculated_up_to_date")
        self.assertEqual(before_row["computed_value"], 500)
        self.assertEqual(before_row["unit"], "s")
        ledger.scenarios["case"]["model_defaults"]["room_volume_m3"]["decision"] = "revisit"
        after = scientific_goal_status(ledger.result(), (output,))
        row = next(row for row in after["calculation_steps"]["steps"] if row["field"] == output)
        self.assertEqual(row["status"], "calculated_needs_updating")
        self.assertEqual(row["points"], .025)
        self.assertEqual(row["computed_value"], 500, "A stale result retains the actual computed value")
        self.assertNotIn("computed_outputs", after)

    def test_uncredited_calculation_prompt_explains_review_and_zero_credit(self):
        for review_status, review in (
            ("rejected", {"admit": False, "reason": "The selected model is incompatible"}),
            ("incomplete", {"admit": False, "completed": False, "reason": "The review could not finish"}),
        ):
            with self.subTest(review_status=review_status):
                initial, _ = completed_removal()
                initial = replace(initial,
                    calculations=tuple({**row, "model_review": review} for row in initial.calculations),
                    estimates=tuple({**row, "model_review": review} for row in initial.estimates))
                run = ResearchRun(ResearchProviders(lambda _: {}, lambda *_: (), lambda _: None),
                    ResearchLimits(), "Explain uncredited arithmetic", ("removal_rate_s",), initial=initial)
                source = ResearchSource(run, initial)
                prompt = {}

                def propose(view):
                    prompt.update(source.inputs(view))
                    return None

                source.propose = propose
                root = run.episode("evaluate_airborne_safety_guideline", source,
                    goal=EpisodeGoal.root(objective="Explain uncredited arithmetic", result_contract={}), focus=())
                root.run(Context(tree=run.tree))
                table = prompt["global_scientific_goal"]["calculation_steps"]
                self.assertEqual(table["calculated_up_to_date"], 1)
                row = next(dict(zip(table["columns"], cells)) for cells in table["rows"]
                           if cells[1] == "removal_rate_s")
                self.assertEqual(row["status"], "calculated_up_to_date")
                self.assertEqual(row["points"], 0)
                self.assertEqual(row["kind"], "requested_output")
                self.assertEqual(row["computed_value"], .002)
                self.assertEqual(row["unit"], "1/s")
                self.assertNotIn("computed_outputs", prompt["global_scientific_goal"])
                self.assertNotIn("ready_to_recalculate", prompt["global_scientific_goal"])
                notes = [table["notes"][index] for index in row["note_ids"]]
                self.assertIn({"kind": "calculation_credit", "review_status": review_status,
                               "reason": review["reason"]}, notes)

    def test_retained_constants_are_not_source_learning_requirements(self):
        constant = "air_density_kg_m3"
        scenarios = {
            "ordinary": {"role": "target", "model_defaults": {constant: {"decision": "retain"}}},
            "exception": {"role": "target", "model_defaults": {constant: {"decision": "revisit"}}},
        }
        self.assertNotIn(constant, research_parameter_requirements("settling_speed_m_s", scenarios, ("ordinary",)))
        self.assertIn(constant, research_parameter_requirements("settling_speed_m_s", scenarios, ("exception",)))
        self.assertIn(constant, research_parameter_requirements("settling_speed_m_s", scenarios))

    def test_refresh_changes_live_score_without_consuming_yield_observations(self):
        def credit(points):
            return Credit(qualities=((GLOBAL_GOAL_PROGRESS, "total", points),), quality_resolution=.025)
        controller = ResearchController((GLOBAL_GOAL_PROGRESS,), point_scores=True, initial=credit(.125))
        controller.refresh(credit(.025))
        self.assertEqual(controller.state().score, .025)
        self.assertEqual(controller.state().history, (.125,))
        self.assertFalse(controller.state().stop)
        step = controller.observe("child_return", credit(.025), is_root=True)
        self.assertAlmostEqual(step.gain, -.1)
        self.assertEqual(controller.state().history, (.125, .025))
        self.assertFalse(step.stop)
        controller.refresh(credit(.125))
        restored = controller.observe("recalculation_return", credit(.125), is_root=True)
        self.assertAlmostEqual(restored.gain, .1)

    def test_child_invalidates_actual_root_before_return_and_exposes_actionable_table(self):
        initial, revised = completed_removal()
        trace = []
        providers = ResearchProviders(lambda _: {"findings": "The selected volume claim changed."},
                                      lambda *_: (), lambda _: None, trace_event=trace.append)
        run = ResearchRun(providers, ResearchLimits(), "Synthetic revision", ("removal_rate_s",), initial=initial)
        found_credit = run.global_goal.preview()["scores"]["found_parameters"]
        parent = ResearchSource(run, initial)
        child = ResearchSource(run, initial)
        acquisition = ResearchSource(run, initial)
        observed = {}
        root_goal = EpisodeGoal.root(objective="Synthetic revision", result_contract={})
        root = run.episode("evaluate_airborne_safety_guideline", parent, goal=root_goal, focus=())
        parent_units = iter((True,))
        child_units = iter((True,))
        child_view = None

        def child_propose(view):
            nonlocal child_view
            child_view = view
            if next(child_units, False):
                return child.leaf(lambda: ResearchResult(assumptions=(revised,)), "revise_premise")
            self.fail("Changed calculation status must return at the ordinary unit boundary")

        previous_after_verdict = child.after_verdict

        def child_after_verdict(unit, contribution, view):
            previous_after_verdict(unit, contribution, view)
            root_controller = next(controller for path, controller in run.live_controllers.items() if len(path) == 1)
            observed["root_score_before_return"] = root_controller.state().score
            observed["root_history_before_return"] = root_controller.state().history
            own_controller = next(controller for path, controller in run.live_controllers.items()
                                  if path[-1][1] == child.controller_key)
            current_view = replace(child_view, controller_state=own_controller.state())
            observed["inputs"] = child.inputs(current_view)
            observed["acquisition_inputs"] = acquisition.inputs(current_view)

        def parent_propose(view):
            if not next(parent_units, False):
                observed["parent_inputs_after_return"] = numeric_references(parent.ledger)
                observed["parent_prompt_after_return"] = parent.inputs(view)
                return None
            contract = parent.contract("Explain the changed premise", subjects=("case",),
                                       parameter="removal_rate_s", deliverable="assumption_hypotheses")
            return parent.edge(run.child(view, child, "investigate", "Revise premise",
                                         contract=contract, proposal={}))

        child.propose = child_propose
        child.after_verdict = child_after_verdict
        parent.propose = parent_propose
        record = root.run(Context(tree=run.tree))
        self.assertAlmostEqual(observed["root_score_before_return"], found_credit + .025)
        self.assertEqual(observed["root_history_before_return"], (found_credit + .125,))
        self.assertAlmostEqual(record.controller_state.score, found_credit + .025)
        self.assertAlmostEqual(record.unit_records[0].controller_step.gain, -.1)
        status = observed["inputs"]["global_scientific_goal"]
        table = status["calculation_steps"]
        cells = next(row for row in table["rows"] if row[1] == "removal_rate_s")
        row = dict(zip(table["columns"], cells))
        self.assertEqual(row["status"], "calculated_needs_updating")
        self.assertTrue(row["ready_to_recalculate"])
        self.assertEqual(row["computed_value"], .002)
        self.assertEqual(row["unit"], "1/s")
        self.assertNotIn("ready_to_recalculate", status)
        self.assertIn("room_volume_m3", row["changed_inputs"])
        self.assertTrue(all(table["notes"][index]["reason"] for index in row["note_ids"]))
        self.assertNotIn("routes", row)
        self.assertNotIn("result_alternatives", row)
        root_status = observed["parent_prompt_after_return"]["global_scientific_goal"]
        self.assertEqual(root_status["calculation_steps"], table)
        self.assertNotIn("computed_outputs", root_status)
        acquisition_status = observed["acquisition_inputs"]["global_scientific_goal"]
        self.assertNotIn("rows", acquisition_status["calculation_steps"])
        self.assertEqual(acquisition_status["calculation_steps"]["points"], table["points"])
        self.assertNotIn("ready_to_recalculate", acquisition_status)
        volume_options = [fact for fact in observed["parent_inputs_after_return"].values()
                          if fact.field == "room_volume_m3" and fact.kind == "model_assumption"]
        self.assertEqual({fact.context for fact in volume_options}, {revised["statement"]})
        self.assertTrue(any(event["event"] == "shared_credit_refreshed" and event["gain"] < 0 for event in trace))

    def test_stale_inherited_snapshot_cannot_restore_old_claim(self):
        initial, revised = completed_removal()
        run = ResearchRun(ResearchProviders(lambda _: {}, lambda *_: (), lambda _: None),
                          ResearchLimits(), "Synthetic stale overlay", ("removal_rate_s",), initial=initial)
        stale_source = ResearchSource(run, initial)
        stale_source.scientific = True
        found_credit = run.global_goal.preview()["scores"]["found_parameters"]
        run.global_goal.commit(ResearchResult(assumptions=(revised,)))
        self.assertAlmostEqual(sum(row[2] for row in stale_source.credit(initial).qualities), found_credit + .025)
        stale_parent_return = replace(initial, findings=("Independent child finding",))
        delta = changed_records(stale_parent_return, initial)
        self.assertEqual(delta.assumptions, ())
        self.assertEqual(delta.calculations, ())
        run.global_goal.commit(delta)
        self.assertEqual(run.global_goal.ledger.assumptions[revised["key"]]["statement"], revised["statement"])
        self.assertAlmostEqual(run.global_goal.preview()["weighted_score"], found_credit + .025)


if __name__ == "__main__":
    unittest.main()
