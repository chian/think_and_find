"""All numeric origins share relevance, selection and committed target progress."""

import unittest
from episodic_workflow.case_reports.research.records import numeric_references

from episodic_workflow.case_reports.research.assumptions import unassessed_assumption
from episodic_workflow.case_reports.research.engine import ResearchLimits, ResearchProviders, ResearchRun, ResearchSource, calculation_decision_view
from episodic_workflow.case_reports.research.goal_state import scientific_goal_status
from episodic_workflow.case_reports.research.model_defaults import initialize_model_defaults
from episodic_workflow.case_reports.research.records import Fact, Ledger, ResearchResult, identity
from episodic_workflow.case_reports.research.relevance import assess_numeric_target_relationships
from episodic_workflow.case_reports.research.scenario_sharing import initialize_parameter_scope
from episodic_workflow.case_reports.research.scenarios import initial_scenarios, load_scenarios
from episodic_workflow.case_reports.research.target_values import project_source_target_values, target_value_candidates
from episodic_workflow.case_reports.research.tools import execute_tool
from episodic_workflow.method_loop import Context, EpisodeGoal
from episodic_workflow.task_executors.scenario_metrics.evaluate import DEFAULT_METRICS, UNITS


def initial_state():
    initial = initialize_model_defaults(initial_scenarios(load_scenarios()[1:4:2]))
    return initialize_parameter_scope(initial, lambda request: {"features": [],
        "dependencies": [{"parameter": field, "factors": [], "reason": "Equivalent test targets"}
                         for field in request["parameters"]]})


def native(field, value, *, subject="native context", context="Actual source context"):
    return Fact(subject, field, value, UNITS[field], "source_model_parameter", "Printed source value",
        source="https://example.test/source", quote=f"{field}={value}", context=context)


def inferred(ledger, field, value, *, refs=(), subject=None):
    return unassessed_assumption({"subject": subject or next(iter(ledger.scenarios)),
        "field": field, "value": value, "unit": UNITS[field], "entry_type": "best_guess",
        "statement": "A conditional target estimate with explicitly uncertain transfer.",
        "basis": "Adapt the source context to the target; this is an estimate, not a source measurement.",
        "source_refs": list(ledger.evidence_options()), "input_refs": list(refs),
        "added_assumptions": ["The stated contextual adjustment is an unverified approximation."]}, ledger)


def project(ledger, grader):
    candidates = target_value_candidates(ledger)
    ledger.merge(assess_numeric_target_relationships(ledger, grader, candidates=candidates))
    ledger.merge(project_source_target_values(ledger, candidates=candidates))
    return scientific_goal_status(ledger.result(), DEFAULT_METRICS)


class CandidatePipelineTests(unittest.TestCase):
    def test_compact_calculation_note_distinguishes_used_and_main_values(self):
        difference = {"subject": "target", "field": "mask_penetration", "used_value": .7,
            "used_unit": "fraction", "current_selected_value": 1, "current_selected_unit": "fraction"}
        table = calculation_decision_view({"steps": [{"subject": "target", "field": "emission_rate_quanta_s",
            "kind": "intermediate", "status": "calculated_needs_updating", "computed_value": .000175,
            "unit": "quanta/s", "points": .125, "parameterization": "cloth-mask sensitivity",
            "matches_main_selection": False, "main_input_differences": [difference]}]}, detailed=True)
        row = dict(zip(table["columns"], table["rows"][0]))
        self.assertFalse(row["matches_main_selection"])
        note = table["notes"][row["note_ids"][0]]
        self.assertEqual(note["parameterization"], "cloth-mask sensitivity")
        self.assertEqual(note["inputs"], [difference])
        self.assertEqual(row["status"], "calculated_needs_updating")

    def test_native_computed_and_inferred_values_compete_through_one_grade_and_cache(self):
        ledger = Ledger(initial_state())
        field, source = "quanta_concentration_exhaled_m3", "observed source event"
        source_inputs = (native("breathing_flow_m3_s", 1.65 / 3600, subject=source),
                         native("quanta_emission_rate_s", 79.3 / 3600, subject=source))
        ledger.merge(ResearchResult(facts=(*source_inputs, native(field, 142)), scenarios=({
            "subject": source, "role": "calibration", "description": "Source event",
            "basis": "Observed context, separate from target cases"},)))
        refs = [key for key, fact in numeric_references(ledger).items() if fact in source_inputs]
        calculation = execute_tool({"action": "calibrate_infectiousness", "subject": source,
            "operator": "cq_from_emission", "source_inputs": refs, "basis": "Conditional source concentration"},
            ledger, DEFAULT_METRICS, review={"admit": True, "reason": "Arithmetic is coherent"})
        ledger.merge(calculation)
        guess = inferred(ledger, field, 29)
        ledger.merge(ResearchResult(assumptions=(guess,)))
        requests = []

        def grade(request):
            requests.append(request)
            value = request["source_profile"]["facts"][0]
            score = {"source_model_parameter": .2, "derived": .6, "inferred_parameter": .9}[value["kind"]]
            return {subject: score for subject in request["target_scenarios"]}

        status = project(ledger, grade)
        group = next(row for row in status["independent_target_values"] if row["parameter"] == field)
        self.assertEqual(group["value"], 29)
        self.assertEqual(group["selected_assumption_key"], guess["key"])
        self.assertEqual(group["source_applicability"]["score"], .9)
        self.assertEqual(ledger.assumptions[guess["key"]]["review"]["verdict"], "unassessed")
        self.assertEqual(sum(a["field"] == field and a["value"] == 29 for a in ledger.assumptions.values()), 1)
        profiles = [request["source_profile"] for request in requests if request["parameter"] == field]
        self.assertEqual({profile["kinds"][0] for profile in profiles},
                         {"source_model_parameter", "derived", "inferred_parameter"})
        self.assertTrue(next(p for p in profiles if "inference" in p)["inference"]["sources"])
        self.assertEqual(next(p for p in profiles if "derivation" in p)["derivation"]["calculations"][0]["operator"],
                         "cq_from_emission")
        count = len(requests)
        self.assertEqual(project(ledger, grade)["recorded_target_inputs"], status["recorded_target_inputs"])
        self.assertEqual(len(requests), count)
        reviewed = {**ledger.assumptions[guess["key"]], "review": {
            "completed": True, "verdict": "unsupported", "criteria": {}}, "reasonableness": 0}
        reviewed["version"] = identity(reviewed)
        ledger.merge(ResearchResult(assumptions=(reviewed,)))
        self.assertEqual(project(ledger, grade)["target_inputs"][guess["subject"]][field]["value"], 29)
        self.assertEqual(len(requests), count, "Scientific review alone is not a new relevance premise")

    def test_ungraded_inference_fills_group_without_a_duplicate_or_support_gate(self):
        for grader in (None, lambda _: (_ for _ in ()).throw(ValueError("No grade available"))):
            ledger = Ledger(initial_state())
            guess = inferred(ledger, "mask_penetration", .7)
            ledger.merge(ResearchResult(assumptions=(guess,)))
            status = project(ledger, grader)
            row = next(row for row in status["independent_target_values"] if row["parameter"] == guess["field"])
            self.assertTrue(row["found"])
            self.assertEqual(row["points"], 1)
            self.assertEqual(row["selected_assumption_key"], guess["key"])
            self.assertIsNone(row["source_applicability"]["score"])
            self.assertEqual(len([a for a in ledger.assumptions.values() if a["field"] == guess["field"]
                                 and a.get("role") == "inferred_assumption"]), 1)

    def test_inferred_replacement_updates_root_prompt_before_child_returns(self):
        initial = initial_state()
        field, requests, observed = "mask_penetration", [], []
        first = native(field, 1, context="Unmasked source model; no cloth-mask performance is measured")

        def grade(request):
            requests.append(request)
            profile = request["source_profile"]
            score = .8 if profile["kinds"] == ["inferred_parameter"] else .12
            return {subject: score for subject in request["target_scenarios"]}

        run = ResearchRun(ResearchProviders(lambda _: {"findings": "Conditional value revised"},
            lambda *_: (), lambda _: None, source_target_assessor=grade), ResearchLimits(),
            "One target candidate pipeline", DEFAULT_METRICS, initial=initial)
        parent = ResearchSource(run, initial)
        root = run.episode("evaluate_airborne_safety_guideline", parent,
            goal=EpisodeGoal.root(objective="Target values", result_contract={}), focus=())
        subjects = tuple(row["subject"] for row in initial.scenarios)
        invocation, guess_key = 0, None

        def parent_propose(view):
            nonlocal invocation, guess_key
            if invocation == 2:
                observed.append((None, parent.inputs(view)))
                return None
            invocation += 1
            child = ResearchSource(run, parent.ledger.result())
            original_after = child.after_verdict
            child_view = None

            def child_propose(view):
                nonlocal child_view, guess_key
                child_view = view
                if invocation == 1:
                    result = ResearchResult(facts=(first,))
                else:
                    guess = inferred(child.ledger, field, .7)
                    guess_key = guess["key"]
                    result = ResearchResult(assumptions=(guess,))
                return child.leaf(lambda: result, "record_numeric_candidate")

            def after(*args):
                original_after(*args)
                root_controller = next(controller for path, controller in run.live_controllers.items() if len(path) == 1)
                observed.append((root_controller.state(), child.inputs(child_view)))

            child.propose, child.after_verdict = child_propose, after
            return parent.edge(run.child(view, child, "investigate", "Mask estimate",
                contract=parent.contract("Conditional mask estimate", parameter=field, subjects=subjects,
                                         deliverable="assumption_hypotheses"), proposal={}))

        parent.propose = parent_propose
        root.run(Context(tree=run.tree))
        before, after = (item[1]["global_scientific_goal"] for item in observed[:2])
        self.assertEqual(before["target_inputs"][subjects[0]][field]["value"], 1)
        self.assertEqual(after["target_inputs"][subjects[0]][field]["value"], .7)
        self.assertEqual(before["recorded_target_inputs"], after["recorded_target_inputs"])
        self.assertEqual(observed[0][0].score, observed[1][0].score)
        self.assertEqual(observed[0][0].history, (0,))
        self.assertEqual(observed[1][0].history, (0, observed[0][0].score))
        self.assertEqual(observed[-1][1]["global_scientific_goal"]["target_inputs"], after["target_inputs"])
        self.assertEqual(len(requests), 2, "Both origins use the same cache across child return")
        self.assertIn(guess_key, run.global_goal.ledger.assumptions)
        self.assertEqual(sum(a["field"] == field and a["value"] == .7
                             for a in run.global_goal.ledger.assumptions.values()), 1)


if __name__ == "__main__":
    unittest.main()
