"""An automatic value survives a real child return and the next parent dispatch."""

import unittest
from episodic_workflow.case_reports.research.records import numeric_references

from episodic_workflow.case_reports.research.engine import ResearchLimits, ResearchProviders, ResearchRun, ResearchSource
from episodic_workflow.case_reports.research.records import ResearchResult
from episodic_workflow.case_reports.research.workflow import GuidelineSource
from episodic_workflow.method_loop import Context, EpisodeGoal
from episodic_workflow.task_executors.scenario_metrics.evaluate import DEFAULT_METRICS
from .test_source_value_handoff import initial_state, source_fact


class DispatchReached(RuntimeError):
    pass


class ParentSourceValueViewTests(unittest.TestCase):
    def test_automatic_value_parent_dispatch_after_real_child_return(self):
        initial = initial_state()
        native = source_fact("breathing_flow_m3_s", .00015, "Quiet seated adults")
        dispatched = []

        def agent(request):
            if request["operation"] == "synthesize_research_findings":
                return {"findings": "A native breathing value is available with conditional applicability."}
            self.assertEqual(request["operation"], "propose_guideline_action")
            dispatched.append(request)
            raise DispatchReached("Real parent input validation passed before dispatch")

        run = ResearchRun(ResearchProviders(agent, lambda *_: (), lambda _: None,
            source_target_assessor=lambda request: {subject: .8 for subject in request["target_scenarios"]}),
            ResearchLimits(), "Use native values for selected cases", DEFAULT_METRICS, initial=initial)
        parent, child = GuidelineSource(run), ResearchSource(run, initial)
        root = run.episode("evaluate_airborne_safety_guideline", parent,
            goal=EpisodeGoal.root(objective="Compute the selected target", result_contract={}),
            focus=(native.field,))
        started = False

        def parent_propose(view):
            nonlocal started
            if started:
                return GuidelineSource.propose(parent, view)
            started = True
            return parent.edge(run.child(view, child, "investigate", "Obtain breathing value",
                contract=parent.contract("Resolve breathing value", parameter=native.field,
                    subjects=tuple(row["subject"] for row in initial.scenarios),
                    deliverable="assumption_hypotheses"), proposal={}))

        def child_propose(view):
            return (child.leaf(lambda: ResearchResult(facts=(native,)), "native_value")
                    if child.evaluated_units == 0 else None)

        parent.propose, child.propose = parent_propose, child_propose
        with self.assertRaises(DispatchReached):
            root.run(Context(tree=run.tree))
        self.assertEqual(len(dispatched), 1)
        request = dispatched[0]
        automatic = next(a for a in parent.ledger.assumptions.values()
                         if a.get("entry_type") == "source_candidate")
        view = next(a for a in request["assumption_options"].values()
                    if a.get("entry_type") == "source_candidate")
        self.assertEqual(view["value"], native.value)
        self.assertEqual(view["source_applicability"]["score"], .8)
        self.assertTrue(all(row["source_contexts"] == [native.context]
                            for row in view["source_applicability"]["relationships"]))
        self.assertEqual(automatic["source_fact_keys"], [native.key])
        self.assertTrue(all(native.key in row["source_value_scores"]
                            for row in automatic["source_applicability"]["relationships"]))
        self.assertEqual(parent.ledger.facts[native.key], native)
        derived_inputs = view["derivation"]["inputs"]
        self.assertEqual(len(derived_inputs), 1)
        self.assertEqual(numeric_references(parent.ledger)[derived_inputs[0]["reference"]], native)
        status = run.global_goal.preview()
        self.assertEqual(status["recorded_target_inputs"]["recorded"], 1)
        self.assertEqual(status["weighted_score"], request["global_scientific_goal"]["weighted_score"])


if __name__ == "__main__":
    unittest.main()
