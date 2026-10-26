"""Useful child work returns through real Episode boundaries without fake observations."""

from dataclasses import replace
import unittest

from episodic_workflow.case_reports.research.acquisition import PageSource
from episodic_workflow.case_reports.research.documents import Document
from episodic_workflow.case_reports.research.engine import ResearchLimits, ResearchProviders, ResearchRun, ResearchSource
from episodic_workflow.case_reports.research.records import Ledger, ResearchResult
from episodic_workflow.case_reports.research.target_values import project_source_target_values
from episodic_workflow.method_loop import Context, EpisodeGoal, EpisodeTree
from episodic_workflow.task_executors.scenario_metrics.evaluate import DEFAULT_METRICS
from episodic_workflow.tests.test_source_value_handoff import initial_state, source_fact
from episodic_workflow.tests.test_default_calculations import planned_ledger, calculate


class CooperativeReturnTests(unittest.TestCase):
    def nested_run(self, fact):
        calls, trace, before_return, investigation_decisions = [], [], [], []

        def agent(request):
            calls.append(request)
            return {"findings": "The acquired result is available for the parent's decision."}

        initial = initial_state()
        run = ResearchRun(ResearchProviders(agent, lambda *_: (), lambda _: None,
            trace_event=trace.append), ResearchLimits(), "Nested useful work", DEFAULT_METRICS, initial=initial)
        root_source = ResearchSource(run, initial)
        root = run.episode("evaluate_airborne_safety_guideline", root_source,
            goal=EpisodeGoal.root(objective="Calculate from acquired values", result_contract={}), focus=())
        grains = ["investigate", "strategy", "search", "page", "lexical_probe"]

        def child_for(parent, view, index):
            source = ResearchSource(run, parent.ledger.result())
            grain = grains[index]

            def propose(child_view):
                if child_view.units_consumed:
                    self.assertEqual(grain, "investigate", "Acquisition must return its new operands immediately")
                    investigation_decisions.append(source.inputs(child_view))
                    run.agent({"operation": "investigation_uses_operands"})
                    return None
                if index == len(grains) - 1:
                    return source.leaf(lambda: ResearchResult(facts=(fact,)), "one_native_value")
                return source.edge(child_for(source, child_view, index + 1))

            source.propose = propose
            if index == len(grains) - 1:
                prior = source.after_verdict

                def after(unit, contribution, unit_view):
                    prior(unit, contribution, unit_view)
                    controller = next(c for path, c in run.live_controllers.items() if len(path) == 1)
                    before_return.append(controller.state())

                source.after_verdict = after
            return run.child(view, source, grain, f"Read useful operands at {grain}",
                contract=parent.contract("Find inputs for concentration", parameter="quanta_concentration_exhaled_m3",
                                         deliverable="source_findings"), proposal={})

        def propose_root(view):
            run.agent(root_source.inputs(view, operation="root_decision"))
            if view.units_consumed:
                return None
            return root_source.edge(child_for(root_source, view, 0))

        root_source.propose = propose_root
        record = root.run(Context(tree=run.tree))
        return run, record, calls, trace, before_return, investigation_decisions

    def test_target_value_unwinds_every_level_then_root_dispatches_next_decision(self):
        run, record, calls, trace, before, investigations = self.nested_run(
            source_fact("breathing_flow_m3_s", .00015))
        self.assertEqual(before[0].history, (0,))
        self.assertEqual(before[0].score, 1)
        self.assertEqual(record.controller_state.history, (0, 1))
        self.assertFalse(investigations, "Investigation returns target progress before another strategy proposal")
        node = record
        while node.unit_records:
            self.assertEqual(node.units_consumed, 1)
            self.assertFalse(node.controller_state.stop, "Cooperative return is not fabricated saturation")
            node = node.unit_records[0].child
            if node is None:
                break
        self.assertEqual([request["operation"] for request in calls].count("root_decision"), 2)
        self.assertEqual(calls[-1]["operation"], "root_decision")
        self.assertEqual(calls[-1]["global_scientific_goal"]["recorded_target_inputs"]["recorded"], 1)
        self.assertEqual(len([e for e in trace if e["event"] == "cooperative_return"]), 5)

    def test_source_operand_returns_to_investigation_even_without_target_credit(self):
        run, record, calls, trace, before, investigations = self.nested_run(
            source_fact("quanta_emission_rate_s", .2))
        self.assertEqual(run.global_goal.preview()["recorded_target_inputs"]["recorded"], 0)
        self.assertEqual(record.controller_state.history, (0, 0))
        self.assertEqual(len(investigations), 1)
        self.assertEqual([e["reason"] for e in trace if e["event"] == "cooperative_return"],
                         ["new_calculation_operands"] * 4)
        self.assertIn("investigation_uses_operands", [request["operation"] for request in calls])

    def test_completed_calculation_returns_without_new_found_value(self):
        ledger = planned_ledger({"effective_radius_m": 1e-6}, ("stokes_particles",))
        calls = []

        def agent(request):
            calls.append(request["operation"])
            return {"findings": "A conditional calculation is available."}

        run = ResearchRun(ResearchProviders(agent, lambda *_: (), lambda _: None), ResearchLimits(),
            "Return arithmetic promptly", ("settling_speed_m_s",), initial=ledger.result())
        parent = ResearchSource(run, ledger.result())
        root = run.episode("evaluate_airborne_safety_guideline", parent,
            goal=EpisodeGoal.root(objective="Compute a conditional quantity", result_contract={}), focus=())
        child = ResearchSource(run, ledger.result())

        def child_propose(view):
            self.assertEqual(view.units_consumed, 0)
            return child.leaf(lambda: calculate(child.ledger, "calculate_geometry_and_removal",
                                                ("settling_speed_m_s",)), "settling")

        def parent_propose(view):
            run.agent(parent.inputs(view, operation="root_decision"))
            if view.units_consumed:
                return None
            return parent.edge(run.child(view, child, "investigate", "Compute settling",
                contract=parent.contract("Compute settling", parameter="settling_speed_m_s"), proposal={}))

        child.propose, parent.propose = child_propose, parent_propose
        record = root.run(Context(tree=run.tree))
        self.assertEqual(run.global_goal.preview()["recorded_target_inputs"]["recorded"], 0)
        self.assertEqual(record.controller_state.history, (0, 1))
        self.assertEqual(child.cooperative_return_reason, "target_progress_changed")
        self.assertEqual(calls[-1], "root_decision")

    def test_relevance_change_for_same_numeric_value_does_not_force_return(self):
        ledger = Ledger(initial_state())
        first = source_fact("breathing_flow_m3_s", .00015)
        ledger.merge(ResearchResult(facts=(first,)))
        ledger.merge(project_source_target_values(ledger))
        run = ResearchRun(ResearchProviders(lambda _: {}, lambda *_: (), lambda _: None,
            source_target_assessor=lambda request: {subject: .9 for subject in request["target_scenarios"]}),
            ResearchLimits(), "Grades do not change arithmetic", DEFAULT_METRICS, initial=ledger.result())
        source = ResearchSource(run, ledger.result())
        episode = run.episode("investigate", source,
            goal=EpisodeGoal.root(objective="Examine changed applicability", result_contract={}), focus=(),
            contract=source.contract("Examine breathing", parameter="breathing_flow_m3_s"))
        decisions = []

        def propose(view):
            decisions.append(view.units_consumed)
            if view.units_consumed:
                return None
            return source.leaf(lambda: ResearchResult(facts=(replace(first, context="New applicability evidence"),)),
                               "same_number")

        source.propose = propose
        episode.run(Context(tree=EpisodeTree(run.grains["investigate"], {})))
        self.assertEqual(decisions, [0, 1])
        self.assertEqual(source.cooperative_return_reason, "")

    def page_run(self, agent, *, prior=(), objective="Inspect source for the assigned uncertainty"):
        text = "abcdefghij" * 45
        run = ResearchRun(ResearchProviders(agent, lambda *_: (), lambda _: text),
            ResearchLimits(text_window_chars=300), "Read a bounded source", DEFAULT_METRICS)
        source = PageSource(run, ResearchResult(reading_attempts=prior),
            candidate={"url": "https://example.test/covered"})
        episode = run.episode("page", source,
            goal=EpisodeGoal.root(objective=objective, result_contract={}), focus=(),
            contract=source.contract("Find source operands", parameter="quanta_concentration_exhaled_m3"))
        tree = EpisodeTree(run.grains["page"], {run.grains["page"]: (
            run.grains["report"], run.grains["lexical_probe"], run.grains["source_table"])})
        return run, source, episode, Context(tree=tree)

    def test_completed_page_closes_after_unread_tail_without_launching_reexamination(self):
        choices, windows = [], []

        def agent(request):
            if request["operation"] == "propose_page_child":
                choices.append(request)
                self.assertLessEqual(len(choices), 2, "Completed source task must not start another reader")
                return {"option": "lexical_probe" if len(choices) == 1 else "report",
                        "query": "absent quantity", "reading_scope": "unexamined",
                        "question": "Read the still-unexamined source material"}
            if request["operation"] == "extract_research_evidence":
                windows.append(request)
                return {"facts": [], "evidence": [], "findings": []}
            return {"findings": "The examined material did not supply the requested input."}

        _, source, episode, context = self.page_run(agent)
        record = episode.run(context)
        self.assertEqual(record.units_consumed, 2)
        self.assertEqual(record.unit_records[0].child.units_consumed, 4)
        self.assertTrue(record.unit_records[0].child.controller_state.stop)
        self.assertEqual(record.unit_records[1].child.units_consumed, 1)
        self.assertEqual(len(windows), 5)
        self.assertEqual(source.cooperative_return_reason, "source_task_examined")
        self.assertEqual(record.controller_state.history, (0, 0, 0))

    def test_separate_task_can_reexamine_only_explicit_focused_chunks(self):
        text = "abcdefghij" * 45
        document = Document("https://example.test/covered", "", text, 300)
        prior = ({"source": document.source, "source_version": document.sha256,
                  "document_characters": len(text), "ranges": [[0, len(text)]],
                  "status": "completed", "question": "A different earlier question", "page_task": "previous-task"},)
        read_ranges, choices = [], []

        def agent(request):
            if request["operation"] == "propose_page_child":
                choices.append(request)
                self.assertNotIn("report", request["available_children"])
                return {"option": "lexical_probe", "query": "specific context",
                        "reading_scope": "reexamine", "reexamine_spans": ["chunk:3"],
                        "question": "Reinterpret this passage for the newly assigned question"}
            if request["operation"] == "extract_research_evidence":
                read_ranges.extend((row["start"], row["end"]) for row in request["source_spans"].values())
                return {"facts": [], "evidence": [], "findings": []}
            return {"findings": "The focused passage was reconsidered under the new question."}

        _, source, episode, context = self.page_run(agent, prior=prior, objective="A genuinely different source question")
        record = episode.run(context)
        self.assertEqual(record.units_consumed, 1)
        self.assertEqual(len(choices), 1)
        self.assertEqual(set(read_ranges), {(200, 300)})
        retained = tuple(source.ledger.reading_attempts.values())
        _, _, repeated, repeated_context = self.page_run(
            lambda _: self.fail("The same completed task must not reopen the source"),
            prior=retained, objective="A genuinely different source question")
        self.assertEqual(repeated.run(repeated_context).units_consumed, 0)

    def test_failed_passage_remains_available_for_focused_retry(self):
        text = "abcdefghij" * 45
        document = Document("https://example.test/covered", "", text, 300)
        base = {"source": document.source, "source_version": document.sha256,
                "document_characters": len(text), "question": "Prior extraction"}
        prior = ({**base, "ranges": [[0, 200], [300, 450]], "status": "completed"},
                 {**base, "ranges": [[200, 300]], "status": "failed"})
        proposals = []

        def agent(request):
            if request["operation"] == "propose_page_child":
                proposals.append(request)
                self.assertEqual(set(request["reexamination_spans"]), {"chunk:3"})
                return {"option": "lexical_probe", "query": "failed passage",
                        "reading_scope": "reexamine", "reexamine_spans": ["chunk:3"],
                        "question": "Retry the passage whose extraction failed"}
            if request["operation"] == "extract_research_evidence":
                return {"facts": [], "evidence": [], "findings": []}
            return {"findings": "The failed passage has now been examined."}

        _, _, episode, context = self.page_run(agent, prior=prior)
        record = episode.run(context)
        self.assertEqual(record.units_consumed, 1)
        self.assertEqual(len(proposals), 1)

    def test_failed_focused_reread_is_retried_despite_prior_complete_coverage(self):
        text = "abcdefghij" * 45
        document = Document("https://example.test/covered", "", text, 300)
        prior = ({"source": document.source, "source_version": document.sha256,
                  "document_characters": len(text), "ranges": [[0, len(text)]],
                  "status": "completed", "question": "Earlier task", "page_task": "earlier"},)
        choices, extractions = [], []

        def agent(request):
            if request["operation"] == "propose_page_child":
                choices.append(request)
                self.assertLessEqual(len(choices), 2)
                return {"option": "lexical_probe", "query": "specific passage",
                        "reading_scope": "reexamine", "reexamine_spans": ["chunk:3"],
                        "question": "Interpret this passage for the new task"}
            if request["operation"] == "extract_research_evidence":
                extractions.append(request)
                if len(extractions) == 1:
                    raise ValueError("Synthetic failed extraction")
                return {"facts": [], "evidence": [], "findings": []}
            return {"findings": "Focused rereading outcome."}

        _, source, episode, context = self.page_run(agent, prior=prior, objective="New source question")
        record = episode.run(context)
        self.assertEqual(record.units_consumed, 2)
        self.assertEqual(len(choices), 2)
        self.assertEqual(len(extractions), 2)
        self.assertEqual(source.cooperative_return_reason, "source_task_examined")


if __name__ == "__main__":
    unittest.main()
