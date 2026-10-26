"""Reading commits advance target values and live ancestor credit immediately."""

from dataclasses import replace
from episodic_workflow.case_reports.research.records import numeric_references
import unittest

from episodic_workflow.case_reports.research.acquisition import PageSource, SearchSource, StrategySource
from episodic_workflow.case_reports.research.admission import extract_and_admit
from episodic_workflow.case_reports.research.engine import ResearchLimits, ResearchProviders, ResearchRun, ResearchSource
from episodic_workflow.case_reports.research.model_defaults import initialize_model_defaults
from episodic_workflow.case_reports.research.records import Fact, Ledger, ResearchResult
from episodic_workflow.case_reports.research.relevance import assess_numeric_target_relationships
from episodic_workflow.case_reports.research.scenario_sharing import initialize_parameter_scope
from episodic_workflow.case_reports.research.scenarios import initial_scenarios, load_scenarios
from episodic_workflow.method_loop import Context, EpisodeGoal, EpisodeTree
from episodic_workflow.task_executors.scenario_metrics.evaluate import DEFAULT_METRICS, UNITS


def initial_state():
    initial = initialize_model_defaults(initial_scenarios(load_scenarios()[:2]))
    return initialize_parameter_scope(initial, lambda request: {"features": [],
        "dependencies": [{"parameter": field, "factors": [], "reason": "Equivalent test targets"}
                         for field in request["parameters"]]})


def source_fact(field, value, context="Source activity"):
    return Fact("one-source-event", field, value, UNITS[field], "source_model_parameter",
                "Native model input", source="https://example.test/native", quote=f"{field}={value}", context=context)


class SourceValueHandoffTests(unittest.TestCase):
    def test_commit_fills_other_fields_and_replaces_best_value_before_parent_return(self):
        initial = initial_state()
        subjects = tuple(row["subject"] for row in initial.scenarios)
        field = "quanta_concentration_exhaled_m3"
        first = source_fact(field, 970, "Singing")
        better = source_fact(field, 29, "Quiet speech")
        flow = source_fact("breathing_flow_m3_s", .00015)
        area = source_fact("mouth_area_m2", .0003)
        grading_requests, extraction_requests, observed, trace = [], [], [], []

        def grade(request):
            grading_requests.append(request)
            self.assertEqual(len(request["source_profile"]["facts"]), 1)
            fact = request["source_profile"]["facts"][0]
            self.assertEqual(request["parameter"], fact["field"])
            return {subject: (.9 if fact["value"] == 29 else .2)
                    for subject in request["target_scenarios"]}

        run = ResearchRun(ResearchProviders(lambda _: {"findings": "Target values updated."},
            lambda *_: (), lambda _: None, source_target_assessor=grade, trace_event=trace.append),
            ResearchLimits(), "Test immediate target handoff", DEFAULT_METRICS, initial=initial)
        parent = ResearchSource(run, initial)
        root = run.episode("evaluate_airborne_safety_guideline", parent,
            goal=EpisodeGoal.root(objective="Target progress", result_contract={}), focus=())
        batches = iter(((first, flow, area), (better,)))
        child = None
        parent_prompts = []
        invocations = 0

        def read_batch(batch):
            def extractor(request):
                extraction_requests.append(request)
                if request["operation"] == "extract_research_evidence":
                    return {"facts": [{key: fact.as_record()[key] for key in
                        ("subject", "field", "value", "unit", "kind", "basis", "quote", "context")}
                        for fact in batch], "evidence": [], "findings": []}
                return {"decisions": [{"option": row["option"], "admit": True,
                                       "reason": "Faithful native source value"}
                                      for row in request["candidates"]]}
            return extract_and_admit(extractor, text="\n".join(fact.quote for fact in batch),
                source=first.source, source_version="native-document", objective="Investigate Cq",
                question="Quiet speech applicability", context=child.context())

        def parent_propose(view):
            nonlocal child, invocations
            if invocations:
                parent_prompts.append(parent.inputs(view))
            batch = next(batches, None)
            if batch is None:
                return None
            invocations += 1
            child = ResearchSource(run, parent.ledger.result())
            original_after = child.after_verdict
            child_view = None

            def child_propose(view):
                nonlocal child_view
                self.assertEqual(view.units_consumed, 0, "Useful target progress must return before another proposal")
                child_view = view
                return child.leaf(lambda: read_batch(batch), "native_values")

            def after_verdict(unit, contribution, view):
                original_after(unit, contribution, view)
                controller = next(controller for path, controller in run.live_controllers.items() if len(path) == 1)
                own = next(controller for path, controller in run.live_controllers.items()
                           if path[-1][1] == child.controller_key)
                observed.append({"score": controller.state().score, "history": controller.state().history,
                    "prompt": child.inputs(replace(child_view, controller_state=own.state())),
                    "numeric": numeric_references(child.ledger)})

            child.propose, child.after_verdict = child_propose, after_verdict
            return parent.edge(run.child(view, child, "investigate", f"Cq investigation {invocations}",
                contract=parent.contract("Investigate Cq", parameter=field, subjects=subjects,
                                         deliverable="assumption_hypotheses"), proposal={}))

        parent.propose = parent_propose
        record = root.run(Context(tree=run.tree))
        first_status = observed[0]["prompt"]["global_scientific_goal"]
        second_status = observed[1]["prompt"]["global_scientific_goal"]
        self.assertEqual(first_status["recorded_target_inputs"]["recorded"], 3)
        self.assertGreater(observed[0]["score"], 0)
        self.assertEqual(observed[0]["score"], first_status["weighted_score"])
        self.assertEqual(observed[0]["history"], (0,))
        self.assertEqual(observed[1]["history"], (0, observed[0]["score"]))
        self.assertEqual(first_status["target_inputs"][subjects[0]][field]["value"], 970)
        self.assertEqual(second_status["target_inputs"][subjects[0]][field]["value"], 29)
        self.assertTrue(all(relation["source_contexts"] == ["Quiet speech"] for relation in
            second_status["target_inputs"][subjects[0]][field]["source_applicability"]["relationships"]))
        self.assertEqual(second_status["recorded_target_inputs"], first_status["recorded_target_inputs"])
        self.assertEqual(observed[1]["score"], observed[0]["score"])
        self.assertEqual(len(grading_requests), 4, "Each new native value is graded once; returns reuse it")
        self.assertEqual([request["operation"] for request in extraction_requests],
                         ["extract_research_evidence", "review_research_evidence"] * 2)
        self.assertEqual({request["parameter"] for request in grading_requests},
                         {field, flow.field, area.field})
        for request in grading_requests:
            self.assertEqual(set(request["target_scenarios"]), set(subjects))
        self.assertTrue(any(event["event"] == "shared_credit_refreshed" for event in trace))
        self.assertEqual(record.controller_state.score, observed[1]["score"])
        self.assertEqual(parent_prompts[-1]["global_scientific_goal"]["target_inputs"],
                         second_status["target_inputs"])
        self.assertEqual(record.units_consumed, 2)
        self.assertEqual([unit.child.units_consumed for unit in record.unit_records], [1, 1])
        self.assertIn("same global scientific score", observed[0]["prompt"]["numeric_yield"]["score_meaning"])
        selected = [fact for fact in observed[1]["numeric"].values()
                    if fact.subject == subjects[0] and fact.field == field]
        self.assertTrue(any(fact.value == 29 for fact in selected))

    def test_missing_or_failed_grade_does_not_prevent_target_progress(self):
        for assessor in (None, lambda _: (_ for _ in ()).throw(ValueError("Unavailable grade"))):
            with self.subTest(assessor=assessor):
                initial = initial_state()
                run = ResearchRun(ResearchProviders(lambda _: {}, lambda *_: (), lambda _: None,
                    source_target_assessor=assessor), ResearchLimits(), "Ungraded native value", DEFAULT_METRICS,
                    initial=initial)
                source = ResearchSource(run, initial)
                source.scientific = True
                source.goal = EpisodeGoal.root(objective="Read native value", result_contract={})
                credit = source.project(ResearchResult(facts=(source_fact("breathing_flow_m3_s", .00015),)))
                status = run.global_goal.preview(source.pending_global)
                self.assertEqual(status["recorded_target_inputs"]["recorded"], 1)
                self.assertGreater(sum(row[2] for row in credit.qualities), 0)
                candidate = next(a for a in source.pending_global.assumptions if a.get("entry_type") == "source_candidate")
                self.assertIsNone(candidate["source_applicability"]["score"])

    def test_value_grades_reused_and_zero_kept_without_threshold(self):
        ledger = Ledger(initial_state())
        fact = source_fact("breathing_flow_m3_s", .00015)
        ledger.merge(ResearchResult(facts=(fact,)))
        requests = []

        def grade(request):
            requests.append(request)
            return {subject: 0.0 for subject in request["target_scenarios"]}

        relations = assess_numeric_target_relationships(ledger, grade)
        ledger.merge(relations)
        self.assertEqual(len(requests), 1)
        self.assertTrue(all(row["source_value_scores"][fact.key] == 0 for row in relations.source_target_relationships))
        self.assertFalse(assess_numeric_target_relationships(ledger, grade).source_target_relationships)
        self.assertEqual(len(requests), 1)
        subject = next(iter(ledger.scenarios))
        ledger.scenarios[subject] = {**ledger.scenarios[subject], "description": "A changed target activity"}
        ledger.merge(assess_numeric_target_relationships(ledger, grade))
        self.assertEqual(len(requests), 2)
        self.assertEqual(set(requests[-1]["target_scenarios"]), {subject})

    def test_partial_grade_failure_preserves_successful_grades_and_all_found_values(self):
        initial = initial_state()
        facts = (source_fact("breathing_flow_m3_s", .00015), source_fact("mouth_area_m2", .0003))

        def grade(request):
            if request["parameter"] == "mouth_area_m2":
                raise ValueError("Second field grading unavailable")
            return {subject: .8 for subject in request["target_scenarios"]}

        run = ResearchRun(ResearchProviders(lambda _: {}, lambda *_: (), lambda _: None,
            source_target_assessor=grade), ResearchLimits(), "Partial grading", DEFAULT_METRICS, initial=initial)
        source = ResearchSource(run, initial)
        source.scientific = True
        source.goal = EpisodeGoal.root(objective="Found native values", result_contract={})
        source.project(ResearchResult(facts=facts))
        status = run.global_goal.preview(source.pending_global)
        self.assertEqual(status["recorded_target_inputs"]["recorded"], 2)
        rows = {a["field"]: a for a in source.pending_global.assumptions if a.get("entry_type") == "source_candidate"}
        self.assertEqual(rows["breathing_flow_m3_s"]["source_applicability"]["score"], .8)
        self.assertIsNone(rows["mouth_area_m2"]["source_applicability"]["score"])
        diagnostic = next(row for row in source.pending_global.diagnostics if row["stage"] == "source_value_relevance")
        self.assertEqual(diagnostic["field"], "mouth_area_m2")

    def test_ready_acquisition_handoff_ends_normally_without_a_model_call(self):
        def unexpected(_):
            self.fail("Ready handoff must not request another model decision")

        run = ResearchRun(ResearchProviders(unexpected, lambda *_: (), lambda _: None),
                          ResearchLimits(), "Ready evidence handoff", DEFAULT_METRICS)
        sources = (StrategySource(run, (), approach="Source evidence"),
                   SearchSource(run, (), query="A prepared query"),
                   PageSource(run, (), candidate={"url": "https://example.test/ready"}))
        for source in sources:
            with self.subTest(source=type(source).__name__):
                source.evidence_handoff_baseline = 0
                source.information_assessment = [{"points": 1}]
                source.evaluated_units = 1
                self.assertTrue(source.evidence_handoff_ready())
                self.assertIsNone(source.propose(None))
        self.assertEqual(run.model_calls, 0)

    def test_calibration_observations_classify_and_trigger_real_local_handoff(self):
        for grain, kind in (("strategy", StrategySource), ("page", PageSource)):
            with self.subTest(grain=grain):
                initial = initial_state()
                classified, graded = [], []

                def agent(request):
                    self.assertEqual(request["operation"], "classify_source_event_types")
                    classified.append(request)
                    return {"groups": [{"type_id": "new", "description": "Observed outbreak",
                        "subjects": list(request["source_subjects"]), "basis": "Native observed event"}]}

                def grade(request):
                    graded.append(request)
                    return {subject: .5 for subject in request["target_scenarios"]}

                run = ResearchRun(ResearchProviders(agent, lambda *_: (), lambda _: None,
                    source_target_assessor=grade), ResearchLimits(), "Calibration source findings",
                    DEFAULT_METRICS, initial=initial)
                source = (kind(run, initial, approach="Observed outbreak") if grain == "strategy"
                          else kind(run, initial, candidate={"url": "https://example.test/native"}))
                contract = source.contract("Find outbreak observations", parameter="quanta_concentration_exhaled_m3",
                    deliverable="source_findings", target_scenarios={row["subject"]: row for row in initial.scenarios})
                episode = run.episode(grain, source,
                    goal=EpisodeGoal.root(objective="Find outbreak observations", result_contract={}),
                    focus=(), contract=contract)
                facts = tuple(replace(source_fact(field, value, "Observed singing outbreak"), kind="source_observation")
                    for field, value in (("index_case_count", 1), ("susceptible_count", 60),
                                         ("secondary_case_count", 52), ("exposure_time_s", 9000)))
                original_propose, first = source.propose, True

                def propose(view):
                    nonlocal first
                    if first:
                        first = False
                        self.assertFalse(source.evidence_handoff_ready())
                        return source.leaf(lambda: ResearchResult(facts=facts), "observed_outbreak")
                    return original_propose(view)

                source.propose = propose
                record = episode.run(Context(tree=EpisodeTree(run.grains[grain], {})))
                self.assertEqual(record.units_consumed, 1)
                self.assertEqual(record.ended_by, "exhausted")
                self.assertTrue(source.evidence_handoff_ready())
                self.assertGreater(sum(row["points"] for row in source.information_assessment), 0)
                self.assertTrue(source.ledger.source_event_types)
                self.assertEqual(len(classified), 1)
                self.assertEqual(len(graded), 1)
                self.assertFalse(any(row.get("status") == "failed"
                                     for row in source.ledger.diagnostics))
                self.assertEqual(run.global_goal.preview()["recorded_target_inputs"]["recorded"], 0)

    def test_local_classification_preserves_central_value_grades_and_other_fields(self):
        initial = initial_state()
        requests = []

        def grade(request):
            requests.append(request)
            return {subject: .7 for subject in request["target_scenarios"]}

        def agent(request):
            self.assertEqual(request["operation"], "classify_source_event_types")
            return {"groups": [{"type_id": "new", "description": "Source population",
                "subjects": list(request["source_subjects"]), "basis": "Native source identity"}]}

        run = ResearchRun(ResearchProviders(agent, lambda *_: (), lambda _: None,
            source_target_assessor=grade), ResearchLimits(), "Preserve value grades", DEFAULT_METRICS,
            initial=initial)
        facts = (source_fact("quanta_concentration_exhaled_m3", 29), source_fact("breathing_flow_m3_s", .00015))
        run.global_goal.ledger.merge(ResearchResult(facts=facts))
        run.global_goal.ledger.merge(assess_numeric_target_relationships(run.global_goal.ledger, grade))
        before = dict(run.global_goal.ledger.source_target_relationships)
        source = StrategySource(run, initial, approach="Source event classification")
        source.parent_contract = source.contract("Find source evidence", parameter="quanta_concentration_exhaled_m3",
            deliverable="source_findings", target_scenarios={row["subject"]: row for row in initial.scenarios})
        source.parameter = "quanta_concentration_exhaled_m3"
        from episodic_workflow.case_reports.research.engine import research_parameter_requirements
        source.parameter_requirements = research_parameter_requirements(source.parameter, source.ledger.scenarios)
        source.goal = EpisodeGoal.root(objective="Classify source", result_contract={})
        classified = source.assess(ResearchResult(facts=facts))
        after = {(row["source"], row["source_subject"], row["target_subject"], row["parameter"]): row
                 for row in classified.source_target_relationships}
        self.assertEqual(after, before)
        self.assertTrue(classified.source_event_types)
        self.assertEqual(len(requests), 2, "Classification must reuse both field-specific grades")

    def test_retained_default_context_keeps_profile_grades_and_local_handoff_consistent(self):
        initial = initial_state()
        requests = []

        def grade(request):
            requests.append(request)
            return {subject: .8 for subject in request["target_scenarios"]}

        def agent(request):
            self.assertEqual(request["operation"], "classify_source_event_types")
            return {"groups": [{"type_id": "new", "description": "Source with physical context",
                "subjects": list(request["source_subjects"]), "basis": "Native source event"}]}

        run = ResearchRun(ResearchProviders(agent, lambda *_: (), lambda _: None,
            source_target_assessor=grade), ResearchLimits(), "Canonical source profile",
            DEFAULT_METRICS, initial=initial)
        parameter = "quanta_concentration_exhaled_m3"
        facts = (source_fact(parameter, 29), source_fact("air_density_kg_m3", 1.204))
        run.global_goal.ledger.merge(ResearchResult(facts=facts))
        run.global_goal.ledger.merge(assess_numeric_target_relationships(run.global_goal.ledger, grade))
        source = StrategySource(run, initial, approach="Examine source context")
        contract = source.contract("Find infectiousness", parameter=parameter,
            deliverable="source_findings", target_scenarios={row["subject"]: row for row in initial.scenarios})
        run.episode("strategy", source,
            goal=EpisodeGoal.root(objective="Find infectiousness", result_contract={}), focus=(), contract=contract)
        self.assertNotIn("air_density_kg_m3", source.parameter_requirements)
        self.assertFalse(source.evidence_handoff_ready())
        source.project(ResearchResult(facts=facts))
        self.assertEqual(len(requests), 1, "Canonical profiles reuse the existing value-specific grade")
        self.assertGreater(sum(row["points"] for row in source.information_assessment), 0)
        scored_fields = {row["field"] for group in source.information_assessment for row in group["fields"]}
        self.assertEqual(scored_fields, {parameter})
        source.evaluated_units = 1
        self.assertTrue(source.evidence_handoff_ready())
        status = run.global_goal.preview(source.pending_global)
        self.assertEqual(status["recorded_target_inputs"]["recorded"], 1)
        found = next(row for row in status["independent_target_values"] if row["parameter"] == parameter)
        self.assertEqual(found["value"], 29)
        self.assertEqual(found["source_applicability"]["score"], .8)
        self.assertTrue(all(row["relevance"] == .8 for row in source.information_assessment))


if __name__ == "__main__":
    unittest.main()
