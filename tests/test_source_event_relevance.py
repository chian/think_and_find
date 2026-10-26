"""Unit checks for source-event grouping and relevance-weighted credit."""

import unittest
from types import MappingProxyType, SimpleNamespace

from episodic_workflow.case_reports.research.acquisition import (
    SOURCE_ASSESSMENT_BATCH_SIZE, PageSource, StrategySource,
)
from episodic_workflow.case_reports.research.engine import ResearchLimits, ResearchProviders, ResearchRun
from episodic_workflow.case_reports.research.goal_state import GLOBAL_GOAL_PROGRESS, ScientificGoalState
from episodic_workflow.case_reports.research.progress import parameter_requirements
from episodic_workflow.case_reports.research.records import Fact, Ledger, ResearchResult, identity
from episodic_workflow.case_reports.research.relevance import (
    assess_source_types, assess_source_types_with_jev, source_profiles,
    target_comparison_context, target_relevance_credit,
)
from episodic_workflow.case_reports.research.scenarios import initial_scenarios, load_scenarios
from episodic_workflow.case_reports.research.workflow import GuidelineSource
from episodic_workflow.method_loop import EpisodeGoal
from episodic_workflow.task_executors.scenario_metrics.evaluate import DEFAULT_METRICS


def fact(source, subject, field, value):
    units = {"attack_probability": "fraction", "exposure_time_s": "s",
             "room_volume_m3": "m3"}
    return Fact(subject, field, value, units[field], "source_observation", "Reported event value",
                source, "Exact reported value", f"Context of {subject}")


def assessed_response(request):
    subjects = list(request["source_subjects"])
    targets = list(request["target_scenarios"])
    return {"groups": [{"type_id": "new", "description": "Group singing",
                         "subjects": subjects,
                         "target_relevance": {target: 0.25 for target in targets},
                         "basis": "Vocal activity differs from the target"}],
            "relationships": [{"source_subject": subject, "target_subject": target,
                "score": 0.25, "basis": "Vocal activity differs from quiet activity",
                "comparisons": [{"field": field, "relation": "partial", "basis": "Qualified context comparison"}
                                for field in request["comparison_variables"][subject][target]]}
                for subject in subjects for target in targets]}


class SourceEventRelevanceTests(unittest.TestCase):
    def setUp(self):
        self.choir = "https://example.test/choir"
        self.classroom = "https://example.test/classroom"
        self.target = {"quiet-room": {"description": "Quiet room, one infectious occupant"}}
        self.result = ResearchResult(facts=(
            fact(self.choir, "choir-event", "attack_probability", 0.6),
            fact(self.choir, "choir-event", "exposure_time_s", 7200),
            fact(self.classroom, "classroom-event", "room_volume_m3", 100),
        ))

    def test_target_context_copies_closed_nested_mappings_for_assessment(self):
        definition = MappingProxyType({"subject": "quiet-room",
            "description": "Quiet room",
            "fixed_parameters": MappingProxyType({"room_volume_m3":
                MappingProxyType({"value": 100, "unit": "m3"})}),
            "fixed_conditions": MappingProxyType({"well_mixed": "yes"}),
            "assumption_needs": ("attack_probability",)})
        context = target_comparison_context(definition, ())
        self.assertEqual(context["fixed_parameters"]["room_volume_m3"]["value"], 100)
        self.assertEqual(context["assumption_needs"], ["attack_probability"])
        identity(context)

    def test_jev_assessment_uses_only_selected_scenario_targets(self):
        targets = {"caller-case-a": {"description": "Quiet room"},
                   "caller-case-b": {"description": "Busy room"}}
        profiles = source_profiles(self.result, {"attack_probability", "exposure_time_s"})
        prompts = []
        def classify(request):
            prompts.append(request)
            return {"groups": [{"type_id": "new", "description": "Choir event",
                "subjects": ["choir-event"], "basis": "One observed event"}]}
        jev_calls = []
        def jev_assess(item):
            jev_calls.append(item)
            return {"caller-case-a": 0.8, "caller-case-b": 0.2}
        groups, relations = assess_source_types_with_jev(
            classify, jev_assess, self.choir, profiles, targets, {}, parameter="attack_probability")
        self.assertEqual(prompts[0]["operation"], "classify_source_event_types")
        self.assertNotIn("target_scenarios", prompts[0])
        self.assertEqual(set(jev_calls[0]["target_scenarios"]), set(targets))
        self.assertEqual({row["target_subject"]: row["score"] for row in relations},
                         {"caller-case-a": 0.8, "caller-case-b": 0.2})
        self.assertTrue(all(row["assessment_method"] == "jev_noul" for row in relations))
        self.assertEqual(groups[0]["target_relevance"],
                         {"caller-case-a": 0.8, "caller-case-b": 0.2})

    def test_source_profiles_preserve_event_identity(self):
        profiles = source_profiles(self.result, {"attack_probability", "exposure_time_s", "room_volume_m3"})
        self.assertEqual(set(profiles), {(self.choir, "choir-event"),
                                         (self.classroom, "classroom-event")})

    def test_qualitative_observation_joins_type_context_without_numeric_credit(self):
        observation = {"kind": "source_evidence", "evidence_role": "observation",
            "source": self.choir, "subject": "choir-seating", "context": "Group singing indoors",
            "fields": ["attack_probability"], "quote": "Singers sat close together."}
        result = ResearchResult(evidence=(observation,))
        profiles = source_profiles(result, {"attack_probability"})
        self.assertIn((self.choir, "choir-seating"), profiles)
        group = {"description": "Group singing", "basis": "Observed activity",
                 "members": [[self.choir, "choir-seating"]],
                 "target_relevance": {"quiet-room": 0.25}}
        credit = target_relevance_credit(result, {"attack_probability"},
                                         {"singing": group}, self.target)[1]
        self.assertEqual(credit.qualities[0][2], 0)

    def test_credit_uses_best_type_not_cross_type_field_pool(self):
        groups = {
            "singing": {"description": "Group singing", "basis": "Activity differs",
                        "members": [[self.choir, "choir-event"]],
                        "target_relevance": {"quiet-room": 0.25}},
            "quiet": {"description": "Quiet classroom", "basis": "Activity matches",
                      "members": [[self.classroom, "classroom-event"]],
                      "target_relevance": {"quiet-room": 1}},
        }
        rows, credit = target_relevance_credit(self.result,
            {"attack_probability", "exposure_time_s", "room_volume_m3"}, groups, self.target)
        self.assertEqual(rows[0]["type_id"], "quiet")
        self.assertEqual(credit.qualities[0][2], 1)
        self.assertEqual([row["field"] for row in rows[0]["fields"]], ["room_volume_m3"])
        groups["singing"]["target_relevance"]["quiet-room"] = 0.75
        rows, improved = target_relevance_credit(self.result,
            {"attack_probability", "exposure_time_s", "room_volume_m3"}, groups, self.target)
        self.assertEqual(rows[0]["type_id"], "singing")
        self.assertEqual(improved.qualities[0][2], 1.5)

    def test_related_events_share_type_without_merging_observations(self):
        second = "https://example.test/related"
        result = ResearchResult(facts=(*self.result.facts,
            fact(second, "second-quiet-event", "attack_probability", 0.2)))
        groups = {"quiet": {"description": "Quiet indoor contact", "basis": "Related event types",
                            "members": [[self.classroom, "classroom-event"],
                                        [second, "second-quiet-event"]],
                            "target_relevance": {"quiet-room": 1}}}
        rows, credit = target_relevance_credit(result,
            {"attack_probability", "room_volume_m3"}, groups, self.target)
        self.assertEqual(credit.qualities[0][2], 2)
        self.assertEqual(rows[0]["member_count"], 2)
        self.assertEqual({fact.subject for fact in result.facts if fact.source in {self.classroom, second}},
                         {"classroom-event", "second-quiet-event"})

    def test_local_group_credit_uses_each_event_relationship(self):
        first = fact(self.choir, "choir-event", "attack_probability", 0.6)
        second = fact(self.classroom, "classroom-event", "attack_probability", 0.2)
        facts = (first, second)
        profiles = source_profiles(ResearchResult(facts=facts), {"attack_probability"})
        target_version = identity(target_comparison_context(
            {**self.target["quiet-room"], "subject": "quiet-room"}, ()))
        relationships = tuple({"source": item.source, "source_subject": item.subject,
            "target_subject": "quiet-room", "parameter": "attack_probability",
            "target_version": target_version,
            "source_profile_version": identity(profiles[(item.source, item.subject)]),
            "score": score, "known": 4, "considered": 4}
            for item, score in ((first, 0.25), (second, 1)))
        result = ResearchResult(facts=facts, source_target_relationships=relationships)
        groups = {"indoor": {"description": "Indoor exposure", "basis": "Shared event type",
            "members": [[first.source, first.subject], [second.source, second.subject]],
            "target_relevance": {"quiet-room": 1}}}
        rows, credit = target_relevance_credit(result, {"attack_probability"}, groups,
                                               self.target, parameter="attack_probability")
        self.assertEqual(rows[0]["points"], 1)
        self.assertEqual(credit.qualities[0][2], 1)

    def test_assessor_must_partition_source_subjects(self):
        profiles = source_profiles(self.result, {"attack_probability", "exposure_time_s", "room_volume_m3"})
        self.target["quiet-room"]["fixed_parameters"] = {"exposure_time_s": {"value": 3600, "unit": "s"}}
        seen = []
        def assess(request):
            seen.append(request)
            return assessed_response(request)
        groups, relationships = assess_source_types(assess, self.choir, profiles, self.target, {}, "Generic disease")
        self.assertEqual(groups[0]["subjects"], ["choir-event"])
        self.assertEqual(relationships[0]["source_subject"], "choir-event")
        duration = next(item for item in relationships[0]["comparisons"] if item["field"] == "exposure_time_s")
        self.assertEqual(duration["source_values"][0]["value"], 7200)
        self.assertEqual(duration["target_value"]["value"], 3600)
        self.assertEqual(seen[0]["disease_of_interest"], "Generic disease")
        with self.assertRaisesRegex(ValueError, "partition"):
            assess_source_types(lambda _: {"groups": []}, self.choir, profiles, self.target, {})

    def test_grouping_survives_ledger_handoff_and_invalid_evidence_loses_credit(self):
        group = {"type_id": "group:1", "description": "Group singing", "basis": "Source context",
                 "members": [[self.choir, "choir-event"]],
                 "target_relevance": {"quiet-room": 0.5}}
        ledger = Ledger(ResearchResult(facts=self.result.facts,
                                       source_event_types=(group,)))
        self.assertEqual(ledger.result().source_event_types, (group,))
        before = target_relevance_credit(ledger.result(),
            {"attack_probability", "exposure_time_s"}, {"group:1": group}, self.target)[1]
        invalid = self.result.facts[0].semantic_record()
        ledger.merge(ResearchResult(assumptions=({"key": "review:1", "subject": "quiet-room",
            "field": "attack_probability", "value": 0.6, "unit": "fraction",
            "learning_evidence": [], "invalidated_evidence": [invalid]},)))
        after = target_relevance_credit(ledger.result(),
            {"attack_probability", "exposure_time_s"}, {"group:1": group}, self.target)[1]
        self.assertEqual(before.qualities[0][2], 1)
        self.assertEqual(after.qualities[0][2], 0.5)

    def test_strategy_assesses_relevance_before_projecting_credit(self):
        requests = []
        def complete(request):
            requests.append(request)
            return assessed_response(request)
        run = SimpleNamespace(complete=complete, context={"disease_of_interest": "Generic disease"},
                              global_goal=ScientificGoalState(()))
        source = StrategySource(run, (), approach="Find related events")
        source.goal = SimpleNamespace(goal_id="strategy:1")
        source.parent_contract = SimpleNamespace(measurements={
            "target_scenarios": self.target, "deliverable": "source_findings"})
        source.parameter_requirements = frozenset({"attack_probability", "exposure_time_s"})
        discovered = ResearchResult(facts=self.result.facts[:2])
        assessed = source.assess(discovered)
        credit = source.credit(assessed)
        self.assertEqual(len(requests), 1)
        self.assertEqual(credit.qualities[0][2], 0.5)
        self.assertEqual(assessed.source_event_types[0]["members"], [[self.choir, "choir-event"]])

    def test_strategy_keeps_completed_group_when_later_review_fails(self):
        def complete(request):
            if request["source"] == self.classroom:
                raise ValueError("Reviewer unavailable")
            return assessed_response(request)
        run = SimpleNamespace(complete=complete, context={"disease_of_interest": "Generic disease"},
                              global_goal=ScientificGoalState(()),
                              model_diagnostic=lambda error: {"category": "provider_failure", "why": str(error)})
        source = StrategySource(run, (), approach="Find related events")
        source.goal = SimpleNamespace(goal_id="strategy:2", objective="Find event evidence")
        source.parent_contract = SimpleNamespace(measurements={
            "target_scenarios": self.target, "deliverable": "source_findings"},
            decision="Find event evidence")
        source.parameter_requirements = frozenset({"attack_probability", "exposure_time_s", "room_volume_m3"})
        assessed = source.assess(self.result)
        self.assertTrue(assessed.evaluated)
        self.assertEqual(len(assessed.source_event_types), 1)
        self.assertEqual(assessed.source_event_types[0]["members"], [[self.choir, "choir-event"]])
        self.assertEqual(source.credit(assessed).qualities[0][2], 0.5)
        self.assertIn("source_event_relevance failed", assessed.issues[0])

    def test_page_assesses_relationship_before_search_or_strategy_returns(self):
        calls = []
        def complete(request):
            calls.append(request)
            return assessed_response(request)
        run = SimpleNamespace(complete=complete,
                              context={"disease_of_interest": "Generic disease"},
                              global_goal=ScientificGoalState(()))
        source = PageSource(run, (), candidate={"url": self.choir})
        source.goal = SimpleNamespace(goal_id="page:1")
        source.parent_contract = SimpleNamespace(measurements={
            "target_scenarios": self.target, "deliverable": "source_findings"})
        source.parameter_requirements = frozenset({"attack_probability", "exposure_time_s"})
        source.parameter = "quanta_concentration_exhaled_m3"
        assessed = source.assess(ResearchResult(facts=self.result.facts[:2]))
        self.assertEqual(len(assessed.source_target_relationships), 1)
        self.assertEqual(assessed.source_target_relationships[0]["source"], self.choir)
        strategy = StrategySource(run, (), approach="Use admitted page findings")
        strategy.goal = SimpleNamespace(goal_id="strategy:later")
        strategy.parent_contract = source.parent_contract
        strategy.parameter_requirements = source.parameter_requirements
        strategy.parameter = source.parameter
        strategy.assess(assessed)
        self.assertEqual(len(calls), 1)

    def test_source_assessment_batches_many_subjects_against_all_targets(self):
        target = {f"target-{number}": {"description": "Quiet room"}
                  for number in range(8)}
        facts = tuple(fact(self.choir, f"event-{number}", "attack_probability", 0.1)
                      for number in range(7))
        calls = []
        def complete(request):
            calls.append(request)
            return assessed_response(request)
        run = SimpleNamespace(complete=complete, context={},
                              global_goal=ScientificGoalState(()))
        source = StrategySource(run, (), approach="Find events")
        source.goal = SimpleNamespace(goal_id="strategy:batch")
        source.parent_contract = SimpleNamespace(measurements={
            "target_scenarios": target, "deliverable": "source_findings"})
        source.parameter_requirements = frozenset({"attack_probability"})
        source.parameter = "attack_probability"
        assessed = source.assess(ResearchResult(facts=facts))
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(len(call["source_subjects"]) <= SOURCE_ASSESSMENT_BATCH_SIZE
                            for call in calls))
        self.assertTrue(all(len(call["target_scenarios"]) == 8 for call in calls))
        self.assertEqual(len(assessed.source_target_relationships), 56)
        self.assertEqual(sum(len(group["members"]) for group in assessed.source_event_types), 7)

    def test_changed_target_value_prompts_page_relationship_reassessment(self):
        source_fact = self.result.facts[0]
        parameter = "attack_probability"
        profile = source_profiles(ResearchResult(facts=(source_fact,)),
                                  {parameter})[(self.choir, "choir-event")]
        old_context = target_comparison_context({**self.target["quiet-room"], "subject": "quiet-room"}, ())
        old = {"source": self.choir, "source_subject": "choir-event", "target_subject": "quiet-room",
               "parameter": parameter, "target_version": identity(old_context),
               "source_profile_version": identity(profile), "score": 0.25,
               "known": 4, "considered": 4}
        group = {"type_id": "type:1", "description": "Indoor event", "basis": "Comparable setting",
                 "members": [[self.choir, "choir-event"]], "target_relevance": {"quiet-room": 0.25}}
        initial = ResearchResult(facts=(source_fact,), source_target_relationships=(old,),
                                 source_event_types=(group,))
        state = ScientificGoalState((), initial)
        state.commit(ResearchResult(assumptions=({"key": "guess:1", "subject": "quiet-room",
            "field": parameter, "value": 0.3, "unit": "fraction", "statement": "Conditional guess",
            "role": "inferred_assumption", "review": {"verdict": "unassessed"}},)))
        calls = []
        def complete(request):
            calls.append(request)
            return assessed_response(request)
        run = SimpleNamespace(complete=complete, context={}, global_goal=state)
        page = PageSource(run, initial, candidate={"url": self.choir})
        page.goal = SimpleNamespace(goal_id="page:2")
        page.parent_contract = SimpleNamespace(measurements={
            "target_scenarios": self.target, "deliverable": "source_findings"})
        page.parameter_requirements = frozenset({parameter})
        page.parameter = parameter
        assessed = page.assess(ResearchResult())
        self.assertEqual(len(calls), 1)
        self.assertNotEqual(assessed.source_target_relationships[0]["target_version"], old["target_version"])

    def test_global_credit_uses_best_admitted_event_and_retracts_invalid_evidence(self):
        initial = initial_scenarios(load_scenarios("classroom-natural-unmasked"))
        state = ScientificGoalState(DEFAULT_METRICS, initial)
        target = initial.scenarios[0]
        parameter = "quanta_concentration_exhaled_m3"
        needs = len(target["assumption_needs"])

        def contribution(source, score):
            source_fact = Fact("source-classroom", parameter, 30, "quanta/m3",
                "source_model_parameter", "Reported fitted value", source,
                "Reported 30 quanta/m3", "Quiet classroom context")
            profile = source_profiles(ResearchResult(facts=(source_fact,)),
                                      parameter_requirements(parameter))[(source, "source-classroom")]
            context = target_comparison_context(target, initial.assumptions)
            relationship = {"source": source, "source_subject": "source-classroom",
                "target_subject": target["subject"], "parameter": parameter,
                "target_version": identity(context), "source_profile_version": identity(profile),
                "score": score, "basis": "Compared source and target activity",
                "source_role": ["source_model_parameter"], "comparisons": [],
                "known": 4, "considered": 4}
            return source_fact, relationship

        first_fact, first_relation = contribution("https://example.test/first", 1)
        first = state.commit(ResearchResult(facts=(first_fact,),
            source_target_relationships=(first_relation,)))
        self.assertAlmostEqual(first["scores"]["source_target_learning"], 1 / needs)
        self.assertEqual(first["source_target_learning"]["covered"], 1)
        second_fact, second_relation = contribution("https://example.test/second", 0.75)
        repeated = state.commit(ResearchResult(facts=(second_fact,),
            source_target_relationships=(second_relation,)))
        self.assertEqual(repeated["scores"]["source_target_learning"],
                         first["scores"]["source_target_learning"])
        self.assertEqual(repeated["weighted_score"], first["weighted_score"])
        state.commit(ResearchResult(assumptions=({"key": "invalid:1", "subject": "evidence-review",
            "field": parameter, "value": 30, "unit": "quanta/m3", "statement": "Invalid citation",
            "invalidated_evidence": [first_fact.semantic_record()]},)))
        corrected = state.preview()
        self.assertAlmostEqual(corrected["scores"]["source_target_learning"], 0.75 / needs)
        self.assertEqual(corrected["source_target_learning"]["by_target"][target["subject"]]
                         ["covered_parameters"][parameter]["source"], "https://example.test/second")

    def test_target_change_stales_old_relationship_until_reassessment(self):
        initial = initial_scenarios(load_scenarios("classroom-natural-unmasked"))
        state = ScientificGoalState(DEFAULT_METRICS, initial)
        target = initial.scenarios[0]
        parameter = "quanta_concentration_exhaled_m3"
        source = "https://example.test/source"
        source_fact = Fact("source-event", parameter, 30, "quanta/m3", "source_model_parameter",
            "Reported fitted value", source, "Reported 30 quanta/m3", "Quiet classroom context")
        profile = source_profiles(ResearchResult(facts=(source_fact,)),
                                  parameter_requirements(parameter))[(source, "source-event")]
        relationship = {"source": source, "source_subject": "source-event",
            "target_subject": target["subject"], "parameter": parameter,
            "target_version": identity(target_comparison_context(target, initial.assumptions)),
            "source_profile_version": identity(profile), "score": 1,
            "basis": "Directly comparable context", "source_role": ["source_model_parameter"],
            "comparisons": [], "known": 4, "considered": 4}
        state.commit(ResearchResult(facts=(source_fact,), source_target_relationships=(relationship,)))
        self.assertGreater(state.preview()["scores"]["source_target_learning"], 0)
        state.commit(ResearchResult(assumptions=({"key": "target:1", "subject": target["subject"],
            "field": parameter, "value": 45, "unit": "quanta/m3", "statement": "Provisional target value",
            "review": {"verdict": "provisional"}},)))
        self.assertEqual(state.preview()["scores"]["source_target_learning"], 0)
        self.assertEqual(state.preview()["recorded_target_inputs"]["recorded"], 1)
        self.assertEqual(state.preview()["target_inputs"][target["subject"]][parameter]["value"], 45)

    def test_root_projection_reads_child_committed_global_credit_without_root_unit(self):
        initial = initial_scenarios(load_scenarios("classroom-natural-unmasked"))
        trace = []
        run = ResearchRun(ResearchProviders(lambda _: {}, lambda *_: (), lambda _: None,
                                            trace_event=trace.append),
                          ResearchLimits(), "Synthetic score projection", DEFAULT_METRICS, initial=initial)
        root = GuidelineSource(run)
        run.episode("evaluate_airborne_safety_guideline", root,
                    goal=EpisodeGoal.root(objective="Synthetic goal", result_contract={}),
                    focus=(GLOBAL_GOAL_PROGRESS,))
        before = run.root_credit_projection()
        target = initial.scenarios[0]
        parameter = "quanta_concentration_exhaled_m3"
        source = "https://example.test/source"
        source_fact = Fact("source-event", parameter, 30, "quanta/m3", "source_model_parameter",
            "Reported fitted value", source, "Reported 30 quanta/m3", "Quiet classroom context")
        profile = source_profiles(ResearchResult(facts=(source_fact,)),
                                  parameter_requirements(parameter))[(source, "source-event")]
        relation = {"source": source, "source_subject": "source-event",
            "target_subject": target["subject"], "parameter": parameter,
            "target_version": identity(target_comparison_context(target, initial.assumptions)),
            "source_profile_version": identity(profile), "score": 1,
            "basis": "Comparable context", "source_role": ["source_model_parameter"],
            "comparisons": [], "known": 4, "considered": 4}
        page = PageSource(run, initial, candidate={"url": source})
        run.episode("page", page,
                    goal=EpisodeGoal.root(objective="Read source", result_contract={}),
                    focus=(), contract=root.contract("Assess source", subjects=[target["subject"]],
                        parameter=parameter, target_scenarios={target["subject"]: target}))
        page.pending = ResearchResult(facts=(source_fact,), source_target_relationships=(relation,))
        page.after_verdict(SimpleNamespace(label="read:1"), SimpleNamespace(episode_update=None),
                           SimpleNamespace(controller_step=SimpleNamespace(gain=0.1, score=0.1, stop=False)))
        self.assertAlmostEqual(run.root_credit_projection() - before,
                               0.1 / len(target["assumption_needs"]))
        self.assertNotIn(source_fact.key, root.ledger.facts)
        assessed = [event for event in trace if event["event"] == "source_target_relationship_assessed"]
        progress = [event for event in trace if event["event"] == "global_goal_progress"]
        self.assertEqual(len(assessed), 1)
        self.assertEqual(trace[0]["event"], "run_started")
        self.assertEqual(trace[0]["target_set"], [target["subject"]])
        self.assertEqual(assessed[0]["relationship"]["target_subject"], target["subject"])
        self.assertEqual(progress[-1]["source_target_learning"]["covered"], 1)
        self.assertGreater(progress[-1]["root_credit_projection"], before)


if __name__ == "__main__":
    unittest.main()
