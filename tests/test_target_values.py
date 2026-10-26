"""Source scalars become usable, graded target alternatives before another search."""

from dataclasses import replace
from episodic_workflow.case_reports.research.records import numeric_references
import unittest

from episodic_workflow.case_reports.research.goal_state import scientific_goal_status
from episodic_workflow.case_reports.research.model_defaults import initialize_model_defaults
from episodic_workflow.case_reports.research.progress import parameter_requirements
from episodic_workflow.case_reports.research.records import Fact, Ledger, ResearchResult, assumption_claim, identity
from episodic_workflow.case_reports.research.relevance import source_profiles, target_comparison_context
from episodic_workflow.case_reports.research.scenario_sharing import initialize_parameter_scope
from episodic_workflow.case_reports.research.scenarios import initial_scenarios, load_scenarios
from episodic_workflow.case_reports.research.target_values import project_source_target_values, target_value_preference
from episodic_workflow.case_reports.research.tools import execute_tool, support_for
from episodic_workflow.task_executors.scenario_metrics.evaluate import DEFAULT_METRICS, UNITS


class TargetValueTests(unittest.TestCase):
    def setUp(self):
        initial = initial_scenarios(load_scenarios()[:2])
        scenarios = tuple({**row, "subject": name} for row, name in
                          zip(initial.scenarios, ("target-alpha", "target-beta")))
        def defaults(request):
            return {"decisions": {subject: {field: {"decision": "retain",
                "reason": "Ordinary physical conditions.", "scenario_evidence": []} for field in fields}
                for subject, fields in request["default_candidates"].items()}}
        planned = initialize_model_defaults(ResearchResult(scenarios=scenarios), defaults)
        scoped = initialize_parameter_scope(planned, lambda request: {"features": [],
            "dependencies": [{"parameter": field, "factors": [],
                "reason": "The fixture stipulates equal values across these cases."}
                for field in request["parameters"]]})
        self.ledger = Ledger(scoped)

    def fact(self, field="breathing_flow_m3_s", value=.00015, source="https://example.org/one"):
        return Fact("native-source-population", field, value, UNITS[field], "source_model_parameter",
            "Reported native scalar.", source=source, quote=f"{field}: {value} {UNITS[field]}",
            context="The source's actual activity and population, retained verbatim.")

    def relation(self, fact, score, *, field=None):
        field = field or fact.field
        profile = source_profiles(self.ledger.result(), parameter_requirements(field))[(fact.source, fact.subject)]
        subject = "target-alpha"
        return {"source": fact.source, "source_subject": fact.subject, "target_subject": subject,
            "parameter": field, "score": score, "known": 1, "considered": 1,
            "source_profile_version": identity(profile),
            "target_version": identity(target_comparison_context(self.ledger.scenarios[subject], ())),
            "basis": "Fixture parameter-specific relevance.", "source_role": [fact.kind]}

    def project(self):
        delta = project_source_target_values(self.ledger)
        self.ledger.merge(delta)
        return delta

    def test_unknown_and_zero_relevance_create_shared_selectable_values_and_global_progress(self):
        first = self.fact()
        second = self.fact("deactivation_rate_s", .001)
        self.ledger.merge(ResearchResult(facts=(first, second)))
        self.ledger.merge(ResearchResult(source_target_relationships=(self.relation(first, 0),)))
        before = scientific_goal_status(self.ledger.result(), DEFAULT_METRICS)
        delta = self.project()
        self.assertEqual(len(delta.assumptions), 2)
        self.assertEqual({a["source_applicability"]["status"] for a in delta.assumptions}, {"unknown", "assessed"})
        self.assertTrue(all(a["reasonableness"] == 0 and not a["learning_evidence"] for a in delta.assumptions))
        after = scientific_goal_status(self.ledger.result(), DEFAULT_METRICS)
        self.assertEqual(after["recorded_target_inputs"]["recorded"] - before["recorded_target_inputs"]["recorded"], 2)
        self.assertGreater(after["weighted_score"], before["weighted_score"])
        for subject in self.ledger.scenarios:
            available = {f.field for f in numeric_references(self.ledger).values()
                         if f.subject == subject and f.kind == "model_assumption"}
            self.assertTrue({first.field, second.field} <= available)
        self.assertEqual(self.ledger.facts[first.key], first)
        self.assertFalse(project_source_target_values(self.ledger).assumptions)

    def test_better_alternative_revises_preference_without_another_found_group(self):
        first, second = self.fact(value=.00015), self.fact(value=.0002, source="https://example.org/two")
        self.ledger.merge(ResearchResult(facts=(first,)))
        self.ledger.merge(ResearchResult(source_target_relationships=(self.relation(first, .2),)))
        self.project()
        before = scientific_goal_status(self.ledger.result(), DEFAULT_METRICS)
        self.ledger.merge(ResearchResult(facts=(second,)))
        self.ledger.merge(ResearchResult(source_target_relationships=(self.relation(second, .8),)))
        self.project()
        alternatives = [a for a in self.ledger.assumptions.values() if a.get("entry_type") == "source_candidate"]
        self.assertEqual(len(alternatives), 2)
        self.assertEqual(max(alternatives, key=target_value_preference)["value"], second.value)
        after = scientific_goal_status(self.ledger.result(), DEFAULT_METRICS)
        self.assertEqual(after["recorded_target_inputs"], before["recorded_target_inputs"])
        self.assertEqual(after["weighted_score"], before["weighted_score"])

    def test_relevance_refresh_keeps_claim_identity_and_same_value_sources_coalesce(self):
        first = self.fact()
        self.ledger.merge(ResearchResult(facts=(first,)))
        original = self.project().assumptions[0]
        self.ledger.merge(ResearchResult(source_target_relationships=(self.relation(first, .7),)))
        refreshed = self.project().assumptions[0]
        self.assertEqual(original["key"], refreshed["key"])
        self.assertEqual(assumption_claim(original), assumption_claim(refreshed))
        self.assertNotEqual(original["version"], refreshed["version"])
        second = replace(first, source="https://example.org/corroboration")
        self.ledger.merge(ResearchResult(facts=(second,)))
        updated = self.project().assumptions[0]
        self.assertEqual(updated["key"], original["key"])
        self.assertEqual(set(updated["source_fact_keys"]), {first.key, second.key})
        self.assertEqual(updated["learning_evidence"], [])

    def test_wrong_parameter_and_stale_relationship_do_not_supply_grade(self):
        fact = self.fact()
        self.ledger.merge(ResearchResult(facts=(fact,)))
        wrong = self.relation(fact, .9, field="quanta_concentration_exhaled_m3")
        stale = {**self.relation(fact, 1), "target_version": "old-target"}
        self.ledger.merge(ResearchResult(source_target_relationships=(wrong, stale)))
        candidate = self.project().assumptions[0]
        self.assertEqual(candidate["source_applicability"]["status"], "unknown")
        self.assertIsNone(candidate["source_applicability"]["score"])

    def test_same_source_subject_values_use_their_own_relevance_not_profile_maximum(self):
        quiet = self.fact("quanta_concentration_exhaled_m3", 29)
        loud = replace(self.fact("quanta_concentration_exhaled_m3", 142),
                       context="The same source population during loud speech.")
        self.ledger.merge(ResearchResult(facts=(quiet, loud)))
        relation = {**self.relation(quiet, .9),
            "source_value_scores": {quiet.key: .9, loud.key: .2},
            "source_value_versions": {fact.key: identity(fact.semantic_record()) for fact in (quiet, loud)}}
        self.ledger.merge(ResearchResult(source_target_relationships=(relation,)))
        self.project()
        candidates = [a for a in self.ledger.assumptions.values() if a.get("entry_type") == "source_candidate"]
        self.assertEqual({a["value"]: a["source_applicability"]["score"] for a in candidates}, {29: .9, 142: .2})
        self.assertEqual(max(candidates, key=target_value_preference)["value"], 29)
        reversed_relation = {**relation, "source_value_scores": {quiet.key: .1, loud.key: .95}, "score": .95}
        self.ledger.merge(ResearchResult(source_target_relationships=(reversed_relation,)))
        self.project()
        candidates = [a for a in self.ledger.assumptions.values() if a.get("entry_type") == "source_candidate"]
        self.assertEqual(max(candidates, key=target_value_preference)["value"], 142)
        status = scientific_goal_status(self.ledger.result(), DEFAULT_METRICS)
        self.assertEqual(status["recorded_target_inputs"]["recorded"], 1)
        group = next(g for g in status["independent_target_values"] if g["parameter"] == quiet.field)
        self.assertEqual(group["value"], 142)

    def test_unknown_grade_never_outranks_assessed_zero_and_ties_keep_existing(self):
        unknown = {"source_applicability": {"score": None}, "reasonableness": 1}
        first = {"source_applicability": {"score": 0}, "reasonableness": 0, "key": "first"}
        second = {"source_applicability": {"score": 0}, "reasonableness": 1, "key": "second"}
        self.assertIs(max((unknown, first, second), key=target_value_preference), first)

    def test_constants_calculated_fields_and_different_quantities_do_not_fill_groups(self):
        facts = tuple(self.fact(field, value) for field, value in (
            ("gravity_m_s2", 9.8), ("air_density_kg_m3", 1.2),
            ("air_dynamic_viscosity_pa_s", 1.8e-5), ("particle_density_kg_m3", 997),
            ("quanta_emission_rate_s", .001), ("settling_speed_m_s", .001),
            ("removal_rate_s", .001), ("removal_capacity_m3_s", .01)))
        self.ledger.merge(ResearchResult(facts=facts))
        self.assertFalse(project_source_target_values(self.ledger).assumptions)
        self.assertEqual(scientific_goal_status(self.ledger.result(), DEFAULT_METRICS)["recorded_target_inputs"]["recorded"], 0)

    def test_invalidated_native_evidence_withdraws_automatic_alternative(self):
        fact = self.fact()
        self.ledger.merge(ResearchResult(facts=(fact,)))
        original = self.project().assumptions[0]
        invalidated = {**original, "invalidated_evidence": [fact.semantic_record()]}
        self.ledger.merge(ResearchResult(assumptions=(invalidated,)))
        withdrawn = self.project().assumptions[0]
        self.assertTrue(withdrawn["source_candidate_unavailable"])
        self.assertEqual(withdrawn["review"]["verdict"], "contradicted")
        self.assertEqual(scientific_goal_status(self.ledger.result(), DEFAULT_METRICS)["recorded_target_inputs"]["recorded"], 0)

    def test_calculation_uses_main_automatic_parent_even_with_stronger_same_value_claim(self):
        fact = self.fact("effective_radius_m", 1e-6)
        self.ledger.merge(ResearchResult(facts=(fact,)))
        automatic = self.project().assumptions[0]
        manual = []
        for subject in self.ledger.scenarios:
            candidate = {**automatic, "key": "manual:" + subject, "subject": subject,
                "entry_type": "best_guess", "statement": "A separately reviewed same-value hypothesis.",
                "reasonableness": 1, "review": {"completed": True, "verdict": "supported"}}
            candidate["version"] = identity(candidate)
            manual.append(candidate)
        self.ledger.merge(ResearchResult(assumptions=tuple(manual)))
        for subject in self.ledger.scenarios:
            options = numeric_references(self.ledger)
            reference, chosen = next((key, option) for key, option in options.items()
                if option.subject == subject and option.field == fact.field
                and option.parents and option.parents[0] == automatic["key"])
            support, support_complete, invalid = support_for(self.ledger, (chosen,), {}, subject)
            self.assertEqual(set(support), {automatic["key"]})
            self.assertEqual((support_complete, invalid), (True, []))
            result = execute_tool({"action": "calculate_geometry_and_removal", "subject": subject,
                "metrics": [],
                "conditions": {"stokes_particles": "Small spherical particles in the Stokes regime."},
                "basis": "Authoritative main-table input."},
                self.ledger, DEFAULT_METRICS, review={"admit": True, "completed": True, "reason": "Kernel execution checked."})
            self.assertTrue(any(f.field == "settling_speed_m_s" for f in result.facts))
            self.assertIn(automatic["key"], result.calculations[0]["assumption_versions"])
            self.assertNotIn("manual:" + subject, result.calculations[0]["assumption_versions"])

    def test_withdrawn_automatic_inputs_disappear_including_persisted_and_shared_options(self):
        fact = self.fact()
        self.ledger.merge(ResearchResult(facts=(fact,)))
        automatic = self.project().assumptions[0]
        old_options = [option for option in numeric_references(self.ledger).values()
                       if option.parents and option.parents[0] == automatic["key"]]
        self.assertEqual(len(old_options), 2)
        # Persisted model facts must not bypass the assumption-option filter.
        self.ledger.merge(ResearchResult(facts=tuple(old_options), assumptions=(
            {**automatic, "source_candidate_unavailable": True},)))
        self.assertFalse(any(option.parents and option.parents[0] == automatic["key"]
                             for option in numeric_references(self.ledger).values()))
        for old in old_options:
            support, support_complete, invalid = support_for(self.ledger, (old,), {}, old.subject)
            self.assertEqual(support, {})
            self.assertEqual(invalid, [automatic["key"]])
            self.assertFalse(support_complete)

    def test_nonparent_fact_keeps_existing_matching_support_fallback(self):
        fact = self.fact()
        self.ledger.merge(ResearchResult(facts=(fact,)))
        automatic = self.project().assumptions[0]
        legacy = Fact(automatic["subject"], fact.field, fact.value, fact.unit,
                      "user_assumption", "Legacy input without explicit parent.")
        support, support_complete, invalid = support_for(self.ledger, (legacy,), {}, legacy.subject)
        self.assertEqual(set(support), {automatic["key"]})
        self.assertEqual((support_complete, invalid), (True, []))

    def test_partial_source_invalidation_does_not_reappear_on_next_projection(self):
        first, second = self.fact(), self.fact(source="https://example.org/two")
        self.ledger.merge(ResearchResult(facts=(first, second)))
        original = self.project().assumptions[0]
        self.ledger.merge(ResearchResult(assumptions=(
            {**original, "invalidated_evidence": [first.semantic_record()]},)))
        updated = self.project().assumptions[0]
        self.assertEqual(updated["source_fact_keys"], [second.key])
        self.assertEqual(updated["invalidated_evidence"], [first.semantic_record()])
        self.assertFalse(self.project().assumptions)
        self.assertEqual(self.ledger.assumptions[original["key"]]["source_fact_keys"], [second.key])

    def test_legacy_whole_profile_grade_cannot_rank_different_native_values(self):
        first, second = self.fact(value=.00015), self.fact(value=.0002)
        self.ledger.merge(ResearchResult(facts=(first, second)))
        self.ledger.merge(ResearchResult(source_target_relationships=(self.relation(first, .9),)))
        candidates = self.project().assumptions
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(a["source_applicability"]["score"] is None for a in candidates))


if __name__ == "__main__":
    unittest.main()
