"""Target discovery count and credit use the same independent-value projection."""

from copy import deepcopy
from episodic_workflow.case_reports.research.records import numeric_references
import unittest

from episodic_workflow.case_reports.research.assumptions import scientific_credit, unassessed_assumption
from episodic_workflow.case_reports.research.goal_state import scientific_goal_status
from episodic_workflow.case_reports.research.progress import found_parameter_progress
from episodic_workflow.case_reports.research.records import Fact, Ledger, ResearchResult, identity
from episodic_workflow.case_reports.research.scenario_sharing import initialize_parameter_scope
from episodic_workflow.task_executors.scenario_metrics.evaluate import UNITS
from episodic_workflow.tests.test_calculation_steps import (
    fixture as calculation_fixture, add_assumption, calculate, rows as calculation_rows,
    OUTPUT as RELAXATION,
)


FIELD = "breathing_flow_m3_s"
OUTPUT = "quanta_emission_rate_s"


def fixture():
    scenarios = tuple({"subject": subject, "description": "Same synthetic respiratory activity",
        "role": "target", "assumption_needs": [FIELD],
        "fixed_parameters": {"quanta_concentration_exhaled_m3": {"value": 20., "unit": "quanta/m3"}},
        "output_contracts": {OUTPUT: {}}} for subject in ("alpha", "beta"))
    scoped = initialize_parameter_scope(ResearchResult(scenarios=scenarios), lambda request: {
        "features": [], "dependencies": [{"parameter": field, "factors": [],
            "reason": "Same declared respiratory activity"} for field in request["parameters"]]})
    return Ledger(scoped)


def add_candidate(ledger, *, subject="alpha", value=.001, verdict="unassessed", score=0.):
    row = unassessed_assumption({"subject": subject, "field": FIELD, "value": value,
        "unit": UNITS[FIELD], "statement": "Conditional respiratory flow", "basis": "Offline numerical hypothesis"}, ledger)
    row["reasonableness"] = score
    row["review"]["verdict"] = verdict
    row["review"]["completed"] = verdict != "unassessed"
    row["version"] = identity(row)
    ledger.merge(ResearchResult(assumptions=(row,)))
    return row


class FoundParameterCreditTests(unittest.TestCase):
    def assert_aligned(self, ledger, count, points):
        status = scientific_goal_status(ledger.result(), (OUTPUT,))
        self.assertEqual(status["recorded_target_inputs"], {"recorded": count, "required": 1})
        self.assertEqual(status["scores"]["found_parameters"], points)
        local = scientific_credit(ledger.result(), (FIELD,), subjects=("beta",), outputs=(OUTPUT,))
        self.assertEqual(sum(row[2] for row in local.qualities), points)
        self.assertEqual(status["scores"]["calculation_steps"], 0.)
        return status

    def test_unreviewed_unsupported_provisional_and_contradicted_are_discovery_not_support(self):
        for verdict in ("unassessed", "unsupported", "provisional", "contradicted", "supported"):
            with self.subTest(verdict=verdict):
                ledger = fixture()
                add_candidate(ledger, verdict=verdict)
                status = self.assert_aligned(ledger, 1, 1.)
                self.assertEqual(status["target_inputs"]["alpha"][FIELD]["review_verdict"], verdict)
                self.assertEqual(status["target_assumptions"]["fulfilled"],
                                 int(verdict in {"supported", "provisional"}))

    def test_source_fact_alone_does_not_fill_target(self):
        ledger = fixture()
        fact = Fact("source setting", FIELD, .001, UNITS[FIELD], "source_observation",
                    "Synthetic measurement", "https://example.test/source", "Flow .001 m3/s", "Synthetic activity")
        ledger.merge(ResearchResult(facts=(fact,)))
        self.assert_aligned(ledger, 0, 0.)
        row = add_candidate(ledger)
        row["entry_type"] = "source_candidate"
        row["source_applicability"] = {"score": 0., "status": "assessed"}
        row["source_fact_keys"] = [fact.key]
        self.assert_aligned(ledger, 1, 1.)

    def test_repetition_alternatives_and_shared_subject_copies_do_not_multiply_credit(self):
        ledger = fixture()
        row = add_candidate(ledger)
        ledger.merge(ResearchResult(assumptions=(row, row)))
        add_candidate(ledger, value=.002)
        add_candidate(ledger, subject="beta")
        self.assert_aligned(ledger, 1, 1.)
        del ledger.assumptions[row["key"]]
        self.assert_aligned(ledger, 1, 1.)

    def test_existing_evidence_bonus_is_once_per_independent_group(self):
        ledger = fixture()
        records = [Fact("source", FIELD, .001, UNITS[FIELD], "source_observation", "Synthetic",
            f"https://example.test/source-{i}", f"Independent measurement {i}", "Synthetic activity").semantic_record()
            for i in range(3)]
        first = add_candidate(ledger)
        first["learning_evidence"] = [{"relation": "supports", "evidence": row} for row in records]
        second = add_candidate(ledger, subject="beta")
        second["learning_evidence"] = list(reversed(first["learning_evidence"]))
        self.assert_aligned(ledger, 1, 1.2)
        first["invalidated_evidence"] = [records[1]]
        self.assert_aligned(ledger, 1, 1.1)

    def test_default_fixed_computed_and_source_only_records_do_not_earn_discovery(self):
        for role in ("scenario_constraint", "request_constraint", "model_default"):
            with self.subTest(role=role):
                ledger = fixture()
                row = add_candidate(ledger)
                row["role"] = role
                self.assert_aligned(ledger, 0, 0.)
        ledger = fixture()
        for field, value in (("quanta_emission_rate_s", .02), ("removal_rate_s", .002),
                             ("removal_capacity_m3_s", .2), ("settling_speed_m_s", .001)):
            row = {"key": field, "version": field, "subject": "alpha", "field": field,
                   "value": value, "unit": UNITS[field], "statement": "Calculated quantity",
                   "basis": "Synthetic derived result", "review": {}, "reasonableness": 0}
            ledger.merge(ResearchResult(assumptions=(row,)))
        self.assert_aligned(ledger, 0, 0.)

    def test_missing_invalid_or_withdrawn_source_candidate_has_no_discovery_floor(self):
        for changes in ({"value": None}, {"value": float("nan")}, {"unit": "wrong"},
                        {"source_candidate_unavailable": True}):
            with self.subTest(changes=changes):
                ledger = fixture()
                row = add_candidate(ledger)
                row.update(changes)
                progress = found_parameter_progress(ledger.result())
                self.assertFalse(progress[0]["found"])
                self.assertEqual(progress[0]["points"], 0.)

    def test_best_value_uses_relevance_independent_of_scientific_review(self):
        ledger = fixture()
        lower = add_candidate(ledger, value=.001, verdict="supported", score=1.)
        lower["source_applicability"] = {"score": .3, "status": "assessed"}
        higher = add_candidate(ledger, value=.002, verdict="unsupported")
        higher["source_applicability"] = {"score": .9, "status": "assessed"}
        status = self.assert_aligned(ledger, 1, 1.)
        self.assertEqual(status["independent_target_values"][0]["value"], .002)
        for subject in ("alpha", "beta"):
            self.assertEqual(status["target_inputs"][subject][FIELD]["value"], .002)
            self.assertEqual(status["target_inputs"][subject][FIELD]["selected_assumption_key"], higher["key"])

    def test_unreviewed_automatic_value_is_available_in_target_route_diagnostics(self):
        from episodic_workflow.tests.test_source_value_handoff import initial_state, source_fact
        from episodic_workflow.case_reports.research.target_values import project_source_target_values

        ledger = Ledger(initial_state())
        ledger.merge(ResearchResult(facts=(source_fact(FIELD, .00015),)))
        ledger.merge(project_source_target_values(ledger))
        status = scientific_goal_status(ledger.result(), ("scenario_reproductive_number",))
        self.assertEqual(status["recorded_target_inputs"]["recorded"], 1)
        for subject, target in status["target_routes"].items():
            self.assertEqual(status["target_inputs"][subject][FIELD]["review_verdict"], "unassessed")
            self.assertTrue(target["alternatives"])
            for route in target["alternatives"]:
                self.assertNotIn(FIELD, route["missing_inputs"])

    def test_promoted_default_counts_only_affected_subject_group(self):
        ledger = fixture()
        row = add_candidate(ledger)
        scenario = deepcopy(ledger.scenarios["alpha"])
        scenario["model_defaults"] = {FIELD: {"decision": "retain", "value": .001, "unit": UNITS[FIELD]}}
        ledger.merge(ResearchResult(scenarios=(scenario,)))
        progress = found_parameter_progress(ledger.result())
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0]["subjects"], ["beta"])
        self.assertFalse(progress[0]["found"])
        self.assertEqual(progress[0]["points"], 0.)

    def test_automatic_best_value_change_stales_only_its_computations(self):
        ledger = calculation_fixture()
        for key, row in tuple(ledger.assumptions.items()):
            if row["field"] == "room_volume_m3":
                del ledger.assumptions[key]
            else:
                row["value"] = 1.
                row["version"] = identity(row)
        ledger.scenarios["target"]["fixed_parameters"]["removal_capacity_m3_s"]["value"] = 1.
        group = next(row for row in found_parameter_progress(ledger.result())
                     if row["parameter"] == "room_volume_m3")
        first = add_assumption(ledger, "room_volume_m3", 10.)
        first.update(entry_type="source_candidate", source_candidate_group=group["group_id"],
                     source_applicability={"score": .3, "status": "assessed"})
        calculate(ledger)
        self.assertEqual(calculation_rows(ledger)[RELAXATION]["computed_value"], 10.)
        first["source_applicability"]["score"] = .5
        first["version"] = identity(first)
        self.assertEqual(calculation_rows(ledger)[RELAXATION]["status"], "calculated_up_to_date")
        second = add_assumption(ledger, "room_volume_m3", 20.)
        second.update(entry_type="source_candidate", source_candidate_group=group["group_id"],
                      source_applicability={"score": .9, "status": "assessed"})
        stale = calculation_rows(ledger)[RELAXATION]
        self.assertEqual(stale["status"], "calculated_needs_updating")
        self.assertEqual(stale["computed_value"], 10.)
        self.assertEqual(stale["points"], .025)
        self.assertTrue(any(change["kind"] == "selected_target_value" and change["selected_value"] == 20.
            for row in stale["result_alternatives"] for change in row["changed_dependencies"]))
        # The public numerical tool automatically consumes the new main value.
        from episodic_workflow.case_reports.research.tools import execute_tool
        ledger.merge(execute_tool({"action": "calculate_geometry_and_removal", "subject": "target",
            "metrics": [RELAXATION], "basis": "Recalculate selected best input",
            "parameterization": "updated selected value"}, ledger, (RELAXATION,),
            review={"admit": True, "reason": "Compatible conditional arithmetic"}))
        current = calculation_rows(ledger)[RELAXATION]
        self.assertEqual(current["computed_value"], 20.)
        self.assertEqual(current["status"], "calculated_up_to_date")
        self.assertEqual(current["points"], .125)

    def test_manual_target_alternative_retains_arithmetic_but_needs_main_value_update(self):
        ledger = calculation_fixture()
        calculate(ledger)
        group = next(row for row in found_parameter_progress(ledger.result())
                     if row["parameter"] == "room_volume_m3")
        source = add_assumption(ledger, "room_volume_m3", 200.)
        source.update(entry_type="source_candidate", source_candidate_group=group["group_id"],
                      source_applicability={"score": .9, "status": "assessed"})
        step = calculation_rows(ledger)[RELAXATION]
        self.assertEqual(step["status"], "calculated_needs_updating")
        self.assertEqual(step["computed_value"], 500.)
        self.assertEqual(step["points"], .025)
        change = step["result_alternatives"][0]["changed_dependencies"][0]
        self.assertEqual((change["previous_value"], change["selected_value"]), (100., 200.))
        self.assertEqual(change["kind"], "selected_target_value")
        self.assertEqual(next(row for row in found_parameter_progress(ledger.result())
                              if row["parameter"] == "room_volume_m3")["value"], 200.)

    def test_same_manual_numeric_value_remains_current_under_automatic_main_selection(self):
        ledger = calculation_fixture()
        calculate(ledger)
        group = next(row for row in found_parameter_progress(ledger.result())
                     if row["parameter"] == "room_volume_m3")
        original = next(row for row in ledger.assumptions.values() if row["field"] == "room_volume_m3")
        source = {**original, "key": "automatic-same-value", "entry_type": "source_candidate",
            "source_candidate_group": group["group_id"], "source_applicability": {"score": .9}}
        source["version"] = identity(source)
        ledger.merge(ResearchResult(assumptions=(source,)))
        self.assertEqual(calculation_rows(ledger)[RELAXATION]["status"], "calculated_up_to_date")
        self.assertEqual(calculation_rows(ledger)[RELAXATION]["points"], .125)

    def test_source_calibration_operands_do_not_follow_target_main_values(self):
        from episodic_workflow.case_reports.research.calculation_steps import calculation_is_up_to_date
        from episodic_workflow.case_reports.research.tools import execute_tool
        ledger = calculation_fixture()
        group = next(row for row in found_parameter_progress(ledger.result())
                     if row["parameter"] == "room_volume_m3")
        source = add_assumption(ledger, "room_volume_m3", 200.)
        source.update(entry_type="source_candidate", source_candidate_group=group["group_id"],
                      source_applicability={"score": .9, "status": "assessed"})
        event = {**ledger.scenarios["target"], "subject": "source-event", "role": "calibration",
                 "fixed_parameters": {}}
        ledger.merge(ResearchResult(scenarios=(event,)))
        for field, value in (("room_volume_m3", 100.), ("removal_capacity_m3_s", .2)):
            candidate = unassessed_assumption({"subject": event["subject"], "field": field,
                "value": value, "unit": UNITS[field], "statement": "Source event premise",
                "basis": "Source-specific physical input"}, ledger)
            ledger.merge(ResearchResult(assumptions=(candidate,)))
        refs = [key for key, fact in numeric_references(ledger).items() if fact.subject == event["subject"]]
        result = execute_tool({"action": "calculate_geometry_and_removal", "subject": event["subject"],
            "source_inputs": refs, "metrics": [RELAXATION], "basis": "Source event arithmetic"},
            ledger, (RELAXATION,), review={"admit": True, "reason": "Consistent event operands"})
        ledger.merge(result)
        self.assertEqual(result.calculations[0]["input_values"]["room_volume_m3"], 100.)
        self.assertEqual(result.calculations[0]["output"][RELAXATION]["value"], 500.)
        self.assertTrue(calculation_is_up_to_date(result.calculations[0], RELAXATION, ledger))


if __name__ == "__main__":
    unittest.main()
