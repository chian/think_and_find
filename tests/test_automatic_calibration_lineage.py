"""Automatic target inputs preserve numerical calibration ancestry before review."""

import unittest
from episodic_workflow.case_reports.research.records import numeric_references

from episodic_workflow.case_reports.research.records import Fact, Ledger, ResearchResult, current_support
from episodic_workflow.case_reports.research.engine import ResearchLimits, ResearchProviders, ResearchRun, ResearchSource
from episodic_workflow.case_reports.research.tools import execute_tool
from episodic_workflow.case_reports.research.workflow import GuidelineSource
from episodic_workflow.case_reports.research.verification import (
    ancestors, calibration_progress, provenance_index, target_calibration_progress,
)
from episodic_workflow.task_executors.scenario_metrics.evaluate import DEFAULT_METRICS, UNITS
from episodic_workflow.method_loop import Context, EpisodeGoal
from .test_derived_target_values import FIELD, REVIEW, SOURCE, calculate, fixture, project


OUTPUT = "quanta_emission_rate_s"


def outbreak_fixture():
    ledger, _ = fixture()
    values = {"secondary_case_count": 3., "susceptible_count": 20., "exposure_time_s": 7200.,
        "index_case_count": 1., "mask_penetration": 1., "relative_transmissibility": 1.,
        "removal_capacity_m3_s": .1}
    observed = {"secondary_case_count", "susceptible_count", "exposure_time_s", "index_case_count"}
    facts = tuple(Fact(SOURCE, field, value, UNITS[field],
        "source_observation" if field in observed else "source_model_parameter",
        "Observed source outcome" if field in observed else "Conditional source-model premise",
        source="https://example.test/outbreak", quote=f"{field}={value}", context="The same source event")
        for field, value in values.items())
    ledger.merge(ResearchResult(facts=facts))
    return ledger, facts


class ParentDispatched(RuntimeError):
    pass


class AutomaticCalibrationLineageTests(unittest.TestCase):
    def target_calculations(self, ledger):
        """Use actual main-table inputs, without a support reviewer."""
        results = []
        for subject, scenario in ledger.scenarios.items():
            if scenario.get("role", "target") != "target":
                continue
            result = execute_tool({"action": "calculate_transmission_metrics", "subject": subject,
                "metrics": [OUTPUT], "parameterization": "automatic-source-estimate",
                "basis": "Conditional target calculation using the automatic source value.",
                "conditions": {**scenario.get("fixed_conditions", {}),
                    "uniform_breathing_and_masks": "Uniform breathing and masks in this conditional estimate."}},
                ledger, DEFAULT_METRICS, review=REVIEW)
            self.assertTrue(any(f.field == OUTPUT and f.subject == subject for f in result.facts))
            ledger.merge(result)
            results.append(result)
        return results

    def assert_numerical_ancestry(self, ledger, source_calculation, targets, native):
        source_value = source_calculation.facts[0]
        candidate = next(a for a in ledger.assumptions.values()
            if a.get("entry_type") == "source_candidate" and a["field"] == FIELD)
        self.assertFalse(candidate["review"]["completed"])
        self.assertEqual(candidate["review"]["verdict"], "unassessed")
        index = provenance_index(ledger.result())
        self.assertIn(source_value.key, index[candidate["key"]]["parents"])
        for target in targets:
            calculation = target.calculations[0]
            support = calculation["metric_support"][OUTPUT]
            lineage = ancestors(support["input_fact_keys"], index)
            self.assertIn(candidate["key"], lineage)
            self.assertIn(source_value.key, lineage)
            self.assertTrue({fact.key for fact in native}.issubset(lineage))
            self.assertEqual(current_support(calculation, ledger.assumptions,
                metric=OUTPUT, ledger=ledger), 0, "Numerical ancestry does not invent support")

    def test_source_quotient_keeps_ancestry_without_claiming_outbreak_inversion(self):
        ledger, native = fixture()
        source_calculation = calculate(ledger)
        ledger.merge(source_calculation)
        project(ledger)
        targets = self.target_calculations(ledger)
        self.assert_numerical_ancestry(ledger, source_calculation, targets, native)
        progress = calibration_progress(ledger.result())
        self.assertEqual(progress["calibrations"], [])
        self.assertFalse(progress["outbreak_calibration_performed"])
        links = target_calibration_progress(ledger.result(), (OUTPUT,))
        self.assertEqual(len(links), 2)
        for target in links.values():
            self.assertFalse(target["coherent_calibrated_solution_available"])
            for alternative in target["alternatives"]:
                self.assertEqual(alternative["computed_quantities"], [OUTPUT])
                self.assertEqual(alternative["calibrated_inputs_by_quantity"][OUTPUT], [])

    def test_actual_outbreak_inversion_links_automatic_value_to_target_outputs(self):
        ledger, facts = outbreak_fixture()
        source_calculation = calculate(ledger, "cq_from_steady_outbreak",
            ("breathing_flow_m3_s", *(fact.field for fact in facts)))
        ledger.merge(source_calculation)
        project(ledger)
        targets = self.target_calculations(ledger)
        self.assert_numerical_ancestry(ledger, source_calculation, targets, facts)
        progress = calibration_progress(ledger.result())
        self.assertTrue(progress["outbreak_calibration_performed"])
        self.assertEqual(len(progress["calibrations"]), 1)
        calibration = progress["calibrations"][0]
        self.assertEqual(calibration["operator"], "cq_from_steady_outbreak")
        self.assertTrue(calibration["outbreak_evidence_acquired"])
        self.assertTrue(calibration["duration_observation_available"])
        links = target_calibration_progress(ledger.result(), (OUTPUT,), calibration=progress)
        for target in links.values():
            self.assertTrue(target["coherent_calibrated_solution_available"])
            self.assertFalse(target["supported_coherent_calibrated_solution_available"])
            for alternative in target["alternatives"]:
                self.assertEqual(alternative["calibrated_inputs_by_quantity"][OUTPUT],
                    [source_calculation.facts[0].key])
                self.assertEqual(alternative["calibrated_support_by_quantity"][OUTPUT], 0)

    def test_real_outbreak_child_return_allows_next_parent_model_dispatch(self):
        self.parent_dispatch(with_target_calculations=False)

    def test_outbreak_and_target_outputs_allow_next_parent_model_dispatch(self):
        self.parent_dispatch(with_target_calculations=True)

    def parent_dispatch(self, *, with_target_calculations):
        ledger, facts = outbreak_fixture()
        initial = ledger.result()
        requests = []

        def agent(request):
            requests.append(request)
            if request["operation"] == "synthesize_research_findings":
                return {"findings": "An outbreak inversion supplies a conditional target concentration."}
            self.assertEqual(request["operation"], "propose_guideline_action")
            raise ParentDispatched("Parent input guard passed")

        run = ResearchRun(ResearchProviders(agent, lambda *_: (), lambda _: None), ResearchLimits(),
            "Use calibrated target inputs", (OUTPUT,) if with_target_calculations else DEFAULT_METRICS,
            initial=initial)
        parent, child = GuidelineSource(run), ResearchSource(run, initial)
        root = run.episode("evaluate_airborne_safety_guideline", parent,
            goal=EpisodeGoal.root(objective="Calculate target estimates", result_contract={}), focus=(FIELD,))
        started = False

        def parent_propose(view):
            nonlocal started
            if started:
                return GuidelineSource.propose(parent, view)
            started = True
            return parent.edge(run.child(view, child, "investigate", "Find concentration",
                contract=parent.contract("Find concentration", parameter=FIELD, subjects=(),
                    deliverable="source_findings"), proposal={}))

        def child_propose(view):
            if child.evaluated_units:
                return None
            def calculate_inputs():
                result = calculate(child.ledger, "cq_from_steady_outbreak",
                    ("breathing_flow_m3_s", *(fact.field for fact in facts)))
                if not with_target_calculations:
                    return result
                working = Ledger(child.ledger.result())
                working.merge(result)
                project(working)
                self.target_calculations(working)
                return working.result()
            return child.leaf(calculate_inputs, "calibrate_infectiousness")

        parent.propose, child.propose = parent_propose, child_propose
        with self.assertRaises(ParentDispatched):
            root.run(Context(tree=run.tree))
        request = requests[-1]
        self.assertEqual(request["operation"], "propose_guideline_action")
        candidate = next(a for a in parent.ledger.assumptions.values()
            if a.get("entry_type") == "source_candidate" and a["field"] == FIELD)
        self.assertEqual(candidate["review"]["verdict"], "unassessed")
        self.assertTrue(calibration_progress(parent.ledger.result())["outbreak_calibration_performed"])
        self.assertTrue(any(f.field == FIELD and f.value == candidate["value"]
                            for f in numeric_references(parent.ledger).values()))
        calibration = request["scientific_progress"]["calibration_progress"]["calibrations"][0]
        self.assertNotIn("output_fact_keys", calibration)
        output = calibration["outputs"][0]
        self.assertEqual({name: output[name] for name in ("subject", "field", "value", "unit")},
            {"subject": SOURCE, "field": FIELD, "value": candidate["value"], "unit": UNITS[FIELD]})
        self.assertNotIn("reference", output)
        if with_target_calculations:
            for target in request["scientific_progress"]["targets"].values():
                for alternative in target["calibration_linkage"]["alternatives"]:
                    self.assertEqual(alternative["calibrated_inputs_by_quantity"][OUTPUT], [output])
            full = target_calibration_progress(parent.ledger.result(), (OUTPUT,))
            raw_key = next(f.key for f in parent.ledger.facts.values()
                           if f.subject == SOURCE and f.field == FIELD and f.value == candidate["value"])
            for target in full.values():
                for alternative in target["alternatives"]:
                    self.assertEqual(alternative["calibrated_inputs_by_quantity"][OUTPUT], [raw_key])


if __name__ == "__main__":
    unittest.main()
