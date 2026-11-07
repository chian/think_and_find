"""Deterministic VOC binding verification; no live research or provider calls.

Run through tools/one_workflow.sh, with --run-dir set to its fresh run directory.
Finite fixtures supply executed sheet changes; they are not research policies.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def _load_checkout():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("episodic_workflow", root / "__init__.py",
        submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    _load_checkout()
    from episodic_workflow.case_reports.research.causal_sheet import CausalSheet
    from episodic_workflow.case_reports.research.computation_yield import (
        ScientificTransitionModel, ScientificVOCController, measured_state)
    from episodic_workflow.case_reports.research.engine import ResearchRun, ResearchProviders, ResearchLimits, TREE
    from episodic_workflow.case_reports.research.episode_contracts import ScientificAttemptCommission
    from episodic_workflow.case_reports.research.records import Credit, ResearchResult
    from episodic_workflow.case_reports.research.workflow import DiscoverySource, ScientificSource
    from episodic_workflow.credit_assignment.research import ResearchController
    from episodic_workflow.credit_assignment.incidence_rarefaction import RAREFACTION_FORMULA_VERSION
    from episodic_workflow.credit_assignment.value_of_computation import HypotheticalObservation, value_of_computation
    from episodic_workflow.method_loop import Context, EpisodeGoal, EpisodeTree

    events = []

    def no_provider(*_):
        raise AssertionError("Verification must not make external provider calls")

    def run_for(agent=no_provider, initial=None):
        return ResearchRun(ResearchProviders(agent=agent, search=no_provider,
            retrieve=no_provider, trace_event=events.append), ResearchLimits(),
            "Deterministic scientific-score fixture.", initial=initial)

    def episode_for(run, source, grain):
        pending, reachable = [grain], set()
        while pending:
            item = pending.pop()
            if item not in reachable:
                reachable.add(item)
                pending.extend(TREE[item])
        run.tree = EpisodeTree(run.grains[grain], {
            run.grains[parent]: tuple(run.grains[child] for child in TREE[parent])
            for parent in reachable if TREE[parent]})
        commission = ScientificAttemptCommission(run.research_request, (), ())
        contract = commission.report_contract() if grain == "scientific" else None
        goal = EpisodeGoal.root(objective=run.research_request, task_context=run.context,
            result_contract={"information": contract.as_record()} if contract else {})
        return run.episode(grain, source, goal=goal, contract=contract)

    def setup(initial=None):
        run = run_for(initial=initial)
        source = ScientificSource(run, run.initial)
        episode = episode_for(run, source, "scientific")
        controller = run.controller((("scientific", episode.key),))
        return run, source, controller

    def inference(name):
        return {"statement": "Fixture causal inference " + name, "variables": ["x:" + name, "y:" + name],
            "intervention": "increase x", "outcome": "y changes", "grain": "fixture scalar", "domain": "fixture"}

    def update(source, key, kind, data, dependencies=()):
        sheet = CausalSheet(source.ledger.causal_sheet.history)
        return ResearchResult(causal_sheet_updates=(sheet.put(key, kind, data, dependencies=dependencies),))

    def complete(source, controller, proposal, result, *, refresh_child=False):
        source.ledger.merge(source.run.global_goal.ledger.result())
        controller.begin()
        source.last_proposal = proposal
        value = source.project(result)
        if refresh_child:
            source.run.global_goal.commit(source.pending)
            before = len(controller.observations)
            controller.refresh(value)
            assert len(controller.observations) == before
        step = controller.observe(proposal.get("action", "fixture"), value, is_root=False)
        source.ledger.merge(source.pending)
        source.run.global_goal.commit(source.pending)
        source.pending = source.pending_global = None
        return step

    def assessment(cls):
        return {**{key: "Deterministic fixture." for key in cls.assessment_fields},
            "record_refs": [], "receipt_refs": [], "detail_refs": []}

    class Verification(unittest.TestCase):
        def test_exactly_two_controllers_and_unchanged_child_formula(self):
            for grain in TREE:
                run = run_for()
                if grain == "discovery":
                    source = DiscoverySource(run)
                    episode = episode_for(run, source, grain)
                elif grain == "scientific":
                    source = ScientificSource(run, run.initial)
                    episode = episode_for(run, source, grain)
                else:
                    source = SimpleNamespace(episode_grain=grain, credit_columns=("fixture",),
                        ledger=SimpleNamespace(result=lambda: ResearchResult()), credit=lambda _: Credit())
                    episode = SimpleNamespace(key="fixture:" + grain)
                    run.controller_specs[episode.key] = source
                controller = run.controller(((grain, episode.key),))
                self.assertIsInstance(controller, ScientificVOCController if grain in
                    {"discovery", "scientific"} else ResearchController)
            self.assertEqual(RAREFACTION_FORMULA_VERSION, "joint_preferential_discovery_v1")
            root = Path(__file__).resolve().parents[1]
            digest = hashlib.sha256((root / "credit_assignment/incidence_rarefaction.py").read_bytes()).hexdigest()
            self.assertEqual(digest, "976d3ec63e24621fddb5aed455924dc5f0f96e8db3559f98f9d36ea85da17f11")
            provenance = json.loads((root / "credit_assignment/VALUE_OF_COMPUTATION_SOURCE.json").read_text())
            self.assertEqual(hashlib.sha256((root / "credit_assignment/value_of_computation.py").read_bytes()).hexdigest(),
                             provenance["sha256"])

        def test_deduplicated_sum_and_signed_corrections(self):
            sheet = CausalSheet()
            sheet.put("G1", "inference", inference("one"))
            sheet.put("G2", "inference", inference("two"))
            run, source, controller = setup(ResearchResult(causal_sheet_updates=tuple(sheet.history)))
            self.assertEqual(run.global_goal.preview()["total_score"], 2)
            before, _ = measured_state(source, source.ledger.result())
            loss = complete(source, controller, {"action": "record"},
                update(source, "G2", "inference", inference("one")))
            self.assertEqual((loss.gain, loss.score), (-1, -1))
            forecast = value_of_computation(controller.model, before, lookahead=1)
            self.assertEqual(next(item for item in forecast.estimates if item.action == "record").expected_gain, -1)
            restored = complete(source, controller, {"action": "record"},
                update(source, "G2", "inference", inference("two")))
            self.assertEqual((restored.gain, restored.score), (1, 0))
            duplicate = complete(source, controller, {"action": "record"},
                update(source, "G2", "inference", inference("two")))
            self.assertEqual(duplicate.gain, 0)
            self.assertEqual([event["score_change"] for event in controller.observations], [-1, 1, 0])
            self.assertEqual(run.global_goal.preview()["total_score"], 2)

        def test_zero_credit_intermediate_has_delayed_payoff_and_provenance(self):
            _, source, controller = setup()
            before, _ = measured_state(source, source.ledger.result())
            premise = {"statement": "Executed fixture check: 2 + 2 = 4.",
                       "scope": "integer fixture", "basis": "deterministic executable check"}
            self.assertEqual(2 + 2, 4)
            preparation = complete(source, controller, {"action": "record"},
                update(source, "checked-premise", "premise", premise))
            after, _ = measured_state(source, source.ledger.result())
            self.assertEqual(preparation.gain, 0)
            self.assertNotEqual(before.key, after.key)
            self.assertNotEqual(before.features, after.features)
            score = complete(source, controller,
                {"action": "record", "records": [{"dependencies": ["checked-premise"]}]},
                update(source, "G", "inference", inference("checked"), ("checked-premise",)))
            self.assertEqual(score.gain, 1)
            report = value_of_computation(controller.model, before, lookahead=2)
            estimate = next(item for item in report.estimates if item.action == "record")
            self.assertEqual((estimate.immediate_gain, estimate.continuation_gain, estimate.expected_gain), (0, 1, 1))
            self.assertEqual(controller.observations[1]["depends_on"], [controller.observations[0]["event_id"]])
            self.assertEqual(len(controller.model.events), 2)

        def test_unknown_forecasts_are_not_zero_observations(self):
            _, source, controller = setup()
            feedback = controller.feedback()
            self.assertTrue(feedback["forecast"]["unmodeled_decisions"])
            self.assertTrue(all(row["expected_gain"] is None for row in feedback["forecast"]["estimates"]))
            self.assertFalse(controller.state().stop)
            before = deepcopy(controller.model.events)
            controller.feedback()
            controller.state()
            self.assertEqual(controller.model.events, before)
            self.assertEqual(controller.observations, [])
            self.assertEqual(controller.history, [0])

        def test_completed_failures_are_measured_zero_and_keep_diagnostics(self):
            _, source, controller = setup()
            failed = ResearchResult(evaluated=False, diagnostics=({"stage": "fixture",
                "category": "provider_failure", "status": "failed", "why": "completed failure"},))
            step = complete(source, controller, {"action": "reasoning"}, failed)
            self.assertFalse(step.evaluated)
            self.assertEqual(step.gain, 0)
            self.assertEqual(controller.failures, 1)
            event = controller.observations[0]
            self.assertTrue(event["yield_observed"])
            self.assertEqual(event["diagnostics"][0]["why"], "completed failure")
            self.assertEqual(next(row for row in controller.feedback()["forecast"]["estimates"]
                                  if row["action"] == "reasoning")["immediate_gain"], 0)
            self.assertFalse(controller.state().stop)  # Other choices remain unknown.

        def test_malformed_completed_actions_still_reach_voc(self):
            responses = [{"action": "record", "records": None},
                         {"action": "inspect", "keys": None},
                         {"action": "reasoning", "commission": None},
                         {"action": {"invalid": "action"}}]
            def agent(request):
                return {**responses.pop(0), "rationale": "Malformed output fixture.",
                        "working_assessment": assessment(ScientificSource)}
            class FiniteScientific(ScientificSource):
                def propose(self, view):
                    return super().propose(view) if responses else None
            run = run_for(agent)
            source = FiniteScientific(run, run.initial)
            episode = episode_for(run, source, "scientific")
            record = episode.run(Context(tree=run.tree))
            controller = run.computation_controller(source)
            self.assertEqual(record.units_consumed, 4)
            self.assertEqual(controller.failures, 4)
            self.assertEqual([item["score_change"] for item in controller.observations], [0, 0, 0, 0])
            self.assertTrue(all(item["yield_observed"] and not item["evaluated"] for item in controller.observations))

        def test_child_shared_refresh_does_not_erase_parent_gain(self):
            _, source, controller = setup()
            step = complete(source, controller, {"action": "reasoning"},
                update(source, "G", "inference", inference("child")), refresh_child=True)
            self.assertEqual(step.gain, 1)
            self.assertEqual(len(controller.observations), 1)
            self.assertEqual(controller.history, [0, 1])
            self.assertEqual(controller.observations[0]["action"], "reasoning")

        def test_checked_calculation_findings_are_distinct_zero_score_states(self):
            _, source, _ = setup()
            before, _ = measured_state(source, source.ledger.result())
            source.last_calculation = {"fixture_check": (2 + 2 == 4), "counterexample": None}
            after, metadata = measured_state(source, source.ledger.result())
            self.assertEqual(before.score, after.score)
            self.assertNotEqual(before.key, after.key)
            self.assertNotEqual(before.features, after.features)
            self.assertEqual(metadata["calculation"]["fixture_check"], True)

        def test_structural_transfer_is_explicit_and_imagined_learning_is_read_only(self):
            run, source, controller = setup()
            complete(source, controller, {"action": "inspect"}, ResearchResult())
            state, _ = measured_state(source, source.ledger.result())
            other = type(state)("another-exact-state", state.score, state.features, state.actions)
            before = deepcopy(controller.model.events)
            prediction = controller.model.predict(other, "inspect")
            self.assertEqual(prediction.basis, "observed_support_only_structural_transfer")
            report = value_of_computation(controller.model, other)
            self.assertEqual(controller.model.events, before)
            self.assertTrue(report.unmodeled_decisions)

        def test_hypothetical_outcomes_update_predictive_probabilities_only_in_forecast(self):
            _, source, controller = setup()
            before, _ = measured_state(source, source.ledger.result())
            complete(source, controller, {"action": "reasoning"}, ResearchResult(evaluated=False))
            complete(source, controller, {"action": "reasoning"},
                update(source, "G", "inference", inference("predictive")))
            observed = controller.model.predict(before, "reasoning")
            self.assertEqual(sum(row.probability for row in observed.transitions if row.next_state.score > 0), .5)
            events_before = deepcopy(controller.model.events)
            imagined = controller.model.predict(before, "reasoning", (
                HypotheticalObservation(before, "reasoning", before),))
            self.assertAlmostEqual(sum(row.probability for row in imagined.transitions if row.next_state.score > 0), 1 / 3)
            self.assertEqual(controller.model.events, events_before)

        def test_scientific_next_decision_receives_forecasts_and_credit_history(self):
            responses = [
                {"action": "record", "rationale": "Save checked intermediate.", "records": [{
                    "key": "P", "kind": "premise", "data": {"statement": "Checked fixture premise.",
                    "scope": "fixture", "basis": "executed fixture"}, "dependencies": []}]},
                {"action": "record", "rationale": "Use the intermediate.", "records": [{
                    "key": "G", "kind": "inference", "data": inference("follow-up"), "dependencies": ["P"]}]},
                {"action": "inspect", "rationale": "Read the resulting state.", "keys": ["G"]}]
            requests = []
            def agent(request):
                requests.append(deepcopy(request))
                return {**responses.pop(0), "working_assessment": assessment(ScientificSource)}
            # Train on actual checked fixture sheet updates, retaining the
            # zero-credit preparation and its subsequent measured registration.
            _, training_source, training = setup()
            complete(training_source, training, responses[0],
                update(training_source, "P", "premise", responses[0]["records"][0]["data"]))
            complete(training_source, training, responses[1],
                update(training_source, "G", "inference", inference("follow-up"), ("P",)))
            class FiniteScientific(ScientificSource):
                def propose(self, view):
                    return super().propose(view) if responses else None
            run = run_for(agent)
            seeded = ScientificTransitionModel()
            for event in training.model.events.values():
                event = deepcopy(event)
                event["event_id"] = "training:" + event["event_id"]
                event["history"]["event_id"] = event["event_id"]
                event["history"]["depends_on"] = ["training:" + key for key in event["history"]["depends_on"]]
                seeded.observe(event)
            run.computation_models["scientific"] = seeded
            source = FiniteScientific(run, run.initial)
            episode = episode_for(run, source, "scientific")
            record = episode.run(Context(tree=run.tree))
            self.assertEqual(record.units_consumed, 3)
            preparation_forecast = next(item for item in requests[0]["value_of_computation"]["forecast"]["estimates"]
                                        if item["action"] == "record")
            self.assertEqual(preparation_forecast["immediate_gain"], 0)
            self.assertEqual(preparation_forecast["continuation_gain"], 1)
            self.assertEqual(requests[0]["value_of_computation"]["credit_history"], [])
            self.assertTrue(requests[0]["value_of_computation"]["forecast_evidence"])
            self.assertEqual({event["score_change"] for event in
                              requests[0]["value_of_computation"]["forecast_evidence"]}, {0, 1})
            feedback = requests[2]["value_of_computation"]
            self.assertEqual(feedback["scope"], "scientific")
            self.assertEqual([event["score_change"] for event in feedback["credit_history"]], [0, 1])
            self.assertEqual(feedback["credit_history"][1]["depends_on"],
                             [feedback["credit_history"][0]["event_id"]])
            self.assertEqual(set(requests[0]["allowed_actions"]),
                {"record", "inspect", "calculate", "reconstruct_theory", "reasoning", "recall"})
            self.assertNotIn("value_of_computation", requests[0]["response_contract"]["properties"]
                             if "properties" in requests[0]["response_contract"] else {})

        def test_scientific_voc_actually_stops_the_episode(self):
            def agent(request):
                return {"action": "inspect", "keys": [], "rationale": "Known empty fixture action.",
                        "working_assessment": assessment(ScientificSource)}
            run = run_for(agent)
            source = ScientificSource(run, run.initial)
            episode = episode_for(run, source, "scientific")
            state, _ = measured_state(source, source.ledger.result())
            model = ScientificTransitionModel()
            # Explicit deterministic training fixture: each available action has
            # an observed empty return. These are model fixtures, not pseudo-credit.
            for action in state.actions:
                model.observe({"event_id": "fixture:" + action, "before": state, "after": state,
                    "action": action, "score_change": 0})
            run.computation_models["scientific"] = model
            with patch.object(ResearchController, "observe", side_effect=AssertionError("Rarefaction used by Scientific")):
                record = episode.run(Context(tree=run.tree))
            self.assertEqual(record.units_consumed, 1)
            self.assertEqual(record.ended_by, "yield_stop")
            self.assertEqual(record.controller_state.reason, "voc_no_gain_within_lookahead")
            self.assertIsNone(record.controller_state.rarefaction)

        def test_discovery_feedback_standing_mode_and_one_observation_per_child(self):
            class ThreeUnitScientific(ScientificSource):
                def __init__(self, *values):
                    super().__init__(*values)
                    self.fixture = iter(("first", "second", "third"))
                def propose(self, view):
                    label = next(self.fixture, None)
                    if label is None:
                        return None
                    self.last_proposal = {"action": "inspect"}
                    return self.leaf(lambda: ResearchResult(), label)
            requests = []
            starts = iter(("fixture start one", "fixture start two"))
            def agent(request):
                self.assertEqual(request["operation"], "propose_discovery_attempt")
                requests.append(deepcopy(request))
                return {"question": next(starts), "inputs": [], "snapshots": [],
                    "rationale": "Finite deterministic child fixture.", "working_assessment": assessment(DiscoverySource)}
            class FiniteDiscovery(DiscoverySource):
                def propose(self, view):
                    return super().propose(view) if len(requests) < 2 else None
            run = run_for(agent)
            source = FiniteDiscovery(run)
            episode = episode_for(run, source, "discovery")
            state, _ = measured_state(source, source.ledger.result())
            model = ScientificTransitionModel()
            for action in state.actions:
                model.observe({"event_id": "fixture:root:" + action, "before": state,
                    "after": state, "action": action, "score_change": 0})
            run.computation_models["discovery"] = model
            with patch("episodic_workflow.case_reports.research.workflow.ScientificSource", ThreeUnitScientific):
                record = episode.run(Context(tree=run.tree))
            self.assertEqual(record.units_consumed, 2)
            self.assertTrue(all(unit.child.units_consumed == 3 for unit in record.unit_records))
            controller = run.computation_controller(source)
            self.assertEqual(len(controller.observations), 2)
            self.assertFalse(controller.state().stop)
            self.assertTrue(requests[1]["value_of_computation"]["standing_investigation"])
            self.assertEqual(len(requests[1]["value_of_computation"]["credit_history"]), 1)
            self.assertEqual(requests[1]["value_of_computation"]["credit_history"][0]["action"], "fresh_start")
            current = controller._current
            for action in current.actions:
                model.observe({"event_id": "fixture:root:current:" + action, "before": current,
                    "after": current, "action": action, "score_change": 0})
            self.assertEqual(controller.feedback()["forecast"]["recommendation"],
                             "no_gain_within_model_and_lookahead")
            self.assertFalse(controller.state().stop)

        def test_forecast_depth_does_not_limit_execution(self):
            choices = iter(("one", "two", "three", "four", "five", "six"))
            class FiniteScientific(ScientificSource):
                def propose(self, view):
                    label = next(choices, None)
                    if label is None:
                        return None
                    self.last_proposal = {"action": "inspect"}
                    return self.leaf(lambda: ResearchResult(), label)
            run = run_for()
            source = FiniteScientific(run, run.initial)
            episode = episode_for(run, source, "scientific")
            record = episode.run(Context(tree=run.tree))
            self.assertEqual(record.units_consumed, 6)
            self.assertEqual(record.ended_by, "exhausted")
            self.assertFalse(record.controller_state.stop)
            self.assertGreater(record.units_consumed, record.controller_state.value_of_computation["forecast"]["lookahead"])

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Verification)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {"checks_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "passed": result.wasSuccessful(), "cases": unittest.defaultTestLoader.getTestCaseNames(Verification)}
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.run_dir / "trace.jsonl").open("w") as stream:
        for event in events:
            stream.write(json.dumps(event) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
