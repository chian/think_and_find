"""Offline regressions for delegated uncertainty and processed communication."""

from dataclasses import replace
import json
from types import SimpleNamespace
import unittest

from episodic_workflow.case_reports.research.assumptions import assess_assumption
from episodic_workflow.case_reports.research.acquisition import StrategySource
from episodic_workflow.case_reports.research.communication import diagnostic_options, validate_summary
from episodic_workflow.case_reports.research.engine import ResearchRun, ResearchSource, ResearchLimits, ResearchProviders
from episodic_workflow.case_reports.research.progress import assess_progress
from episodic_workflow.case_reports.research.records import Ledger, ResearchResult
from episodic_workflow.case_reports.research.workflow import InvestigationSource
from episodic_workflow.credit_assignment.research import ResearchController
from episodic_workflow.method_loop import EpisodeCompletion, EpisodeGoal
from episodic_workflow.task_executors._http import ProviderError
from episodic_workflow.tests.test_assumption_support import candidate, fixture_ledger, FIELD
from episodic_workflow.tests.test_approved_tree import execute_binding
from episodic_workflow.tests.test_research import run_synthetic, supported_review, synthetic_synthesis


def binding(agent, *, inherited=ResearchResult(), question="Bound infectiousness for the target activity", focus=(FIELD,)):
    run = ResearchRun(ResearchProviders(agent, lambda *_: (), lambda _: None), ResearchLimits(),
                      "Synthetic contract test", ("indoor_reproductive_number",))
    source = ResearchSource(run, inherited)
    contract = source.contract(question, focus, subjects=["target"])
    goal = EpisodeGoal.root(objective=question, result_contract={"fields": list(focus)})
    run.episode("report", source, goal=goal, focus=focus, contract=contract)
    return source


def evidence(quote="The activity limits the plausible interval.", **changes):
    return {"subject": "reference", "fields": [FIELD], "source": "https://example.test/source",
            "quote": quote, "claim": "Synthetic qualitative constraint", "context": "Synthetic comparable activity",
            "basis": "Software fixture only", **changes}


def judgment(request, status="bounded", refs=None):
    return {"judgments": [{"field": field, "status": status, "answer": "Synthetic contract-relative answer",
        "evidence": list(request["new_evidence"] if refs is None else refs),
        "limitations": ["No permission to calculate follows from acquisition credit"]} for field in request["focus"]]}


def accept(source, result):
    credit = source.project(result)
    source.ledger.merge(source.pending)
    source.pending = None
    return credit


class ContractProgressTests(unittest.TestCase):
    def test_tagged_dependency_paragraph_is_not_numeric_resolution(self):
        fields = ("ventilation_flow_m3_s", "filter_efficiency", "room_volume_m3", "floor_area_m2",
                  "room_height_m", "breathing_flow_m3_s", "mask_penetration", FIELD, "quanta_emission_rate_s")
        requests = []
        def reviewer(request):
            requests.append(request)
            return judgment(request, "background")
        source = binding(reviewer, focus=fields, question="Establish justified numeric inputs for the target room")
        paragraph = evidence("Exposure depends on ventilation, geometry, breathing, masks and infectiousness.", fields=list(fields))
        credit = accept(source, ResearchResult(evidence=(paragraph,)))
        self.assertTrue(source.ledger.evidence)
        self.assertTrue(credit.evaluated)
        self.assertFalse(credit.identities)
        self.assertTrue(all(score == 0 for _, _, score in credit.qualities))
        self.assertEqual(requests[0]["parent_contract"]["decision"], source.parent_contract.decision)
        self.assertIn("qualitative dependency list earns zero", requests[0]["instruction"])

    def test_qualitative_resolution_can_progress_without_admitting_a_parameter(self):
        source = binding(lambda request: judgment(request))
        credit = accept(source, ResearchResult(evidence=(evidence(),)))
        self.assertEqual(credit.qualities[0][2], .5)
        self.assertFalse(source.ledger.facts)
        self.assertFalse(source.ledger.assumptions)
        self.assertFalse(source.ledger.estimates)

    def test_each_parent_assesses_its_own_question_not_child_grade(self):
        calls = []
        def reviewer(request):
            calls.append(request)
            status = "resolved" if request["parent_contract"]["decision"] == "Identify the dependency" else "background"
            return judgment(request, status)
        result = ResearchResult(evidence=(evidence("The equation depends on infectiousness."),))
        child = binding(reviewer, question="Identify the dependency")
        parent = binding(reviewer, question="Establish a numeric infectiousness bound")
        self.assertEqual(accept(child, result).qualities[0][2], 1)
        self.assertEqual(accept(parent, result).qualities[0][2], 0)
        self.assertEqual(len(calls), 2)

    def test_duplicate_or_paraphrased_records_do_not_create_new_yield(self):
        calls = []
        def reviewer(request):
            calls.append(request)
            return judgment(request)
        source = binding(reviewer)
        row = evidence()
        controller = ResearchController((FIELD,))
        first = controller.observe("first", accept(source, ResearchResult(evidence=(row,))), is_root=False)
        second = controller.observe("copy", accept(source, ResearchResult(evidence=(
            {**row, "claim": "Paraphrased", "source": "https://example.test/mirror"},))), is_root=False)
        self.assertEqual(first.score, .5)
        self.assertEqual(second.gain, 0)
        self.assertEqual(len(calls), 1)

    def test_inherited_evidence_does_not_earn_acquisition_credit_again(self):
        initial = ResearchResult(evidence=(evidence(),))
        source = binding(lambda _: self.fail("Unchanged inherited evidence needs no review"), inherited=initial)
        self.assertEqual(accept(source, initial).qualities[0][2], 0)

    def test_positive_credit_requires_new_valid_evidence_references(self):
        initial = fixture_ledger().result()
        current = replace(initial, evidence=(evidence(),))
        for refs in ([], ["evidence:999"], ["evidence:1"]):
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                assess_progress(lambda r: judgment(r, refs=refs), contract=None, objective="Bound a value",
                    focus=(FIELD,), initial=initial, current=current, previous=[], context={})

    def test_malformed_assessment_keeps_evidence_and_failure_information(self):
        source = binding(lambda _: {"judgments": [{"field": FIELD}]})
        credit = source.project(ResearchResult(evidence=(evidence(),)))
        self.assertFalse(credit.evaluated)
        self.assertEqual(credit.qualities[0][2], 0)
        self.assertTrue(source.pending.evidence)
        diagnostic = source.pending.diagnostics[-1]
        self.assertEqual(diagnostic["stage"], "mandatory_information_assessment")
        self.assertEqual(diagnostic["requested_fields"], [FIELD])
        self.assertEqual(diagnostic["question"], source.parent_contract.decision)

    def test_contrary_evidence_can_retract_current_information_score(self):
        def reviewer(request):
            return judgment(request, "resolved" if len(request["evidence"]) == 1 else "bounded")
        source = binding(reviewer)
        controller = ResearchController((FIELD,))
        controller.observe("first", accept(source, ResearchResult(evidence=(evidence(),))), is_root=False)
        after = controller.observe("contrary", accept(source, ResearchResult(evidence=(
            evidence("The original bound excludes a relevant activity."),))), is_root=False)
        self.assertEqual(after.gain, -.5)
        self.assertFalse(after.stop)

    def test_uncertainty_learning_does_not_turn_an_unsupported_claim_into_an_input(self):
        ledger = fixture_ledger()
        def review(uncertainty_grade):
            def agent(request):
                result = supported_review(request, grade=1, verdict="unsupported")
                result["criteria"]["support"]["grade"] = 0
                result["criteria"]["support"]["evidence"] = []
                result["criteria"]["uncertainty"]["grade"] = uncertainty_grade
                return result
            return assess_assumption(agent, candidate(), ledger, {})
        first, second = review(1), review(2)
        self.assertEqual(first.assumptions[0]["resolution_credit"], 3 / 8)
        self.assertEqual(second.assumptions[0]["resolution_credit"], 4 / 8)
        self.assertEqual(second.assumptions[0]["reasonableness"], 0)
        self.assertFalse(second.facts)
        ledger.merge(second)
        output = ledger.result().credit(("indoor_reproductive_number",), scientific=True)
        self.assertEqual(output.qualities[0][2], 0)

    def test_yield_resolution_recognizes_one_criterion_improvement(self):
        ledger = fixture_ledger()
        def review(request):
            result = supported_review(request, grade=1, verdict="unsupported")
            result["criteria"]["support"].update(grade=0, evidence=[])
            return result
        ledger.merge(assess_assumption(review, candidate(), ledger, {}))
        initial = ledger.result().credit((FIELD,), scientific=True)
        controller = ResearchController((FIELD,), capacity=1, window=3, initial=initial)
        for n in range(2):
            controller.observe(str(n), initial, is_root=True)
        def narrowed(request):
            response = review(request)
            response["criteria"]["uncertainty"]["grade"] = 2
            return response
        ledger.merge(assess_assumption(narrowed, candidate(), ledger, {}))
        step = controller.observe("bounded uncertainty", ledger.result().credit((FIELD,), scientific=True), is_root=True)
        self.assertEqual(step.gain, 1 / 8)
        self.assertFalse(step.stop)
        self.assertLess(controller.state().rarefaction["score_resolution"], 1 / 8)


class AssumptionRetentionTests(unittest.TestCase):
    def source(self, reviewer):
        run = ResearchRun(ResearchProviders(reviewer, lambda *_: (), lambda _: None), ResearchLimits(),
                          "Synthetic retention test", ("indoor_reproductive_number",))
        source = InvestigationSource(run, fixture_ledger().result())
        run.episode("investigate", source,
            goal=EpisodeGoal.root(objective="Assess the target assumption", result_contract={}), focus=(FIELD,))
        proposal = candidate()
        proposal.pop("evidence")
        source.candidates[source.candidate_key(proposal)] = proposal
        return source

    def step(self, source, result):
        source.staged = Ledger(source.ledger.result())
        credit = source.project(result)
        source.after_verdict(None, None, None)
        return credit

    def test_failed_reassessment_preserves_prior_learning_and_parent_credit(self):
        for error in (ProviderError("Synthetic refusal"), ValueError("Synthetic invalid review"),
                      OSError("Synthetic transport failure")):
            with self.subTest(error=type(error).__name__):
                calls = []
                def reviewer(request):
                    calls.append(request)
                    if len(calls) > 1:
                        raise error
                    return supported_review(request, grade=1, verdict="provisional")
                source = self.source(reviewer)
                first = self.step(source, ResearchResult())
                prior = dict(next(iter(source.ledger.assumptions.values())))
                extra = evidence("Additional methods context, not an invalidation.")
                failed = self.step(source, ResearchResult(evidence=(extra,)))
                self.assertFalse(failed.evaluated)
                self.assertEqual(next(iter(source.ledger.assumptions.values())), prior)
                self.assertIn(extra, list(source.ledger.evidence.values()))
                self.assertIn(extra, list(calls[-1]["evidence"].values()))
                self.assertEqual(failed.qualities, first.qualities)
                returned = source.build_result(SimpleNamespace(ended_by="source_failed", units_consumed=2))
                self.assertTrue(returned.evaluated)
                parent = ResearchController((FIELD,), initial=first)
                step = parent.observe("partial investigation", returned.credit((FIELD,), scientific=True), is_root=True)
                self.assertEqual(step.score, .5)
                self.assertEqual(step.gain, 0)
                diagnostic = returned.diagnostics[-1]
                state = diagnostic["assessment_state"]
                self.assertEqual(state["retained_version"], prior["version"])
                self.assertEqual(state["retained_basis_version"], prior["assessment_basis_version"])
                self.assertNotEqual(state["unassessed_basis_version"], prior["assessment_basis_version"])
                self.assertIn("not reassessed", state["meaning"])
                options = diagnostic_options([diagnostic])
                summary = validate_summary({"findings": [], "uncertainties": [], "failures": [{
                    "references": list(options), "meaning": "Earlier judgment retained; new context is unreviewed.",
                    "not_established": "No new support or contradiction was established."}]}, options)
                self.assertEqual(summary["failures"][0]["basis"][0]["assessment_state"], state)

    def test_operational_limit_does_not_withdraw_completed_assessment(self):
        source = self.source(supported_review)
        self.step(source, ResearchResult())
        prior = dict(next(iter(source.ledger.assumptions.values())))
        failed = self.step(source, ResearchResult(evidence=(evidence("New context awaiting review."),)))
        self.assertFalse(failed.evaluated)
        self.assertEqual(next(iter(source.ledger.assumptions.values())), prior)
        self.assertEqual(source.run.model_calls, 1)
        self.assertEqual(source.ledger.diagnostics[-1]["category"], "operational_limit")

    def test_one_failed_reassessment_does_not_erase_independent_partial_progress(self):
        reviewing_again = False
        def reviewer(request):
            if reviewing_again and request["candidate"]["value"] == 30:
                raise ProviderError("Synthetic failed reassessment")
            return supported_review(request, grade=2 if reviewing_again else 1,
                                    verdict="supported" if reviewing_again else "provisional")
        source = self.source(reviewer)
        first = self.step(source, ResearchResult())
        prior = dict(next(iter(source.ledger.assumptions.values())))
        alternative = candidate(value=25)
        alternative.pop("evidence")
        source.candidates[source.candidate_key(alternative)] = alternative
        reviewing_again = True
        after = self.step(source, ResearchResult(evidence=(evidence("Additional applicable evidence."),)))
        self.assertTrue(after.evaluated)
        self.assertEqual(source.ledger.assumptions[prior["key"]], prior)
        self.assertTrue(any(f.value == 25 and f.kind == "model_assumption" for f in source.ledger.facts.values()))
        parent = ResearchController((FIELD,), initial=first)
        self.assertEqual(parent.observe("partial success", after, is_root=True).gain, .5)

    def test_failed_first_assessment_does_not_create_credit_or_a_judgment(self):
        def reviewer(_):
            raise ProviderError("Synthetic first assessment failure")
        source = self.source(reviewer)
        credit = self.step(source, ResearchResult())
        self.assertFalse(credit.evaluated)
        self.assertEqual(credit.qualities[0][2], 0)
        hypothesis = next(iter(source.ledger.assumptions.values()))
        self.assertEqual(hypothesis["review"]["verdict"], "unassessed")
        self.assertFalse(hypothesis["review"]["completed"])
        self.assertFalse(hypothesis["evidence"])
        self.assertFalse(any(f.kind == "model_assumption" for f in source.ledger.facts.values()))
        self.assertIsNone(source.ledger.diagnostics[-1]["assessment_state"]["retained_version"])
        self.assertTrue(source.ledger.diagnostics[-1]["assessment_state"]["hypothesis_available"])

    def test_later_completed_review_can_correct_prior_credit_downward(self):
        responses = iter(("supported", "fail", "provisional"))
        calls = []
        def reviewer(request):
            calls.append(request)
            response = next(responses)
            if response == "fail":
                raise ProviderError("Synthetic unfinished review")
            return supported_review(request, grade=2 if response == "supported" else 1, verdict=response)
        source = self.source(reviewer)
        first = self.step(source, ResearchResult())
        prior = dict(next(iter(source.ledger.assumptions.values())))
        extra = evidence("Evidence that corrects the earlier applicability claim.")
        self.step(source, ResearchResult(evidence=(extra,)))
        corrected = self.step(source, ResearchResult())
        current = next(iter(source.ledger.assumptions.values()))
        self.assertNotEqual(current["version"], prior["version"])
        self.assertNotEqual(current["assessment_basis_version"], prior["assessment_basis_version"])
        self.assertEqual(calls[-1]["previous_assessments"][0]["version"], prior["version"])
        self.assertIn(extra, list(calls[-1]["evidence"].values()))
        parent = ResearchController((FIELD,), initial=first)
        self.assertEqual(parent.observe("corrected conclusion", corrected, is_root=True).gain, -.5)


class CommunicationTests(unittest.TestCase):
    def test_search_query_contract_matches_literal_provider_execution(self):
        requests, searches = [], []
        query = '"Synthetic methods paper" supplementary material'
        objective = "Determine whether the source provides an applicable bound and explain remaining gaps."
        def agent(request):
            requests.append(request)
            if request["operation"] == "propose_strategy_query":
                return {"query": query, "objective": objective}
            if request["operation"] == "synthesize_research_findings":
                return synthetic_synthesis(request)
            self.fail(request["operation"])
        def search(text, limit):
            searches.append((text, limit))
            return ()
        run = ResearchRun(ResearchProviders(agent, search, lambda _: self.fail("No source was selected")),
                          ResearchLimits(), "Synthetic query contract test", (FIELD,))
        execute_binding(run, StrategySource(run, (), approach="Seek a source-backed bound"), "strategy")
        self.assertEqual(searches, [(query, run.limits.search_limit)])
        proposal = requests[0]
        self.assertIn("Passed unchanged", proposal["search_execution"]["query"])
        self.assertIn("not instructions to an agent", proposal["instruction"])
        self.assertIn("not browsing or extraction instructions", proposal["response_contract"]["properties"]["query"]["description"])
        self.assertEqual(requests[1]["child_objective"], objective)

    def test_exact_unadmitted_claims_reach_every_acquisition_level(self):
        agent, _ = run_synthetic()
        proposed = next(r for r in agent.requests if r["operation"] == "assess_assumption_reasonableness")["candidate"]
        operations = {"propose_strategy_query", "choose_research_source", "propose_page_child",
                      "extract_research_evidence", "review_research_evidence", "assess_information_progress"}
        visited = set()
        for request in agent.requests:
            if request["operation"] not in operations or request.get("source") == "request":
                continue
            visited.add(request["operation"])
            claims = request["context"]["parent_information_contract"]["measurements"]["unadmitted_hypotheses"]
            self.assertIn(proposed, claims)
            self.assertNotIn("model_call_costs", request["context"])
            self.assertIn("critical_distance_m", request["context"]["implemented_model"]["quantities"])
        self.assertEqual(visited, operations)
        query = next(r for r in agent.requests if r["operation"] == "propose_strategy_query")
        self.assertFalse(query["admitted_facts"])
        self.assertFalse(query["assumption_options"])

    def test_failure_synthesis_cannot_erase_unprocessed_useful_information(self):
        diagnostic = {"stage": "retrieval", "category": "provider_failure", "source": "https://example.test/a",
                      "why": "Only an abstract was returned; methods were not accessible"}
        options = diagnostic_options([diagnostic])
        summary = validate_summary({"findings": [], "uncertainties": [], "failures": []}, options)
        self.assertIn("methods were not accessible", json.dumps(summary["failures"]))
        self.assertIn("does not establish a scientific conclusion", summary["failures"][0]["not_established"])
        with self.assertRaises(ValueError):
            validate_summary({"findings": [], "uncertainties": [], "failures": [{
                "references": ["failure:999"], "meaning": "Invented cause", "not_established": "Unknown"}]}, options)

    def test_rolling_summary_replaces_stale_uncertainty_without_deleting_audit(self):
        requests = []
        def agent(request):
            requests.append(request)
            response = synthetic_synthesis(request)
            response["child_summary"].update(findings=["The source bounds the target input."], uncertainties=[])
            response["parent_summary"].update(findings=["Earlier activity mismatch still matters; the new source bounds the input."],
                                              uncertainties=["Applicability to a different activity remains open."])
            return response
        parent = binding(agent, inherited=ResearchResult(findings=("Earlier activity mismatch still matters.",),
                                                       issues=("No source bound is available.",)))
        child = ResearchSource(parent.run, parent.ledger.result())
        goal = EpisodeGoal.child(parent.goal, objective="Find a target bound", result_contract={"fields": [FIELD]})
        episode = parent.run.episode("strategy", child, goal=goal, focus=(FIELD,),
                                     contract=parent.contract("Bound the target input", (FIELD,)))
        result = ResearchResult(evidence=(evidence(),), findings=("The new source supplies a bound.",),
                                issues=("No source bound is available.",))
        report = parent.synthesize(result, EpisodeCompletion("exhausted", "", 1), episode.request)
        parent.ledger.merge(result)
        self.assertEqual(report["unresolved"], [])
        self.assertIn("No source bound is available.", parent.ledger.issues)
        context = parent.communication_context()
        self.assertEqual(context["uncertainties"], ["Applicability to a different activity remains open."])
        self.assertEqual(len(context["findings"]), 1)
        self.assertIn("Earlier activity mismatch", context["findings"][0])
        descendant = ResearchSource(parent.run, parent.ledger.result())
        descendant.inherit_communication(parent)
        self.assertEqual(descendant.communication_context()["uncertainties"], context["uncertainties"])
        self.assertIn("missing scenario parameters", requests[0]["instruction"])

    def test_processed_failure_contains_scope_and_inference_not_repeated_raw_prose(self):
        raw = "Long diagnostic prose. " * 100
        options = diagnostic_options([{"stage": "retrieval", "category": "provider_failure",
            "source": "https://example.test/a", "why": raw}])
        summary = validate_summary({"findings": [], "uncertainties": [], "failures": [{
            "references": list(options), "meaning": "Full methods could not be read; access cause unknown.",
            "not_established": "The parameter's absence from the methods is not established."}]}, options)
        self.assertNotIn(raw, json.dumps(summary))
        self.assertIn("https://example.test/a", json.dumps(summary))
        self.assertIn("not established", summary["failures"][0]["not_established"])

    def test_synthesis_failure_preserves_prior_summary_and_new_unsynthesized_records(self):
        parent = binding(lambda _: {"child_summary": {}})
        parent.decision_summary["findings"] = ["Previously established limitation"]
        child = ResearchSource(parent.run)
        goal = EpisodeGoal.child(parent.goal, objective="Find evidence", result_contract={"fields": [FIELD]})
        episode = parent.run.episode("strategy", child, goal=goal, focus=(FIELD,),
                                     contract=parent.contract("Resolve uncertainty", (FIELD,)))
        result = ResearchResult(findings=("New source conclusion",), issues=("New unresolved question",))
        report = parent.synthesize(result, EpisodeCompletion("exhausted", "", 1), episode.request)
        parent.ledger.merge(result)
        self.assertEqual(report["findings"], ["New source conclusion"])
        self.assertIn("Previously established limitation", parent.communication_context()["findings"])
        self.assertIn("New source conclusion", parent.communication_context()["findings"])
        self.assertTrue(any(d["stage"] == "synthesis" for d in report["diagnostics"]))
