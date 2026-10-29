"""Offline expected generalizability bounds demonstration; synthetic, not discoveries.

Uses ordinary sheet records, scoped commissions and atomic result merges. No
agent, provider, unit-test runner or live research episode is started.
"""
import importlib.util
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "episodic_workflow", root / "__init__.py", submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from episodic_workflow.case_reports.research.causal_sheet import CausalSheet
    from episodic_workflow.case_reports.research.causal_commission import SheetCommission, read_sheet_result, record_schema
    from episodic_workflow.case_reports.research.records import Ledger, ResearchResult

    sheet = CausalSheet()
    sheet.put("G", "inference", dict(statement="Increasing input x increases response y",
        variables=["x", "y"], intervention="Increase x while holding model parameters fixed",
        outcome="Direction of change in y", grain="Deterministic scalar response",
        domain="Initial scope hint: positive response inputs; maintained expectations are separate rows"))

    def theory(key, mechanism):
        sheet.put(key, "theory", dict(name=key, mechanism=mechanism,
            predictions=["Model-specific response to increased x"], admission="candidate", same_as="",
            distinction="Synthetic model; substantive theoretical identity is not established", compared_with=[]))

    def application(key, model, context, direction="supports"):
        sheet.put(key, "application", dict(inference="G", theory=model, context_key=key,
            context=context, context_distinction="Explicitly stipulated mathematical circumstances",
            direction=direction, inference_use="Predict the sign of the response to increasing x",
            consequence_without_inference="The sign must instead be derived from the model"))

    theory("linear", "y=x for x>0")
    application("linear_use", "linear", "x increases from 1 to 2; y increases from 1 to 2")
    initial = sheet.put("B", "expectation", dict(inference="G", bounds="Positive inputs in linear response models",
        basis="provisional_judgment", rationale="Provisional extrapolation from the stipulated linear form; no literature claim",
        grounds=[], revision=None))
    initial_ref = sheet.anchor("B")
    original_G = sheet.anchor("G")
    theory("quadratic", "y=x*x for x>0")
    application("quadratic_use", "quadratic", "x increases from 1 to 2; y increases from 1 to 4")

    def revision(prior, triggers, changes, change, repair_kind, proposed_rule, general_rules,
                 rule_reasoning, retained=(), lost=(), predictions=()):
        return dict(prior=prior, triggers=triggers, changes=changes, change=change,
            repair_kind=repair_kind, proposed_rule=proposed_rule, general_rules=general_rules,
            rule_reasoning=rule_reasoning, retained=list(retained), lost=list(lost),
            new_predictions=list(predictions), limitations="Synthetic proposed argument; no host proof or empirical validation")

    def consequence(key, effect):
        return dict(application=sheet.anchor(key), effect=effect)

    def commissioned(key, data, inputs, snapshots=()):
        commission = SheetCommission("Explain this change in expected generalizability bounds and its consequences",
                                    inputs=tuple(inputs), snapshots=tuple(snapshots))
        request = commission.request(sheet)
        result = read_sheet_result(sheet, commission, [dict(key=key, kind="expectation", data=data, dependencies=[])],
                                   actor="synthetic-reasoner")
        sheet.merge(result.causal_sheet_updates)
        return request, sheet.return_page(result.causal_sheet_updates)

    expanded = dict(inference="G", bounds="Provisional expansion to positive-input response models, including nonlinear ones",
        basis="provisional_judgment", rationale="Agreement of linear and quadratic examples motivates a conjecture, not a proof",
        grounds=["linear_use", "quadratic_use"], revision=revision(initial_ref, [sheet.anchor("quadratic_use")],
            ["expand_bounds"], "Expand beyond linear response", "not_a_repair",
            "Conjecture that positive-input response models preserve the positive direction",
            "The linear and quadratic equations; no universal monotonicity rule has been established",
            "Both models give positive changes; extrapolating to other nonlinear forms remains provisional",
            retained=[consequence("linear_use", "Keeps y:1 to 2")],
            predictions=["Other positive-input nonlinear models should also increase y; this can fail"]))
    _, expansion_return = commissioned("B", expanded, ["B", "linear_use", "quadratic_use"])
    expanded_ref = sheet.anchor("B")
    theory("saturating", "y=x-x*x on 0<x<1")
    application("failed_use", "saturating", "x:3/5 to 4/5 gives y:6/25 to 4/25", "possible_reversal")
    sheet.put("counterargument", "reasoning", dict(author="synthetic-author", premises=["failed_use"],
        steps=["At x=3/5, x-x*x=6/25", "At x=4/5, x-x*x=4/25", "Increasing x decreased y"],
        conclusion="The expanded expectation fails for this stipulated response model",
        limitations="Arithmetic argument; not a discovery or established natural-phenomenon causal reversal"))
    break_data = dict(expectation=expanded_ref, triggers=[sheet.anchor("counterargument")],
        expected="Positive inputs were expected to preserve the increasing response direction",
        observed="The stipulated nonlinear response decreases from 6/25 to 4/25",
        conflict="This application fits the stated positive-input expectation but contradicts G",
        limitations="Counterexample to a synthetic conjecture; empirical relevance remains open")
    break_commission = SheetCommission("Record the failed expectation without deciding how to repair it",
                                       inputs=("B", "counterargument"))
    break_result = read_sheet_result(sheet, break_commission,
        [dict(key="break", kind="expectation_break", data=break_data, dependencies=[])], actor="synthetic-reader")
    sheet.merge(break_result.causal_sheet_updates)
    unreviewed_break = sheet.return_page(break_result.causal_sheet_updates)

    rule_revision = revision(expanded_ref, [sheet.anchor("break"), sheet.anchor("counterargument")],
        ["shrink_bounds", "repair_theory"], "Propose a monotonicity condition within the differentiable response rules",
        "missing_dependency", "For differentiable y=f(x), restrict the increasing claim to intervals where f'(x)>0",
        "Derivative sign and the mean value theorem over differentiable intervals",
        "For x-x*x, f'(x)=1-2*x; positive direction holds below 1/2 and reverses above 1/2. The boundary follows the general rule",
        retained=[consequence("linear_use", "Derivative 1 keeps the earlier positive response"),
                  consequence("quadratic_use", "Derivative 2*x stays positive at positive inputs")],
        lost=[consequence("failed_use", "The original increasing claim no longer covers x>1/2; its failed prediction remains recorded")],
        predictions=["For the saturating model, increases confined below 1/2 increase y; above 1/2 they decrease y"])
    rule_data = dict(inference="G", bounds="Differentiable response intervals with positive derivative throughout",
        basis="provisional_judgment", rationale="Proposed general-rule repair to assess; derivative condition has not been independently reviewed",
        grounds=[], revision=rule_revision)
    _, rule_return = commissioned("B", rule_data, ["B", "break", "counterargument", "linear_use", "quadratic_use", "failed_use"])
    repaired_ref = sheet.anchor("B")
    patch_revision = revision(expanded_ref, [sheet.anchor("break")], ["repair_theory"],
        "Explore a competing exceptional patch from the same failed expectation", "exceptional_patch",
        "Replace x-x*x with x-x*x+2*x only for the named saturating model",
        "The original model equations contain no rule selecting this model-specific extra term",
        "The added term makes the derivative 3-2*x positive on (0,1), but its coefficient and selection lack an independent derivation",
        retained=[consequence("linear_use", "Unchanged linear equation preserves its prediction"),
                  consequence("quadratic_use", "Unchanged quadratic equation preserves its prediction")],
        lost=[consequence("failed_use", "The original saturating model's y values are discarded, rather than explained")],
        predictions=["The patched response at x=3/5 is 36/25 and at x=4/5 is 44/25; these are untested new predictions"])
    patch_data = dict(inference="G", bounds=expanded["bounds"], basis="provisional_judgment",
        rationale="Competing patch proposal, not an accepted reconciliation", grounds=[], revision=patch_revision)
    patch_request, patch_return = commissioned("B_patch", patch_data,
        ["break", "linear_use", "quadratic_use", "failed_use"], snapshots=[expanded_ref])

    # Updating G invalidates live assessments, but the historical failure and its
    # triggering argument remain addressable without becoming a dependency cycle.
    sheet.put("G", "inference", {**sheet.rows["G"]["data"],
        "statement": "For y=f(x), the response direction follows derivative sign on differentiable intervals"})
    stale_after_G = {key: sheet.current(key) for key in ("B", "B_patch", "failed_use", "break")}
    reconsidered = {**rule_data, "bounds": "Differentiable intervals with a defined derivative sign",
        "revision": revision(repaired_ref, [sheet.anchor("counterargument"), sheet.anchor("break")],
            ["change_inference", "expand_bounds"], "Revise G itself to predict the response direction rather than universally assert increase",
            "general_rule", "Use derivative sign to predict either increasing or decreasing response",
            "The derivative sign rule already proposed in the preceding revision",
            "Generalizing the conditional directional claim restores coverage of the descending interval without restoring the original false increasing prediction",
            retained=[consequence("linear_use", "Keeps positive response prediction"),
                      consequence("quadratic_use", "Keeps positive response prediction")],
            lost=[consequence("failed_use", "Original positive prediction remains false; the revised G predicts negative response")],
            predictions=["The saturating model changes response direction at x=1/2"])}
    _, reconsidered_return = commissioned("B", reconsidered,
        ["B", "counterargument", "break", "linear_use", "quadratic_use", "failed_use"])

    support_data = dict(inference=original_G, expectation=expanded_ref,
        grounds=["linear_use", "quadratic_use"],
        argument="The original increasing claim still explains the linear and positive-input quadratic applications",
        limitations="Support for G as it was; not certainty or a probability of universal truth")
    support_commission = SheetCommission("Assess continued support for the original claim after its break",
        inputs=("linear_use", "quadratic_use"), snapshots=(original_G, expanded_ref))
    support_result = read_sheet_result(sheet, support_commission,
        [dict(key="original_support", kind="expectation_support", data=support_data, dependencies=[])],
        actor="synthetic-reader")
    sheet.merge(support_result.causal_sheet_updates)
    first_support = sheet.anchor("original_support")
    supported_after_change = sheet.usable("original_support")
    if not supported_after_change:
        raise RuntimeError("A newer G version incorrectly erased support for the original G")
    theory("cubic", "y=x*x*x for x>0")
    sheet.put("later_argument", "reasoning", dict(author="synthetic-author", premises=["cubic"],
        steps=["For x>0 the derivative of x*x*x is 3*x*x>0"],
        conclusion="This later model also supports the original increasing claim over positive inputs",
        limitations="Stipulated model, not a literature-grounded hypothesis or universal proof"))
    sheet.put("original_support", "expectation_support", {**support_data,
        "grounds": [*support_data["grounds"], "later_argument"],
        "argument": support_data["argument"] + "; the later cubic model supplies another aligned argument"})
    later_support = sheet.return_page([sheet.rows["original_support"]])
    break_identity_count = len(sheet.of_kind("expectation_break"))
    sheet.put("linear", "theory", {**sheet.rows["linear"]["data"],
        "mechanism": "The original linear response premise has been withdrawn for this model"})
    after_ground_retraction = sheet.usable("original_support")
    if after_ground_retraction:
        raise RuntimeError("Historical G pin incorrectly protected retracted scientific grounds")

    rejected = {}
    def reject(name, operation):
        try:
            operation()
        except ValueError as exc:
            rejected[name] = str(exc)
        else:
            raise RuntimeError(f"Demonstration expected rejection: {name}")
    reject("silent_rewrite", lambda: sheet.put("B", "expectation", {**reconsidered, "revision": None}))
    reject("unknown_snapshot", lambda: sheet.snapshot_view([dict(key="B", version="invented")]))
    reject("label_without_rule", lambda: sheet.put("B_bad", "expectation",
        {**patch_data, "revision": {**patch_revision, "changes": ["change_inference"], "proposed_rule": ""}}))
    unscoped = SheetCommission("Current inputs do not authorize hidden historical input", inputs=("B", "break"))
    reject("uncommissioned_history", lambda: read_sheet_result(sheet, unscoped,
        [dict(key="hidden", kind="expectation", dependencies=[], data={**patch_data,
            "revision": {**patch_revision, "changes": ["change_inference"], "retained": [], "lost": []}})], actor="synthetic-reader"))
    before_atomic = len(sheet.history)
    candidate = sheet.propose("temporary", "premise", dict(statement="Temporary", scope="Demo", basis="Synthetic"))
    reject("atomic_bad_delta", lambda: sheet.merge([candidate, {"malformed": True}]))
    if len(sheet.history) != before_atomic or "temporary" in sheet.rows:
        raise RuntimeError("Atomic merge failed")
    # Replay through the actual result ledger consumer, with every audit reference.
    replayed = Ledger(ResearchResult(causal_sheet_updates=tuple(sheet.history))).causal_sheet
    if replayed.rows != sheet.rows or replayed.snapshot(initial_ref) != initial:
        raise RuntimeError("Ledger replay lost current state or historical expectation")
    # Import the full action producer, and check the model-facing schema is present.
    from episodic_workflow.case_reports.research.workflow import COMMISSION_SCHEMA
    if "snapshots" not in COMMISSION_SCHEMA["properties"] or not record_schema("expectation"):
        raise RuntimeError("Missing bounded action contracts")
    print(json.dumps({"example": "SYNTHETIC definitions and proposals, not scientific discoveries",
        "initial": sheet.snapshot_view([initial_ref]), "expansion": expansion_return,
        "failure_before_revision": unreviewed_break,
        "general_rule_proposal": rule_return, "exceptional_patch_proposal": patch_return,
        "competing_proposal_historical_inputs": patch_request["historical_inputs"],
        "live_currentness_after_G_changed": stale_after_G,
        "reconsidered_G": reconsidered_return, "reconsidered_bounds_current": sheet.current("B"),
        "support_for_original_G_after_revision": supported_after_change,
        "later_support_same_identity": later_support,
        "earlier_support_retrievable": sheet.snapshot_view([first_support]),
        "break_identities_after_later_support": break_identity_count,
        "support_usable_after_ground_retraction": after_ground_retraction,
        "old_expectation_and_failure_retained": sheet.snapshot_view([expanded_ref, sheet.anchor("break")]),
        "rejections": rejected, "atomic_merge_preserved": True, "ledger_replay_preserved": True,
        "limitations": "Host validates fields, links, versioning and scope; scientific rule validity, bound membership, repair classification and empirical support need reasoned review. No ordinal scores or global aggregate."}, indent=2))


if __name__ == "__main__":
    main()
