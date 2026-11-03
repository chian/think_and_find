"""Persistent discovery, scientific attempts and reasoning own distinct decisions."""
from dataclasses import replace
from copy import deepcopy
from fractions import Fraction

from ...method_loop import Context, EpisodeGoal
from .acquisition import SearchStrategySource
from .episode_contracts import SearchStrategyCommission, ScientificAttemptCommission
from .causal_commission import SheetCommission, read_sheet_result, SNAPSHOT_SCHEMA
from .causal_sheet import CausalSheet
from .engine import ResearchLimits, ResearchProviders, ResearchRun, ResearchSource, changed_records, selected_choice
from .records import ResearchResult
from .report import ResearchReport
from ...credit_assignment.causal_sheet import CRITERIA, CRITERION_KINDS
from .grading import GRADING_FIELDS
from .sheet_tools import SCIENTIFIC_KINDS, SHEET_TOOLS, _sheet_tool_contract, _execute_sheet_tool
from .task import RESEARCH_OBJECTIVE

DISCOVERY_TASK = ("Launch repeated independent attempts at the generic scientific objective. "
    "Choose a starting question for each attempt from unresolved candidates or a fresh natural-phenomenon direction. "
    "Use the shared scientific sheet and the latest attempt's returned changes to avoid restarting an unchanged, "
    "unproductive question. A previous attempt's yield is not scientific completion. "
    "Do not research or grade at this level; each scientific attempt owns those decisions. "
    "The shared score counts G registration, expected generalizability bounds registration, "
    "theory/context application registration, support for G and causal change. Repeated scored identities add zero.")

SCIENTIFIC_TASK = RESEARCH_OBJECTIVE + """
Own one scientific attempt, starting from the commissioned question, while pursuing the
generic discovery objective. Use the current subscoring sheets to decide where
exploration, deeper reasoning, evidence, comparison,
repair assessment, or executable calculation could improve the assessment. You may
introduce, revisit, or set aside candidates as outcomes warrant. Reuse an existing
inference identity for the same claim; renaming a candidate is not new scientific work.
Choose a Search Strategy child for source evidence or a Reasoning child for a specific logical problem.
Use returned source findings, their bearing on G and expected generalizability bounds,
and unresolved questions to formulate the Reasoning commission. Supply selected record keys
and exact historical snapshots; the child receives its own assignment, not your transcript.
When an executable result reports an error, inspect the receipt and its protocol.
Repeating an unchanged immutable protocol returns the same receipt and earns no new credit. Correct
the formal expression in a newly keyed protocol, then reconnect any source formalization
and scoring operands that depended on the old protocol. If you commission Reasoning for
the repair, include the protocol and the source formalization or other expression-bearing
records the child must inspect, along with the failed receipts.
When a sourced theoretical statement could support or challenge G, commission its
formalization and comparison where the source permits it. Keep an empirical
measurement as a typed observation rather than inventing a causal equation for it.
For an empty sheet, Search Strategy may start with inputs=[] and snapshots=[].
For an already formalized observation need, commission a support_assessment key among
Search Strategy inputs. Readers return evidence and grading needs to Page. Page can select the
fixed observation comparison or record source-grounded scientific operands and invoke graders
before returning to Search. Source findings, grader outcomes and unresolved needs travel upward. There is no prescribed sequence or
mandatory reasoning library. Source evidence comes through reading; never invent a
citation or observation. Scientific merit and investigation progress are separate:
useful negative findings can resolve uncertainty. The numerical controller governs
the end of this attempt. A recorded candidate or a completed search is not an established example.
"""


COMMISSION_SCHEMA = {"type": "object", "additionalProperties": False,
    "required": ["question", "inputs", "assignments"],
    "properties": {
        "question": {"type": "string", "minLength": 1},
        "inputs": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
        "assignments": {"type": "array", "maxItems": 12, "items": {
            "type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "string"}}},
        "calculation_context": {"type": "string"},
        "snapshots": {"type": "array", "maxItems": 8, "items": SNAPSHOT_SCHEMA}}}


def _reasoning_commission_schema(source, visible):
    sheet = source.ledger.causal_sheet
    schema = deepcopy(COMMISSION_SCHEMA)
    schema["properties"]["inputs"]["items"] = {"enum": sorted(visible)} if visible else False
    schema["properties"]["inputs"]["description"] = "Exact existing keys; put the logical task in question."
    schema["properties"]["calculation_context"] = {"enum": ["", *(key for key in visible
        if sheet.rows[key]["kind"] in {"expectation_support", "support_contribution", "causal_change", *GRADING_FIELDS})]}
    pairs = [(key, criterion) for key in sorted(visible) for criterion, kind in CRITERION_KINDS.items()
             if sheet.rows[key]["kind"] == kind]
    schema["properties"]["assignments"]["description"] = "Explicit [sheet_key, calculation_criterion] pairs for the scoped job; [] for exploration with no assigned calculation."
    schema["properties"]["assignments"]["items"] = {"oneOf": [{"type": "array", "minItems": 2, "maxItems": 2,
        "prefixItems": [{"const": key}, {"const": criterion}]} for key, criterion in pairs]} if pairs else False
    return schema


class DiscoverySource(ResearchSource):
    """Keep the shared ledger and select a new start after each completed attempt."""

    instructions = DISCOVERY_TASK
    assessment_fields = {
        "attempt_landscape": "What completed attempts establish about promising fresh starting questions; retain duplication concerns.",
        "open_directions": "Unresolved natural-phenomenon directions worth exploring and why."}
    credit_columns = ("global_goal_progress",)

    def __init__(self, run):
        super().__init__(run, run.initial)
        self.starting_questions = []

    def credit(self, result):
        return self.run.global_goal.credit(result)

    def propose(self, view):
        sheet = self.ledger.causal_sheet
        scientific = [(key, row) for key, row in sheet.rows.items() if row["kind"] != "evidence"]
        index = [{"key": key, "kind": row["kind"], "version": row["version"],
                  "current": sheet.current(key),
                  "label": str(row["data"].get("name", row["data"].get("statement",
                      row["data"].get("bounds", row["data"].get("argument", "")))))[:160]}
                 for key, row in scientific]
        keys = [row["key"] for row in index]
        schema = {"type": "object", "additionalProperties": False,
            "required": ["question", "inputs", "snapshots", "rationale"],
            "properties": {
                "question": {"type": "string", "minLength": 1,
                    "description": "One specific starting scientific question for a fresh attempt; the child still owns the generic objective."},
                "inputs": {"type": "array", "maxItems": 8, "uniqueItems": True,
                    "items": {"enum": keys} if keys else False},
                "snapshots": {"type": "array", "maxItems": 8, "items": SNAPSHOT_SCHEMA},
                "rationale": {"type": "string", "minLength": 1}}}
        proposal = self.decide(self.bounded_inputs(view, {
            "task": DISCOVERY_TASK, "objective": str(view.goal.objective),
            "operation": "propose_discovery_attempt",
            "instruction": DISCOVERY_TASK,
            "candidate_index": index,
            "previous_starts": {"questions": self.starting_questions[-8:],
                                "total": len(self.starting_questions)},
            "global_score": self.run.root_credit_projection(),
            "response_contract": schema}))
        self.last_proposal = proposal
        if proposal.get("action") == "recall":
            return self.recall(proposal)
        commission = ScientificAttemptCommission(proposal["question"],
            proposal["inputs"], proposal["snapshots"])
        commission.request(sheet)
        self.starting_questions.append(commission.question)
        child = ScientificSource(self.run, self.ledger.result())
        return self.edge(self.run.child(view, child, "scientific", commission.question,
            contract=commission.report_contract(), proposal={"starting_question": commission.question,
                                                          "inputs": list(commission.inputs)}))


class ScientificSource(ResearchSource):
    instructions = SCIENTIFIC_TASK
    assessment_fields = {
        "scientific_answer": "Current answer about the candidate causal inference, its role in theories and possible causal changes, with provenance references.",
        "expected_generalizability_bounds": "What was expected, what supports or breaks that expectation and why; preserve revisions.",
        "qualifications": "What remains conditional, disputed or unverified.",
        "next_uncertainties": "Specific evidence or reasoning that could change the answer."}
    commission_type = ScientificAttemptCommission
    record_kinds = SCIENTIFIC_KINDS
    credit_columns = ("attempt_scientific_gain",)

    def credit(self, result):
        from .episode_credit import scientific_attempt_credit
        return scientific_attempt_credit(result, self.initial)

    def parent_report(self, result, completion=None, request=None):
        from .episode_contracts import literature_report
        from ...credit_assignment.causal_progress import compile_episode_points
        report = literature_report(self, result, completion)
        delta = changed_records(result, self.initial)
        changed = [row for row in delta.causal_sheet_updates if row["kind"] != "evidence"]
        after = compile_episode_points(CausalSheet(tuple(result.causal_sheet_updates)))
        before = compile_episode_points(CausalSheet(tuple(self.initial.causal_sheet_updates)))
        component_changes = {name: Fraction(value) - Fraction(before["components"][name])
                             for name, value in after["components"].items()}
        report.update(starting_question=self.commission.question,
            scientific_changes={"selected": [ref for ref in self.working_assessment["record_refs"]
                if self.ledger.causal_sheet.rows[ref["key"]]["kind"] != "evidence"], "total": len(changed)},
            score_change=float(sum(component_changes.values(), Fraction())),
            score_component_changes={name: float(value) for name, value in component_changes.items()})
        return report

    def propose(self, view):
        schema, visible = _sheet_tool_contract(self, self.record_kinds)
        actions = ["record", "inspect", "calculate", "search_strategy", "reasoning"]
        if schema["properties"]["assessment"] is not False:
            actions.append("evaluate")
        schema["properties"]["action"] = {"enum": actions}
        commission_schema = _reasoning_commission_schema(self, visible)
        schema["properties"].update({
            "commission": commission_schema,
            "search_commission": {"type": "object", "additionalProperties": False,
                "required": ["question", "inputs", "snapshots"],
                "properties": {name: deepcopy(commission_schema["properties"][name])
                               for name in ("question", "inputs", "snapshots")}},
            "approach": {"type": "string", "minLength": 1}})
        schema["allOf"].extend([
            {"if": {"properties": {"action": {"const": "search_strategy"}}},
             "then": {"required": ["search_commission", "approach"]}},
            {"if": {"properties": {"action": {"const": "reasoning"}}},
             "then": {"required": ["commission"]}}])
        proposal = self.decide(self.inputs(view, operation="propose_scientific_action",
            instruction=SCIENTIFIC_TASK + SHEET_TOOLS, allowed_actions=actions,
            response_contract=schema, calculation_criteria=CRITERIA))
        self.last_proposal = proposal
        if proposal.get("action") == "recall":
            return self.recall(proposal)
        action = selected_choice(proposal.get("action"), actions, "action")
        if action == "search_strategy":
            commission = SearchStrategyCommission(**proposal["search_commission"])
            commission.request(self.ledger.causal_sheet)
            approach = proposal.get("approach")
            if not isinstance(approach, str) or not approach.strip():
                raise ValueError("Search Strategy needs an initial approach")
            child = SearchStrategySource(self.run, self.ledger.result(), approach=approach)
            return self.edge(self.run.child(view, child, "search_strategy", commission.question,
                contract=commission.report_contract(), proposal={"approach": approach, "question": commission.question}))
        if action == "reasoning":
            commission = SheetCommission(**proposal["commission"])
            commission.request(self.ledger.causal_sheet)
            child = ReasoningSource(self.run, self.ledger.result())
            return self.edge(self.run.child(view, child, "reasoning", commission.question,
                contract=commission.report_contract(), proposal={"question": commission.question}))
        return _execute_sheet_tool(self, proposal, visible)

    def record_results(self, records):
        staged = CausalSheet(self.ledger.causal_sheet.history)
        updates = []
        for row in records:
            data = dict(row["data"])
            if row["kind"] == "reasoning":
                data["author"] = self.actor
            updates.append(staged.put(row["key"], row["kind"], data, dependencies=row["dependencies"]))
        return ResearchResult(causal_sheet_updates=tuple(updates))

    def catalog_rows(self):
        # Put scientific objects before their potentially much larger source
        # evidence catalog, so an attempt can address existing candidates.
        return sorted(self.ledger.causal_sheet.rows.items(),
                      key=lambda item: item[1]["kind"] == "evidence")

    def authorize_inspection(self, keys):
        pass

    def authorize_snapshots(self, snapshots):
        pass

    def authorize_evaluation(self, assessment):
        pass  # The scientific parent owns all admitted protocols.


class ReasoningSource(ResearchSource):
    commission_type = SheetCommission
    assessment_fields = {
        "logical_answer": "Answer to the commissioned derivation, comparison, repair or countermodel question.",
        "premises_and_derivation": "Public essential steps and premises, with exact references to full derivations.",
        "qualifications": "Conditions, counterexamples and limits; executable verification is distinct from an authored claim.",
        "unresolved_premises": "Missing premises or checks that the parent could commission."}
    instructions = ("Resolve the commissioned logical question through repeated choices of derivation, "
        "countermodel, comparison, repair analysis, independent argument checking, or formal calculation. "
        "Selected inputs and historical snapshots in this call contain their complete records. "
        "Recent own-action outcomes identify what prior inspections exposed, what records were admitted, and the credit change. "
        "Use the supplied source findings and explicit premises. Distinguish conclusions conditional on "
        "premises from empirically supported claims. Return public steps, conclusion, limitations and "
        "unresolved premises so the scientific parent can commission further evidence. Reconsider the "
        "derivation when a calculation or counterexample defeats it. You have sheet tools, not search "
        "children. Model critiques are reasoning claims, never verification receipts or credit gates. "
        "An invalid argument or countermodel can be useful; do not force a positive result.")
    record_kinds = list(SCIENTIFIC_KINDS)
    credit_columns = ("reasoning_scientific_gain",)

    def credit(self, result):
        from .episode_credit import reasoning_credit
        return reasoning_credit(result, self.initial, self.commission)

    def propose(self, view):
        schema, visible = _sheet_tool_contract(self, self.record_kinds)
        actions = ["record", "inspect", "calculate"]
        if schema["properties"]["assessment"] is not False:
            actions.append("evaluate")
        schema["properties"]["action"] = {"enum": actions}
        proposal = self.decide(self.inputs(view, operation="propose_reasoning_action",
            instruction=self.instructions + SHEET_TOOLS, allowed_actions=actions,
            response_contract=schema, calculation_criteria=CRITERIA))
        self.last_proposal = proposal
        if proposal.get("action") == "recall":
            return self.recall(proposal)
        selected_choice(proposal.get("action"), actions, "action")
        return _execute_sheet_tool(self, proposal, visible)

    def record_results(self, records):
        initial_keys = {r["key"] for r in self.initial.causal_sheet_updates}
        commission = replace(self.commission,
            inputs=tuple(dict.fromkeys([*self.commission.inputs, *self.selected_keys])),
            snapshots=tuple(self.scoped_snapshots()))
        result = read_sheet_result(self.ledger.causal_sheet, commission, records,
            actor=self.actor, owned_subjects=self.ledger.causal_sheet.rows.keys() - initial_keys)
        return replace(result, diagnostics=tuple(self.run.identify_diagnostic({**diagnostic,
            "model_call": self.run.model_calls}) for diagnostic in result.diagnostics))

    def authorize_evaluation(self, assessment):
        initial_keys = {r["key"] for r in self.initial.causal_sheet_updates}
        assigned = any(subject == assessment or self.ledger.causal_sheet._depends_on(subject, assessment)
                       for subject, _ in self.commission.assignments)
        if not assigned and assessment in initial_keys:
            raise ValueError("Reasoning may evaluate assigned protocols, operands of assigned subsheets, or protocols it introduced")

    def catalog_rows(self):
        sheet = self.ledger.causal_sheet
        initial_keys = {row["key"] for row in self.initial.causal_sheet_updates}
        keys = set(self.commission.inputs) | (sheet.rows.keys() - initial_keys)
        keys.update(subject for subject, _ in self.commission.assignments)
        if self.commission.calculation_context:
            keys.add(self.commission.calculation_context)
        # The catalogue exposes public operands reachable from the commissioned
        # job. Detail is still selected explicitly through bounded inspect calls.
        todo = list(keys)
        while todo:
            operand = todo.pop()
            for dependency in sheet.rows[operand]["dependencies"]:
                if dependency not in keys:
                    keys.add(dependency)
                    todo.append(dependency)
        return [(key, row) for key, row in sheet.rows.items() if key in keys]

    def authorize_inspection(self, keys):
        if not set(keys) <= {key for key, _ in self.catalog_rows()}:
            raise ValueError("Reasoning needs explicitly commissioned inputs or its own records")

    def authorize_snapshots(self, snapshots):
        allowed = {(ref["key"], ref["version"]) for ref in self.commission.snapshots}
        allowed.update((key, row["version"]) for key, row in self.catalog_rows())
        if any((ref["key"], ref["version"]) not in allowed for ref in snapshots):
            raise ValueError("Reasoning needs explicitly commissioned historical versions")

    def parent_report(self, result, completion=None, request=None):
        from .episode_contracts import reasoning_report
        return reasoning_report(self, result, completion)


def run_research(*, research_request=RESEARCH_OBJECTIVE, providers, limits=None):
    if not isinstance(research_request, str) or not research_request.strip():
        raise ValueError("Supply a scientific research request")
    run = ResearchRun(providers, limits or ResearchLimits(), research_request)
    source = DiscoverySource(run)
    goal = EpisodeGoal.root(objective=research_request,
        result_contract={"result": "Candidate examples assessed through inference-specific theory, generalizability and reversal subsheets",
                         "uncertainty_retained": True}, task_context=run.context)
    episode = run.episode("discovery", source, goal=goal)
    record = episode.run(Context(tree=run.tree))
    return record, ResearchReport.from_result(source.build_result(record), record, run)
