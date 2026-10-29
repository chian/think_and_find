"""Scientific allocation and reasoning are distinct iterative decision owners."""
from dataclasses import replace
from copy import deepcopy

from ...method_loop import Context, EpisodeGoal
from .acquisition import SearchStrategySource
from .episode_contracts import SearchStrategyCommission
from .causal_commission import SheetCommission, record_schema, read_sheet_result, SNAPSHOT_SCHEMA
from .causal_sheet import CausalSheet
from .engine import ResearchLimits, ResearchProviders, ResearchRun, ResearchSource, selected_choice
from .records import ResearchResult
from .report import ResearchReport
from ...credit_assignment.causal_sheet import CRITERIA, CRITERION_KINDS
from .importance import evaluate_cases
from .grading import GRADING_FIELDS
from .task import RESEARCH_OBJECTIVE

SCIENTIFIC_TASK = RESEARCH_OBJECTIVE + """
Own the global discovery assessment across candidate inferences. Use the current
subscoring sheets to decide where exploration, deeper reasoning, evidence, comparison,
repair assessment, or executable calculation could improve the assessment. You may
introduce, revisit, or set aside candidates as outcomes warrant. Reuse an existing
inference identity for the same claim; renaming a candidate is not new scientific work.
Choose a Search Strategy child for source evidence or a Reasoning child for a specific logical problem.
Use returned source findings, their bearing on G and expected generalizability bounds,
and unresolved questions to formulate the Reasoning commission. Supply selected record keys
and exact historical snapshots; the child receives its own assignment, not your transcript.
When a sourced theoretical statement could support or challenge G, commission its
formalization and comparison where the source permits it. Keep an empirical
measurement as a typed observation rather than inventing a causal equation for it.
For an empty sheet, Search Strategy may start with inputs=[] and snapshots=[].
For an already formalized observation need, commission a support_assessment key among
Search Strategy inputs. Source readers can then match native quantity, units and numeric
conditions and run that fixed comparison immediately. Broad source findings return for interpretation. There is no prescribed sequence or
mandatory reasoning library. Source evidence comes through reading; never invent a
citation or observation. Scientific merit and investigation progress are separate:
useful negative findings can resolve uncertainty. The numerical controller governs
continuation. A recorded candidate or a completed search is not an established example.
"""


def _records_schema(kinds):
    return {"type": "array", "maxItems": 8, "items": {"oneOf": [
        {"type": "object", "additionalProperties": False,
         "required": ["key", "kind", "data", "dependencies"],
         "properties": {"key": {"type": "string", "minLength": 1, "maxLength": 160},
                        "kind": {"const": kind}, "data": record_schema(kind),
                        "dependencies": {"type": "array", "items": {"type": "string"}}}}
        for kind in kinds]}}


COMMISSION_SCHEMA = {"type": "object", "additionalProperties": False,
    "required": ["question", "inputs", "assignments"],
    "properties": {
        "question": {"type": "string", "minLength": 1},
        "inputs": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
        "assignments": {"type": "array", "maxItems": 12, "items": {
            "type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "string"}}},
        "calculation_context": {"type": "string"},
        "snapshots": {"type": "array", "maxItems": 8, "items": SNAPSHOT_SCHEMA}}}


SCIENTIFIC_KINDS = ["inference", "premise", "theory", "application", "reasoning", "comparison", "repair", "importance", "source_formalization",
    "expectation", "expectation_break", "expectation_support", "support_contribution", "causal_change", *GRADING_FIELDS]


def _visible_keys(source):
    sheet = source.ledger.causal_sheet
    keys = {key for key, _ in source.catalog_rows()[source.catalog_offset:source.catalog_offset + 20]}
    keys.update(source.selected_keys)
    if source.commission:
        keys.update(source.commission.inputs)
        keys.update(subject for subject, _ in source.commission.assignments)
        if source.commission.calculation_context:
            keys.add(source.commission.calculation_context)
    keys.update(sheet.inferences_for(source.selected_keys))
    return keys


def _sheet_tool_contract(source, kinds):
    """Schemas of primitive sheet operations; these do not define child roles."""
    sheet = source.ledger.causal_sheet
    visible = _visible_keys(source)
    gradable = [key for key in visible if sheet.rows[key]["kind"] == "importance"]
    contexts = ["", *(key for key in visible if sheet.rows[key]["kind"] in
                     {"expectation_support", "support_contribution", "causal_change", *GRADING_FIELDS})]
    return {"type": "object", "additionalProperties": False,
        "required": ["action", "rationale"], "properties": {
            "action": {"enum": ["record", "inspect", "calculate", "evaluate"]},
            "rationale": {"type": "string", "minLength": 1},
            "records": _records_schema(kinds),
            "keys": {"type": "array", "maxItems": 8, "uniqueItems": True, "items": {"type": "string"}},
            "snapshots": {"type": "array", "maxItems": 8, "items": SNAPSHOT_SCHEMA},
            "assessment": {"enum": gradable} if gradable else False,
            "evaluation_context": {"enum": contexts},
            "case_ids": {"type": "array", "minItems": 1, "maxItems": 8, "uniqueItems": True,
                         "items": {"type": "string"}},
            "calculation_offset": {"type": "integer", "minimum": 0},
            "calculation_limit": {"type": "integer", "minimum": 1, "maximum": 8},
            "catalog_offset": {"type": "integer", "minimum": 0}},
        "allOf": [{"if": {"properties": {"action": {"const": action}}}, "then": {"required": required}}
                  for action, required in (("record", ["records"]), ("inspect", ["keys"]),
                    ("calculate", ["keys"]), ("evaluate", ["assessment", "case_ids"]))]}, visible


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


SHEET_TOOLS = ("Use the following scientific sheet tools within your assigned responsibility. "
        "Only exact catalog keys belong in reference fields; put scientific instructions in question. "
        "Record public scientific derivations rather than a plan to reason later. "
        "Every record must contain exactly its data schema fields; use empty arrays/strings where appropriate. "
        "Dependencies name existing scientific keys or earlier records in the same batch, not prose. "
        "Inspect explicitly selected records when you need detail; catalog pagination is available through catalog_offset. "
        "Inspect accepts snapshots=[{key, version}] for up to eight immutable historical records; keys=[] is allowed. "
        "A different context of the same theory is an application, not a distinct theory. "
        "Each application explicitly names its candidate inference and inference_version (empty selects current G). New hypotheses can reference a commissioned historical G version to support original G after a break.  Preserve distinct candidates and their subsheets; revising one invalidates only assessments depending on it. "
        "Unknown and failed calculations remain unresolved. Define G importance using an immutable importance protocol. "
        "The protocol's circumstance domain fixes a measurement denominator; expectation rows record G's revisable expected generalizability bounds. "
        "inference.domain retains an initial scope hint; bounds-only changes belong in expectation records so they do not change G. "
        "Propose and revise expected generalizability bounds to discover and explain broken expectations, not to certify an absolute domain. "
        "Use literature or reasoning across literature where possible; explicitly provisional judgment is permitted for these bounds only. "
        "As new hypotheses appear, retain the previous expectation, the triggering finding, and the reason for a revision. "
        "Record whether a repair follows through better definitions or a missing dependency within general rules, or introduces a special rule/fudge factor outside them. "
        "Choose equations, a linked pair in an SCM, or discrete simulation/master equations without a required order. "
        "When a sourced theoretical statement bears on G, record source_formalization tied to its evidence and the G importance protocol. "
        "Use the same formal method and named circumstance inputs; give the source-derived equations or rules, focal G realization, "
        "declared G alteration, answer, quantity/units, public derivation and limitations. Do not copy G's model as a source claim "
        "unless the source actually supports that mapping. A source measurement is an observation, not a causal model to invent. "
        "The host compares executed source and G predictions and their common intervention on declared circumstances; "
        "the model cannot assign a match verdict. Conflicts within expected generalizability bounds remain visible. "
        "Use evaluate with one assessment and one to eight predeclared case_ids; the host computes paired answers. "
        "For original G after revision, select evaluation_context as its scoped support, change or executable assessment key. "
        "Use calculate with one to eight scoring-sheet keys to inspect computed results. "
        "The exact_typed_answer comparator compares Boolean or exact rational answers; define a Boolean answer expression for threshold questions. "
        "Expressions support numeric literals, named variables, arithmetic + - * / and integer powers [-16,16], "
        "comparisons, and/or/not, conditional expressions, abs/min/max. Equations are explicit acyclic assignments; "
        "simulation equations update synchronously for steps from initial expressions. No imports, arbitrary code or implicit solvers. "
        "Provide integer or rational/decimal text inputs and positive weights. A sample frequency is never a proof over continuous space. "
        "Use expectation_test to execute one realization of B_G: exact expectation snapshot, immutable scope predicate, expected answer target, "
        "and original-G claim_relation over baseline/altered answers plus the shared claim_intervention. The host checks that the paired answers "
        "realize that intervention. An empty/unrepresentable G relation stays unresolved. "
        "Use support_assessment to connect an expectation_support, its application and importance protocol to that same expectation_test. "
        "Set source_formalization to the sourced formal claim key when counting a theoretical hypothesis; use an empty string "
        "for an observation-only assessment. Ground the formalization in the expectation_support argument and public reasoning. "
        "Supply prediction quantity/units and selected source observations with their declared case IDs; the host calculates dependence and alignment. "
        "Native source conditions must match the numeric circumstances; source uncertainty is reported, never invented as a scoring tolerance. "
        "Use support_contribution.checks to reference these assessments. Text references alone earn no support. Additional aligned observations can "
        "support a dependent application even when the altered prediction also aligns. Source copies and parameter contexts do not mint hypotheses. "
        "Use repair_assessment to compare complete candidate and alternative protocols against the original test, including the target expectation_test. "
        "Keep original circumstances, targets and weights; shrinking bounds must leave excluded failures visible. Separate declared construction and test cases. "
        "The original G relation and intervention are frozen from the source test, not inferred from a new label or different equation text. "
        "Use causal_change.assessment to reference the repair comparison. Numerical change credit uses support for original G and the measured restored "
        "original-scope fraction requiring a G change among the compared preserving alternatives. Empty or unexecuted alternatives leave necessity unresolved. "
        "The component aggregation and restoration scaling convention is exposed in policy_details; do not enter numerical merit grades. "
        "Later support for original G increases that existing change's value; never rediscover its break to claim the gain. "
        "Use calculate for summary results; calculation_offset and calculation_limit retrieve bounded detail pages. "
        "Do not revise G merely to rephrase it or revise its expected bounds; use expectation/support revisions for those changes. "
        "Define scientific operands rather than assigning grades, importance results, or a global merit score.")


def _execute_sheet_tool(source, proposal, visible_keys):
    """Execute a selected primitive tool, independent of decision ownership."""
    sheet = source.ledger.causal_sheet
    action = proposal["action"]
    if action == "record":
        records = proposal.get("records")
        if not isinstance(records, list) or not records or len(records) > 8:
            raise ValueError("Record action requires one to eight scientific records")
        for row in records:
            if row.get("kind") not in source.record_kinds:
                raise ValueError("Record kind is outside this episode's scientific responsibility")
        def record():
            result = source.record_results(records)
            # The next decision sees the actual outcomes, not the entire trace.
            source.selected_keys = list(dict.fromkeys(r["key"] for r in result.causal_sheet_updates))[-2:]
            return result
        return source.leaf(record, "scientific_reasoning")
    if action == "inspect":
        keys = proposal["keys"]
        if not isinstance(keys, list) or len(keys) > 8:
            raise ValueError("Inspect at most eight explicitly selected records")
        source.authorize_inspection(keys)
        sheet.view(keys)
        snapshots = proposal.get("snapshots", [])
        source.authorize_snapshots(snapshots)
        combined = {(ref["key"], ref["version"]): ref for ref in
                    [*(source.commission.snapshots if source.commission else ()), *snapshots]}
        sheet.snapshot_view(list(combined.values()))
        source.selected_keys = keys
        source.selected_snapshots = snapshots
        offset = proposal.get("catalog_offset", source.catalog_offset)
        if type(offset) is not int or offset < 0:
            raise ValueError("catalog_offset must be nonnegative")
        source.catalog_offset = offset
        return source.leaf(lambda: ResearchResult(), "inspect_sheet")
    if action == "evaluate":
        assessment = proposal["assessment"]
        sheet._require_kind(assessment, "importance")
        source.authorize_evaluation(assessment)
        evaluation_context = proposal.get("evaluation_context") or (source.commission.calculation_context if source.commission else None)
        if evaluation_context and evaluation_context not in visible_keys:
            raise ValueError("Select an explicitly visible historical claim context")
        return source.leaf(lambda: ResearchResult(causal_sheet_updates=tuple(
            evaluate_cases(sheet, assessment, proposal["case_ids"], context=evaluation_context))), "evaluate_importance")
    if action == "calculate":
        keys = proposal["keys"]
        source.authorize_inspection(keys)
        def calculate():
            context = proposal.get("evaluation_context") or (source.commission.calculation_context if source.commission else None)
            if context and context not in visible_keys:
                raise ValueError("Select an explicitly visible historical claim context")
            source.last_calculation = sheet.calculation_view(keys, context=context, offset=proposal.get("calculation_offset"), limit=proposal.get("calculation_limit", 8))
            return ResearchResult()
        return source.leaf(calculate, "calculate_scoring_sheet")
    raise ValueError("Unknown sheet tool")


class ScientificSource(ResearchSource):
    instructions = SCIENTIFIC_TASK
    record_kinds = SCIENTIFIC_KINDS
    credit_columns = ("global_goal_progress",)

    def credit(self, result):
        return self.run.global_goal.credit(result)

    def propose(self, view):
        actions = ["record", "inspect", "calculate", "evaluate", "search_strategy", "reasoning"]
        schema, visible = _sheet_tool_contract(self, self.record_kinds)
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
        proposal = self.run.agent(self.inputs(view, operation="propose_scientific_action",
            instruction=SCIENTIFIC_TASK + SHEET_TOOLS, allowed_actions=actions,
            response_contract=schema, calculation_criteria=CRITERIA))
        self.last_proposal = proposal
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
        return list(self.ledger.causal_sheet.rows.items())

    def authorize_inspection(self, keys):
        pass

    def authorize_snapshots(self, snapshots):
        pass

    def authorize_evaluation(self, assessment):
        pass  # The scientific parent owns all admitted protocols.


class ReasoningSource(ResearchSource):
    commission_type = SheetCommission
    instructions = ("Resolve the commissioned logical question through repeated choices of derivation, "
        "countermodel, comparison, repair analysis, independent argument checking, or formal calculation. "
        "Use the supplied source findings and explicit premises. Distinguish conclusions conditional on "
        "premises from empirically supported claims. Return public steps, conclusion, limitations and "
        "unresolved premises so the scientific parent can commission further evidence. Reconsider the "
        "derivation when a calculation or counterexample defeats it. You have sheet tools, not search "
        "children. Model critiques are reasoning claims, never verification receipts or credit gates. "
        "An invalid argument or countermodel can be useful; do not force a positive result.")
    record_kinds = list(SCIENTIFIC_KINDS)
    credit_columns = ("reasoning_scientific_gain", "assigned_calculation_coverage")

    def credit(self, result):
        from .episode_credit import reasoning_credit
        return reasoning_credit(result, self.initial, self.commission)

    def propose(self, view):
        actions = ["record", "inspect", "calculate", "evaluate"]
        schema, visible = _sheet_tool_contract(self, self.record_kinds)
        proposal = self.run.agent(self.inputs(view, operation="propose_reasoning_action",
            instruction=self.instructions + SHEET_TOOLS, allowed_actions=actions,
            response_contract=schema, calculation_criteria=CRITERIA))
        self.last_proposal = proposal
        selected_choice(proposal.get("action"), actions, "action")
        return _execute_sheet_tool(self, proposal, visible)

    def record_results(self, records):
        initial_keys = {r["key"] for r in self.initial.causal_sheet_updates}
        commission = replace(self.commission,
            inputs=tuple(dict.fromkeys([*self.commission.inputs, *self.selected_keys])),
            snapshots=tuple(self.scoped_snapshots()))
        return read_sheet_result(self.ledger.causal_sheet, commission, records,
            actor=self.actor, owned_subjects=self.ledger.causal_sheet.rows.keys() - initial_keys)

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
    source = ScientificSource(run)
    goal = EpisodeGoal.root(objective=research_request,
        result_contract={"result": "Candidate examples assessed through inference-specific theory, generalizability and reversal subsheets",
                         "uncertainty_retained": True}, task_context=run.context)
    episode = run.episode("scientific", source, goal=goal)
    record = episode.run(Context(tree=run.tree))
    return record, ResearchReport.from_result(source.build_result(record), record, run)
