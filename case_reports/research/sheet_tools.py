"""Primitive sheet tools shared by episodes with their own decision ownership."""
from .causal_commission import record_schema, SNAPSHOT_SCHEMA
from .records import ResearchResult
from .importance import evaluate_cases
from .grading import GRADING_FIELDS

def _records_schema(kinds):
    return {"type": "array", "maxItems": 8, "items": {"oneOf": [
        {"type": "object", "additionalProperties": False,
         "required": ["key", "kind", "data", "dependencies"],
         "properties": {"key": {"type": "string", "minLength": 1, "maxLength": 160},
                        "kind": {"const": kind}, "data": record_schema(kind),
                        "dependencies": {"type": "array", "items": {"type": "string"}}}}
        for kind in kinds]}}


SCIENTIFIC_KINDS = ["inference", "premise", "theory", "application", "reasoning", "comparison", "repair", "importance", "source_formalization",
    "expectation", "expectation_break", "expectation_support", "support_contribution", "causal_change", *GRADING_FIELDS]


def _visible_keys(source):
    sheet = source.ledger.causal_sheet
    keys = {key for key, _ in source.catalog_rows()[source.catalog_offset:source.catalog_offset + 20]}
    keys.update(source.selected_keys)
    if source.commission:
        keys.update(source.commission.inputs)
        keys.update(subject for subject, _ in getattr(source.commission, "assignments", ()))
        if getattr(source.commission, "calculation_context", ""):
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


SHEET_TOOLS = ("Use the following scientific sheet tools within your assigned responsibility. "
        "Registering G earns 1 point, recording its expected generalizability bounds earns 1 point, "
        "and registering a theory/context application earns 1 point, each once per scored identity. "
        "Support earns 0.1 per qualifying unit. Executing calculations, extracting source claims or observations, "
        "and writing support arguments earn zero points themselves. Calculation findings earn credit through "
        "support or causal change, whose formula is specified below. Repeating a calculation remains available "
        "and adds no credit for the same finding. "
        "Reference only commissioned inputs, selected sheet inputs, your own admitted records, or records "
        "created earlier in the same batch. The catalog is an index; inspect another catalog key before using it. "
        "Put scientific instructions in question. "
        "reasoning.premises contains record keys, never sentences: put a new assumption's text in "
        "premise.statement and list that premise's key, or list selected evidence keys for source claims. "
        "comparison.left and comparison.right must be two distinct application record keys for the same G; "
        "never use an inference, theory, evidence or reasoning key there. If two applications do not exist, "
        "record a public reasoning argument and its unresolved comparison instead of forcing a comparison record. "
        "A repair references a comparison key; construction_evidence and test_evidence contain only evidence keys. "
        "theory.compared_with contains only theory keys. importance.application contains an application key. "
        "If one proposed record is rejected, earlier and later independently valid records can still be admitted; "
        "read the admitted changes and the rejected record's diagnostic before repairing the batch. "
        "You may reference records you admitted in earlier decisions of this episode. "
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
        "Expressions support numeric literals, named variables, arithmetic + - * /, integer powers [-16,16] written with **, "
        "comparisons, and/or/not, conditional expressions, abs/min/max; ^ is unsupported. Equations are explicit acyclic assignments; "
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
        "A result receipt, including one with an execution error, is final for that immutable protocol and case. "
        "Repeating the same case returns that receipt. A changed expression requires a corrected protocol under a new key. "
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
        evaluation_context = proposal.get("evaluation_context") or getattr(source.commission, "calculation_context", None)
        if evaluation_context and evaluation_context not in visible_keys:
            raise ValueError("Select an explicitly visible historical claim context")
        return source.leaf(lambda: ResearchResult(causal_sheet_updates=tuple(
            evaluate_cases(sheet, assessment, proposal["case_ids"], context=evaluation_context))), "evaluate_importance")
    if action == "calculate":
        keys = proposal["keys"]
        source.authorize_inspection(keys)
        def calculate():
            context = proposal.get("evaluation_context") or getattr(source.commission, "calculation_context", None)
            if context and context not in visible_keys:
                raise ValueError("Select an explicitly visible historical claim context")
            source.last_calculation = sheet.calculation_view(keys, context=context, offset=proposal.get("calculation_offset"), limit=proposal.get("calculation_limit", 8))
            return ResearchResult()
        return source.leaf(calculate, "calculate_scoring_sheet")
    raise ValueError("Unknown sheet tool")


