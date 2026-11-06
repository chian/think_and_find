"""Reconstruct a theory, pursue a substantive gap, and repair typed output.

Each source uses the common Episode loop. Literature acquisition and reasoning
remain children with their own decisions and request-relative returns.
"""
from copy import deepcopy

from .acquisition import LiteratureSource, SearchStrategySource, PageSource
from .causal_sheet import CausalSheet
from .causal_commission import SheetCommission, SNAPSHOT_SCHEMA
from .engine import ResearchSource, selected_choice
from .episode_contracts import (TheoryReconstructionCommission, TheoryGapCommission,
    TheoryFormatRepairCommission, SearchStrategyCommission, PageCommission)
from .records import ResearchResult
from .theory_runtime import (theory_context, theory_credit, theory_report, integration_schema,
    integrate_theory, request_schema, validate_request)


RECONSTRUCTION_TASK = """Reconstruct the commissioned theory as a connected scientific account.
Own the evolving theory fragment and decide what to extract, connect, check or investigate next.
The parent has supplied an initial vocabulary and a scientific question, not an equation to fill in.
Discover entities and defined properties, variable/parameter distinctions, relationships,
mechanisms, assumptions, and the derivations or approximations connecting them. Use the supplied
closed language and exact source evidence references. Retain native scope and conditions.
Preserve the parent's existing quantity meanings. Introduce additional source-defined quantities
explicitly; never silently rename a quantity or change its role. New IDs do not establish novelty.
Source-grounded claims remain scientific assertions; type validation is not proof of source fidelity.
Mathematical normalization compares represented expressions only within its reported capabilities.
Use integrate_theory for typed additions. Existing components are immutable; a correction gets a
new ID with an explicit supersedes link. Reference existing components to retain explanatory
dependencies. Record a substantive gap from an actual missing, ambiguous or conflicting connection.
A gap may be resolved by evidence that a theory intentionally leaves a relationship unspecified.
Choose search_strategy for broad reconstruction or a concentrated literature question, read_source
to revisit an already identified paper, repair_gap to delegate investigation of recorded gaps, and
reasoning for an explicit derivation or check using supplied premises. Returned literature must
extend this same fragment or report what remains unresolved. No fixed action sequence is required.
Local yield counts new source-grounded requested contributions with duplicate identities removed;
it is independent of global G credit. The numerical estimator owns continuation.
"""

GAP_TASK = """Resolve the commissioned substantive gaps in an existing theory reconstruction.
The typed request identifies the actual components and questions that generated this investigation.
Decide whether a targeted reread, a search through other literature, or reasoning can clarify them.
Use the shared definitions and exact source references. Integrate new relationships, assumptions,
scope qualifications, derivations, or explicit statements of what the theory leaves unspecified.
Return a new gap component superseding the old one when reporting sourced resolutions; its
resolutions must reference the actual returned components. Keep remaining questions explicit.
Missing evidence is not a negative finding. A validator error is an output-repair problem, not a
scientific gap. Do not invent a functional form to fill an unspecified relationship. This episode
owns repeated choices about its assigned gaps; Search Strategy owns query choices within each
literature commission. Local credit measures request contributions, not global G points.
"""

FORMAT_REPAIR_TASK = """Correct the rejected theory additions using the exact validator feedback,
the supplied fragment, the closed language and the supplied source evidence. Correct declarations,
references or constructor types; preserve the proposed scientific content and valid components.
Do not invent source evidence or replace a substantive uncertainty with a made-up equation.
Return a complete corrected additions object. Validation will run again and return its exact
errors. The repair has its own numerical continuation estimator; unresolved errors return to its
parent if it yields. Successful type validation does not establish scientific truth.
"""


def reconstruction_commission_schema():
    from .theory_language import fragment_schema
    return {"type": "object", "additionalProperties": False,
        "required": ["question", "inputs", "snapshots", "theory_request", "seed"],
        "properties": {
            "question": {"type": "string", "minLength": 1},
            "inputs": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
            "snapshots": {"type": "array", "items": SNAPSHOT_SCHEMA},
            "theory_request": request_schema(),
            "seed": {"anyOf": [{"type": "null"}, fragment_schema()],
                     "description": "For a new reconstruction, declare its identity, original question and initial vocabulary; empty maps are permitted. For an existing fragment use null."}}}


def _work_schema(source, *, gap_child):
    text = {"type": "string", "minLength": 1}
    common = {"question": text, "theory_request": request_schema(), "rationale": text}
    search = {"type": "object", "additionalProperties": False,
        "required": ["action", "question", "theory_request", "approach", "rationale"],
        "properties": {**common, "action": {"const": "search_strategy"}, "approach": text}}
    read = {"type": "object", "additionalProperties": False,
        "required": ["action", "question", "theory_request", "source", "rationale"],
        "properties": {**common, "action": {"const": "read_source"}, "source": text}}
    reasoning = {"type": "object", "additionalProperties": False,
        "required": ["action", "question", "inputs", "rationale"], "properties": {
            "action": {"const": "reasoning"}, "question": text, "rationale": text,
            "inputs": {"type": "array", "uniqueItems": True, "items": text}}}
    branches = [integration_schema(), search, read, reasoning]
    if gap_child:
        branches.append({"type": "object", "additionalProperties": False,
            "required": ["action", "question", "theory_request", "rationale"],
            "properties": {**common, "action": {"const": "repair_gap"}}})
    return {"oneOf": branches}


def _commission(source, cls, proposal):
    request = validate_request(proposal["theory_request"], source.ledger.causal_sheet)
    if request["reconstruction"] != source.commission.theory_request["reconstruction"]:
        raise ValueError("Child literature work must address this same theory reconstruction")
    inputs = tuple(dict.fromkeys([*source.commission.inputs, *source.selected_keys]))
    commission = cls(question=proposal["question"], inputs=inputs,
        snapshots=tuple(source.scoped_snapshots()), theory_request=request)
    commission.request(source.ledger.causal_sheet)
    return commission


def _propose_work(source, view, *, gap_child):
    context = theory_context(source)
    sheet = source.ledger.causal_sheet
    known_sources = {data["source"] for _, row in sheet.of_kind("evidence") for data in [row["data"]]}
    proposal = source.decide(source.inputs(view, operation=("propose_theory_reconstruction"
        if gap_child else "propose_theory_gap_repair"), instruction=source.instructions,
        **context, known_sources=sorted(known_sources),
        available_reasoning_inputs=[{"key": key, "kind": row["kind"], "version": row["version"]}
                                    for key, row in sheet.rows.items()],
        response_contract=_work_schema(source, gap_child=gap_child)))
    source.last_proposal = proposal
    action = selected_choice(proposal.get("action"),
        ["integrate_theory", "search_strategy", "read_source", "reasoning", "recall",
         *(["repair_gap"] if gap_child else [])], "action")
    if action == "recall":
        return source.recall(proposal)
    if action == "integrate_theory":
        return source.leaf(lambda: integrate_theory(source, proposal["additions"]), "integrate_theory")
    if action == "reasoning":
        from .workflow import ReasoningSource
        commission = SheetCommission(proposal["question"], tuple(proposal["inputs"]))
        commission.request(sheet)
        child = ReasoningSource(source.run, source.ledger.result())
        return source.edge(source.run.child(view, child, "reasoning", commission.question,
            contract=commission.report_contract(), proposal=proposal))
    if action == "repair_gap":
        commission = _commission(source, TheoryGapCommission, proposal)
        fragment = context["theory_fragment"]
        if not commission.theory_request["focus"] or not any(
                key in fragment["gaps"] for key in commission.theory_request["focus"]):
            raise ValueError("A gap repair must identify an actual recorded reconstruction gap")
        child, grain = TheoryGapSource(source.run, source.ledger.result()), "theory_gap"
    elif action == "read_source":
        commission = _commission(source, PageCommission, proposal)
        url = proposal["source"]
        if url not in known_sources and url not in source.run.documents:
            raise ValueError("A focused reread needs an identified source; use Search Strategy to discover sources")
        child = PageSource(source.run, source.ledger.result(), candidate={"url": url})
        grain = "page"
    else:
        commission = _commission(source, SearchStrategyCommission, proposal)
        child = SearchStrategySource(source.run, source.ledger.result(), approach=proposal["approach"])
        grain = "search_strategy"
    return source.edge(source.run.child(view, child, grain, commission.question,
        contract=commission.report_contract(), proposal=proposal))


class TheoryReconstructionSource(LiteratureSource):
    commission_type = TheoryReconstructionCommission
    instructions = RECONSTRUCTION_TASK
    assessment_fields = {
        "theory_account": "How the extracted entities, mechanisms and relationships explain the commissioned phenomenon.",
        "assumptions_and_approximations": "Which source-grounded assumptions and approximations connect levels of description.",
        "qualifications": "Disagreements, scope qualifications and limits of the reconstruction.",
        "remaining_gaps": "Specific unresolved dependencies and which investigation could resolve them."}
    credit_columns = ("reconstruction_contribution",)

    def credit(self, result):
        return theory_credit(result, self.initial, self.commission.theory_request, self.credit_columns[0])

    def propose(self, view):
        key = self.commission.theory_request["reconstruction"]
        if key not in self.ledger.causal_sheet.rows:
            def initialize():
                self.commission.request(self.ledger.causal_sheet)
                sheet = CausalSheet(self.ledger.causal_sheet.history)
                return ResearchResult(causal_sheet_updates=(sheet.put(key, "theory_fragment", self.commission.seed),))
            return self.leaf(initialize, "declare_theory_vocabulary")
        return _propose_work(self, view, gap_child=True)

    def parent_report(self, result, completion=None, request=None):
        report = super().parent_report(result, completion, request)
        report.update(theory_report(self, result))
        return report


class TheoryGapSource(LiteratureSource):
    commission_type = TheoryGapCommission
    instructions = GAP_TASK
    assessment_fields = {
        "gap_answer": "What returned literature and reasoning establish about the commissioned missing connection.",
        "qualifications": "Scope, conflicts and assumptions of the proposed resolution.",
        "remaining_gaps": "Unresolved questions, including relationships intentionally left unspecified."}
    credit_columns = ("gap_contribution",)

    def credit(self, result):
        return theory_credit(result, self.initial, self.commission.theory_request, self.credit_columns[0])

    def propose(self, view):
        return _propose_work(self, view, gap_child=False)

    def parent_report(self, result, completion=None, request=None):
        report = super().parent_report(result, completion, request)
        report.update(theory_report(self, result))
        return report


class TheoryFormatRepairSource(ResearchSource):
    commission_type = TheoryFormatRepairCommission
    instructions = FORMAT_REPAIR_TASK
    assessment_fields = {"repair_answer": "Which exact output errors have been corrected and what remains unresolved."}
    credit_columns = ("format_repair_contribution",)

    def __init__(self, run, inherited, *, rejected):
        super().__init__(run, inherited)
        self.theory_repair = deepcopy(rejected)
        self.repaired = False

    def credit(self, result):
        return theory_credit(result, self.initial, self.commission.theory_request, self.credit_columns[0])

    def propose(self, view):
        if self.repaired:
            return None
        def correct():
            from .theory_language import additions_schema
            from ...task_executors.evidence_extraction.argo import ModelOutputError
            prompt = {"operation": "repair_theory_output", "instruction": self.instructions,
                "question": self.commission.question, **theory_context(self),
                "rejected_additions": self.theory_repair["additions"],
                "validator_errors": self.theory_repair["errors"],
                "response_contract": additions_schema()}
            if "raw_response" in self.theory_repair:
                prompt["rejected_response"] = self.theory_repair["raw_response"]
            try:
                response = self.run.complete(prompt)
            except ModelOutputError as exc:
                self.theory_repair = {**self.theory_repair,
                    "raw_response": exc.raw_response, "errors": [str(exc)]}
                return self.failure_result(exc, "theory_output_repair")
            result = integrate_theory(self, response)
            self.repaired = self.theory_repair is None
            self.working_assessment["repair_answer"] = ("Typed additions accepted by the validator."
                if self.repaired else "; ".join(self.theory_repair["errors"]))
            return result
        return self.leaf(correct, "correct_theory_output")

    def parent_report(self, result, completion=None, request=None):
        report = super().parent_report(result, completion, request)
        report.update(theory_report(self, result))
        report["repair_status"] = {"repaired": self.repaired, "repair": deepcopy(self.theory_repair)}
        return report


def repair_theory_unit(source, view):
    source_records = theory_context(source)["source_evidence"]
    commission = TheoryFormatRepairCommission(question=source.commission.question,
        inputs=tuple(dict.fromkeys([*source.commission.inputs, *source.selected_keys,
                                   *(row["key"] for row in source_records)])),
        snapshots=tuple(source.scoped_snapshots()),
        theory_request=source.commission.theory_request)
    child = TheoryFormatRepairSource(source.run, source.ledger.result(), rejected=source.theory_repair)
    return source.edge(source.run.child(view, child, "theory_format_repair", commission.question,
        contract=commission.report_contract(), proposal={"operation": "repair_theory_output"}))
