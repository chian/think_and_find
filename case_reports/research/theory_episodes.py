"""Scientific's iterative reconstruction, repair and comparison children.

Information needs become explicit goals returned through Scientific and Root.
These episodes use scientific records and formal tools, never acquisition loops.
"""
from copy import deepcopy
from fractions import Fraction

from .causal_sheet import CausalSheet
from .causal_commission import SheetCommission, SNAPSHOT_SCHEMA, read_sheet_result
from .engine import ResearchSource, selected_choice
from .episode_contracts import (TheoryReconstructionCommission, TheoryRepairCommission,
    TheoryComparisonCommission, TheoryFormatRepairCommission)
from .episode_memory import job_report
from .records import Credit, Ledger, ResearchResult
from .sheet_tools import SCIENTIFIC_KINDS, SHEET_TOOLS, _sheet_tool_contract, _execute_sheet_tool
from .theory_runtime import (theory_context, theory_credit, theory_report, integration_schema,
    integrate_theory, request_schema)


SCIENTIFIC_CHILD_INSTRUCTIONS = """
Own the commissioned scientific question through repeated decisions. Supplied
records contain source evidence and scientific definitions. State public
premises, derivations, conclusions and limitations. A theory can contain many
causal assertions: register relevant Gs with defined quantities, interventions,
outcomes and grain, retaining their connection to the theory. Assumptions can
legitimately define conditions behind expected generalizability bounds. Do not
force every assertion into an executable equation or manufacture a replacement.
Record an examination after public source-grounded reasoning has addressed a
selected G and theoretical inputs. Reference its actual reasoning and grounds;
supported, challenged and unresolved conclusions can all record a first
examination. A plan to reason later is not a completed scientific finding.

Use scientific-object lookup for existing theories, Gs, bounds, comparisons,
repairs and formal results. Choose Lean for a particular equivalence question
between formal assertions under shared definitions and explicit assumptions.
Its checked result or unresolved obligation informs your next decision; failure
to establish equivalence does not establish inequivalence. Choose SCM/equation/
simulation calculations for specified interventions or changed relationships
whose consequences bear on your question. Both tools must reference the actual
scientific objects and conditions under examination. They are available methods,
not compulsory stages. Formal consequences do not establish empirical premises;
calling a tool or writing a verdict supplies no numerical merit grade.

Use causal_change to define an explicit task comparison and compare_tasks to execute its
cases when a scoped alteration or applicability difference is ready to assess. Keep G and
the original expected generalizability bounds pinned; extensions beyond those bounds can
earn credit. Record original, candidate and reference models with scientific source bindings,
the task quantity/units/accuracy requirements and public reasoning. The host calculates
gradual improvement from task errors without an original-support multiplier, a reversal
requirement or a prerequisite to exhaust competing repairs. Keep successful tasks, losses,
nonrepresented events and unresolved outcomes separate. Numerical ODE event comparison is
available alongside Lean and SCM; it is not a mandatory episode or empirical validation.

Use create_discovery_goal for missing evidence: state the pointed question,
scientific definitions, acceptable answer form, source requirements and
executable typed requirements. Use native requirements for literal source-field
matches only. Mechanistic content belongs in a scientific requirement: its
theory_structure predicate checks the source-grounded component and its typed
reference closure. A source candidate is retrieval information, not fulfillment
of that scientific requirement. Equivalence and G dependence still require
their actual Lean or SCM checks. Scientific returns that goal through Root to
Discovery. Do not search for literature, infer missing measurements or convert
unavailable evidence into absence. Useful partial findings and negative results
remain findings. The numerical iterator owns continuation.
"""

RECONSTRUCTION_TASK = """Reconstruct the commissioned theory from supplied text and evidence.
Repeatedly choose which source-defined entity, quantity, relationship, mechanism,
assumption or derivation to formalize or check next. The commission identifies
an explanatory account, not an equation-shaped blank. Preserve native scope and
qualifications and the supplied meanings of shared quantities. Distinguish
variables, parameters and literal constants. Use integrate_theory for connected
typed declarations and record for scientific Gs, premises, theory identities and
applications. Multiple Gs can constitute a theory; new IDs alone are not novel
science. Existing components cannot be overwritten: corrections explicitly
supersede earlier components while retaining their provenance. Record missing
connections as gaps and create pointed information goals. Reconstruct usable
accounts with their limitations; historical-source completeness is not an
independent task unless commissioned.
""" + SCIENTIFIC_CHILD_INSTRUCTIONS

REPAIR_TASK = """Develop and examine repairs to the commissioned scientific account.
Repeatedly choose a concrete modification, retained success, affected prediction
or additional consequence to examine. Begin with the identified failure,
original expected generalizability bounds, source targets and premises. A repair
may improve definitions or add a dependency while preserving G, or may omit,
aggregate or reformulate G. Keep these outcomes distinct without forcing a sign
reversal or denying a mechanism's existence. Use integrate_theory for explicit
amended components and record for repair arguments and executable comparisons.
Formal tool results should inform which candidate or premise to reconsider.
Retain original failures, successful G-preserving alternatives, fitting
assumptions, prediction losses and information costs. Accommodation and
independent corroboration are different findings. Scope failed attempts to the
actual alternatives and evidence; no result proves all future repairs impossible.
This task repairs scientific explanations, not malformed JSON.
""" + SCIENTIFIC_CHILD_INSTRUCTIONS

COMPARISON_TASK = """Compare the supplied theories and their uses of commissioned Gs.
Repeatedly choose which connection, transfer, condition or consequence needs
comparison. Look up relevant accounts, Gs, definitions, assumptions, expected
generalizability bounds, repairs and versions. Explain their correspondence
while preserving native quantities, scale, framing and success criteria.
Identical tasks or opposite signs on identical observables are not required.
Define the actual equivalence or causal-consequence question before choosing
Lean or SCM tools. Checked equivalence can establish that forms express the
same declared relation under their assumptions; renaming or a failed proof
attempt cannot establish an alteration of G. Separate formal predictions,
observed outcomes and source-to-model fidelity. Record what changed, why a
transfer succeeded or failed, and what examined repairs retain or lose. Use
executable support/change operands where they represent the actual question;
zero or unresolved numerical change does not refute a useful explained failure
or successful G-preserving repair. Return missing-information goals to Scientific.
""" + SCIENTIFIC_CHILD_INSTRUCTIONS

FORMAT_REPAIR_TASK = """Correct rejected theory additions using exact validator feedback,
the supplied fragment, closed language and evidence. Correct constructor types,
declarations and references while retaining valid components and proposed
scientific content. Do not invent evidence or replace uncertainty with a made-up
equation. Return corrected additions with unresolved errors explicit. Validation
establishes representation consistency, not scientific truth. The ordinary
numerical iterator owns repeated correction.
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
                "description": "Declare a new reconstruction's identity, original question and vocabulary. Use null for an existing fragment."}}}


def repair_commission_schema():
    schema = reconstruction_commission_schema()
    schema["properties"]["seed"]["description"] = (
        "Use null to amend an existing theory. A separate candidate reconstruction "
        "may declare its own explicit seed and input references.")
    return schema


def comparison_commission_schema():
    text = {"type": "string", "minLength": 1}
    return {"type": "object", "additionalProperties": False,
        "required": ["question", "inputs", "assignments", "snapshots", "calculation_context"],
        "properties": {"question": text,
            "inputs": {"type": "array", "uniqueItems": True, "items": text},
            "assignments": {"type": "array", "items": {"type": "array",
                "minItems": 2, "maxItems": 2, "items": text}},
            "snapshots": {"type": "array", "items": SNAPSHOT_SCHEMA},
            "calculation_context": {"type": "string"}}}


def _branches(schema):
    return schema.get("oneOf", [schema])


class _ScopedScientificSource(ResearchSource):
    """Shared primitive operations; concrete sources own their scientific jobs."""
    record_kinds = list(SCIENTIFIC_KINDS)

    def _commission_inputs(self):
        keys = [*self.commission.inputs, *self.selected_keys]
        request = getattr(self.commission, "theory_request", None)
        if request and request["reconstruction"] in self.ledger.causal_sheet.rows:
            keys.append(request["reconstruction"])
        return tuple(dict.fromkeys(keys))

    def _sheet_commission(self):
        return SheetCommission(self.commission.question, self._commission_inputs(),
            assignments=tuple(getattr(self.commission, "assignments", ())),
            snapshots=tuple(self.scoped_snapshots()),
            calculation_context=getattr(self.commission, "calculation_context", ""))

    def record_results(self, records):
        from dataclasses import replace
        initial = {row["key"] for row in self.initial.causal_sheet_updates}
        result = read_sheet_result(self.ledger.causal_sheet, self._sheet_commission(), records,
            actor=self.actor, owned_subjects=self.ledger.causal_sheet.rows.keys() - initial)
        return replace(result, diagnostics=tuple(self.run.identify_diagnostic({**diagnostic,
            "model_call": self.run.model_calls}) for diagnostic in result.diagnostics))

    def catalog_rows(self):
        sheet = self.ledger.causal_sheet
        initial = {row["key"] for row in self.initial.causal_sheet_updates}
        keys = set(self._commission_inputs()) | (sheet.rows.keys() - initial)
        keys.update(subject for subject, _ in getattr(self.commission, "assignments", ()))
        context = getattr(self.commission, "calculation_context", "")
        if context:
            keys.add(context)
        todo = list(keys)
        while todo:
            for dependency in sheet.rows[todo.pop()]["dependencies"]:
                if dependency not in keys:
                    keys.add(dependency)
                    todo.append(dependency)
        return [(key, row) for key, row in sheet.rows.items() if key in keys]

    def authorize_inspection(self, keys):
        if not set(keys) <= {key for key, _ in self.catalog_rows()}:
            raise ValueError("Inspect commissioned or explicitly looked-up scientific objects")

    def authorize_snapshots(self, snapshots):
        allowed = {(ref["key"], ref["version"]) for ref in self.scoped_snapshots()}
        allowed.update((key, row["version"]) for key, row in self.catalog_rows())
        if any((ref["key"], ref["version"]) not in allowed for ref in snapshots):
            raise ValueError("Select commissioned or explicitly looked-up historical scientific objects")

    def authorize_evaluation(self, assessment):
        if assessment not in {key for key, _ in self.catalog_rows()}:
            raise ValueError("Evaluate a commissioned, created or explicitly looked-up protocol")

    def _schema(self):
        from .scientific_tools import tool_schema
        from .discovery_goals import discovery_goal_action_schema
        schema, visible = _sheet_tool_contract(self, self.record_kinds)
        actions = ["record", "inspect", "calculate"]
        if schema["properties"]["assessment"] is not False:
            actions.append("evaluate")
        schema["properties"]["action"] = {"enum": actions}
        branches = [schema, *_branches(tool_schema()), *_branches(discovery_goal_action_schema())]
        return {"oneOf": branches}, visible, [*actions, "scientific_tool", "create_discovery_goal"]

    def _work(self, view, operation, **context):
        from .scientific_tools import TOOL_INSTRUCTIONS, execute_tool
        from .discovery_goals import create_discovery_goal, DISCOVERY_GOAL_INSTRUCTION
        schema, visible, actions = self._schema()
        proposal = self.decide(self.inputs(view, operation=operation,
            instruction=self.instructions + SHEET_TOOLS + TOOL_INSTRUCTIONS + DISCOVERY_GOAL_INSTRUCTION,
            allowed_actions=actions, response_contract=schema, **context))
        self.last_proposal = proposal
        action = selected_choice(proposal.get("action"), [*actions, "recall"], "action")
        if action == "recall":
            return self.recall(proposal)
        if action == "scientific_tool":
            return self.leaf(lambda: execute_tool(self, proposal), "scientific_tool:" + proposal["name"])
        if action == "create_discovery_goal":
            return self.leaf(lambda: create_discovery_goal(self, proposal), "create_discovery_goal")
        if action == "integrate_theory":
            return self.leaf(lambda: integrate_theory(self, proposal["additions"]), "integrate_theory")
        return _execute_sheet_tool(self, proposal, visible)

    def _scientific_credit(self, result):
        from ...credit_assignment.causal_progress import compile_episode_points
        from ...credit_assignment.causal_support import controller_projection
        after = compile_episode_points(Ledger(result).causal_sheet)
        before = compile_episode_points(Ledger(self.initial).causal_sheet)["points"]
        points = {key: Fraction(value) - Fraction(before.get(key, "0"))
                  for key, value in after["points"].items()
                  if Fraction(value) > Fraction(before.get(key, "0"))}
        projection = controller_projection(points, after["resolution"])
        if not projection["supported"]:
            return Credit(evaluated=False)
        return Credit(evaluated=result.evaluated, quality_resolution=projection["resolution"],
            qualities=tuple((self.credit_columns[0], key, value)
                           for key, value in sorted(projection["values"].items())))

    def parent_report(self, result, completion=None, request=None):
        report = job_report(self, result, completion)
        initial = {row["key"] for row in self.initial.causal_sheet_updates}
        report["discovery_goals"] = [{"key": key, "version": row["version"]}
            for key, row in Ledger(result).causal_sheet.of_kind("discovery_goal") if key not in initial]
        return report


class _TypedTheorySource(_ScopedScientificSource):
    """Typed integration and corrective transport shared by two scientific jobs."""

    def _schema(self):
        schema, visible, actions = super()._schema()
        schema["oneOf"].append(integration_schema())
        return schema, visible, [*actions, "integrate_theory"]

    def bounded_inputs(self, view, declared):
        declared = deepcopy(declared)
        fragment = self.ledger.causal_sheet.rows.get(self.commission.theory_request["reconstruction"])
        shown = ({(fragment["key"], fragment["version"]): "theory_fragment"}
                 if fragment is not None and "theory_fragment" in declared else {})
        for field in ("source_evidence", "selected_sheet_inputs", "historical_sheet_inputs",
                      "scientific_inputs", "historical_inputs"):
            if field not in declared:
                continue
            projected = []
            for row in declared[field]:
                ref = (row["key"], row["version"])
                if ref in shown:
                    projected.append({**{name: value for name, value in row.items() if name != "data"},
                                      "content_in": shown[ref]})
                else:
                    projected.append(row)
                    shown[ref] = field
            declared[field] = projected
        return super().bounded_inputs(view, declared)

    def after_verdict(self, unit, contribution, view):
        super().after_verdict(unit, contribution, view)
        if contribution.episode_update is not None:
            returned = contribution.episode_update.report.as_record()
            if returned["job"]["responsibility"] == "TheoryFormatRepairSource":
                self.theory_repair = returned["repair_status"]["repair"]

    def _typed_work(self, view, operation):
        key = self.commission.theory_request["reconstruction"]
        if key not in self.ledger.causal_sheet.rows:
            def initialize():
                self.commission.request(self.ledger.causal_sheet)
                sheet = CausalSheet(self.ledger.causal_sheet.history)
                return ResearchResult(causal_sheet_updates=(sheet.put(key, "theory_fragment", self.commission.seed),))
            return self.leaf(initialize, "declare_theory_vocabulary")
        if getattr(self, "theory_repair", None) is not None:
            return repair_theory_unit(self, view)
        return self._work(view, operation, **theory_context(self))

    def parent_report(self, result, completion=None, request=None):
        report = super().parent_report(result, completion, request)
        report.update(theory_report(self, result))
        return report


class TheoryReconstructionSource(_TypedTheorySource):
    commission_type = TheoryReconstructionCommission
    instructions = RECONSTRUCTION_TASK
    assessment_fields = {
        "theory_account": "The connected source-grounded account, its entities, quantities, causal assertions and Gs.",
        "assumptions_and_approximations": "Definitions and conditions behind expected generalizability bounds, with source qualifications.",
        "formal_checks": "What chosen equivalence/intervention checks established, with exact tool and scientific references.",
        "qualifications": "What remains conditional, disputed or unverified.",
        "information_needs": "Pointed missing-information goals returned to Scientific and Root for Discovery."}
    credit_columns = ("reconstruction_contribution",)

    def credit(self, result):
        return theory_credit(result, self.initial, self.commission.theory_request, self.credit_columns[0])

    def propose(self, view):
        return self._typed_work(view, "propose_theory_reconstruction")


class TheoryRepairSource(_TypedTheorySource):
    commission_type = TheoryRepairCommission
    instructions = REPAIR_TASK
    assessment_fields = {
        "repair_account": "Concrete proposed and examined changes and their relation to the failure and affected Gs.",
        "tested_consequences": "Actual checks, restored predictions, retained successes, losses and additional consequences.",
        "G_preservation_or_alteration": "What establishes preservation, alteration or unresolved status of each examined G.",
        "qualifications": "Fitting assumptions, information costs, scope and limits of corroboration and numerical comparison.",
        "information_needs": "Specific missing evidence or premises for Scientific and Root to request from Discovery."}
    credit_columns = ("repair_scientific_gain",)

    def credit(self, result):
        return self._scientific_credit(result)

    def propose(self, view):
        return self._typed_work(view, "propose_theory_repair")


class TheoryComparisonSource(_ScopedScientificSource):
    commission_type = TheoryComparisonCommission
    instructions = COMPARISON_TASK
    assessment_fields = {
        "comparison_account": "The theories, tasks and Gs and their correspondence through native definitions and conditions.",
        "established_results": "Source-grounded outcomes, executed checks, failures and useful G alterations.",
        "repair_consequences": "Examined preserving/altering repairs, retained successes, prediction losses and information costs.",
        "qualifications": "Scope and unresolved source-to-formalism correspondence; zero score does not refute broader findings.",
        "information_needs": "Pointed information goals for Scientific and Root to send to Discovery."}
    credit_columns = ("comparison_scientific_gain",)

    def credit(self, result):
        return self._scientific_credit(result)

    def propose(self, view):
        return self._work(view, "propose_theory_comparison")


class TheoryFormatRepairSource(ResearchSource):
    commission_type = TheoryFormatRepairCommission
    instructions = FORMAT_REPAIR_TASK
    assessment_fields = {"repair_answer": "Which representation errors were corrected and which validator errors remain unresolved."}
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
    records = theory_context(source)["source_evidence"]
    commission = TheoryFormatRepairCommission(question=source.commission.question,
        inputs=tuple(dict.fromkeys([*source.commission.inputs, *source.selected_keys,
                                   *(row["key"] for row in records)])),
        snapshots=tuple(source.scoped_snapshots()), theory_request=source.commission.theory_request, seed=None)
    child = TheoryFormatRepairSource(source.run, source.ledger.result(), rejected=source.theory_repair)
    return source.edge(source.run.child(view, child, "theory_format_repair", commission.question,
        contract=commission.report_contract(), proposal={"operation": "repair_theory_output"}))
