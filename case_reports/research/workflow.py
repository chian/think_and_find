"""Root coordinates sibling information acquisition and scientific investigation."""
from fractions import Fraction

from ...method_loop import Context, EpisodeGoal
from .engine import ResearchLimits, ResearchRun, ResearchSource, changed_records, selected_choice
from .records import ResearchResult, Ledger
from .causal_sheet import CausalSheet
from .episode_contracts import (ScientificCommission, DiscoveryCommission,
    TheoryReconstructionCommission, TheoryRepairCommission, TheoryComparisonCommission)
from .acquisition import DiscoverySource
from .theory_episodes import (TheoryReconstructionSource, TheoryRepairSource, TheoryComparisonSource,
    reconstruction_commission_schema, repair_commission_schema, comparison_commission_schema)
from .discovery_goals import (discovery_goal_action_schema, create_discovery_goal,
    compile_discovery_points, DISCOVERY_GOAL_INSTRUCTION)
from .scientific_tools import tool_schema, execute_tool, TOOL_INSTRUCTIONS
from .sheet_tools import SCIENTIFIC_KINDS, SHEET_TOOLS, _sheet_tool_contract, _execute_sheet_tool
from .episode_memory import job_report
from .task import RESEARCH_OBJECTIVE
from .report import ResearchReport


ROOT_TASK = """Coordinate the scientific investigation by choosing between sibling Discovery
and Scientific jobs. Discovery acquires new information for explicit scientific
criteria; Scientific formalizes theories, repairs them and compares uses of G.
Use new information, pending scientific questions and actual completed outcomes
to choose the next commission. Their exchange runs through you. There is no
mandatory order or fixed acquisition-then-analysis itinerary. Pass selected
scientific records, exact source evidence and explicit information goals, never
parent histories or recursively copied reports. Do not launch another open-ended
literature job when the available findings instead need Scientific interpretation.

For a fresh direction, create_discovery_goal records a scientifically specific
information request with an answer JSON Schema and executable criteria. Choose
a natural phenomenon; no topic, author or theory is prescribed. Initial source
requirements can request a source-defined equation, causal relation or observation.
Do not invent an unobserved G or demand an unknown equation be filled in. A
native requirement checks only its explicit source_claim or source_value
predicate. It certifies a native type match, not a scientific meaning hidden in
the criterion ID or the investigation's prose. Use a scientific requirement
when the request concerns a mechanism, entity's role, theoretical relationship,
or use of particular quantities. A theory_structure predicate constrains the
actual typed component and its referenced entities, quantities and equations;
its native candidate filter only helps retrieval. Until that source-grounded
structure exists the requirement stays pending. Dependence on G, equivalence
and distinction between theories retain their separate executed checks.

For an existing Scientific goal, forward its definitions, answer type and criteria
unchanged to Discovery. Forward Discovery's qualified findings and unresolved
correspondences to Scientific. You can revisit either arm with the shared record
versions. Discovery's local credit measures fulfillment of its commissioned
typed requirements. Native matches report native_fields_only; scientific
fulfillment reports scientific_requirement. Root credits a verified source operand or scientific finding once
globally; another goal or criterion for the same result adds no point. The first
source-grounded public examination of a G earns a progress point once. Existing
support points retain their separate meaning. Graduated applicability-difference and
useful-alteration credit comes from executed task comparisons, including useful extensions
beyond the original expected generalizability bounds. A scoped improvement need not repair
every task; support for old G, a relation reversal and rejected alternatives are not
prerequisites. Commission Scientific to record and execute these operands when available
results warrant them, rather than repeatedly producing unscored summaries. New queries,
episode visits and tool calls alone earn no such points.
Use VOC forecasts and observed outcomes; the numerical controller owns continuation.
"""

SCIENTIFIC_TASK = """Own scientific work on supplied sources and shared scientific objects.
Your four responsibilities are: (1) convert source text into formalized theories
and their Gs, (2) develop repairs informed by supplied text, (3) compare theories
to establish useful G alterations, and (4) provide Root with pointed information
goals for Discovery. Choose Theory Reconstruction, Theory Repair or Theory
Comparison as iterative children for those responsibilities. Each can perform
its own scoped reasoning and use the same scientific tools available to you.
Scientific is not a literature-search planner. Missing evidence becomes a
create_discovery_goal action returned through Root to sibling Discovery.

A theory can contain many causal assertions. Register relevant Gs and their
defined quantities, native conditions and expected generalizability bounds.
Reconstruct the supplied source excerpts, not just titles or prior summaries.
Preserve each theory's native question and quantities; comparisons need justified
mappings, not identical settings. Useful failures, G-preserving repairs and
refutations remain findings even without a numerical change score. A new
reconstruction version does not erase previous version-bound scientific credit.

Use lookup for theories, Gs, bounds, comparisons, repairs and results. For a
specific equivalence question, Lean checks the two formal statements under
explicitly source-bound definitions and assumptions. For a causal question,
SCM/equation/simulation tools execute the chosen models and interventions. Tool
receipts inform the next scientific choice; an unsuccessful proof is unresolved,
not inequivalence, and a model calculation does not validate its empirical premises.

Information goals define exact scientific inputs, a custom answer type and
individually identified typed requirements. Literal native requirements can
verify source fields immediately. Scientific requirements use theory_structure
over source-grounded typed components, or the separate equivalence, dependence
and distinct-theory checks. A native candidate match cannot fulfill a scientific
requirement. If 'another theory uses G'
requires interpretation, formalize its actual use, compare it, and link the
executed result to the discovery finding so the goal verifier can resolve it.
Never replace that work with a model-authored pass verdict or numerical grade.
Both supported and challenged Gs can receive first-assessment progress; subsequent
support and useful alterations use calculated credit. Use causal_change and compare_tasks
for scoped applicability differences and gradual task-error improvements. Preserve the
original bounds, task losses and missing early events. A whole-theory repair or failed
competing alternatives are not required; public reasoning alone is not an executed score.
"""

TEXT = {"type": "string", "minLength": 1}
SCIENTIFIC_COMMISSION_SCHEMA = {"type": "object", "additionalProperties": False,
    "required": ["question", "inputs", "snapshots"],
    "properties": {"question": TEXT, "inputs": {"type": "array", "uniqueItems": True, "items": TEXT},
                   "snapshots": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                       "required": ["key", "version"], "properties": {"key": TEXT, "version": TEXT}}}}}


def _expand_scientific_inputs(sheet, keys):
    """Forward selected source excerpts and goal definitions, not recursive reports."""
    expanded, todo = set(), list(keys)
    while todo:
        key = todo.pop()
        if key in expanded:
            continue
        if key not in sheet.rows:
            raise ValueError(f"Unknown scientific commission input: {key}")
        expanded.add(key)
        row = sheet.rows[key]
        if row["kind"] == "discovery_finding":
            data = row["data"]
            todo.append(data["goal"])
            todo.extend(item["evidence"]["key"] for item in data["source_claims"])
            todo.extend(item["evidence"]["key"] for item in data.get("source_observations", ()))
            todo.extend(ref["key"] for ref in data["scientific_links"])
        elif row["kind"] == "discovery_goal":
            todo.extend(ref["key"] for ref in row["data"]["inputs"])
    return tuple(sorted(expanded))


class RootSource(ResearchSource):
    instructions = ROOT_TASK
    assessment_fields = {
        "investigation": "What Discovery and Scientific returns establish about the investigation, with qualifications.",
        "next_work": "Which existing information goal or scientific uncertainty deserves the next commission and why."}
    credit_columns = ("global_goal_progress",)

    def __init__(self, run):
        super().__init__(run, run.initial)

    def credit(self, result):
        return self.run.global_goal.credit(result)

    def catalog_rows(self):
        return sorted(self.ledger.causal_sheet.rows.items(), key=lambda item: item[1]["kind"] == "evidence")

    def propose(self, view):
        sheet = self.ledger.causal_sheet
        goals = [{"key": key, "version": row["version"], "question": row["data"]["question"],
                  "criteria": [item["id"] for item in row["data"]["criteria"]]}
                 for key, row in sheet.of_kind("discovery_goal")]
        schema = {"oneOf": [discovery_goal_action_schema(),
            {"type": "object", "additionalProperties": False,
             "required": ["action", "goal", "rationale"],
             "properties": {"action": {"const": "discovery"}, "goal": TEXT, "rationale": TEXT}},
            {"type": "object", "additionalProperties": False,
             "required": ["action", "commission", "rationale"],
             "properties": {"action": {"const": "scientific"}, "commission": SCIENTIFIC_COMMISSION_SCHEMA,
                            "rationale": TEXT}}]}
        proposal = self.decide(self.inputs(view, operation="propose_root_action", instruction=ROOT_TASK + DISCOVERY_GOAL_INSTRUCTION,
            discovery_goals=goals, discovery_progress=compile_discovery_points(sheet),
            allowed_actions=["create_discovery_goal", "discovery", "scientific"], response_contract=schema))
        self.last_proposal = proposal
        action = proposal.get("action")
        if action == "recall":
            return self.recall(proposal)
        if action == "create_discovery_goal":
            return self.leaf(lambda: create_discovery_goal(self, proposal), "create_discovery_goal")
        if action == "discovery":
            key = proposal["goal"]
            sheet._require_kind(key, "discovery_goal")
            data = sheet.rows[key]["data"]
            inputs = tuple(dict.fromkeys([key, *(ref["key"] for ref in data["inputs"])]))
            commission = DiscoveryCommission(data["question"], inputs, tuple(data["inputs"]), discovery_goal=key)
            self.selected_keys = list(inputs)
            child = DiscoverySource(self.run, self.ledger.result())
            return self.edge(self.run.child(view, child, "discovery", commission.question,
                contract=commission.report_contract(), proposal=proposal))
        selected_choice(action, ["scientific"], "action")
        raw = proposal["commission"]
        commission = ScientificCommission(raw["question"], _expand_scientific_inputs(sheet, raw["inputs"]), raw["snapshots"])
        commission.request(sheet)
        self.selected_keys = list(commission.inputs)
        child = ScientificSource(self.run, self.ledger.result())
        return self.edge(self.run.child(view, child, "scientific", commission.question,
            contract=commission.report_contract(), proposal=proposal))


class ScientificSource(ResearchSource):
    instructions = SCIENTIFIC_TASK
    assessment_fields = {
        "scientific_answer": "What reconstructed theories, comparisons and repairs establish about G, with exact source and calculation references.",
        "expected_generalizability_bounds": "What was expected, which findings support or challenge it, and justified revisions.",
        "qualifications": "Unresolved scientific premises, source limitations and the scope of formal results.",
        "discovery_directions": "Pointed missing-information goals and why their criteria would advance this investigation."}
    commission_type = ScientificCommission
    record_kinds = SCIENTIFIC_KINDS
    credit_columns = ("scientific_gain",)

    def credit(self, result):
        from .episode_credit import scientific_credit
        return scientific_credit(result, self.initial)

    def parent_report(self, result, completion=None, request=None):
        report = job_report(self, result, completion)
        delta = changed_records(result, self.initial)
        report["discovery_goals"] = [{"key": row["key"], "version": row["version"]}
            for row in delta.causal_sheet_updates if row["kind"] == "discovery_goal"]
        report["pending_discovery"] = [item for item in compile_discovery_points(Ledger(result).causal_sheet)["outcomes"]
            if any(check["status"] == "needs_scientific_assessment" for check in item["results"])]
        return report

    def propose(self, view):
        primitive, visible = _sheet_tool_contract(self, self.record_kinds)
        formal_tools = tool_schema()
        branches = [primitive, discovery_goal_action_schema(), *formal_tools.get("oneOf", [formal_tools])]
        commissions = {
            "reconstruct_theory": (reconstruction_commission_schema(), TheoryReconstructionCommission, TheoryReconstructionSource, "theory_reconstruction"),
            "repair_theory": (repair_commission_schema(), TheoryRepairCommission, TheoryRepairSource, "theory_repair"),
            "compare_theories": (comparison_commission_schema(), TheoryComparisonCommission, TheoryComparisonSource, "theory_comparison")}
        for action, (schema, _, _, _) in commissions.items():
            branches.append({"type": "object", "additionalProperties": False,
                "required": ["action", "commission", "rationale"],
                "properties": {"action": {"const": action}, "commission": schema, "rationale": TEXT}})
        proposal = self.decide(self.inputs(view, operation="propose_scientific_action",
            instruction=SCIENTIFIC_TASK + SHEET_TOOLS + TOOL_INSTRUCTIONS + DISCOVERY_GOAL_INSTRUCTION,
            discovery_progress=compile_discovery_points(self.ledger.causal_sheet),
            allowed_actions=["record", "inspect", "calculate", "evaluate", "create_discovery_goal", "scientific_tool", *commissions],
            response_contract={"oneOf": branches}))
        self.last_proposal = proposal
        action = proposal.get("action")
        if action == "recall":
            return self.recall(proposal)
        if action == "create_discovery_goal":
            return self.leaf(lambda: create_discovery_goal(self, proposal), "create_discovery_goal")
        if action == "scientific_tool":
            return self.leaf(lambda: execute_tool(self, proposal), "scientific_tool:" + proposal["name"])
        if action in commissions:
            _, cls, source_class, grain = commissions[action]
            commission = cls(**proposal["commission"])
            commission.request(self.ledger.causal_sheet)
            child = source_class(self.run, self.ledger.result())
            return self.edge(self.run.child(view, child, grain, commission.question,
                contract=commission.report_contract(), proposal=proposal))
        return _execute_sheet_tool(self, proposal, visible)

    def record_results(self, records):
        sheet = CausalSheet(self.ledger.causal_sheet.history)
        updates = []
        for row in records:
            data = dict(row["data"])
            if row["kind"] == "reasoning":
                data["author"] = self.actor
            updates.append(sheet.put(row["key"], row["kind"], data, dependencies=row["dependencies"]))
        return ResearchResult(causal_sheet_updates=tuple(updates))

    def catalog_rows(self):
        return sorted(self.ledger.causal_sheet.rows.items(), key=lambda item: item[1]["kind"] == "evidence")

    def authorize_inspection(self, keys):
        pass

    def authorize_snapshots(self, snapshots):
        pass

    def authorize_evaluation(self, assessment):
        pass


def run_research(*, research_request=RESEARCH_OBJECTIVE, providers, limits=None):
    if not isinstance(research_request, str) or not research_request.strip():
        raise ValueError("Supply a scientific research request")
    run = ResearchRun(providers, limits or ResearchLimits(), research_request)
    source = RootSource(run)
    goal = EpisodeGoal.root(objective=research_request,
        result_contract={"result": "Source-grounded reconstructed theories, G assessments, comparisons and useful alterations with explicit uncertainty",
                         "uncertainty_retained": True}, task_context=run.context)
    episode = run.episode("root", source, goal=goal)
    record = episode.run(Context(tree=run.tree))
    return record, ResearchReport.from_result(source.build_result(record), record, run)
