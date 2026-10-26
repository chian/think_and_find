"""The scientific episode tree, composed through one generic method_loop."""

from dataclasses import replace

from ...method_loop import Context, EpisodeGoal
from ...task_executors._http import ProviderError
from ...task_executors.scenario_metrics.catalog import METRICS, metric_catalog_record
from ...task_executors.scenario_metrics.dependencies import input_fields
from ...task_executors.scenario_metrics.evaluate import CONDITIONS, DEFAULT_METRICS, UNITS
from .acquisition import StrategySource
from .admission import extract_and_admit
from .assumptions import (assumption_candidate,
                          candidate_response_schema, frame_scenario, register_candidate, text)
from .communication import decision_records
from .engine import CONSISTENCY, ResearchLimits, ResearchProviders, ResearchRun, ResearchSource, selected_choice
from .goal_state import GLOBAL_GOAL_PROGRESS
from .modeling import DERIVATIONS
from .model_defaults import MODEL_DEFAULTS, initialize_model_defaults
from .records import Ledger, ResearchResult, identity, numeric_references
from .report import ResearchReport, scientific_progress
from .scenarios import initial_scenarios, load_scenarios
from .tools import TOOLS, TOOL_METRICS, resolve_calculation_inputs, execute_tool, tool_options
from .verification import CALIBRATION_PROGRESS


OBJECTIVE = (
    "Derive conditional indoor transmission safety estimates in the sense of "
    "Bazant and Bush for the requested scenarios. Retain the provenance of physical and infectiousness inputs; "
    "report allowable room time, risk-based safe distance, and R_in only, with their exposure "
    "conditions and regimes. Intermediate quantities serve these outputs, not additional report goals. "
    "Use one risk model per named parameterization and background regime: fix distance to solve time, "
    "fix time to solve distance, and fix both to evaluate R_in. "
    "Resolve uncertainty through evidence for assumptions, not comparison or matching to published "
    "answers. A jet/background concentration crossover is not the requested safe distance."
)


def investigable_parameters(ledger, focus):
    """Retained physical baselines are supplied, not new investigation assignments."""
    from .scenario_sharing import needed_fields
    needed = set().union(*(needed_fields(scenario) for scenario in ledger.scenarios.values()))
    return (set(focus) & (UNITS.keys() | CONDITIONS.keys())) - (MODEL_DEFAULTS.keys() - needed)


def admit_scopes(run, proposal, inherited, context):
    ledger = Ledger(inherited)
    scenarios = proposal.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError("Scenario proposals must be a list of named descriptions, not numeric dictionaries")
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("Scenario descriptions must be objects")
        if run.fixed_scenarios:
            if (scenario.get("role") != "calibration" or scenario.get("subject") in ledger.scenarios
                    or scenario.get("subject") not in calibration_subjects(ledger)):
                raise ValueError("Target cases are fixed. Only a distinct calibration context identified by admitted source observations may be registered; existing scopes cannot be replaced")
        framed = frame_scenario(run.complete, scenario, ledger, context)
        if not framed.scenarios:
            raise ValueError(f"Scenario {scenario.get('subject')!r} was not admitted: "
                             + "; ".join(framed.findings))
        ledger.merge(framed)
    return ledger


def calibration_subjects(ledger):
    observed = {f.subject for f in ledger.facts.values() if f.kind == "source_observation" and f.source}
    observed.update(e["subject"] for e in ledger.evidence.values()
                    if e.get("kind") == "source_evidence" and e.get("evidence_role") == "observation" and e.get("source"))
    return sorted(observed - ledger.scenarios.keys())


def source_calculation_view(ledger, numeric_catalog=None):
    """Native operands are exposed only for source/calibration calculations."""
    if numeric_catalog is None:
        numeric_catalog = decision_records(ledger, target_subjects=(name for name, definition in ledger.scenarios.items()
            if definition.get("role", "target") == "target"))["numeric"]
    records = [{"reference": key, **numeric_catalog[key]}
               for key, fact in numeric_references(ledger).items()
               if fact.subject not in ledger.scenarios
               or ledger.scenarios[fact.subject].get("role", "target") != "target"]
    return {"native_input_records": records,
            "contract": "Source/calibration calculations select source_inputs references for their actual source subject. Target calculations omit numeric references and automatically use the main table, fixed inputs and retained defaults."}


def calculation_leaf(source, proposal, staged, context):
    """Execute the deterministic tool without a calculation model review."""
    action = proposal["action"]

    def calculate():
        result = execute_tool(proposal, staged, source.run.metrics)
        staged.merge(result)
        return staged.result(evaluated=result.evaluated)

    return source.leaf(calculate, action)


class GuidelineSource(ResearchSource):
    def __init__(self, run):
        super().__init__(run, run.initial or ())
        self.request_read = False
        self.request_evidence = ResearchResult()
        self.output_fields = run.metrics

    def proposal_schema(self, *, ledger=None, actions=None, focus=None):
        ledger = self.ledger if ledger is None else ledger
        actions = ["investigate", *self.run.tools] if actions is None else actions
        focus = self.focus if focus is None else focus
        prose = {"type": "string", "minLength": 1}
        options = {key: fact for key, fact in numeric_references(ledger).items()
                   if fact.subject not in ledger.scenarios
                   or ledger.scenarios[fact.subject].get("role", "target") != "target"}
        def strings(description, choices=None, minimum=0):
            item = prose if choices is None else ({"type": "string", "enum": list(choices)} if choices else False)
            return {"type": "array", "minItems": minimum, "uniqueItems": True,
                    "items": item, "description": description}
        schema = {"type": "object", "additionalProperties": False, "required": ["action"],
            "properties": {
                "action": {"type": "string", "enum": actions},
                "objective": {**prose, "description": "Required information need for an investigation child; optional for a calculation whose rationale belongs in basis"},
                "parameter": {"type": "string", "enum": sorted(investigable_parameters(ledger, focus)),
                    "description": "For investigation: the calculation parameter or model condition this branch serves. Shared credit tracks all independent target values and calculation steps."},
                "deliverable": {"type": "string", "enum": ["source_findings", "assumption_hypotheses"],
                    "description": "For investigation: source_findings requests observations, bounds or applicability evidence; assumption_hypotheses requests recorded parameter or condition estimates with their provenance for the named subjects."},
                "subjects": strings("Existing or newly proposed scenario names whose uncertainty the investigation should resolve"),
                "scenarios": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                    "required": ["subject", "description", "basis", "role"], "properties": {
                        "subject": prose, "description": {**prose, "description": "A case or conditional experiment, not a calculation topic or numeric parameter records"},
                        "basis": {**prose, "description": "Alignment with the original request"},
                        "role": {"type": "string", "enum": ["target", "calibration", "validation"]}}}},
                "subject": {**prose, "description": "Reviewed scenario for the selected tool"},
                "source_inputs": strings("For a source/calibration calculation only: native_input_records references, one value per field. Omit for target calculations, whose inputs bind automatically.",
                                 options, minimum=1),
                "basis": {**prose, "description": "Named parameterization or comparison rationale"},
                "parameterization": {**prose, "description": "Stable physical-model name, reused across tools; it never overrides main-table numeric inputs"},
                "metrics": strings("Omit to use this tool's requested outputs, or choose only names permitted by its action-specific schema below. Geometry-derived input parameters are emitted automatically, not selected here.",
                                   sorted(set(METRICS) | {"removal_rate_s"})),
                "operator": {"type": "string", "enum": list(DERIVATIONS)},
                "prediction": {**prose, "description": "Existing calculation option for comparison"},
                "observation": {**prose, "description": "Existing observed fact option for comparison"},
                "metric": {"type": "string", "enum": sorted(METRICS)},
            }, "allOf": [
                {"if": {"properties": {"action": {"const": "investigate"}}},
                 "then": {"required": ["objective", "subjects", "deliverable", "parameter"]}},
                {"if": {"properties": {"action": {"enum": [name for name in TOOLS if name != "compare_predictions_with_observations"]}}},
                 "then": {"required": ["subject", "basis"]}},
                {"if": {"properties": {"action": {"const": "calibrate_infectiousness"}}},
                 "then": {"required": ["operator"]}},
                {"if": {"properties": {"action": {"const": "compare_predictions_with_observations"}}},
                 "then": {"required": ["prediction", "observation", "metric", "basis"]}},
                *({"if": {"properties": {"action": {"const": tool}}},
                   "then": {"required": ["parameterization"], "properties": {"metrics": strings("Output names executable by this selected tool",
                       sorted(choices), minimum=0 if tool == "calculate_geometry_and_removal" else 1)}}}
                  for tool, choices in TOOL_METRICS.items()),
            ]}
        from .scenario_sharing import needed_fields
        for field in MODEL_DEFAULTS:
            eligible = [subject for subject, definition in ledger.scenarios.items()
                        if field in needed_fields(definition)]
            schema["allOf"].append({
                "if": {"required": ["action", "parameter"], "properties": {
                    "action": {"const": "investigate"}, "parameter": {"const": field}}},
                "then": {"properties": {"subjects": strings(
                    "Only cases whose upfront constants review promoted this physical property", eligible)}}})
        if self.run.fixed_scenarios:
            properties = schema["properties"]
            for name in ("prediction", "observation", "metric"):
                properties.pop(name)
            events = calibration_subjects(ledger)
            scopes = [*ledger.scenarios, *events]
            properties["scenarios"]["description"] = "Register an unregistered event identified by admitted observations as role=calibration, using its exact source subject. Include this declaration in the same proposal as a calculation for that event; scope review runs before calculation. Existing scenarios need no declaration. Target cases remain fixed."
            properties["scenarios"]["items"]["properties"]["role"] = {"const": "calibration"}
            properties["scenarios"]["items"]["properties"]["subject"] = {"enum": events} if events else False
            properties["subject"] = {"type": "string", "enum": scopes,
                "description": "Exact registered scenario subject, or an observed event registered in this proposal's scenarios. Except for transfer_parameter, every selected numeric input must have this exact subject; event aliases are not interchangeable."}
            properties["subjects"] = strings("Existing cases or observed calibration events registered in this proposal",
                                            scopes)
            properties["basis"]["description"] = "Named conditional parameterization and its rationale"
            schema["allOf"] = [row for row in schema["allOf"] if row["if"]["properties"]["action"].get("const")
                               != "compare_predictions_with_observations"]
            schema["allOf"].extend({
                "if": {"required": ["subject"], "properties": {"subject": {"const": subject}},
                       "not": {"required": ["operator"], "properties": {"operator": {"const": "transfer_parameter"}}}},
                "then": {"properties": {
                    "source_inputs": strings("Native operands belonging to this source/calibration case",
                                     [key for key, fact in options.items() if fact.subject == subject], minimum=1)}}}
                for subject, definition in ledger.scenarios.items())
        target_subjects = [subject for subject, definition in ledger.scenarios.items()
                           if definition.get("role", "target") == "target"]
        schema["allOf"].append({
            "if": {"required": ["subject"], "properties": {"subject": {"enum": target_subjects}}},
            "then": {"properties": {"source_inputs": False,
                "operator": {"type": "string", "enum": [name for name in DERIVATIONS
                                                         if name != "transfer_parameter"]}}},
            "else": {"if": {"required": ["action"], "properties": {
                "action": {"enum": [name for name in TOOLS if name != "compare_predictions_with_observations"]}}},
                "then": {"required": ["source_inputs"]}}})
        return schema

    def propose(self, view):
        targets = [definition["subject"] for definition in (
            self.run.initial.scenarios if self.run.initial is not None else self.ledger.scenarios.values())
            if definition.get("role", "target") == "target"]
        records = decision_records(self.ledger, target_subjects=targets)
        assumptions = records["assumptions"]
        proposal = self.run.agent(self.inputs(view, operation="propose_guideline_action",
            allowed_actions=["investigate", *self.run.tools], tools=self.run.tools,
            assumption_options={key: value for key, value in assumptions.items() if value["value"] is not None},
            source_calculation_inputs=source_calculation_view(self.ledger, records["numeric"]),
            condition_hypotheses=[value for value in assumptions.values() if value["value"] is None],
            metric_catalog={k: metric_catalog_record(k) for k in METRICS},
            canonical_fields=UNITS, model_conditions=CONDITIONS, derivations=DERIVATIONS,
            calculations=records["calculations"],
            **({"observed_fact_options": {key: row for key, row in records["facts"].items()
                                         if row["kind"] == "source_observation"}}
               if "compare_predictions_with_observations" in self.run.tools else {}),
            scientific_progress=scientific_progress(self.ledger.result(), self.run.metrics, decision_view=True),
            instruction=(
                (
                'Choose one useful action toward the original request, informed by prior results. At each root '
                'decision, try a requested calculation as soon as selected numeric facts or recorded hypotheses and '
                'physically applicable conditions make it executable. Also consider a ready intermediate calculation '
                'in calculation_steps; it can advance the requested outputs before all their inputs are found. '
                'Geometry with metrics=[] computes available derived inputs from the target main table. Prefer a new '
                'calculation to another investigation when it would fill a missing requested output using '
                'the current main-table inputs. Record its input provenance without waiting for additional '
                'evidence. If a calculation lacks a specific physical input, request a narrowly scoped '
                'assumption_hypotheses investigation to propose a qualified value, then reconsider calculation when '
                'it returns. Do not spend another source_findings branch collecting every possible context detail '
                'first. This run performs the scientific investigation using the existing workflow and calculation '
                'kernels; an implementation plan or input glossary is not its result. No fixed action sequence, no '
                'stop/report/assessment action. If context.fixed_scenarios is true, the caller has selected and '
                'defined every TARGET case. Do not discover, propose, replace or alter target scenarios. Their '
                'fixed_parameters and fixed_conditions define scope, not empirical evidence; use the supplied options '
                "without reassessing those givens. Investigate the remaining assumptions. Read each case's "
                'output_contracts. The upfront constants judge recorded model_defaults before scenario sharing. '
                'Retained physical baselines are supplied automatically with their conditional provenance and earn no '
                'discovery credit; do not investigate or reassess them. Only defaults explicitly promoted for a case '
                'remain parameter needs. Name the calculation parameter served by each investigation in parameter. '
                'Shared credit tracks the predefined independent target values and calculation steps; strategies '
                'cannot add requirements by naming more settings, sources or topics. Search choices remain yours. '
                'Credit measures found independent parameters, qualifying evidence learning and completed calculation '
                'steps. Read global_scientific_goal.calculation_steps.credit_policy for the numerical rates and '
                'eligibility requirements. Requested final outputs receive the requested_output_multiplier; '
                "intermediate calculations use the base rates. The table's points and credit components already "
                'include that multiplier. Compare earned credit with the credit available from completing missing '
                'requested outputs or updating stale results. Each requested output contributes separately; there is '
                "no additional all-task completion bonus. The caller's target cases were compared before research and "
                'equivalent parameter needs were grouped as independent unknowns. A numeric_option shared from '
                "another case belongs to the selected target's group and is bound automatically when it is the main "
                'selected value. Do not request a duplicate hypothesis for that group. Give each investigation a '
                'specific question and only the context necessary to answer it. For investigate, include '
                'calculation_deferrals identifying one or more requested-output rows affected by this choice, using '
                'the defined reason codes and current table references. Use improving_evidence_support when choosing '
                'stronger evidence even though calculation is possible. This records your decision; it does not '
                'change readiness or credit. Use [] only when the table has no requested-output rows; omit it for '
                'calculation actions. Name only the target subjects involved; use [] for source discovery without a '
                'target assignment. An empty subjects list leaves target applicability open; it does not prohibit '
                'estimates for relevant existing targets or registered source calibration contexts. The child '
                'contract retains the full caller-defined target set, including fixed inputs and conditions, while '
                "subjects marks this episode's local investigation. This lets source discovery assess applicability "
                'across the requested cases even with subjects=[]. Do not pass your whole requirement list to a '
                'child. Its return contains structured parameter_results (numbers, units, origin, status and declared '
                'references), reviews (verdict, completed and criterion grades), calculations (output statuses, '
                'values and units), reading_coverage (counts), operations (closed stage, category, status and count), '
                'and completion (ended_by and evaluated). These records come from existing work-call JSON; there is '
                'no separate summary call or free-form child message. Use the declared terms and definitions to '
                'interpret them when choosing the next action. Specify deliverable=source_findings when you need '
                'observed events, bounds or applicability evidence without a new target hypothesis. Specify '
                'assumption_hypotheses when you need recorded parameter estimates with their rationale and '
                'provenance. Evidence discovery and hypothesis justification have different information contracts; '
                'either child may form a useful conditional estimate or execute an available calculation. Useful '
                'target-value changes return control at an ordinary episode boundary so you can reconsider the whole '
                'target table. Outbreak calibration is an available way to derive a missing infectiousness value, '
                'not an additional report requirement. If choosing that method, acquire observed outbreak counts, '
                "duration and context through investigate's search children. Source observations retain their "
                'actual event subject. Register a separate calibration context in scenarios using that exact subject '
                'and role=calibration; this does not add a target case. Use calibrate_infectiousness with cq_from_steady_outbreak or '
                'cq_from_transient_outbreak. That tool contains the calibrate_cq_from_outbreak subtask. Eligible '
                'derived source Cq automatically enters matching target alternatives with its calculation lineage and '
                'value-specific relevance. Subsequent target calculations bind the main selected value automatically; '
                'a second transfer proposal is not required. Use cq_activity_rescaling only when an explicit activity '
                'adjustment is justified by its inputs. Assess transfer compatibility, including the runtime '
                "disease/variant and activity, without double scaling. A source's published Cq is not an outbreak "
                'inversion performed by this run. An emission-to-breathing quotient is also a usable conditional '
                'estimate, with its own provenance. Missing calibration does not block found target values, '
                'bounded best guesses, target calculations, or final reporting. An inversion is not required after '
                'the authoritative main table already supplies the needed input. '
                'calculate_conditional_limits exposes solve_time through allowable_room_time_s and '
                'transient_allowable_room_time_s: FIX neighbor_distance_m and SOLVE duration. Its solve_distance '
                'operation uses safe_distance_m and transient_safe_distance_m: FIX exposure_time_s and SOLVE '
                'separation. calculate_transmission_metrics exposes evaluate_R_in through '
                'scenario_reproductive_number and transient_scenario_reproductive_number: FIX BOTH exposure_time_s '
                'and neighbor_distance_m. All three operations use the same risk model and assumptions within a '
                'background regime. Background-only legacy outputs and the concentration crossover do not substitute '
                'for these requested answers. Before execution, the tool binds caller-supplied reference inputs, '
                'retained defaults and the authoritative main-table scientific values. These reference inputs are not '
                'the unknown being solved for and are not discoveries to research or reassess. Never copy the '
                'reference separation as a calculated safe distance or replace it with random spacing. Omitted '
                'target-case reference inputs are filled by the scenario loader using the declared reference '
                "defaults, not selected by the agent. These defaults never supply an outbreak's exposure duration. "
                'calculation_variables remain free variables for explicitly requested legacy relationships, not '
                'scalar safety answers. In particular, do not invent an exposure duration to fill a risk or distance '
                'output. This restriction applies to declared calculation variables, not to proposing reasoned '
                'physical hypotheses. A source range need not choose a scalar for the investigator: choosing and '
                'qualifying that hypothesis is its responsibility. Calibration contexts must be grounded in acquired '
                'observations, not invented outbreaks. Reference existing subjects without redeclaring or '
                'paraphrasing their definitions. Investigate unresolved assumptions, conflicting evidence, or '
                'prediction discrepancies. Target calculations take the target subject, tool and requested metrics. '
                'Model conditions bind exclusively from that scenario; do not supply or override conditions. '
                'Calculations automatically use the main table, including unsupported main values; do not supply '
                'facts or select numeric alternatives. Source/calibration calculations retain their own native '
                'operands through source_calculation_inputs. Native source values and eligible source calculations '
                'automatically create target input alternatives. The main table selects the alternative with the '
                'highest source-scenario relevance for that value and parameter; every target calculation binds that '
                'value. Low or unknown relevance does not block a found value or its discovery credit. Additional '
                'inferred hypotheses originate in investigate, not as hidden literals in a tool call. Evidence '
                'provenance stays attached to each input. Calculating the main estimate can expose a discrepancy '
                'without independently validating it. Recalculate when main values or model claims change. Competing numeric candidates '
                'remain in the evidence history and may become the main value through relevance ranking; they cannot '
                'be selected separately for a target calculation. Inspect global_scientific_goal.calculation_steps, '
                "including each row's value, units, status, credit, ready_to_calculate and ready_to_recalculate "
                'flags. Changed dependencies remove current-result credit; retained method credit records only prior '
                'successful work, not present applicability. At each ordinary decision, consider a useful ready '
                'recalculation alongside other ways to advance the Goal. Develop numerical results using the current '
                'main-table values, and update affected results when those values change. Requested numerical outputs '
                'are complete when every requested calculation row is calculated_up_to_date. No additional support '
                'assessment or calibration is needed to report those results. Do not wait for more evidence before using the '
                'calculation tools. This is a scientific objective, not a fixed action order. A calculation for a '
                'scoped case can reveal concrete gaps relevant to the other fixed cases; resolving every uncertainty '
                'across all cases is not a prerequisite to that calculation. Use one stable parameterization name for '
                'compatible input assumptions across tools; a tool or output-family name is not a different physical '
                'case. Distinguish steady, transient and near-field regimes explicitly without fragmenting an '
                'otherwise shared parameterization. Unchanged inputs reuse current computed results; relevance '
                'changes alone do not require arithmetic again. Read per-quantity coverage and recorded dependencies: '
                'a changed input affects only outputs that use it. Each calculation tool owns '
                "the output names in its action-specific metrics schema. Omit metrics to use that tool's requested "
                'outputs. Geometry/removal derivations also emit available input parameters automatically; do not put '
                'those input names in metrics. The same parameter derivations run inside the limit and transmission '
                'tools, so a separate geometry step is not required when the selected inputs suffice. '
                'input_requirements shows executable routes and their missing inputs/conditions, not a requirement to '
                'search for an already implemented equation. An absent removal term requires an explicit hypothesis, '
                'including when the proposed term is zero; it is never silently zero. Routes also declare '
                'parameter_constraints enforced by execution. If the selected case falls outside every implemented '
                'route, identify that capability gap explicitly. A differently worded condition or another supporting '
                'source cannot register a new kernel or override those constraints. Evidence can inform a needed '
                'extension without making it executable in this run. Such a gap does not resolve the requested '
                'quantity, but it is not a prerequisite to strengthening other still-uncertain conclusions. For all '
                'three operations, inspect the alternative routes and their physical hypotheses. The '
                'negligible_near_field condition explicitly selects a spatially uniform risk model for time, distance '
                'AND R_in; mask penetration alone does not justify or select it. A well-mixed background does not '
                'establish negligible local exposure. Do not assume away local risk merely to fill coverage. A zero '
                'minimum under that approximation means no positive separation is imposed by the model, not that '
                'contact is safe. Retain the hypothesis and its uncertainty. The transient jet formula adds an '
                'immediately established local term to the time-evolving background. Record the approximation; it is '
                'not an extra review prerequisite. A duration formula alone does not satisfy a requested scalar '
                'output contract. A target scenario identifies the case or conditional experiment receiving '
                'estimates. A model extension or unresolved equation is an investigation focus for a case, not a '
                'separate target requiring its own complete report. Only when cases are not fixed and the request '
                'leaves them open, explicitly scoped study cases may instantiate that open scope; do not substitute '
                'them for specified cases or claim they establish results for every possible case. Distinguish '
                'requester constraints from unresolved scientific assumptions and named conditional study choices. An '
                'unspecified input is not automatically something only the requester can supply: the investigation '
                'branch may propose and justify bounded assumptions where appropriate. Such choices are neither '
                'universal defaults nor facts about an unspecified real room. Preserve all requested outputs and case '
                'constraints. General applicability does not mean one universal set of parameter values. Each target '
                'uses its own main-table values with their rationale and provenance distinguished from requester '
                'givens and source observations. Missing justification does not block calculation or final reporting. '
                'scientific_progress uses the same selected calculation rows as the main progress and credit table. '
                'requested_outputs_complete means all requested rows for a target are calculated_up_to_date; '
                'quantity_coverage exposes outputs still absent or needing updating across cases. Progress on other quantities does not resolve an absent output or '
                "make it optional. Use the child's structured records to distinguish missing proposed inputs, "
                'uncertain evidence and execution errors; missing direct evidence does not prohibit a recorded '
                'estimate. Use the requested calculation statuses to identify remaining numerical work. '
                'New evidence and relevance records do not require repeating unchanged arithmetic. A current result '
                'meeting its output contract earns method and current-result credit. Read the recorded '
                'calculation_steps.credit_policy for the existing rates; there is no support approval prerequisite '
                'for calculation or final reporting. Copying a reference input, computing a legacy metric, repeating arithmetic, or excluding a '
                'kernel does not answer a missing requested operation. There is no fixed action order; apply the '
                'newly executable calculation preference above at each root decision. The registered equations are '
                'already executable kernels; verifying their scientific basis may resolve uncertainty, but '
                'transcribing them is not itself a prerequisite for tool availability or a numerical safety '
                'conclusion. The requested output quantities are model outputs, not facts to search for verbatim. '
                'Return an instance of the JSON Schema; omit fields not used by the selected action. '
            )
            ),
            response_contract=self.proposal_schema()))
        proposal = self.resolve_decision_references(proposal)
        self.last_proposal = proposal
        action = selected_choice(proposal.get("action"), ["investigate", *self.run.tools], "action")
        objective = text(proposal.get("objective"), "objective") if action == "investigate" or "objective" in proposal else None
        staged = admit_scopes(self.run, proposal, self.ledger.result(), self.context())
        context = {**self.context(), "defined_scenarios": dict(staged.scenarios)}
        if not self.request_read:
            self.request_evidence = extract_and_admit(self.run.agent, text=self.run.research_request, source="request",
                objective=OBJECTIVE, context=context,
                question="Admit only explicit requester-defined constraints. Questions asking for unknown scientific quantities are not evidence of their answers.")
            self.request_read = True
        staged.merge(self.request_evidence)
        if action == "investigate":
            parameter = selected_choice(proposal.get("parameter"),
                investigable_parameters(staged, self.focus), "parameter")
            deliverable = selected_choice(proposal.get("deliverable"),
                ("source_findings", "assumption_hypotheses"), "deliverable")
            source = InvestigationSource(self.run, staged.result())
            subjects = proposal.get("subjects")
            if not isinstance(subjects, list) or any(s not in staged.scenarios for s in subjects):
                raise ValueError("Investigation subjects must be reviewed scenarios")
            if parameter in MODEL_DEFAULTS:
                from .scenario_sharing import needed_fields
                if any(parameter not in needed_fields(staged.scenarios[subject]) for subject in subjects):
                    raise ValueError("A retained or caller-fixed physical baseline is not an investigation need for that case")
            source.credit_subjects = tuple(subjects) or None
            if deliverable == "assumption_hypotheses" and not subjects:
                raise ValueError("An assumption investigation must name its assigned scenario subjects")
            targets = {name: definition for name, definition in staged.scenarios.items()
                       if definition.get("role") == "target"}
            target_scenarios = {name: {
                "description": targets[name]["description"],
                "fixed_parameters": targets[name].get("fixed_parameters", {}),
                "fixed_conditions": targets[name].get("fixed_conditions", {}),
                "assumption_needs": targets[name].get("assumption_needs", ()),
                "model_defaults": targets[name].get("model_defaults", {}),
            } for name in targets}
            return self.edge(self.run.child(view, source, "investigate", objective,
                proposal=proposal, contract=self.contract(
                    f"Decide which assumptions/model choices or calculations to revisit: {objective}",
                    subjects=subjects, deliverable=deliverable,
                    meaning=objective, parameter=parameter, target_scenarios=target_scenarios)))

        return calculation_leaf(self, proposal, staged, context)


class InvestigationSource(ResearchSource):
    def __init__(self, run, inherited):
        super().__init__(run, inherited)
        self.candidates = {}
        self.staged = Ledger(inherited)
        self.output_fields = ()

    def uses_scientific_credit(self):
        # The child answers its assignment; the root projects scientific credit.
        return False

    @staticmethod
    def candidate_key(candidate):
        return identity([candidate.get("subject"), candidate.get("field"), candidate.get("value"),
                         candidate.get("statement") if candidate.get("field") in CONDITIONS else None])

    def inference_options(self):
        """Inputs retain native fields; the assignment constrains inferred outputs."""
        return self.ledger.evidence_options(), numeric_references(self.ledger)

    def hypothesis_scenarios(self, ledger=None, *, include_observed=False):
        """An open source assignment still permits hypotheses in reviewed scopes."""
        ledger = self.ledger if ledger is None else ledger
        if not self.run.fixed_scenarios:
            return None
        subjects = set(self.requested_subjects) if self.requested_subjects else {
            name for name, definition in ledger.scenarios.items()
            if definition.get("role", "target") == "target"}
        subjects.update(name for name, definition in ledger.scenarios.items()
                        if definition.get("role") == "calibration")
        scenarios = {name: definition for name, definition in ledger.scenarios.items()
                     if name in subjects}
        if include_observed:
            # As with tool subjects, these require a same-proposal declaration
            # and successful admit_scopes review before runtime validation.
            scenarios.update((name, {}) for name in calibration_subjects(ledger))
        return scenarios

    def proposal_schema(self):
        prose = {"type": "string", "minLength": 1}
        evidence, numeric = self.inference_options()
        schema = GuidelineSource.proposal_schema(self, ledger=self.ledger,
            actions=["evidence_strategy", *self.run.tools], focus=())
        schema.pop("required")
        schema["anyOf"] = [{"required": ["action"]},
                           {"required": ["candidates"], "properties": {"candidates": {"minItems": 1}}}]
        for condition in schema["allOf"]:
            if "action" in condition["if"].get("properties", {}):
                condition["if"]["required"] = ["action"]
        schema["properties"].update({
            "approach": {**prose, "description": "Evidence strategy and its rationale"},
            "objective": {**prose, "description": "Information this strategy must resolve for the parent"},
            "candidates": {"type": "array", "description": "Your hypotheses for eligible target cases or source calibration contexts. An observed event needs a same-proposal calibration declaration and scope review. Omit action and supply nonempty candidates to register estimates using accumulated evidence, without another search. Candidates are registered and ranked by value-specific relevance; no separate parameter-support review follows. An empty array proposes no new hypotheses.",
                "items": candidate_response_schema(
                    scenarios=self.hypothesis_scenarios(include_observed=True),
                    evidence=evidence, numeric_inputs=numeric)},
        })
        for branch in schema["properties"]["candidates"]["items"]["oneOf"]:
            fields = branch["properties"]["field"]["enum"]
            branch["properties"]["field"]["enum"] = [
                field for field in fields if field in self.parameter_requirements]
        schema["allOf"].append({"if": {"required": ["action"], "properties": {"action": {"const": "evidence_strategy"}}},
                                 "then": {"required": ["approach", "objective", "candidates"]}})
        return schema

    def propose(self, view):
        self.staged = Ledger(self.ledger.result())
        evidence, numeric = self.inference_options()
        targets = [definition["subject"] for definition in (
            self.run.initial.scenarios if self.run.initial is not None else self.ledger.scenarios.values())
            if definition.get("role", "target") == "target"]
        records = decision_records(self.ledger, target_subjects=targets)
        proposal = self.run.agent(self.inputs(view, operation="propose_investigation_strategy",
            registered_assumptions=records["assumptions"],
            canonical_fields=UNITS, model_conditions=CONDITIONS,
            allowed_actions=["evidence_strategy", *self.run.tools], tools=self.run.tools,
            inference_inputs={
                "evidence_options": records["evidence"],
                "numeric_options": records["numeric"],
                "selection_contract": "Candidate source_refs select these evidence_options; input_refs select these numeric_options. These references identify existing records, not new evidence or permission to relabel a measurement.",
            },
            source_calculation_inputs=source_calculation_view(self.ledger, records["numeric"]),
            condition_hypotheses=[value for value in records["assumptions"].values()
                                  if value["value"] is None],
            calculations=records["calculations"],
            **({"observed_fact_options": {key: row for key, row in records["facts"].items()
                                         if row["kind"] == "source_observation"}}
               if "compare_predictions_with_observations" in self.run.tools else {}),

            episode_responsibilities={
                "this_investigation": "Meet the parent's information contract through recorded estimates from accumulated evidence and chosen evidence strategies. When existing inputs permit a requested calculation, execute its deterministic tool here and share that progress with the whole Goal.",
                "strategy_child": "Acquire native source values and evidence. Each numeric commit automatically creates matching target alternatives, selects the highest value-specific source relevance, and updates shared credit. Inferred hypotheses remain this investigation's responsibility.",
                "scientific_parent": "Use shared calculation and evidence progress to select the next investigation or calculation.",
            },
            instruction=(
                (
                "You are the investigation episode, responsible for connecting source evidence to the parent's "
                'scientific Goal. Native values and eligible source calculations automatically enter matching '
                'independent target groups; their highest value-specific source relevance selects the main-table '
                'value and updates global credit without another proposal or review. Target calculations bind those '
                'main values automatically. Propose an inferred hypothesis only when inference or another candidate '
                'is needed. Use the information you already have to infer missing parameters: recognize related '
                'quantity names, convert units, recast measurements, or calculate an estimate from other quantities '
                'when the physical reasoning permits it. Record each inferred numeric candidate as '
                'entry_type=best_guess, with its own canonical value and unit, source_refs and input_refs from '
                'inference_inputs, the reasoning and arithmetic in basis, and any added_assumptions. Preserve the '
                'native meaning of each source measurement; your inference is a separate parameter claim. To submit '
                'candidates using accumulated evidence now, omit action and return a nonempty candidates array. '
                'Registration and value-specific relevance projection follow immediately; no separate '
                'parameter-support review runs. Empty references honestly declare an unsupported guess; they do not '
                'manufacture evidence. Choose evidence_strategy when you need additional evidence, or choose an '
                'available calculation tool when selected existing inputs and physically applicable conditions can '
                'produce a requested result or a ready intermediate step. Geometry with metrics=[] computes available '
                'derived inputs from the target main table. Choose the target subject, tool and metrics. Model '
                'conditions bind exclusively from the scenario; do not supply or override them. Do not supply facts '
                'or numeric alternatives. Source/calibration tools use source_calculation_inputs for their native '
                'operands. A new hypothesis must be registered and ranked before it can become the main input; it '
                'cannot be inserted into a target tool call. Prefer a newly executable calculation to '
                'another long evidence strategy when it advances the shared Goal. A calculation here retains its '
                'source and assumption lineage and returns to the parent in the structured calculations records. '
                "Inspect global_scientific_goal.calculation_steps, including each row's value, units, status, credit, "
                'ready_to_calculate and ready_to_recalculate flags: changed dependencies remove current-result '
                'credit, while retained method credit records prior successful work, not present applicability. '
                'Consider useful ready recalculations at the next ordinary decision; this does not prescribe an '
                'action sequence. For evidence_strategy, include calculation_deferrals identifying the '
                'requested-output rows affected by further investigation, with defined reason codes and current table '
                'references. improving_evidence_support remains valid when calculation is possible. Use [] only when '
                'no requested-output rows exist. Omit calculation_deferrals for calculation actions and '
                'candidates-only inference. These records do not change readiness or credit. Choose candidate '
                "hypotheses and an evidence-acquisition approach for this parent's scientific uncertainty. Put "
                "hypotheses you propose in this response's candidates array. Read the parent's deliverable: "
                'source_findings needs source-grounded answers, bounds and applicability limits and does not require '
                'inventing a target hypothesis; it permits useful conditional estimates and calculations. '
                'assumption_hypotheses needs recorded values or condition hypotheses. An outbreak inversion is an '
                'available inference method, not an additional completion requirement. A published input, a derived source estimate and a '
                'bounded best guess retain their distinct provenance. An emission-to-breathing quotient does not '
                'become an independent outbreak inversion. Use the available estimate while keeping that limitation '
                'explicit. Automatic source-value entries already supply numeric alternatives. Under either contract, '
                'any hypotheses you choose to propose must be recorded in the candidates array. Asking the strategy '
                'child to construct a parameterization, choose values or independently assess assumptions does not '
                'perform that work: its contract extracts source values automatically but does not perform new '
                'physical inferences. Admitted source values and ranges can inform your own conditional choices with '
                'an explicit applicability rationale; they need not already be labeled with the target subject. A '
                'source range may inform a scalar estimate without selecting one for you. Identify '
                'which uncertainty would change that choice or its support. The prohibition on arbitrary exposure '
                'durations or scenario replacement does not prohibit physical-parameter estimation within a fixed '
                "case. Distinguish the case's declared calculation_variables from missing physical inputs. For the "
                'latter, reason from available ranges, transferable measurements, dimensional constraints or clearly '
                'labeled guesses; name the applicability premise and what could overturn it. A range can motivate one '
                'qualified hypothesis or competing input candidates without constituting a direct measurement. Do not '
                'defer a whole requested output merely because the source did not select a scalar for you. If a '
                'proposed approximation would contradict the fixed case or the available model, explain that specific '
                'conflict rather than inventing a compatible case. Model constraints conveyed by the parent are '
                'execution limits, not evidence grades. A new hypothesis can supply a missing parameter or qualify an '
                'applicable regime, but cannot add a kernel or override its enforced constraints. Distinguish '
                'evidence that would inform a model extension from an input the parent can calculate with now. An '
                'empty candidates array is appropriate when this iteration is acquiring information without proposing '
                'a new hypothesis. Check the shared target table: automatically extracted values may already have '
                'filled the inputs. Use previous failures to understand what remains unknown. You may propose or '
                'revise named candidate assumptions and alternatives, but cannot mark them supported or assign '
                'credit. A candidate statement describes a physical/scenario claim, not a research instruction or a '
                'rule to use a model only after evidence is found. Existing registered_assumptions remain recorded '
                'automatically. Use candidates=[] when this iteration proposes no new or revised hypothesis; do not '
                "reconstruct or repeat an unchanged claim's unseen prose. Only new or materially revised hypotheses "
                'need full candidate fields. Put a material premise revision in the claim statement or identified '
                'source/input provenance. New evidence remains available with its provenance without reproposing an '
                'unchanged claim. Changing a statement changes claim identity and can invalidate calculations. Source '
                'extraction and assumption inference have different contracts: extracted numeric facts must be '
                'printed in a source, whereas a candidate may be an inferred conditional value with a named rationale '
                'to test against numeric or qualitative evidence. A formula inventory alone is not a justified target '
                'parameterization. When a final parameter is not directly reported, consider measurements of its '
                'ingredients, related systems, geometric or physical constraints, and explicit approximations. '
                'Explain the inference, differences in context and uncertainty. A guess can be proposed before '
                'justification exists; retain its estimate origin and rationale. If selected by relevance as the main '
                'value, the parent can calculate its consequences. Do not restrict all evidence searches to a '
                'reference paper merely because it defines the model. Choose the approach from the information needed '
                'and prior search outcomes. A numeric candidate needs a finite JSON number, not a string or null. If '
                'a parameter has no proposed value yet, omit that numeric candidate rather than using null. Explain '
                'in approach which requested quantity remains unresolved and what reasoning or evidence is needed to '
                'form a hypothesis; missing direct measurement alone does not prohibit an inferred choice. Null is '
                'only for nonnumeric model conditions. All numeric origins use the same registration, '
                'source-relevance grading and main-value selection. No separate parameter-support review follows '
                'registration or a returned strategy. The system has already compared the selected cases and grouped '
                'parameter needs that can use one value. Propose one candidate for an independent group; do not '
                "duplicate it across that group's cases. Retained model_defaults were reviewed before grouping and "
                'are already supplied conditional physical premises. Do not propose or reassess them; only promoted '
                "default properties appear among the case's remaining parameter needs. A candidate is a hypothesis, "
                "never reported evidence. When context.fixed_scenarios is true, preserve the caller's target cases "
                'and their fixed_parameters and fixed_conditions. You may register a distinct source event identified '
                'by admitted observations as a calibration scenario, using its exact source subject and '
                'role=calibration. For a calculation on an unregistered event, include that scenarios declaration in '
                'the same proposal as the calculation; scope review runs before execution. Reuse an already '
                'registered subject without redeclaring it. The calculation subject must exactly match the registered '
                "scenario and, except for transfer_parameter, every selected numeric input's subject. Source-event "
                'grouping does not make different subject labels interchangeable for calculation. Source evidence '
                'retains its actual event identity; explain its applicability to the selected target cases. No '
                'stopping or separate assessment action. A scenario is a case or conditional experiment, not an '
                'equation or investigation topic. Every candidate subject must match a reviewed case. Source-event '
                'labels alone do not establish a scope; source evidence retains its original identity and '
                'applicability must be assessed for the target. Fixed-mode scenario registration is limited to these '
                "source-grounded calibration contexts; the target set remains fixed.  The parent's information "
                'contract and requested subjects are in context. Source or calibration subjects can supply evidence, '
                'but new templates do not substitute for resolving assumptions for the requested subjects. Explain '
                'applicability to those subjects. Propose hypotheses for your assigned parameter or its executable '
                "dependencies, within the candidate schema's eligible subjects. With subjects=[], existing target "
                'cases remain eligible; an empty assignment is not a ban on estimates. Registered source calibration '
                'contexts are eligible for their own missing operands. Keep source hypotheses on their actual source '
                'subject and target hypotheses on the target they describe. At each decision, inspect '
                'independent_target_values for missing inputs and the shared calculation_steps for ready work. Prefer '
                'a bounded useful estimate or executable calculation over another source search that only repeats '
                'known caveats. New main-table target values or changed selected values can return this episode at '
                'the next ordinary boundary, letting the parent choose among all remaining parameters; no stop action '
                'is needed. The parent-assigned parameter guides the question; shared credit covers all predefined '
                'independent target values and calculation steps. Values for other target parameters also count '
                'automatically. Additional source settings are not extra requirements. Retain competing parameter '
                'values with their evidence and conditions. Highest value-specific source relevance selects the main '
                'value, and only that value feeds target calculations. Do not select a different value per '
                'calculation or by evidence count. Give the strategy child a specific evidence question and the '
                'necessary hypothesis context in objective, not your whole assignment or a field checklist. Its '
                'structured return contains parameter_results, reviews, calculations, reading_coverage, operations '
                'and completion, with declared terms and definitions. Use its numeric results, review grades, output '
                'statuses, coverage counts and operation outcomes to decide whether this approach is informing the '
                'assumption. Existing work-call JSON supplies the return; no separate summary call or free-form child '
                'message is produced. Return an instance of the JSON Schema. '
            )
            ), response_contract=self.proposal_schema()))
        proposal = self.resolve_decision_references(proposal)
        self.last_proposal = proposal
        action = (selected_choice(proposal["action"], ["evidence_strategy", *self.run.tools], "action")
                  if "action" in proposal else None)
        if action is not None and action != "evidence_strategy":
            if proposal.get("candidates"):
                raise ValueError("Register new candidates before calculating; target inputs come from the main table")
            self.staged = admit_scopes(self.run, proposal, self.ledger.result(), self.context())
            context = {**self.context(), "defined_scenarios": dict(self.staged.scenarios)}
            return calculation_leaf(self, proposal, self.staged, context)
        self.staged = admit_scopes(self.run, proposal, self.ledger.result(), self.context())
        if "candidates" not in proposal:
            raise ValueError("Investigation proposals require a candidates array. Put this episode's hypotheses there; instructions asking its evidence-acquisition child to propose them do not register hypotheses. Use [] when none are proposed.")
        candidates = proposal["candidates"]
        if not isinstance(candidates, list) or (action is None and not candidates):
            raise ValueError("Candidate assumptions must be a list, nonempty when no action is selected")
        for candidate in candidates:
            if (isinstance(candidate, dict) and candidate.get("field") in UNITS
                    and candidate.get("entry_type") != "best_guess"):
                raise ValueError("An inferred numeric candidate must declare entry_type=best_guess")
            if isinstance(candidate, dict) and candidate.get("entry_type") == "best_guess":
                if "derivation" in candidate:
                    raise ValueError("New candidate provenance must select the supplied source_refs and input_refs")
                for name, options in (("source_refs", evidence), ("input_refs", numeric)):
                    refs = candidate.get(name)
                    if not isinstance(refs, list) or any(not isinstance(ref, str) or ref not in options for ref in refs):
                        raise ValueError(f"Candidate {name} must select this investigation's inference_inputs")
        validated = [assumption_candidate(candidate, self.staged) for candidate in candidates]
        if any(c["field"] not in self.parameter_requirements for c in validated):
            raise ValueError("Candidate assumptions must serve the assigned parameter or its executable dependencies")
        eligible = self.hypothesis_scenarios(self.staged)
        if eligible is not None and any(c["subject"] not in eligible for c in validated):
            raise ValueError("Candidate assumptions must belong to an assigned target or reviewed calibration context")
        canonical = []
        for candidate in validated:
            registration = register_candidate(candidate, self.staged)
            self.staged.merge(registration)
            registered = registration.assumptions[0]
            canonical.append({name: registered[name] for name in (
                "subject", "field", "value", "unit", "statement", "basis", "entry_type", "derivation")
                if name in registered})
        changed_candidates = [candidate for candidate in canonical
                              if self.candidates.get(self.candidate_key(candidate)) != candidate]
        self.candidates.update((self.candidate_key(candidate), candidate) for candidate in canonical)
        for candidate in changed_candidates:
            self.run.emit_trace("target_input_candidate", status="proposed_not_admitted",
                episode_id=self.goal.goal_id, investigated_target_subjects=list(self.requested_subjects),
                subject=candidate["subject"], field=candidate["field"], value=candidate["value"],
                unit=candidate["unit"], statement=candidate["statement"],
                entry_type=candidate.get("entry_type"), derivation=candidate.get("derivation"),
                model_calls=self.run.model_calls)
        if action is None:
            return self.leaf(lambda: self.staged.result(), "inferred_parameters")
        approach, objective = text(proposal.get("approach"), "approach"), text(proposal.get("objective"), "objective")
        source = StrategySource(self.run, self.staged.result(), approach=approach)
        return self.edge(self.run.child(view, source, "strategy", objective,
            proposal=proposal, contract=self.contract(
                f"Find source values, bounds and context relevant to the assigned parameter: {objective}",
                subjects=self.requested_subjects,
                target_scenarios=self.parent_contract.measurements.get("target_scenarios"))))

    def assess(self, result):
        """Retain registered claims before common numeric grading and selection."""
        working = Ledger(self.staged.result())
        working.merge(result)
        # Also support restored episodes whose candidates predate registration.
        for key, candidate in self.candidates.items():
            if key not in working.assumptions:
                working.merge(register_candidate(candidate, working))
        return super().assess(working.result(evaluated=result.evaluated,
                                            stop_reason=result.stop_reason))



def run_guideline_research(*, disease, providers, research_request="", metrics=DEFAULT_METRICS, limits=None, scenario="bazant-bush"):
    """Caller-defined cases; evidence and scientific answers start empty."""
    disease = text(disease, "disease of interest")
    if not isinstance(research_request, str):
        raise ValueError("Additional research instructions must be text")
    metrics = tuple(dict.fromkeys(metrics))
    if set(metrics) != set(DEFAULT_METRICS):
        raise ValueError("The safety-guideline workflow requires allowable time, safe distance and R_in "
                         "in both steady and initially clean-room regimes. Omit --metric to use the "
                         "complete output contract; scenario selection does not reduce these quantities.")
    request = (f"Disease of interest: {disease}. Estimate allowable room time, risk-based safe distance, "
               "and R_in for the caller-selected fixed scenarios. Fix separation to solve time; "
               "fix duration to solve separation; fix both to evaluate R_in under the same model. "
               "Use the scenario's reference inputs for these calculations. Investigate evidence for the "
               "disease-dependent assumptions; the scenario catalog provides no infectiousness answers. "
               "Do not discover or replace scenarios, assign arbitrary exposure durations, or target "
               "published numerical answers. " + research_request).strip()
    from .scenario_sharing import initialize_parameter_scope
    initial = initialize_model_defaults(initial_scenarios(load_scenarios(scenario)),
        providers.scenario_parameter_assessor, disease=disease)
    initial = initialize_parameter_scope(initial,
        providers.scenario_parameter_assessor, disease=disease)
    return _run_research(research_request=request, providers=providers, metrics=DEFAULT_METRICS,
                         limits=limits, initial=initial, disease=disease,
                         scenario_selection=scenario)


def _run_research(*, research_request, providers, metrics=DEFAULT_METRICS, limits=None, initial=None,
                  disease=None, scenario_selection=None):
    """Shared method_loop composition; unfixed mode is used by synthetic fixtures."""
    if not isinstance(research_request, str) or not research_request.strip():
        raise ValueError("research_request must be a nonempty natural-language prompt")
    metrics = tuple(dict.fromkeys(metrics))
    if not metrics or any(m not in METRICS and m != "removal_rate_s" for m in metrics):
        raise ValueError("Unknown requested metric")
    run = ResearchRun(providers, limits or ResearchLimits(), research_request, metrics,
                      initial=initial, scenario_selection=scenario_selection)
    if disease is not None:
        run.context["disease_of_interest"] = disease
    variables = (set.intersection(*(set(s.get("calculation_variables", ())) for s in initial.scenarios))
                 if initial and initial.scenarios else set())
    focus = tuple(sorted((input_fields(metrics, parameters=UNITS) - variables) | set(metrics) |
                         {CALIBRATION_PROGRESS, GLOBAL_GOAL_PROGRESS} |
                         (set() if run.fixed_scenarios else {CONSISTENCY})))
    goal = EpisodeGoal.root(objective=f"{OBJECTIVE}\n\nResearch request: {research_request}",
        result_contract={"metrics": list(metrics),
            "requested_output_completion": "Every requested calculation row is calculated_up_to_date",
            "scenario_output_contracts": {s["subject"]: s.get("output_contracts", {})
                                          for s in initial.scenarios} if initial else {},
            "input_provenance_retained": True,
            "input_roles": "Requester constraints, source observations, source-model choices and named hypotheses remain distinct; "
                           "evidence claims require retained provenance, while hypotheses may be calculated before support exists."},
        task_context=run.context)
    source = GuidelineSource(run)
    episode = run.episode("evaluate_airborne_safety_guideline", source, goal=goal, focus=focus)
    record = episode.run(Context(tree=run.tree))
    return record, ResearchReport.from_result(source.build_result(record), record, run)
