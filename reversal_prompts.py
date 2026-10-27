"""Decision-specific model messages for the biological discovery episodes.

Stored request dictionaries are the intermediate representation. This module
selects semantic fields, gives durable references temporary labels, and renders
an instruction prompt. The reverse label map stays with the provider call.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json


MAX_ITERATION_CHARS = 48000
MAX_PROMPT_CHARS = 64000
MAX_CONTRACT_CHARS = 32000
SEARCH_DESCRIPTION_CHARS = 1200


@dataclass(frozen=True)
class RenderedModelCall:
    instructions: str
    iteration_message: str
    prompt: str
    response_contract: dict
    decode_response: object


def _fields(value, names):
    if not isinstance(value, Mapping):
        return value
    return {name: value[name] for name in names if name in value}


def _answer(value):
    return _fields(value, ('conclusion', 'comparison_notes', 'explanation',
                           'inference_pair', 'argument', 'evidence_refs', 'uncertainties',
                           'next_search_need')) if value else None


def _theory_answer(value, *, compact=False):
    fields = (('conclusion', 'cross_case_mapping', 'case_refs', 'evidence_refs',
               'next_frame_gap', 'next_evidence_need', 'avoid_pattern') if compact else
              ('conclusion', 'cross_case_mapping', 'explanation', 'case_refs',
               'evidence_refs', 'uncertainties', 'next_frame_gap',
               'next_evidence_need', 'avoid_pattern'))
    return _fields(value, fields) if value else None


def _claim(value):
    return _fields(value, ('id', 'title', 'source', 'claim', 'quote', 'side', 'source_kind'))


def _short_text(value, limit):
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[:limit].rstrip() + f' [and {len(value)-limit} omitted characters]'


def _compact_text(value, limit):
    if isinstance(value, Mapping):
        return {key: _compact_text(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_text(item, limit) for item in value]
    return _short_text(value, limit)


def _measurement_claim(value, cited, limit):
    claim = _claim(value)
    result = _fields(claim, ('id', 'source', 'source_kind', 'side'))
    result['claim'] = _short_text(claim.get('claim', ''), limit if cited else limit // 2)
    if cited:
        result['quote'] = _short_text(claim.get('quote', ''), limit)
    return result


def _known_material_brief(value, limit):
    argument = value.get('argument') or {}
    generalization = argument.get('generalization') or {}
    return {'conclusion': _short_text(value.get('conclusion', ''), limit),
            'rule': _short_text(generalization.get('rule', ''), limit),
            'theories': [_short_text(item.get('theory', ''), limit // 2)
                         for item in generalization.get('supports', [])[:8]],
            'target_theory': _short_text(generalization.get('target_theory', ''), limit)}


def _previous_measurement_brief(value, limit):
    if not value:
        return None
    argument = value.get('argument') or {}
    generalization = argument.get('generalization') or {}
    return {'conclusion': _short_text(value.get('conclusion', ''), limit),
            'rule': _short_text(generalization.get('rule', ''), limit),
            'domain': _short_text(generalization.get('proposed_domain', ''), limit),
            'frame_conclusions': {
                name: _short_text(next((step.get('claim', '') for step in
                    (argument.get(name) or {}).get('steps', [])
                    if step.get('id') == (argument.get(name) or {}).get('conclusion_step_id')), ''), limit)
                for name in ('frame_one', 'frame_two')},
            'supports': [{key: _short_text(item.get(key, ''), limit)
                          for key in ('id', 'question', 'theory', 'consequence',
                                      'relation_to_rule', 'status')}
                         for item in generalization.get('supports', [])],
            'repairs': [{'id': item.get('id'), 'status': item.get('status'),
                         'formal_change': _short_text(
                             (item.get('addition') or {}).get('formal_change', ''), limit)}
                        for item in generalization.get('repair_tests', [])],
            'evidence_refs': value.get('evidence_refs', [])}


def _measurement_context(request):
    answer = request['answer']
    cited = set(answer.get('evidence_refs') or [])
    known = request.get('known_materials') or {}
    recent_known = list(known.items())[-8:]
    source_claims = request['source_claims']
    cited_claims = [item for item in source_claims if item['id'] in cited]
    uncited_claims = [item for item in source_claims if item['id'] not in cited]
    def core(value):
        return _fields(value, ('conclusion', 'argument')) if value else None
    for limit in (450, 320, 240, 180, 150, 120, 90):
        selected = {
            'cited_source_claims': [_measurement_claim(item, True, limit)
                                    for item in cited_claims],
            'other_claim_summaries': [{
                'id': item['id'], 'claim': _short_text(item.get('claim', ''), 80)}
                for item in uncited_claims[:8]],
            'unshown_other_claim_count': max(0, len(uncited_claims)-8),
            'inquiry_generalization': _compact_text(
                request.get('inquiry_generalization'), limit),
            'proposed_answer': _compact_text(core(answer), limit),
            'previous_answer': _previous_measurement_brief(
                request.get('previous_answer'), limit),
            'known_materials': {key: _known_material_brief(value, limit)
                                for key, value in recent_known},
            'unshown_known_material_count': len(known) - len(recent_known),
            'text_field_limit': limit,
        }
        if len(json.dumps(selected, ensure_ascii=False, separators=(',', ':'))) <= MAX_ITERATION_CHARS - 1000:
            return selected
    raise ValueError('Review argument is too large for a bounded measurement view')


def _pair_brief(value):
    return _fields(value, ('frame_one_causal_inference',
                           'frame_two_causal_inference',
                           'proposed_translation', 'causal_tension'))


def _acquisition_update(value):
    if not value:
        return None
    delta = value['delta']
    return {'new_readings': delta['new_source_readings'],
            'new_claims': delta['new_source_claims'],
            'new_inquiries': delta['new_inquiries'],
            'ended_by': value['completion']['ended_by']}


def _case(value, *, facts=False):
    if not isinstance(value, Mapping):
        return value
    selected = _fields(value, ('key', 'title', 'topic', 'theme_id', 'causal_structure',
                               'generalization_defect', 'identity_rationale',
                               'inference_pair',
                               'frame_one_question', 'frame_one_claim',
                               'frame_two_question', 'frame_two_claim',
                               'reading_notes'))
    if value.get('current_answer'):
        selected['current_answer'] = _answer(value['current_answer'])
    if facts and value.get('facts'):
        selected['source_claims'] = [_claim(item) for item in value['facts']]
    if value.get('occurrences'):
        selected['occurrences'] = []
        for item in value['occurrences']:
            occurrence=_fields(item, ('title', 'causal_structure',
                'generalization_defect', 'frame_one_question', 'frame_one_claim',
                'frame_two_question', 'frame_two_claim','identity_rationale'))
            if item.get('inference_pair'):
                occurrence['inference_pair']=_pair_brief(item['inference_pair'])
            selected['occurrences'].append(occurrence)
    return selected


def _select(request):
    """Choose only content needed for this operation's next decision."""
    op = request['operation']
    if op == 'form_search':
        context = ({'focus': request['focus']} if request.get('focus') else {})
        cross_case=(_fields(request['cross_case_feedback'],
                            ('next_frame_gap','next_evidence_need','avoid_pattern'))
                    if request.get('cross_case_feedback') else None)
        feedback = [{'query': item.get('literal_query'),
                     'new_inquiries': item.get('new_candidate_count', 0),
                     'new_claims': item.get('new_claim_count', 0),
                     'new_readings': item.get('new_reading_count', 0),
                     'learning_share': item.get('learning_share')}
                    for item in request.get('search_feedback', [])
                    if item.get('literal_query')]
        measured_queries = {item['query'] for item in feedback}
        return {'acquisition_scope': request['acquisition_scope'],
                **context, 'cross_case_feedback':cross_case,
                'recent_queries': [query for query in
                    request.get('previous_queries', [])[-4:] if query not in measured_queries],
                'measured_search_feedback': feedback[-12:],
                'unshown_search_feedback_count':max(0,len(feedback)-12)}
    if op == 'choose_source':
        return {'information_need': request['information_need'],
                'candidate_pages': [{**_fields(item, ('option_id', 'provider_rank',
                    'title', 'url')),
                    'description':str(item.get('description',''))[:SEARCH_DESCRIPTION_CHARS],
                    'description_characters':len(str(item.get('description',''))),
                    'relevance_probability': (item.get('jev_assessment') or {}).get('relevance_probability')}
                    for item in request['candidates']],
                'previous_page_update': _acquisition_update(request.get('previous_page_update'))}
    if op == 'choose_reading':
        summary = request['source_summary']
        return {'information_need': request['information_need'],
                'source': _fields(summary, ('title', 'source', 'characters', 'opening',
                    'unexamined_characters', 'fully_examined_chunks', 'outline', 'tables')),
                'available_modes': request['available_modes'],
                'previous_reading_update': _acquisition_update(request.get('previous_reading_update'))}
    if op == 'extract_claims_and_discover':
        return {'reading_question': request['question'],
                'source_title': request['source_title'],
                'source_spans': {key: item['text'] for key, item in request['source_spans'].items()},
                'retained_context': {'anchors': [_fields(item, ('span', 'quote')) for item in
                                                request['retained_context'].get('anchors', [])],
                                     'unresolved': request['retained_context'].get('unresolved', [])}}
    if op == 'synthesize_discovery':
        fresh = set(request.get('new_fact_ids', []))
        claims = [{**_claim(item), 'new_this_cycle': item['id'] in fresh}
                  for item in request['source_claims']]
        return {'source_claims': claims,
                'unshown_fact_count': request['unshown_fact_count'],
                'current_cross_case_decision_feedback':(_fields(request['cross_case_account'],
                    ('next_frame_gap','next_evidence_need','avoid_pattern'))
                    if request.get('cross_case_account') else None),
                'unshown_inquiry_count':request.get('unshown_inquiry_count',0),
                'known_inquiries': [_fields(item, ('key', 'title',
                    'frame_one_question','frame_two_question',
                    'causal_structure', 'generalization_defect','inference_pair'))
                    for item in request['known_inquiries']]}
    if op == 'resolve_candidate_identities':
        return {'known_biological_inquiries': [_case(item) for item in
                                               request['known_conceptual_inquiries']],
                'unshown_inquiry_count': request.get('unshown_inquiry_count',0),
                'candidate_occurrences': [_fields(item, ('label', 'title', 'topic',
                    'causal_structure', 'generalization_defect',
                    'frame_one_question', 'frame_one_claim',
                    'frame_two_question', 'frame_two_claim',
                    'inference_pair',
                    'identity_rationale', 'same_case_as'))
                    for item in request['candidate_occurrences']]}
    if op.startswith('assess_case_'):
        hypothesis = request['discovered_hypothesis']
        return {'source_claims': [_claim(item) for item in request['source_claims']],
                'current_answer': _answer(request.get('current_answer')),
                'discovered_hypothesis': _case(hypothesis)}
    if op == 'measure_answer_quality':
        return _measurement_context(request)
    if op == 'develop_theory':
        return {'source_claims': [_claim(item) for item in request['source_claims']],
                'reviewed_cases': [_case(item) for item in request['reviewed_cases']],
                'current_account': _theory_answer(request.get('current_answer'),compact=True),
                'unshown_fact_count': request['unshown_fact_count']}
    if op == 'measure_theory_quality':
        return {'source_claims': [_claim(item) for item in request['source_claims']],
                'reviewed_cases': [_case(item) for item in request['reviewed_cases']],
                'proposed_account': _theory_answer(request['answer']),
                'previous_account': _theory_answer(request.get('previous_answer'),compact=True),
                'known_materials': {key: _theory_answer(value) for key, value in
                                    request['known_materials'].items()}}
    raise ValueError(f'No instruction template for model operation {op}')


INSTRUCTION_TEMPLATES = {
    'form_search': '''Develop one bounded search hypothesis for the numerically selected scope. For general discovery, seek a source-grounded causal rule with an explicit biological domain and a theory it could support. For an existing inquiry, use the stated rule and domain to seek either a genuinely distinct theory it supports, independent evidence for a target theory, or evidence discriminating a concrete repair. Search across organisms when the rule claims to generalize across them; do not use organism change itself as the putative failure. If repeated queries find no new warrant, change the biological question, scale, or plausible target. State the rule, its proposed domain, the theory or repair to investigate, discriminating evidence, literal query, and concise information need. Measured query outcomes guide scheduling, not scientific truth. The numerical controller decides continuation.''',

    'choose_source': '''Choose exactly one option_id from the unprocessed candidate pages for the stated information need. Prefer a source likely to establish the causal inference whose independent warrant or cross-frame mapping is currently missing, including a study from another paper or literature. Compare titles, descriptions, provider rank, and measured relevance probability; a topical match does not prove the source contains the needed causal result. An empirical result can warrant one frame without recognizing the pluralist problem. Do not invent source content from a snippet. Source selection is an acquisition decision, not scientific admission or a decision to stop. Give a concise reading objective tied to the missing inference.''',

    'choose_reading': '''Report reading advances through bounded windows in document order and retains exact earlier anchors; lexical reading ranks all eligible source chunks for a literal query; table reading searches all remaining table rows, not only the preview. Choose the mode that best addresses this information need in this source. After report reading, lexical reading may deliberately reexamine a changed or unresolved question. A table key must name a detected table. State a source-local question that can guide extraction. Do not infer that unread material lacks evidence, and do not treat the opening preview as the whole document.''',

    'extract_claims_and_discover': '''Extract source-reported claims and exact quotations contained within one supplied span. Do not concatenate spans or quote retained context as new evidence. Classify a fact as frame_one, frame_two, or mapping according to which causal claim it informs; that label is not proof. Source_kind describes the document actually supplied: a study title or reference listed in a review remains review material, never a primary result. Primary means the supplied document itself reports original empirical or theoretical research for the claim. A cited title alone is a search lead, not evidence that its reported result was verified. Return no facts when the span supports none. Retain unresolved questions and exact current-span anchors. After extraction, propose source-grounded candidates, including provisional ones. In inference_pair, state the source causal inference, a proposed generalizable rule, its biological domain and scope rationale, any target inference and attempted transport, and exact warrant labels for each side. An empty warrant list means that side is unresolved. A source fact does not automatically license broad generalization. A later result refuting an unwarranted guess is not a second valid causal description. A candidate must cite at least one fact label from this response; side-specific labels must resolve to extracted facts. Candidate labels must be unique. same_case_as concerns the same proposed rule and domain, even when later support comes from another organism covered by that domain. Empty candidates is valid.''',

    'synthesize_discovery': '''Reason across acquired source claims to identify or revise proposed generalizable causal rules. For each candidate, fill inference_pair with the exact source inference, generalizable_rule, proposed_domain, domain_rationale, distinct target inference if available, proposed translation, and specific tension. Cite supplied source claims separately for each inference. If target warrant is missing, leave its warrant list empty and state the open question. Seek hypotheses or theories that the same rule supports through an explicit derivation and, where needed, a formal addition that preserves earlier predictions. A different organism can be a valid support or challenge when the rule's domain includes it; organism change alone is not the causal failure. Distinguish evidence invalidating an unsupported guess from two useful constructions. Current theory is decision feedback, not evidence. Use an existing inquiry identity for evidence concerning the same proposed rule and domain. Every candidate cites supplied claims; unshown claims are not evidence of absence.''',

    'resolve_candidate_identities': '''Assign two independent addresses. same_case_as identifies one proposed generalizable causal rule and its biological domain: merge later occurrences that add a distinct supported theory, target, or repair test to that same rule, including occurrences in different organisms when the rule claims to span them. Do not merge genuinely different rules merely because the topic or wording resembles. same_theme_as groups the specific kind of causal translation or repair failure across different rules. A change of species alone creates neither a new rule nor a new theme. Preserve separate evidence and conclusions for genuinely different rules. Prior aliases are identity memory. This is address resolution, not scientific admission; retain weak, strong, negative, and unresolved occurrences. Resolve equivalents within this batch to an earlier candidate label. No candidate is excluded by this operation.''',

    'assess_case_construct': '''Build or revise the strongest source-grounded argument around one proposed generalizable inference. In frame_one, show source reports and inferential steps warranting the local causal rule. In generalization, specify that rule, its premises, variable meanings, abstraction map, proposed biological domain, domain warrant and exact citations. Add distinct supported theories with their causal questions, predictions, explicit derivations, typed graph edits for every nonempty addition, relation_to_rule application/component/restatement/speculative, and warranted/provisional/unsupported status; explain how each addition preserves the core rule and earlier uses. A plausible upstream mechanism or next experiment is provisional until independently supported. In frame_two, warrant an independent target theory if evidence is available. State whether the target falls in the proposed domain, the transported prediction, and the causal conflict. Test concrete repairs and record their typed graph edits, formal changes, independent constraints, target predictions and IDs of earlier supported theories whose use is tested. If a repair works and yields a distinct warranted application, include that target as a supported theory with its formal addition. If no target or repair is warranted, keep it provisional, with empty citations and open conclusions. Do not claim that no possible global graph exists. State the next evidence need. Build, challenge and revise in any order.''',

    'assess_case_challenge': '''Challenge the local inference, proposed domain, every supported theory and its derivation. Check whether each claimed new theory has a genuinely distinct prediction and cited warrant, and whether its formal addition preserves earlier uses. Ask whether the target theory is independently warranted and lies inside the justified domain. Test the strongest simple repairs, including context variables, missing dependencies, scale maps or parameter assumptions where relevant. For each repair, ask whether it predicts the target with independently constrained inputs and retains the prior theories. If one does, mark it works and narrow the conclusion; if uncertain, mark unresolved. An organism switch by itself, a later refutation of an unwarranted guess, or mere computational difficulty is not a failed generalization. Preserve useful local findings and state discriminating evidence.''',

    'measure_answer_quality': '''Measure the explicit generalization and challenge on continuous task-fit and epistemic coordinates. Rate each source and inference step under step_warrants. Under support_warrants, independently set reviewed_relation for each theory: application only when it answers a genuinely new causal question with a source-warranted consequence beyond the proof and predictions already used to establish G; component for a conjunct, subpath, measurement readout, explanatory restatement, or original prediction of G; restatement for a paraphrase or extra citation; speculative for a plausible but untested mechanism. Do not defer to the answer's declared relation_to_rule. Also set evidence_basis: direct_result only when the supplied primary source itself reports the relevant finding or derivation; secondary_summary when the supplied review actually summarizes it; citation_title when only a cited title or reference list mentions the finding; unsupported otherwise. A title quoted in a review remains citation_title even if its extracted source_kind was mislabeled primary. In the rationale, explain both verdicts and then rate source warrant, derivation from the same rule, new consequence, and preservation under its formal addition. Components, restatements, speculation, citation titles, and unsupported claims receive no reuse credit. Check the declared status; an unwarranted theory is provisional. Under repair_warrants, rate the evidence and reasoning for each tested repair's target prediction, independent constraints, and effect on prior uses. Compare the reviewed rule with inquiry_generalization and previous_answer; a materially different rule needs an explicit source-grounded revision rather than receiving reuse credit under the old inquiry. Rate generalization_warrant and target_domain_warrant from the stated scope, not from a blanket organism rule. Rate target_theory_warrant independently. repair_audit and predictive_loss require concrete comparisons; a viable simple repair gets zero large-failure credit. The review handoff abbreviates long text fields, shows full claim excerpts only for cited claims, and may summarize or omit other claims; treat truncation and unshown claims as uncertainty, not evidence of absence. A careful negative answer can score well for source fidelity while scoring low on task fit. Assess previous and proposed answers against the same claims. Scores do not admit, reject, or stop research. Treat cosmetic rewrites as the same material.''',

    'develop_theory': '''Develop or revise an account of pragmatic causal pluralism from acquired inferences and reviewed cases. Compare explicitly stated generalizable rules, the distinct theories each supports, their formal additions, and independently warranted targets where transport and plausible repairs lose usefulness. Explain why each original and target theory remains useful for its own question. A change of organism can supply a relevant test only when the proposed domain includes it; organism difference itself is not an explanation. Do not turn a refuted unsupported hypothesis into two valid causalities. Separate source reports, case interpretations and cross-case inference. Cite only supplied claims and cases. Provide three single-sentence decision fields of at most 40 words each. next_frame_gap names an unresolved generalizable rule, support derivation, target-domain test or repair class. next_evidence_need names independent evidence that could resolve it. avoid_pattern names an unproductive shortcut. These guide discovery, not truth or stopping.''',

    'measure_theory_quality': '''Rate a cross-case account for explicit source-grounded generalizable inferences, distinct supported theories, formal additions, warranted target theories, and the predictive consequences of tested repairs. A list of opposite effects, organism switches, refuted unsupported guesses, or generic emergence claims is weak theory fit. Keep source fidelity and uncertainty on separate coordinates. Scores are continuous credit coordinates, never scientific gates or stopping decisions.''',
}


CAUSAL_TRANSFER_FRAMEWORK = '''The research object is a proposed generalizable causal inference G. State its source-grounded premises, causal rule, intervention and outcome meanings, abstraction or grain, and proposed domain. Explain why available evidence makes that scope plausible. A source fact is not automatically a universal rule. The domain of G determines which organisms and scales are relevant: an ecological rule about herbivores may properly span different animals; a lineage-specific rule may not. Changing organism alone is not a reversal, and a target outside G's justified domain is not a valid failure of generalization.

Show distinct theories or hypotheses H_i that G supports. A counted application answers a new causal question with a prediction beyond G's own source proof; a conjunct, subpath, measurement readout, or restatement of G is a component, not reuse. For each, state the question, prediction or causal consequence, derivation from G, exact source warrant, relation_to_rule, and any addition Delta_i. An addition must use graph edits: add_variable, add_edge, change_mapping, or constrain_parameter, naming the terms and expression for each edit. The kind none uses an empty edit list. Explain why earlier supported uses remain valid. A paraphrase, another citation of the same consequence, a merely plausible mechanism, or a suggested research direction does not add support. Mark each theory warranted only when its consequence and derivation have actual source warrant; otherwise mark it provisional or unsupported. The system may accumulate these supports in any order.

For a challenge, state an independently warranted target theory H_star, the exact inference transported from G, and the prediction that fails or becomes unhelpful. Test concrete plausible repairs: context variables, missing dependencies, scale mappings, or parameter fits as appropriate. For each, state typed graph edits and their formal change, independent constraints, target prediction, and the IDs of earlier supported theories whose use is tested. A repair that predicts the target and preserves useful prior predictions is an ordinary extension. If it yields a distinct, warranted new theory, record that theory as a support with the formal addition; the repair earns no failure bonus. A stronger finding shows that plausible tested repairs require target-specific fitting, lose predictive discrimination, or damage prior support. Do not claim that no conceivable larger graph exists. A microscopic statement may remain true while failing as a useful predictive component of a coarse-grained theory.

The argument's two causal frames remain useful: the first warrants G locally and the second warrants the target theory. Each frame has source-report and inference steps with explicit premises and citations. The translation states which inference from G is carried into the target and how actors, interventions, outcomes, and relations map. The fixed-category alternative is the strongest simple repair, not an assertion that context matters. Keep provisional gaps explicit. This is an objective and credit grammar, not a fixed research schedule.'''


GENERALIZATION_SCOPE_RULE = '''Always name the nature of the proposed generalizability before treating a target as a test. Specify the rule's domain in biological and causal terms, why the source inference might extend to that domain, and what changes of organism, population, or scale the rule claims to tolerate. Search may cross organisms when the stated and warranted domain calls for it. Rate a challenge by its fit to that domain and its failure of predictive or explanatory reuse, never by organism difference alone. In argument.generalization, record supported theories, typed formal additions, target-domain reasoning, and tested repairs with source references. A repair that still predicts and preserves earlier uses cannot earn a large failure reward.'''


TRANSFER_OPERATIONS = frozenset({
    'form_search', 'extract_claims_and_discover', 'synthesize_discovery',
    'resolve_candidate_identities', 'assess_case_construct',
    'assess_case_challenge', 'measure_answer_quality',
    'develop_theory', 'measure_theory_quality',
})


TRANSFER_STAGE_GUIDANCE = {
    'form_search': '''Use the current rule, domain, credit components and evidence gaps to choose a consequential query: a new supported consequence, a target within the claimed domain, or a repair's discriminating prediction. Choose based on what could most improve the argument; there is no required order. A query is a hypothesis about evidence, not a claimed finding.''',
    'extract_claims_and_discover': '''Separate source facts from a candidate generalizable rule. In inference_pair, name generalizable_rule, proposed_domain and domain_rationale explicitly. An unsupported target remains open; do not fabricate an opposed inference.''',
    'synthesize_discovery': '''Prefer a distinct, independently testable consequence of the same rule or a target theory and repair test over a topical analogy. Keep domain warrant and source citations visible.''',
    'resolve_candidate_identities': '''Group uses of the same rule under one inquiry even when the rule's warranted domain includes different organisms. Keep different rules distinct even if they share organisms or a failure theme.''',
    'assess_case_construct': '''The generalization object carries the rule, domain, supporting theories and typed additions, target-domain comparison, and repair tests. Source_report steps cite exact claims; inference steps name earlier premises. A counted application needs a distinct causal question and source-supported consequence beyond the source proof of the rule, an explicit derivation from that rule, relation_to_rule application, and warranted status. Mark subpaths, conjuncts and readouts of the original model as component or restatement. A research direction or plausible mechanism is provisional. A repair needs named graph edits, a formal change and tested_support_ids linking it to earlier supported theories; a failed repair with no tested prior use cannot support a large failure bonus. Empty arrays are valid for provisional work.''',
    'assess_case_challenge': '''Seek the strongest useful repair. A context variable or added edge counts as an ordinary extension when it predicts the target and preserves earlier support. A claim that repairs fail needs explicit, source-grounded tests of independent constraint and predictive use.''',
    'measure_answer_quality': '''Rate each support under support_warrants and each repair under repair_warrants. Penalize components of the original rule counted as separate theories, duplicate consequences, speculative mechanisms presented as warranted theories, unwarranted derivations, target-only fitting and claims of universal impossibility. A target outside the warranted domain gives no failure bonus.''',
    'develop_theory': '''Compare the same rule's uses and losses of use across reviewed arguments. Say which formal additions still preserve generalization and which target theory demands a different causal construction.''',
    'measure_theory_quality': '''Credit a theory of generalization limits only when it preserves the distinct uses and repair outcomes in the cited reviewed cases.''',
}


def _kind(value):
    if value.startswith('case_'):
        return 'inquiry'
    if value.startswith('theme_'):
        return 'theme'
    if value.startswith('evidence:'):
        return 'evidence'
    if value.startswith('answer:'):
        return 'material'
    if value.startswith('page:'):
        return 'page'
    if value.startswith('chunk:'):
        return 'span'
    if value.startswith('table:'):
        return 'table'
    return None


def _identities(value, register):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and _kind(key):
                register(key)
            _identities(item, register)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _identities(item, register)
    elif isinstance(value, str) and _kind(value):
        register(value)


def _replace(value, aliases):
    if isinstance(value, Mapping):
        return {_replace(key, aliases): _replace(item, aliases) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace(item, aliases) for item in value]
    if isinstance(value, tuple):
        return [_replace(item, aliases) for item in value]
    if isinstance(value, str):
        if value in aliases:
            return aliases[value]
        for identity, label in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
            value = value.replace(identity, label)
        return value
    return value


def _render_value(value, depth=0):
    pad = '  ' * depth
    if isinstance(value, Mapping):
        lines = []
        for key, item in value.items():
            if item is None or item == [] or item == {}:
                continue
            label = key.replace('_', ' ')
            if isinstance(item, (Mapping, list)):
                lines.append(f'{pad}{label}:')
                lines.extend(_render_value(item, depth + 1))
            else:
                lines.append(f'{pad}{label}: {item}')
        return lines
    if isinstance(value, list):
        lines = []
        for index, item in enumerate(value, 1):
            if isinstance(item, (Mapping, list)):
                lines.append(f'{pad}{index}.')
                lines.extend(_render_value(item, depth + 1))
            else:
                lines.append(f'{pad}{index}. {item}')
        return lines
    return [f'{pad}{value}']


def _search_context_lines(selected):
    lines = [f'scope: {selected["acquisition_scope"]}']
    focus = selected.get('focus')
    if focus:
        for name, value in focus.items():
            if value:
                if isinstance(value,(Mapping,list)):
                    lines.append(name.replace('_',' ')+':')
                    lines.extend(_render_value(value,1))
                else:
                    lines.append(f'{name.replace("_", " ")}: {" ".join(str(value).split())}')
    cross_case = selected.get('cross_case_feedback')
    if cross_case:
        lines.append('current cross-case decision feedback:')
        for name in ('next_frame_gap','next_evidence_need','avoid_pattern'):
            if cross_case.get(name):
                lines.append(f'- {name.replace("_", " ")}: {" ".join(str(cross_case[name]).split())}')
    if selected.get('recent_queries'):
        lines.append('recent queries:')
        lines.extend(f'- {query}' for query in selected['recent_queries'])
    if selected.get('measured_search_feedback'):
        lines.append('measured feedback (query | readings | claims | inquiries | learning share):')
        lines.extend(f'- {item["query"]} | {item["new_readings"]} | {item["new_claims"]} | '
                     f'{item["new_inquiries"]} | {item["learning_share"]}'
                     for item in selected['measured_search_feedback'])
    if selected.get('prior_response_failures'):
        lines.append('response failures:')
        lines.extend(f'- {item["error_type"]}: {item["error"]}'
                     for item in selected['prior_response_failures'])
    return lines


def render_reversal_request(request):
    """Keep standing instructions and selected iteration updates distinct."""
    selected = _select(request)
    aliases = {}
    counts = {}

    def register(identity):
        if identity not in aliases:
            kind = _kind(identity)
            counts[kind] = counts.get(kind, 0) + 1
            aliases[identity] = f'ref_{kind}_{counts[kind]}'

    _identities(selected, register)
    # A response contract may contain an admissible reference absent from the
    # selected content; its temporary label still belongs to this call only.
    _identities(request['response_contract'], register)
    selected = _replace(selected, aliases)
    contract = _replace(request['response_contract'], aliases)
    op = request['operation']
    if op not in INSTRUCTION_TEMPLATES:
        raise ValueError(f'No instruction template for model operation {op}')
    instructions = 'Standing research task:\n' + request['research_question']
    if op in TRANSFER_OPERATIONS:
        instructions += '\n\nPragmatic causal frames and inference translation:\n' + CAUSAL_TRANSFER_FRAMEWORK
        instructions += '\n\nExplicit generalizability requirement:\n' + GENERALIZATION_SCOPE_RULE
    instructions += '\n\nOperation instructions:\n' + INSTRUCTION_TEMPLATES[op]
    if op in TRANSFER_STAGE_GUIDANCE:
        instructions += '\n\nStage-specific use of the framework:\n' + TRANSFER_STAGE_GUIDANCE[op]
    if op in ('measure_answer_quality','measure_theory_quality'):
        heading='Candidate-fit dimensions' if op=='measure_answer_quality' else 'Theory-fit dimensions'
        dimensions = (heading+':\n'
                      + '\n'.join(f'- {name}: {meaning}' for name, meaning in request['rubric'].items())
                      + '\n\nEpistemic-quality dimensions:\n'
                      + '\n'.join(f'- {name}: {meaning}' for name, meaning in
                                  request['quality_rubric'].items()))
        instructions += '\n\n' + dimensions
    if request.get('prior_attempts'):
        failures = [{'error_type':item.get('error_type'),
                     'error':str(item.get('error', '')).splitlines()[0][:240]}
                    for item in request['prior_attempts']]
        selected['prior_response_failures'] = failures
        instructions += '\n\nIf the iteration message includes a prior response failure, correct that contract error in this response.'
    if op == 'measure_answer_quality':
        message = json.dumps(selected, ensure_ascii=False, separators=(',', ':'))
    else:
        context_lines = (_search_context_lines(selected) if op == 'form_search'
                         else _render_value(selected))
        message = '\n'.join(context_lines)
    if len(message) > MAX_ITERATION_CHARS:
        raise ValueError(f'{op} Episode context exceeds its {MAX_ITERATION_CHARS}-character bound; select addressable argument records')
    if aliases:
        instructions += '\n\nUse only the temporary reference labels supplied for this call. Do not invent a reference.'
    prompt = instructions + '\n\nIteration message:\n' + message
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f'{op} model prompt exceeds its {MAX_PROMPT_CHARS}-character bound')
    if len(json.dumps(contract, ensure_ascii=False)) > MAX_CONTRACT_CHARS:
        raise ValueError(f'{op} response contract exceeds its {MAX_CONTRACT_CHARS}-character bound')
    reverse = {label: identity for identity, label in aliases.items()}
    return RenderedModelCall(instructions, message, prompt, contract,
                             lambda response: _replace(response, reverse))
