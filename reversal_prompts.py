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
        return {'source_claims': [_claim(item) for item in request['source_claims']],
                'proposed_answer': _answer(request['answer']),
                'previous_answer': _answer(request.get('previous_answer')),
                'known_materials': {key: _answer(value) for key, value in
                                    request['known_materials'].items()}}
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
    'form_search': '''Develop one bounded search hypothesis for the numerically selected scope. Seek a biological process with two independently warranted causal inferences at question-specific grains; the inferences may be established in different papers, and papers need not recognize the philosophical tension. Identify the inference proposed for transport and the causal actor, relation, or intervention that may change its meaning. For general discovery, search broadly for a structural comparison in a new biological system or scale, without naming an unacquired case as a finding. For an existing inquiry, use priority_dimension as numerical decision feedback: seek source evidence for that missing causal inference or mapping, preferably a study independent of the source already supporting the other side. A source that merely disproves a previously imagined causal hypothesis is useful negative evidence, not warrant for two valid descriptions. If recent focused queries repeat one hypothesis without finding independent warrant, change the causal literature or organismal/collective grain being sought. Measured query outcomes guide scheduling, not scientific truth. State the paired-inference hypothesis, discriminating evidence, literal query, and short information need. The numerical controller decides continuation.''',

    'choose_source': '''Choose exactly one option_id from the unprocessed candidate pages for the stated information need. Prefer a source likely to establish the causal inference whose independent warrant or cross-frame mapping is currently missing, including a study from another paper or literature. Compare titles, descriptions, provider rank, and measured relevance probability; a topical match does not prove the source contains the needed causal result. An empirical result can warrant one frame without recognizing the pluralist problem. Do not invent source content from a snippet. Source selection is an acquisition decision, not scientific admission or a decision to stop. Give a concise reading objective tied to the missing inference.''',

    'choose_reading': '''Report reading advances through bounded windows in document order and retains exact earlier anchors; lexical reading ranks all eligible source chunks for a literal query; table reading searches all remaining table rows, not only the preview. Choose the mode that best addresses this information need in this source. After report reading, lexical reading may deliberately reexamine a changed or unresolved question. A table key must name a detected table. State a source-local question that can guide extraction. Do not infer that unread material lacks evidence, and do not treat the opening preview as the whole document.''',

    'extract_claims_and_discover': '''Extract source-reported claims and exact quotations contained within one supplied span. Do not concatenate spans or quote retained context as new evidence. Classify a fact as frame_one, frame_two, or mapping according to which causal description or relation between descriptions it informs; this label is source organization, not proof that either frame is complete. Primary means original empirical or theoretical research for the claim; an introduction describing another study is not primary evidence of that result. Return no facts when the span supports none. Retain concise unresolved questions and exact current-span anchors for interpretation across reading windows.

After extraction, propose source-grounded frame comparisons, including provisional and weak ones. Fill inference_pair with a causal inference genuinely warranted by each frame, exact fact labels warranting each, the proposed translation of the first inference, and the causal tension with the second. Empty warrant lists mean the side is unresolved; write that plainly in its inference field. A source may warrant only one side. Do not infer causation from a method that merely traces descent, association, location, or sequence. A later experiment falsifying that unwarranted inference is not a second valid causal description. A different paper may supply the missing side later, so state the research question precisely instead of fabricating its evidence. Every candidate must cite at least one fact label from this response; the side-specific warrant labels must resolve to facts extracted in this response. Candidate labels must be unique. same_case_as concerns the same biological process and question pair; a later identity operation compares against existing inquiries and independently groups conceptual themes. Empty candidates is valid. Discover cases from sources rather than a preset taxonomy.''',

    'synthesize_discovery': '''Reason across acquired source claims, including claims from different papers, to discover or revise question-specific causal-frame comparisons. For each proposed case, fill inference_pair: one causal inference actually warranted by the first frame, one independently warranted by the second, the inference proposed for transport, and the specific tension caused by translating actors or relations. Cite exact supplied source claims separately for each inference. If one side lacks warrant, give it an empty warrant list and state unresolved; retain the candidate as research memory while aiming subsequent search at that side. A paper need not state the philosophical problem itself. Test whether the first claim really licenses the transported prediction. If the second paper merely falsifies an earlier unsupported causal hypothesis, record that as invalidation, not pragmatic causal pluralism. A fixed-variable moderator or different outcome alone is also weak. Current theory is decision feedback, not evidence. Use an existing biological inquiry identity for new evidence on the same process and question pair; separate organisms can share a conceptual theme without sharing source facts. Every candidate must cite supplied claims; unshown claims are not evidence of absence. Record search hypotheses aimed at missing independent warrants or disputed category mappings, without a preset taxonomy.''',

    'resolve_candidate_identities': '''Assign two independent addresses to every candidate occurrence. same_case_as identifies the biological inquiry: use an existing case when the biological process and proposed inference translation are the same, even if a formerly unresolved second frame gains independent warrant and its question is sharpened. Do not merge different biological systems or different causal tensions merely because their abstract wording resembles. same_theme_as identifies the abstract causal translation problem: use a known case or earlier candidate whose theme matches, even across species or scales. For example, movement out of a compartment versus death of a cell may recur in different systems; those are separate cases in one theme, with separate source evidence and answers, and their identical credit regions are counted once. A new theme needs a specific difference in the construction of actors or relations and the attempted inference. When same_case_as points to an existing case, its theme is inherited, so choose a same_theme_as consistent with it. Prior biological-case and conceptual-theme aliases are identity memory. This is address resolution, not scientific admission: retain weak, strong, negative, and unresolved occurrences. Resolve equivalents within this batch to an earlier candidate label. Do not infer novelty from different wording or from a discovery proposal saying new. No candidate is excluded by this operation.''',

    'assess_case_construct': '''Develop the strongest defensible source-grounded account of the explicit linked inference pair. State each biological question, the actors and relations its grain constructs, the causal inference warranted there, and exact source claims warranting that inference. The two warrants may come from different papers; do not ask one paper to articulate both sides. State the prediction licensed by carrying the first inference to the second frame, then show whether the second warranted inference conflicts because the actor, relation, intervention, or system boundary changes. Check that the first source really licensed the transported prediction: a later experiment rejecting an unsupported ancestry, association, or location hypothesis is invalidation, not two valid causal descriptions. If either side remains unsupported, keep the pair explicitly unresolved. Explain each grain's pragmatic use and the biology of any mapping failure, separating source reports from your interpretation. State consequential uncertainties and the missing independent evidence. Positive, negative, and unresolved accounts remain research answers; issue no admission or stopping decision.''',

    'assess_case_challenge': '''Try to break the linked inference pair, using all supplied sources. For each side, ask whether its cited source establishes the exact causal inference written in inference_pair rather than a true but unrelated description. Test whether the first inference actually entails the proposed transported prediction. If the second finding simply refutes an earlier unsupported hypothesis, say so and put the warranted replacement claim in the second frame without pretending the rejected claim remains valid. Also test fixed-variable modifiers, changed measurements, and different outcomes. Correct the pair, evidence refs, and account when a narrower, negative, or unresolved conclusion is warranted. Keep valid source findings and identify independent evidence that could establish the missing side or mapping. This is an improvement attempt, not admission or a stopping decision.''',

    'measure_answer_quality': '''Measure the explicit linked inference pair on the declared continuous task-fit and epistemic-quality coordinates. For each warrant score, check the exact causal inference and its side-specific citations, not merely a true claim at that grain. For inference_reversal, ask whether the first warranted inference actually licenses proposed_translation and whether the second independently warranted inference is incompatible with it because the categories or relations change. Score near zero when the purported transported rule was never warranted and new evidence merely invalidates it, even if tracing or description in the first frame is useful. A careful negative answer can score highly for source fidelity and uncertainty while scoring low on task fit. Assess previous and proposed answers against the same claims; do not copy old scores. Specific uncertainty can improve quality; generic hedging cannot. Scores never admit, reject, or stop research. Resolve substantive material identity: paraphrase, verbosity, fresh rating, or longer quotation of the same assertion does not create a new contribution. A supported change to the paired causal claims or their translation can be new. Every attempt is stored.''',

    'develop_theory': '''Develop or revise one coherent account of pragmatic causal pluralism from acquired biological evidence and reviewed cases. Compare explicit linked causal inferences whose separate warrants may come from different papers. Explain how question-specific actors and relations make each inference useful and why transporting one to the other fails. Do not treat a new experiment overturning an earlier unsupported hypothesis as two valid causalities. Do not merely assemble effect reversals, say context matters, or claim a finer grain is fundamentally wrong. For collective effects, describe how interactions create the collective actor or outcome and what each grain leaves out. Separate source reports, case interpretations, and cross-case inference; retain negative cases and consequential uncertainty. Cite only supplied claims and cases. Provide three complete, single-sentence decision fields of at most 40 words each, with no paragraphs or literal search queries. next_frame_gap names a generalizable unresolved category or relation mapping, without names from the current case. next_evidence_need names the type of independent source finding that could establish a missing causal inference or revise the mapping in another biological system. avoid_pattern names a previously unproductive shortcut. These fields steer general discovery, not scientific conclusions. This is iterative reasoning, not admission or stopping.''',

    'measure_theory_quality': '''Measure the cross-case account continuously on the declared theory-fit and epistemic-quality coordinates. Strong theory fit requires source-backed linked causal inferences that both remain valid for their questions and a defended reason their translation fails. A catalog of newly falsified hypotheses, ordinary effect differences, or generic emergence does not supply that. Assess category construction and inference translation, not length or confidence. Compare previous and proposed accounts against the same claims; when no prior account exists, no prior measurement is due. A warranted negative or unresolved account can have high epistemic quality and low theory fit. Resolve material identity against known accounts: paraphrase, fresh rating, longer citation, another fact-store version, or changed next-search directive alone adds no scientific contribution. A corrected category mapping, independently warranted second inference, or changed inferential conclusion can be new. Every attempt is retained. Scores are continuous credit coordinates, never scientific gates or stopping decisions.''',
}


CAUSAL_TRANSFER_FRAMEWORK = '''A biological cause is stated using entities or categories and relations defined at a chosen grain. The grain is chosen for a biological question: a category such as organism, cell, lineage, or population makes some interactions visible and coarse-grains others. A well-supported causal claim at one grain can be genuinely useful without being a universally transportable component. The target is pragmatic causal pluralism: two descriptions can both be adequate for their own questions while a causal inference carried between them breaks because the actor, category, relation, system boundary, or intervention meaning changes. Do not declare one grain fundamentally wrong merely because it fails another question.

Record two frames as a reasoning grammar, not a scientific admission checklist. For each frame, state its biological question, chosen entities or categories, causal inference, and source warrant. The warrants may come from different papers; one paper need not contain or recognize both viewpoints. In inference_pair, name the exact inference licensed within frame one, the corresponding inference warranted within frame two, the first inference's proposed translation, and the causal tension. Cite source claims separately for each side. Ask whether the first warrant really licenses that translation. If a later experiment only rejects an earlier unsupported causal guess, one viewpoint has been corrected rather than two valid causalities exposed. A true descriptive claim about ancestry, association, or location cannot silently become a causal fate, mechanism, or intervention claim. Explain why each grain is pragmatically useful and what it hides. A frame or mapping may be provisional. Keep source reports, interpretive choices, and projected inferences distinct; never invent a second finding to complete the grammar. There is no universal list of sufficient context variables.

A partial SCM formalism can display the problem: a variable X_q and edge X_q → Y_q belong to a question-indexed causal description M_q, whose variables are constructed by a grain g_q. Translating an edge from M_1 to M_2 assumes that actors, interventions, and relations survive that change of grain. They need not. For collective phenomena, the macro actor can be constituted by interactions among its units, so intervention on a named unit need not mean intervention on the collective variable. Use an SCM to locate a failed inference, not to insist on one privileged ontology or to claim metaphysical irreducibility from an unexplained pattern.

A reviewed case records an editable causal argument. Each frame defines its question, actors, boundary, intervention, outcome, scope, and pragmatic use. Its steps distinguish source reports from inferences, cite exact source claims, and name earlier premises. The conclusion points to a step, or remains open. The translation maps the particular inference, names the bridge premise, and identifies a mistaken prediction, intervention choice, or causal role. A fixed-category alternative adds a cofactor, context variable, or dependency while keeping the same variable and intervention meanings. If that alternative explains the apparent reversal, the case is a near miss. A strong case explains the exact semantic reconstruction that the ordinary alternative misses. Build, challenge, and revise these links in any order; the representation is an objective for reasoning, not a mandated research sequence.

Schematic examples use placeholders and supply no named biological cases. Collective case: a property P gives an individual unit U access to resource R; as many U express P, their interactions construct a shared structure S that changes access to R. The individual-level and collective-level causal descriptions may both be true, while treating the individual advantage as a fixed population component fails. Relational-category case: an organism's role is defined by its interactions in a network; transporting its role category by organism name into a differently organized network can fail even when the organism remains identifiable. Mechanistic case: a useful coarse relation A → B summarizes a distributed pathway for one problem; another problem resolves feedback and interacting participants so that A and B no longer name a stable causal component. Near miss: paper one shows that X marks a lineage and paper two shows Y causes fate; if paper one never supported X causing fate, paper two has rejected an unwarranted guess, not reversed a valid causal inference. Another near miss: the same fixed actors X and Y show different effect sizes under moderator Z; that is ordinary effect modification unless the relevant entities or relations change. Significance versus nonsignificance alone says nothing about pragmatic causal pluralism. Judge every actual biological case from acquired evidence, not these placeholders.'''


TRANSFER_OPERATIONS = frozenset({
    'form_search', 'extract_claims_and_discover', 'synthesize_discovery',
    'resolve_candidate_identities', 'assess_case_construct',
    'assess_case_challenge', 'measure_answer_quality',
    'develop_theory', 'measure_theory_quality',
})


TRANSFER_STAGE_GUIDANCE = {
    'form_search': '''Use priority_dimension and the side-specific warrant claims to aim the next focused query at the weaker causal inference or translation. Seek another source that could establish that inference in its own explanatory frame; a study repeating the same invalidation is not independent warrant. In frame_search_hypothesis, state the linked inferences and proposed category change. In discriminating_evidence, name an observation that could support or defeat their mapping. General discovery may explore a new structural pattern but must not name an unacquired case as a finding.''',
    'extract_claims_and_discover': '''Treat an exact quote as support only for the relation it reports. Extraction from one source need not establish both frames. In a candidate, frame_one_question and frame_two_question name the problems; inference_pair states each causal inference and its own warrant labels, the proposed translation, and the tension. Use empty warrant labels for an unresolved side. Use causal_structure for category construction and generalization_defect for why the attempted transport matters. A new experiment disproving a causal claim never warranted by the earlier source is a near miss, not completed pragmatic causal pluralism.''',
    'synthesize_discovery': '''Connect claims from one or more papers into a linked inference pair. For each candidate, state both biological questions, the exact causal inference warranted in each, its separate source refs, and the proposed translation. A true first-frame claim about another outcome cannot warrant the prediction being tested in the second frame. If one causal inference lacks support, mark it unresolved with empty warrant refs; search hypotheses should seek literature that would independently establish that side, not another refutation of the same hypothesis. Current cross-case decision feedback guides the open question but is not evidence.''',
    'resolve_candidate_identities': '''For same_case_as, compare biological system, process, and question pair so source claims for one case cannot enter another case's answer. For same_theme_as, compare the category construction and specific translation failure across cases. A different species alone does not create a new theme; identical named actors do not make different translation defects the same.''',
    'assess_case_construct': '''Build or revise the argument at whatever part can now be warranted. In each frame, source_report steps cite exact claims; inference steps name earlier step IDs as premises and state the rule that reaches their conclusion. One step may suffice when it actually carries the inference; adding steps for appearance is useless. An open side has an empty conclusion_step_id. The translation names the particular source and target steps, maps actors, interventions, outcomes and relations, states the bridge premise, and describes the consequential inference that would fail. The fixed_category_alternative gives the strongest specific account using stable variables plus context or dependencies and says whether it reconciles the findings. Do not assert that no conceivable global graph exists. Keep unsupported parts explicitly provisional, then state the next evidence need.''',
    'assess_case_challenge': '''Inspect every cited step and premise link. Correct any source_report that is only descriptive while its inference step claims a causal intervention. Test whether the source inference licenses the transported consequence. Specify an ordinary fixed-category graph with added context or dependencies and determine whether it already reconciles the observations; if so, mark the candidate a near miss. Preserve useful local inferences and unresolved links. Revise the structured argument and next evidence need rather than issuing an admission or stopping decision.''',
    'measure_answer_quality': '''Assess every argument step under step_warrants, citing source claims that bear on it and scoring the actual inferential link, not its length or fluency. An unsupported premise makes every dependent conclusion weak. Rate first_frame_warrant and second_frame_warrant for the whole coherent derivations. Rate translation_specificity for the exact map and bridge; inference_reversal for a consequential failure between still-valid local inferences; fixed_category_challenge for whether a specified stable-variable graph is defeated; and pragmatic_pluralism for the distinct uses. A cofactor, omitted dependency, or refuted unsupported hypothesis earns low reversal credit. A careful negative answer can score highly for source fidelity and uncertainty. Assess previous and proposed answers against the same claims; repeated ratings and cosmetic step splits are the same material. Scores are continuous and never admit, reject, or stop research.''',
    'develop_theory': '''Identify patterns only from linked inference pairs whose separate warrants survive scrutiny. Explain each grain's pragmatic value and limits. Treat a refuted earlier hypothesis as an evidence correction, not pluralist causation. State the next independently supportable causal inference or category mapping whose evidence could change the account.''',
    'measure_theory_quality': '''Score the account's biological warrant and explanation of linked causal inferences that are both useful but do not transport. A list of falsified hypotheses, empirical oppositions, generic emergence language, or an assertion that one grain is fundamentally wrong is weak theory fit. Keep source fidelity and uncertainty on their separate coordinates.''',
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
