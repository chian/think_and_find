"""Question-driven biological case discovery, composed from the existing Episode."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

from .method_loop import (ChildEpisodeUnit, ClosedRecord, Context, Episode, EpisodeGoal,
                          EpisodeRequest, EpisodeTree, GoalPreview, GoalProposal, Grain,
                          Leaf, ReportContract, SourceEnd)
from .method_loop.episode_binding.web_search_binding import SearchPageProposer
from .case_reports.research.documents import Document, Span, rank_spans
from .case_reports.research.model_tiers import ModelTier
from .configured_providers import reversal_research_providers
from .task_executors.outbreak_search.queries import source_locator, source_text
from .credit_assignment.attainment import AttainmentArchive, hypervolume
from .causal_argument import (argument_brief, argument_contract, argument_progress, argument_refs,
                              normalize_argument, step_ids)
from .reversal_prompts import MAX_ITERATION_CHARS


RUBRIC = {
    'generalization_warrant': 'Measure whether the same particular source-grounded inference is about a natural biological phenomenon: name the biological cause or contrast, biological outcome, causal relation, variable meanings, and justified domain. Laboratory and engineered organisms are natural systems for this purpose. A model may warrant the inference when its mapping to the biology is supported. Score 0 for an algorithm-only rule whose purported cause is changing a model input or constraint and whose outcome is only a calculated output, even if its operation is mechanically certain; also score 0 for an uncited universal restatement or unrelated substituted rule. Score 0.5 for an incompletely warranted biological scope and 1 for a well-specified, supported biological inference. A changed rule needs an explicit source-grounded revision. The domain determines relevant targets.',
    'target_domain_warrant': 'Measure whether the target theory falls within the proposed inference domain or a source-justified extension of it: 0 an unsupported leap beyond that domain; 0.5 plausible but uncertain inclusion; 1 well-supported inclusion. A different organism is allowed when the rule genuinely generalizes across organisms; organism change alone is not failure.',
    'target_theory_warrant': 'Measure independent empirical or theoretical warrant for the target hypothesis or theory and its causal predictions: 0 imagined or merely the negation of the proposed rule; 0.5 partial support; 1 a useful independently warranted target theory.',
    'repair_audit': 'Measure whether the argument explicitly tests the strongest plausible repairs to the proposed inference, including formally stated context variables, dependencies, mappings, or fitted assumptions where relevant: 0 no concrete repair, a viable repair ignored, or a bare assertion of impossibility; 0.5 a partial, source-grounded audit; 1 the pertinent simple repairs are specified and their prediction and preservation effects are compared.',
    'predictive_loss': 'Measure whether proposed repairs lose reusable predictive or explanatory value: 0 a simple independently constrained repair predicts the target and retains earlier supported uses; 0.5 a consequential but unresolved loss; 1 tested plausible repairs need target-specific fitting, lose independent constraint or discrimination, or damage prior supported hypotheses. Do not claim universal impossibility.',
    'first_frame_warrant': 'Continuously measure source support for the first biological causal inference in the explicit inference pair, at the grain suited to its question: 0 merely descriptive, invented, refuted, or only the fact that changing an algorithm input changes its computed output; 0.5 a partly warranted biological causal relation; 1 a warranted inference among biological actors and relations. A model prediction needs a supported mapping to the natural phenomenon. A true claim about a different outcome does not warrant the inference proposed for transport.',
    'second_frame_warrant': 'Continuously measure independent source support for the second causal inference in the explicit inference pair: 0 imagined, merely descriptive, or only the negation of a rejected first hypothesis; 0.5 partly warranted; 1 a well-supported causal inference with its own actors and relations. Independent warrant may come from the same paper or other papers; paper count is not a scientific gate.',
    'translation_specificity': 'Measure whether the proposed translation names the exact source inference, maps its actors, intervention, outcome, and relation, and states a plausible bridge premise: 0 no particular inference or map; 0.5 a partial map with a consequential gap; 1 a specific, biologically intelligible attempted transport. A merely imaginable but unlicensed prediction is weak.',
    'inference_reversal': 'Measure whether transporting the source-grounded generalizable inference yields a consequentially incompatible prediction in an independently warranted target theory within its justified domain: 0 the transported prediction was never licensed, organism change alone explains the contrast, a prior unsupported hypothesis was falsified, or the outcomes are unrelated; 0.5 a plausible linked failure with a major gap; 1 a clear linked failure while both local causal constructions remain useful.',
    'category_relation_break': 'Continuously measure whether the linked reversal arises from how actors, categories, relations, system boundaries, or intervention meanings change across grains: 0 routine fixed-variable modifier, contrary measurement, or invalidated hypothesis; 0.5 a plausible but unresolved category or relation mapping; 1 a warranted consequential break in the explicit causal translation.',
    'fixed_category_challenge': 'Measure a specified rival account that keeps actor, variable, intervention, and outcome meanings fixed while adding a cofactor, context variable, or dependency: 0 a useful simple repair reconciles the target or was not examined; 0.5 a specified rival with unresolved discrimination; 1 a source-grounded reason the repair loses useful predictions or requires reconstructing categories or relations. Do not claim no possible larger graph can describe the biology.',
    'pragmatic_pluralism': 'Continuously measure why both causal inferences in the explicit pair remain useful for their own questions: 0 one is simply false or a descriptive claim was promoted into an unwarranted causal hypothesis; 0.5 both have some use but their distinct causal reach is unclear; 1 both are warranted and pragmatically appropriate despite their noninterchangeability. These anchors are continuous scores, never admission rules.',
}
QUALITY_RUBRIC = {
    'source_fidelity': 'Measure faithfulness of consequential claims and inferences to the supplied evidence: 0 fabricated, contradicted, or unsupported central inference; 0.5 partly grounded with important source claims mishandled; 1 a well-grounded account that separates reported findings from its interpretation. Empty or evasive answers are not fully faithful substantive accounts.',
    'uncertainty_discipline': 'Measure whether strength and scope match the evidence: 0 unsupported certainty or misleading scope; 0.5 useful but incompletely traced limitations; 1 specific uncertainties with explicit consequences for the frame comparison. Generic hedging without consequences is not progress. A warranted negative account can score fully.',
    'explanatory_usefulness': 'Measure usefulness for the question of pragmatically valid causal descriptions at different grains: 0 irrelevant or misleading; 0.5 a partially informative comparison with important inference gaps; 1 a clear explanation of what each frame makes causal, why an inference does or does not translate, and what remains unresolved. More words or routine biology detail do not improve this score.',
}
THEORY_RUBRIC = {
    'cross_case_grounding': 'Measure whether the account compares acquired generalizable inferences, their supported theories, and exact source claims: 0 generic philosophy or organism differences alone; 0.5 partly grounded comparison; 1 well-supported comparison retaining differences in scope and repair outcomes.',
    'category_construction': 'Measure whether the account explains how each question and grain constructs its actors, categories, interventions, and relations: 0 categories taken as fixed names; 0.5 some construction explained; 1 clear, empirically anchored comparison of category choices.',
    'causal_nontransportability': 'Measure whether a reusable biological causal inference supports distinct hypotheses yet loses predictive or explanatory use in a warranted target theory after concrete repairs are tested: 0 an organism switch alone, a rejected unsupported guess, a useful routine modifier, or only a contrast between algorithms and their calculated outputs; 0.5 an incompletely warranted biological failure; 1 a specific supported failure of biological transport and repair without claiming metaphysical irreducibility.',
    'pragmatic_research_use': 'Measure whether the account explains why each grain suits its biological problem and identifies concrete evidence that could revise the proposed mapping: 0 no use or test; 0.5 partial rationale or broad test; 1 consequential problem-relative rationale and discriminating research directions.',
}
QUESTION = '''Discover the scope and limits of generalizable biological causal inferences.
A causal explanation relates entities or categories constructed at a grain suited
to a biological question. Begin with a natural phenomenon in an organism,
population, community, or other biological system, including laboratory or
engineered systems. Identify a particular source-grounded causal inference
that plausibly generalizes: state its rule, variables, intervention meanings,
abstraction, proposed domain, and the evidence for that scope. The cause and
outcome must refer to the biological system. Changing an input to an algorithm
and obtaining a different calculated output establishes how the algorithm
operates; without a warranted model-to-system mapping it is not a biological
causal inference. Models can help formulate or test biological inferences.
Derive distinct
useful hypotheses or theories it supports. When a support needs a small addition,
state the changed variables, dependencies, or mapping and show that the core rule
and prior predictions survive. Seek an independently warranted target theory
where transporting that rule fails because its causal categories, relations,
scale, or intervention meaning no longer serve the target question. Test concrete
repairs and show whether they retain independent constraint and predictive use.
A contrary empirical finding alone is not the phenomenon.
A routine moderator such as temperature, an omitted blocker, a significant
effect beside a nonsignificant one, or two unrelated outcomes is weak evidence
of this problem while the causal categories and question remain fixed.
The inference's justified domain decides which organisms and scales to examine.
An ecological rule about herbivores may apply across different animals. A change
of organism alone is a routine context difference, not a structural failure;
an unsupported leap outside the proposed rule's domain is not a reversal.
Do not declare one grain fundamentally wrong or infer irreducibility from an
unexplained effect. Assess why each grain is pragmatically appropriate and where
its explanatory reach ends. Context can matter when it reorganizes the relevant
entities and relations; there is no universal sufficient-context checklist.
Discover diverse cases by searching and reading, without a preset list of
organisms, molecules, case names, or mechanism categories. Keep source links,
uncertainties, provisional cases, and well-supported negative assessments.
An SCM component may expose a failed translation without defining the only
real entities or supplying a complete account of collective phenomena.'''
ROOT = 'collection'
COLUMNS = (*QUALITY_RUBRIC,'argument_progress')
THEORY_COLUMNS = (*QUALITY_RUBRIC,'theory_progress')
CREDIT_VERSION = 'question-discovery-v12.9'
WINDOW = 4
STOP_EPSILON = 0.001
ROOT_STOP_YIELD = 0.0
UCB_EXPLORATION = 0.01
YIELD_PRIOR = 0.01
YIELD_UPDATE = 0.25
WINDOW_CHARS = 12000
SEARCH_LIMIT = 3
SYNTHESIS_CONTEXT_CHARS = 24000
THEORY_CONTEXT_CHARS = 8000
IDENTITY_CATALOG_CHARS = 12000
SYNTHESIS_CATALOG_CHARS = 10000
SIDES = ('frame_one', 'frame_two', 'mapping')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def quality_utility(measurements):
    """Numerical report preference, independent of historical method credit."""
    return math.prod((1+measurements[name]['score'])/2 for name in QUALITY_RUBRIC)


def material_vector(material):
    return tuple(material['epistemic_quality'][name]['score'] for name in QUALITY_RUBRIC)+(material['argument_progress']['score'],)


def theory_vector(material):
    return tuple(material['epistemic_quality'][name]['score'] for name in QUALITY_RUBRIC)+(material['theory_progress']['score'],)


def derive_theory_progress(theory_fit, cited_case_progress):
    """Limit synthesis credit to progress demonstrated by its cited cases."""
    interpretation=min(theory_fit[name]['score'] for name in (
        'cross_case_grounding','category_construction','pragmatic_research_use'))
    nontransport=theory_fit['causal_nontransportability']['score']
    def pooled(name):
        return 1.0-math.exp(-math.fsum(
            progress[name] for progress in cited_case_progress.values()))
    generalization=min(interpretation,pooled('generalization_strength'))
    support=min(interpretation,pooled('support_breadth'))
    challenge=min(interpretation,nontransport,pooled('challenge_strength'))
    score=0.05*generalization+0.20*support+0.75*challenge
    cited_ceiling=math.fsum(progress['score'] for progress in cited_case_progress.values())
    if score>cited_ceiling+1e-9:
        raise ValueError('Theory progress cannot exceed its cited case progress')
    return {'score':score,'generalization_strength':generalization,
            'support_breadth':support,'challenge_strength':challenge,
            'source_case_ids':list(cited_case_progress),
            'cited_case_progress_ceiling':cited_ceiling,
            'cited_case_count':len(cited_case_progress)}


def string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} requires nonempty text')
    return value


def select_identity_catalog(catalog, candidates, budget):
    """Select whole, relevant identity anchors within an Episode view budget."""
    def words(value):
        return set(re.findall(r'[a-z0-9]{3,}', value.lower()))

    query=' '.join(str(candidate.get(field,'')) for candidate in candidates for field in
                   ('title','topic','causal_structure','generalization_defect',
                    'frame_one_question','frame_two_question'))
    query+=' '+' '.join(str(candidate.get('inference_pair',{}).get(field,''))
                       for candidate in candidates for field in
                       ('generalizable_rule','proposed_domain'))
    query_words=words(query)
    direct={candidate.get('same_case_as') for candidate in candidates}
    def score(row):
        title_words=words(row['title']+' '+row['topic'])
        full_words=words(' '.join(str(row[field]) for field in
            ('causal_structure','generalization_defect','frame_one_question',
             'frame_two_question'))+' '+str(row.get('inference_pair',{})))
        return 3*len(query_words & title_words)+len(query_words & full_words)

    ordered=sorted(enumerate(catalog),key=lambda item:(
        item[1]['key'] not in direct,-score(item[1]),item[0]))
    selected=[]
    for _,row in ordered:
        if len(json.dumps([*selected,row],ensure_ascii=False))<=budget:
            selected.append(row)
        elif row['key'] in direct:
            raise ValueError('A proposed known identity does not fit the bounded identity view')
    return selected,[row['key'] for row in catalog if row not in selected]


def schema(properties, required=None):
    return {'type': 'object', 'additionalProperties': False,
            'required': list(properties if required is None else required), 'properties': properties}


TEXT = {'type': 'string', 'minLength': 1}


def inference_pair_contract(reference_items=None):
    citations={'type':'array','uniqueItems':True,'items':reference_items or TEXT}
    return schema({'frame_one_causal_inference':TEXT,'frame_one_warrant_refs':citations,
        'frame_two_causal_inference':TEXT,'frame_two_warrant_refs':citations,
        'proposed_translation':TEXT,'causal_tension':TEXT,
        'generalizable_rule':TEXT,'proposed_domain':TEXT,'domain_rationale':TEXT})


def candidate_contract(reference_field,reference_items=None):
    return schema({'label':TEXT,'same_case_as':TEXT,'identity_rationale':TEXT,
        'title':TEXT,'topic':TEXT,'causal_structure':TEXT,'generalization_defect':TEXT,
        'frame_one_question':TEXT,'frame_one_claim':TEXT,
        'frame_two_question':TEXT,'frame_two_claim':TEXT,
        'inference_pair':inference_pair_contract(reference_items),
        reference_field:{'type':'array','uniqueItems':True,'minItems':1,
                         'items':reference_items or TEXT}})


def normalize_inference_pair(pair,references):
    if not isinstance(pair,dict):
        raise ValueError('An inference pair must be a structured record')
    result={}
    for name in ('frame_one_causal_inference','frame_two_causal_inference',
                 'proposed_translation','causal_tension','generalizable_rule',
                 'proposed_domain','domain_rationale'):
        result[name]=string(pair.get(name),name)
    for name in ('frame_one_warrant_refs','frame_two_warrant_refs'):
        refs=pair.get(name)
        if not isinstance(refs,list) or any(ref not in references for ref in refs):
            raise ValueError(name+' must cite available source claims')
        result[name]=list(dict.fromkeys(references[ref] for ref in refs))
    return result


@dataclass(frozen=True)
class Record(ClosedRecord):
    value: dict

    def as_record(self):
        return self.value


@dataclass(frozen=True)
class Step:
    stop: bool
    score: float
    gain: float
    predicted_next_gain: float
    reason: str
    goal_commit_ids: tuple = ()
    requests_transition: bool = False


class Controller:
    """Local volume for child work; action-level expected marginal yield at root."""
    epoch = CREDIT_VERSION

    def __init__(self, run, keys, mode):
        self.run, self.keys, self.mode = run, keys, mode
        self.history = [run.state.snapshot(keys)['score']]
        self.stop = False
        self.reason = 'continue'

    def observe(self, unit_label, value, *, is_root):
        item = value.as_record()
        previous = self.history[-1]
        score = item['score']
        if score < previous-1e-12:
            raise ValueError('Historical attained credit cannot decrease')
        self.history.append(score)
        gains = [b-a for a,b in zip(self.history, self.history[1:])]
        scale = 1.0
        if self.mode=='discovery':
            projected=self.run.project_feedback(score-previous)
            prediction=max(projected.get(key,{}).get('expected_yield',YIELD_PRIOR)
                           for key in (ROOT,*self.run.state.data))
            saturated=len(gains)>=WINDOW and prediction==ROOT_STOP_YIELD
        else:
            prediction=sum(gains[-WINDOW:])/WINDOW/scale
            saturated=len(gains)>=WINDOW and prediction<=STOP_EPSILON
        # Root observations are complete acquisition-and-reasoning cycles.
        self.stop = saturated
        self.reason = 'low_predicted_credit' if self.stop else 'continue'
        return Step(self.stop, score, score-previous, prediction, self.reason,
                    tuple(item.get('commit_ids', ())))

    def state(self):
        return RecordState(self.stop, self.reason, tuple(self.history))

    def transitioned(self, epoch):
        raise ValueError('This registered experiment has no controller epoch mutation')


@dataclass(frozen=True)
class RecordState:
    stop: bool
    reason: str
    history: tuple

    def as_record(self):
        return {'stop': self.stop, 'reason': self.reason, 'history': list(self.history)}


class GoalState:
    """Source-grounded inquiries, immutable measurements, and collection credit."""
    def __init__(self,run):
        self.run=run
        self.data={}
        self.facts={}
        self.events={}
        self.archives={}
        self.collection_stats={}
        self.fact_origins={}
        self.material_origins={}
        self.credited_refs={}
        self.theory={'review_attempts':[],'materials':{},'current':None}
        self.theory_archive=None

    @property
    def state_id(self):
        return digest({'cases':self.data,'facts':self.facts,'events':self.events,
                       'attainment':self.archives,'fact_origins':self.fact_origins,
                       'material_origins':self.material_origins,
                       'credited_refs':self.credited_refs,'theory':self.theory,
                       'theory_archive':self.theory_archive})

    @property
    def evaluator_policy(self):
        return digest({'credit_version':CREDIT_VERSION,'target':self.run.target,
            'question':self.run.question,'candidate_fit':RUBRIC,'theory_fit':THEORY_RUBRIC,
            'quality':QUALITY_RUBRIC,
            'model_tiers':self.run.providers['model_tiers'],
            'implementation':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                for name in ('reversal_task.py','reversal_prompts.py','causal_argument.py','configured_providers.py',
                             'credit_assignment/attainment.py',
                             'method_loop/episode_binding/web_search_binding.py',
                             'task_executors/evidence_extraction/argo.py',
                             'task_executors/evidence_extraction/codex.py')}})

    def evidence_version(self,key,data=None,facts=None):
        records=self.facts if facts is None else facts
        rows=self.data if data is None else data
        return digest(records if key==ROOT else {ref:records[ref] for ref in rows[key]['fact_ids']})

    def case(self,key,data=None,facts=None):
        rows=self.data if data is None else data
        records=self.facts if facts is None else facts
        row=rows[key];current=row['current']
        material=row['materials'].get(current['material_id']) if current else None
        return {'key':key,'title':row['title'],'topic':row['topic'],
            'theme_id':row['theme_id'],
            'causal_structure':row['causal_structure'],'generalization_defect':row['generalization_defect'],
            'inference_pair':row['inference_pair'],
            'frame_one_question':row['frame_one_question'],
            'frame_one_claim':row['frame_one_claim'],
            'frame_two_question':row['frame_two_question'],
            'frame_two_claim':row['frame_two_claim'],
            'identity_rationale':row['identity_rationale'],'occurrences':row['occurrences'],
            'facts':[records[ref] for ref in row['fact_ids']],
            'evidence_version':self.evidence_version(key,rows,records),
            'current_answer':current['answer'] if current else None,
            'material_id':current['material_id'] if current else None,
            'measurements':current['candidate_fit'] if current else {},
            'epistemic_quality':current['epistemic_quality'] if current else {},
            'argument_progress':current['argument_progress'] if current else None,
            'credited_vector':material_vector(material) if material else (0.0,)*len(COLUMNS),
            'materials':row['materials'],'review_count':len(row['review_attempts']),
            'reading_notes':row['reading_notes']}

    def catalog(self,data=None,facts=None):
        rows=self.data if data is None else data
        result=[]
        for key in rows:
            case=self.case(key,rows,facts)
            result.append({name:case[name] for name in ('key','title','topic','theme_id','causal_structure',
                'generalization_defect','frame_one_question','frame_one_claim',
                'frame_two_question','frame_two_claim','inference_pair',
                'identity_rationale','current_answer','measurements',
                'epistemic_quality','argument_progress','review_count','reading_notes')})
        return result

    def identity_catalog(self):
        """Compact identity anchors; full occurrences remain addressable in state."""
        pair_fields=('frame_one_causal_inference','frame_two_causal_inference',
                     'proposed_translation','causal_tension','generalizable_rule',
                     'proposed_domain','domain_rationale')
        return [{'key':key,'theme_id':row['theme_id'],'title':row['title'],
                 'topic':row['topic'],
                 'causal_structure':row['causal_structure'],
                 'generalization_defect':row['generalization_defect'],
                 'frame_one_question':row['frame_one_question'],
                 'frame_two_question':row['frame_two_question'],
                 'inference_pair':{name:row['inference_pair'][name]
                                   for name in pair_fields},
                 'occurrence_count':len(row['occurrences'])}
                for key,row in self.data.items()]

    def snapshot(self,keys=None,data=None,archives=None,theory_archive=None):
        rows=self.data if data is None else data
        saved=self.archives if archives is None else archives
        if keys is None:
            by_theme={}
            for key,item in saved.items():
                theme=rows[key]['theme_id']
                by_theme.setdefault(theme,[]).extend(
                    AttainmentArchive.from_record(item).records.values())
            score=math.fsum(hypervolume(vectors) for vectors in by_theme.values())
            active_theory=self.theory_archive if theory_archive is None else theory_archive
            if active_theory is not None:
                score+=AttainmentArchive.from_record(active_theory).score
        else:
            stored=saved.get(keys[0])
            score=AttainmentArchive.from_record(stored).score if stored else 0.0
        selected=list(rows) if keys is None else list(keys)
        return {'score':score,'candidate_inquiries':len(rows),
                'conceptual_themes':len({row['theme_id'] for row in rows.values()}),
                'reviewed_inquiries':sum(bool(rows[k]['materials']) for k in selected),
                'columns':COLUMNS,'collection_stats':self.collection_stats if keys is None else {}}

    def remember(self,data,archives,key,material_id):
        started=time.monotonic()
        local=AttainmentArchive.from_record(archives.get(key,AttainmentArchive(COLUMNS).as_record()))
        local.add('initial',(0.0,)*len(COLUMNS))
        local.add(material_id,material_vector(data[key]['materials'][material_id]))
        archives[key]=local.as_record()
        return {'aggregation':'sum_theme_union_hypervolumes',
                'groups':len({row['theme_id'] for row in data.values() if row['materials']}),
                'materials':sum(len(row['materials']) for row in data.values()),
                'seconds':time.monotonic()-started}

    def as_record(self):
        return {'schema_version':19,'credit_version':CREDIT_VERSION,
            'evaluator_policy':self.evaluator_policy,'question':self.run.question,
            'target':self.run.target,'cases':self.data,'facts':self.facts,'events':self.events,
            'attainment':self.archives,'collection_stats':self.collection_stats,
            'fact_origins':self.fact_origins,'material_origins':self.material_origins,
            'credited_refs':self.credited_refs,
            'theory':self.theory,'theory_archive':self.theory_archive,
            'search_memory':{'queries':self.run.queries,'action_stats':self.run.action_stats,
                             'cycle_outcomes':self.run.cycle_outcomes}}

    def restore(self,record):
        if (record.get('schema_version')!=19 or record.get('credit_version')!=CREDIT_VERSION
                or record.get('evaluator_policy')!=self.evaluator_policy
                or record.get('question')!=self.run.question or record.get('target')!=self.run.target):
            raise ValueError('State is incompatible with this discovery task and evaluator policy')
        expected={key for key,row in record['cases'].items() if row['materials']}
        if any(not row.get('theme_id','').startswith('theme_') for row in record['cases'].values()):
            raise ValueError('Every biological inquiry requires a conceptual theme')
        for row in record['cases'].values():
            normalize_inference_pair(row['inference_pair'],{ref:ref for ref in row['fact_ids']})
        if set(record['attainment'])!=expected:
            raise ValueError('Every credited inquiry requires one local archive')
        for item in record['attainment'].values():
            if AttainmentArchive.from_record(item).columns!=tuple(COLUMNS):
                raise ValueError('Credit columns do not match this discovery task')
        if record['theory_archive'] is not None:
            if AttainmentArchive.from_record(record['theory_archive']).columns!=tuple(THEORY_COLUMNS):
                raise ValueError('Theory credit columns do not match')
        if bool(record['theory']['materials'])!=(record['theory_archive'] is not None):
            raise ValueError('Theory materials and credit archive must agree')
        for case,origins in record['material_origins'].items():
            if case not in record['cases'] or not set(origins)<=set(record['cases'][case]['materials']):
                raise ValueError('Material origins must resolve to known case materials')
        for field,attribute in (('cases','data'),('facts','facts'),('events','events'),
                                ('attainment','archives'),('collection_stats','collection_stats'),
                                ('fact_origins','fact_origins'),('material_origins','material_origins'),
                                ('credited_refs','credited_refs'),
                                ('theory','theory'),('theory_archive','theory_archive')):
            setattr(self,attribute,json.loads(json.dumps(record[field])))
        for field in ('queries','action_stats','cycle_outcomes'):
            setattr(self.run,field,json.loads(json.dumps(record['search_memory'][field])))

    def preview(self,proposal,unit_ref):
        payload=proposal.payload;key=payload['key'];event=payload['event']
        event_id=digest({'inquiry':key,'event':event})
        data=json.loads(json.dumps(self.data));facts=json.loads(json.dumps(self.facts))
        events=json.loads(json.dumps(self.events));archives=json.loads(json.dumps(self.archives))
        theory=json.loads(json.dumps(self.theory))
        theory_archive=json.loads(json.dumps(self.theory_archive))
        stats=self.collection_stats;preference=None;diagnostic=None
        fresh=event_id not in events
        if fresh:
            events[event_id]={'inquiry':key,**event}
            for fact in event.get('facts',()):facts[fact['id']]=fact
            if key!=ROOT and key in data and event.get('kind')=='source_reading':
                data[key]['fact_ids']=list(dict.fromkeys([*data[key]['fact_ids'],*[f['id'] for f in event['facts']]]))
                data[key]['reading_notes'].extend(event.get('unresolved_questions',[]))
            for candidate in event.get('candidates',()):
                identity=candidate['case_id'];row=data.setdefault(identity,{
                    'title':candidate['title'],'topic':candidate['topic'],
                    'theme_id':candidate['theme_id'],
                    'causal_structure':candidate['causal_structure'],
                    'generalization_defect':candidate['generalization_defect'],
                    'inference_pair':candidate['inference_pair'],
                    'frame_one_question':candidate['frame_one_question'],
                    'frame_one_claim':candidate['frame_one_claim'],
                    'frame_two_question':candidate['frame_two_question'],
                    'frame_two_claim':candidate['frame_two_claim'],
                    'identity_rationale':candidate['identity_rationale'],
                    'fact_ids':[],'occurrences':[],'reading_notes':[],
                    'review_attempts':[],'materials':{},'current':None})
                if row['theme_id']!=candidate['theme_id']:
                    raise ValueError('One biological inquiry cannot change conceptual theme')
                row['fact_ids']=list(dict.fromkeys([*row['fact_ids'],*candidate['evidence_refs']]))
                occurrence={**candidate,'source':event['source'],'source_version':event['version']}
                if digest(occurrence) not in {digest(o) for o in row['occurrences']}:
                    row['occurrences'].append(occurrence)
                row['reading_notes']=list(dict.fromkeys([*row['reading_notes'],*event.get('unresolved_questions',[])]))
            if 'review' in event:
                row=data[key];review=event['review'];row['review_attempts'].append(event)
                previous=row['current'];first=review['material_id'] not in row['materials']
                if event['previous_material_id']!=(previous['material_id'] if previous else None):
                    raise ValueError('Paired measurement no longer matches the current report')
                row['materials'].setdefault(review['material_id'],review)
                proposed_u=quality_utility(review['epistemic_quality'])
                prior_u=None
                if previous:
                    previous={**previous,**event['previous_measurements']}
                    prior_u=quality_utility(previous['epistemic_quality'])
                row['current']=review if previous is None or proposed_u>prior_u else previous
                preference={'proposed_material_id':review['material_id'],
                    'previous_material_id':previous['material_id'] if previous else None,
                    'preferred_material_id':row['current']['material_id'],
                    'proposed_utility':proposed_u,'previous_utility':prior_u}
                if first:
                    before=self.snapshot((key,),data,archives)['score']
                    stats=self.remember(data,archives,key,review['material_id'])
                    if previous:
                        new=material_vector(review);old=material_vector(previous)
                        paid=material_vector(row['materials'][previous['material_id']])
                        if (all(a>=b for a,b in zip(new,old)) and any(a>b for a,b in zip(new,old))
                                and all(a>=b for a,b in zip(paid,new))
                                and self.snapshot((key,),data,archives)['score']-before<=1e-12):
                            diagnostic={'condition':'optimistic paid vector suppresses contemporary improvement',
                                        'prior_paid':paid,'prior_current':old,'proposed':new}
            if 'theory_review' in event:
                review=event['theory_review']
                case_refs=event.get('case_material_refs',{})
                if set(case_refs)!=set(review['answer']['case_refs']):
                    raise ValueError('Theory case-material lineage must match cited cases')
                for case_id,material_id in case_refs.items():
                    if case_id not in data or not data[case_id]['current'] or data[case_id]['current']['material_id']!=material_id:
                        raise ValueError('Theory case-material lineage must name the current cited account')
                expected_progress=derive_theory_progress(review['theory_fit'],{
                    case_id:data[case_id]['current']['argument_progress'] for case_id in case_refs})
                if review.get('theory_progress')!=expected_progress:
                    raise ValueError('Theory credit must match cited case progress')
                theory['review_attempts'].append(event)
                prior=theory['current']
                if event['previous_material_id']!=(prior['material_id'] if prior else None):
                    raise ValueError('Theory measurement no longer matches the current account')
                first=review['material_id'] not in theory['materials']
                theory['materials'].setdefault(review['material_id'],review)
                proposed_u=quality_utility(review['epistemic_quality'])
                previous_u=quality_utility(event['previous_measurements']['epistemic_quality']) if prior else None
                theory['current']=review if prior is None or proposed_u>previous_u else prior
                if first:
                    local=AttainmentArchive.from_record(theory_archive) if theory_archive else AttainmentArchive(THEORY_COLUMNS)
                    local.add('initial',(0.0,)*len(THEORY_COLUMNS))
                    local.add(review['material_id'],theory_vector(review))
                    theory_archive=local.as_record()
        ids=(event_id,) if fresh else ()
        keys=payload.get('scope_keys')
        snapshot=self.snapshot(keys,data,archives,theory_archive)
        return GoalPreview(Record({**snapshot,'commit_ids':ids}),ids,self.state_id,
            no_commit_result=self.snapshot(keys),
            token=(data,facts,events,archives,theory,theory_archive,stats,key,event_id,event,snapshot,preference,diagnostic))

    def commit(self,preview,result_ids):
        data,facts,events,archives,theory,theory_archive,stats,key,event_id,event,snapshot,preference,diagnostic=preview.token
        if tuple(result_ids)!=(event_id,):
            raise ValueError('A complete validated event is committed atomically')
        old_facts=set(self.facts)
        old_root_score=self.snapshot()['score']
        old_archive=self.theory_archive if 'theory_review' in event else self.archives.get(key)
        old_score=(AttainmentArchive.from_record(old_archive).score
                   if old_archive else 0.0)
        self.data,self.facts,self.events,self.archives=data,facts,events,archives
        self.theory,self.theory_archive=theory,theory_archive
        self.collection_stats=stats
        if self.run.pending_cycle is not None:
            cycle_id=self.run.pending_cycle['cycle_id']
            for fact in event.get('facts',()):
                if fact['id'] not in old_facts:
                    self.fact_origins.setdefault(fact['id'],cycle_id)
            if 'review' in event:
                self.material_origins.setdefault(key,{}).setdefault(
                    event['review']['material_id'],cycle_id)
            if 'review' in event or 'theory_review' in event:
                is_theory='theory_review' in event
                review=event['theory_review'] if is_theory else event['review']
                refs=set(review['answer']['evidence_refs'])
                for issue in review['answer']['uncertainties']:
                    refs.update(issue['evidence_refs'])
                for dimension in (('theory_fit' if is_theory else 'candidate_fit'),'epistemic_quality'):
                    for rating in review[dimension].values():
                        refs.update(rating['evidence_refs'])
                used=set(self.credited_refs.get(key,()))
                fresh=sorted(refs-used)
                current_archive=self.theory_archive if is_theory else self.archives[key]
                new_score=AttainmentArchive.from_record(current_archive).score
                gain=max(0.0,self.snapshot()['score']-old_root_score)
                if gain>0:
                    self.credited_refs[key]=sorted(used|refs)
                    owners={self.fact_origins[ref] for ref in fresh
                            if ref in self.fact_origins}|{cycle_id}
                    case_lineage={}
                    if is_theory:
                        for case_id,material_id in event['case_material_refs'].items():
                            origin=self.material_origins.get(case_id,{}).get(material_id)
                            if origin is not None:
                                owners.add(origin)
                                case_lineage[case_id]={'material_id':material_id,'origin_cycle':origin}
                    self.run.pending_credit_events.append({'inquiry':key,
                        'material_id':review['material_id'],'local_gain':gain,
                        'newly_used_evidence_refs':fresh,
                        'cited_case_material_origins':case_lineage,
                        'eligible_cycles':sorted(owners)})
        self.run.record({'event':'goal_committed','case':key,'result_identity':event_id,
                         'contribution':event,'credit':snapshot})
        if preference:self.run.record({'event':'report_preference','case':key,**preference})
        if diagnostic:self.run.record({'event':'credit_calibration_diagnostic','case':key,**diagnostic})
        self.run.save_state()
        return snapshot


class Source:
    """One Episode source template with task-level proposal/execute parts."""
    def __init__(self, run, mode, key, **parts):
        self.run, self.mode, self.key, self.parts = run, mode, key, parts
        self.position = 0
        self.reading_attempts = []
        self.feedback = []
        self.initial_fact_count = self.fact_count()
        self.initial_candidate_count = len(run.state.data)
        self.initial_candidate_events = self.candidate_events()

    def fact_count(self):
        return (len(self.run.state.facts) if self.key==ROOT else
                len(self.run.state.data[self.key]['fact_ids']))

    def candidate_events(self):
        return sum(bool(event.get('candidates')) for event in self.run.state.events.values())

    def next(self, view):
        if self.run.unavailable:
            return SourceEnd('source_failed')
        # Return new evidence to the parent investigation rather than keeping it
        # below a large acquisition subtree. This is an information handoff, not
        # a claim that the source has been exhausted or the case has passed.
        if self.mode in ('find','page','report','lexical','table') and view.units_consumed:
            handoff=(self.candidate_events()>self.initial_candidate_events if self.key==ROOT
                     else self.fact_count()>self.initial_fact_count)
            if handoff or len(self.run.state.data)>self.initial_candidate_count:
                return None
        try:
            return getattr(self, 'next_'+self.mode)(view)
        except Exception as exc:
            self.feedback.append({'error_type':type(exc).__name__, 'error':str(exc),
                                  'rejected_output':str(getattr(exc,'raw_response',''))[:WINDOW_CHARS]})
            self.run.record({'event': 'source_failed', 'mode': self.mode, 'case': self.key,
                             'error_type': type(exc).__name__, 'error': str(exc)})
            # Failures are observed zero-yield units, not negative scientific evidence.
            if self.mode == 'discovery':
                raise
            return self.run.leaf(self.key, {'kind':'failure', 'stage':self.mode,
                                 'error':str(exc)}, f'failure:{view.units_consumed}')

    def ask(self, operation, **request):
        failures=[{'error_type':item['error_type'],'error':item['error']}
                  for item in self.feedback[-WINDOW:]]
        return self.run.ask(operation,self.key,prior_attempts=failures,
            **request)

    def next_discovery(self,view):
        return self.run.child(view,'cycle',ROOT,
            'Discover or improve source-grounded comparisons of pragmatically useful causal frames')

    def next_cycle(self, view):
        position=self.position
        self.position+=1
        if position==0:
            self.parts['action']=self.run.choose_action()
            key=self.parts['action']['key']
            return self.run.child(view,'find',key,
                self.run.question if key==ROOT else self.run.state.data[key]['topic'])
        if position==1:
            return self.run.child(view,'synthesize',ROOT,
                'Compare question-specific causal frames across acquired evidence and improve search hypotheses')
        if position==2:
            key=self.run.choose_inquiry(self.parts['action']['key'])
            self.parts['reviewed_inquiry']=key
            if key is None:
                self.position=4
                return (self.run.child(view,'theory',ROOT,
                    'Revise the cross-case account of pragmatic causal pluralism')
                    if self.run.state.facts else None)
            return self.run.child(view,'think',key,'Develop or correct the source-grounded comparison',
                                  version=self.run.state.evidence_version(key))
        if position==3 and self.run.state.facts:
            return self.run.child(view,'theory',ROOT,
                'Revise the cross-case account of pragmatic causal pluralism')
        return None

    def next_synthesize(self,view):
        if self.position:return None
        self.position+=1
        return self.run.action(ROOT,self.run.synthesize,'cross_source_discovery_reasoning')

    def next_think(self, view):
        if self.position:
            return None
        role=('construct','challenge')[self.run.state.case(self.key)['review_count']%2]
        self.position += 1
        return self.run.action(self.key, lambda: self.run.review(self.key, role, self.parts['version']),
                               f'reason:{role}')

    def next_theory(self, view):
        if self.position:return None
        self.position+=1
        return self.run.action(ROOT,self.run.theorize,'cross_case_reasoning')

    def next_find(self, view):
        if 'proposer' not in self.parts:
            case = self.run.state.case(self.key) if self.key!=ROOT else {}
            pair=((case.get('current_answer') or {}).get('inference_pair')
                  or case.get('inference_pair'))
            scores=case.get('measurements') or {}
            cited=([self.run.state.facts[ref] for ref in dict.fromkeys([
                    *pair['frame_one_warrant_refs'],*pair['frame_two_warrant_refs']])
                    if ref in self.run.state.facts] if pair else [])
            argument=(case.get('current_answer') or {}).get('argument') or {}
            generalization=argument.get('generalization') or {}
            focus=({'topic':case['topic'],
                    'frame_one_question':case['frame_one_question'],
                    'frame_one_claim':case['frame_one_claim'],
                    'frame_two_question':case['frame_two_question'],
                    'frame_two_claim':case['frame_two_claim'],
                    'inference_pair':pair,
                    'current_fit':{name:row['score'] for name,row in scores.items()},
                    'credit_components':case.get('argument_progress'),
                    'active_generalization':({
                        'rule':generalization.get('rule'),
                        'proposed_domain':generalization.get('proposed_domain'),
                        'supported_theories':[{
                            name:item[name] for name in ('id','theory','consequence')}
                            for item in generalization.get('supports',[])[:6]],
                        'tested_repairs':[{
                            'id':item['id'],'status':item['status'],
                            'formal_change':item['addition']['formal_change']}
                            for item in generalization.get('repair_tests',[])[:6]]}
                        if generalization else None),
                    'paired_warrant_claims':[{'side':('frame_one' if fact['id'] in
                        pair['frame_one_warrant_refs'] else 'frame_two'),
                        'id':fact['id'],'source':fact['source'],'claim':fact['claim']}
                        for fact in cited[:8]],
                    'next_search_need':(case.get('current_answer') or {}).get('next_search_need')}
                   if self.key!=ROOT else None)
            current_theory=self.run.state.theory['current']
            theory_feedback=({name:current_theory['answer'][name] for name in
                              ('next_frame_gap','next_evidence_need','avoid_pattern')}
                             if self.key==ROOT and current_theory else None)
            prior = self.run.queries.setdefault(self.key,[])
            feedback=[{'literal_query':item.get('literal_query'),
                       'new_candidate_count':item.get('new_candidate_count',0),
                       'new_claim_count':item.get('new_claim_count',0),
                       'new_reading_count':item.get('new_reading_count',0),
                       'learning_share':item.get('attributed_gain',0.0)}
                      for item in self.run.cycle_outcomes[-WINDOW:]]
            answer = self.ask('form_search',
                acquisition_scope='general discovery' if self.key==ROOT else 'existing inquiry',
                previous_queries=prior[-WINDOW:],focus=focus,
                cross_case_feedback=theory_feedback,
                search_feedback=feedback,
                response_contract=schema({'frame_search_hypothesis':TEXT,
                    'discriminating_evidence':TEXT,
                    'query':{'type':'string','minLength':1,'maxLength':240},
                    'information_need':{'type':'string','minLength':1,'maxLength':160}}))
            query = string(answer.get('query'),'query')
            need = string(answer.get('information_need'),'information_need')
            hypothesis=string(answer.get('frame_search_hypothesis'),'frame search hypothesis')
            discriminator=string(answer.get('discriminating_evidence'),'discriminating evidence')
            if len(query)>240 or '\n' in query or len(need)>160 or '\n' in need:
                raise ValueError('Search decision must contain one bounded query and one bounded information need')
            prior.append(query)
            if self.run.pending_cycle is not None:
                self.run.pending_cycle.update(literal_query=query,information_need=need,
                    frame_search_hypothesis=hypothesis,discriminating_evidence=discriminator)
            self.run.record({'event':'reasoning_directed_query','case':self.key,'query':query,
                             'information_need':need,
                             'frame_search_hypothesis':hypothesis,
                             'discriminating_evidence':discriminator,
                             'prior_candidates':list(self.run.state.data),
                             'prior_evidence_version':self.run.state.evidence_version(self.key),
                             'prior_answer':case.get('current_answer'),
                             'prior_measurements':case.get('measurements',{})})
            def select(goal,candidates,history):
                return self.ask('choose_source',tier=ModelTier.FAST,
                    information_need=need,candidates=candidates,
                    previous_page_update=self.parts.get('previous_page_update'),
                    response_contract=schema({'option_id':{'type':'string','enum':[c['option_id'] for c in candidates]},
                                              'objective':TEXT,'rationale':TEXT}))
            self.parts.update(proposer=SearchPageProposer(query=query,limit=SEARCH_LIMIT,
                provider_search=self.run.providers['search'],jev=self.run.providers['assessor'],select_page=select,
                standing_task=self.run.question,information_need=need,
                assessment_instructions='Could this source inform how a biological question constructs causal actors and relations, or how two useful descriptions fail to translate? Measure possible contribution, including unresolved or negative comparisons, without an eligibility cutoff.',
                assessment_criteria={'true':'Potential source-grounded contribution to comparing pragmatically useful causal frames or their translation.',
                                     'false':'No apparent contribution to the declared research question.'}),
                information_need=need)
        if view.child_reports:
            self.parts['previous_page_update']=view.child_reports[-1].as_record()
        candidate = asyncio.run(self.parts['proposer'].next(view.goal))
        if candidate is None:
            return None
        return self.run.child(view,'page',self.key,self.parts['information_need'],candidate=candidate)

    def next_page(self, view):
        if 'document' not in self.parts:
            candidate=self.parts['candidate']
            locator=source_locator(candidate)
            raw=self.run.providers['retrieve'](candidate)
            text=raw if isinstance(raw,str) else source_text(raw)
            if not text.strip():
                raise ValueError('Retrieval returned no document body')
            fresh=Document(locator,str(candidate.get('title','')),text,WINDOW_CHARS)
            document=self.run.documents.setdefault((locator,fresh.sha256),fresh)
            self.parts.update(document=document,attempted=set())
            self.run.store_source(document, raw)
        document=self.parts['document']
        attempted=self.parts['attempted']
        options=[m for m in ('report','lexical','table') if m not in attempted and (m!='table' or document.tables)]
        if not options:
            return None
        reading_update=(view.child_reports[-1].as_record() if view.child_reports else None)
        answer=self.ask('choose_reading',tier=ModelTier.FAST,
            information_need=str(view.goal.objective),source_summary=document.summary(),
            available_modes=options,previous_reading_update=reading_update,
            response_contract=schema({'mode':{'type':'string','enum':options},'query':{'type':'string'},
                                      'table_key':{'type':'string'},'question':TEXT}))
        mode=answer.get('mode')
        if mode not in options:
            raise ValueError('Undeclared reading mode')
        attempted.add(mode)
        query=answer.get('query','')
        if not isinstance(query,str):
            raise ValueError('The source-local query must be text')
        if mode=='report':
            spans=list(document.remaining())
        elif mode=='lexical':
            if not query.strip():
                raise ValueError('Lexical reading requires a literal query')
            eligible=document.remaining() or document.remaining('reexamine')
            spans=list(rank_spans(eligible,query))
        else:
            table=next((t for t in document.tables if t.key==answer.get('table_key')),None)
            if table is None:
                raise ValueError('Table selection must name a detected table key')
            spans=[table.header,*[s for s,values in zip(table.rows,table.cells)
                                       if any(query.casefold() in v.casefold() for v in values)]]
        return self.run.child(view,mode,self.key,string(answer.get('question'),'question'),
                              document=document,spans=spans,
                              memory=self.run.reading_memory.setdefault(document.sha256,
                                  {'anchors':[],'unresolved':[]}))

    def next_read(self,view):
        spans=self.parts['spans']
        if self.position>=len(spans):
            return None
        selected=[]
        size=0
        while self.position<len(spans):
            span=spans[self.position]
            if selected and size+len(span.text)>WINDOW_CHARS:
                break
            if len(span.text)>WINDOW_CHARS:
                raise ValueError('One source span exceeds the declared reading window')
            selected.append(span)
            size+=len(span.text)
            self.position+=1
        return self.run.action(self.key,
            lambda:self.run.read(self.key,self.parts,selected,str(view.goal.objective)),
            f'{self.mode}_window:{self.position}')

    next_report=next_read
    next_lexical=next_read
    next_table=next_read


class Run:
    def __init__(self,config,output,progress,*,question=QUESTION,target=10):
        self.question=string(question,'research question')
        if type(target) is not int or target<1:
            raise ValueError('Collection target must be a positive integer')
        self.target=target
        self.output=Path(output).expanduser()
        self.output.mkdir(parents=True,exist_ok=False)
        (self.output/'sources').mkdir()
        self.progress=progress
        self.sequence=0;self.episode_number=0;self.unavailable=None
        self.operation_attempts={};self.operation_successes={};self.failure_streaks={}
        self.events=(self.output/'events.jsonl').open('x')
        self.providers=reversal_research_providers(config,record=self.record)
        self.state=GoalState(self)
        self.queries={ROOT:[]};self.documents={};self.reading_memory={};self.sources={}
        self.action_stats={};self.cycle_outcomes=[];self.pending_cycle=None;self.cycle_initial_facts=set()
        self.pending_credit_events=[]
        self.grains={name:Grain(name,f'{name} task work unit','DiscoveryResult',
            lambda path,name=name:Controller(self,self.scope_keys(path[-1][1].split(':',1)[0]),name))
            for name in ('discovery','cycle','synthesize','think','theory','find','page','report','lexical','table')}
        g=self.grains
        self.tree=EpisodeTree(g['discovery'],{g['discovery']:(g['cycle'],),
            g['cycle']:(g['think'],g['theory'],g['synthesize'],g['find']),g['find']:(g['page'],),
            g['page']:(g['report'],g['lexical'],g['table'])})
        self.context=Context(tree=self.tree,goal_state=self.state)

    @staticmethod
    def scope_keys(key):
        return None if key==ROOT else (key,)

    def expected_yield(self,key):
        """Forecast from credited learning shares on actual action visits."""
        return self.action_stats.get(key,{}).get('expected_yield',YIELD_PRIOR)

    def feedback_allocation(self,root_gain):
        """Allocate a single gain to distinct evidence-origin and reasoning cycles.

        This is retrospective learning feedback, not another method credit.
        An origin gets at most one share per newly credited material, regardless
        of its citation count. The current reasoning cycle is always eligible.
        """
        if not math.isfinite(root_gain) or root_gain < -1e-12:
            raise ValueError('Root gain must be finite and nonnegative')
        cycle_id=self.pending_cycle['cycle_id']
        gain=max(0.0,root_gain)
        events=[event for event in self.pending_credit_events if event['local_gain']>0]
        total=math.fsum(event['local_gain'] for event in events)
        if gain==0:
            return {cycle_id:0.0},[]
        if total==0:
            return {cycle_id:gain},[{'reason':'untraced_gain','cycle_shares':{cycle_id:gain}}]
        shares={}
        audit=[]
        paid=0.0
        for index,event in enumerate(events):
            amount=(gain*event['local_gain']/total if index<len(events)-1 else gain-paid)
            paid+=amount
            owners=event['eligible_cycles']
            if not owners or cycle_id not in owners:
                raise ValueError('Current reasoning cycle must own a learning share')
            distribution={}
            allotted=0.0
            for position,owner in enumerate(owners):
                part=(amount/len(owners) if position<len(owners)-1 else amount-allotted)
                distribution[owner]=part
                shares[owner]=shares.get(owner,0.0)+part
                allotted+=part
            audit.append({**event,'method_gain':amount,'cycle_shares':distribution})
        return shares,audit

    def project_feedback(self,root_gain):
        """Recompute forecasts in visit order after a hypothetical cycle gain.

        This pure projection is used by the root controller before on_unit
        installs the same allocation. Delayed shares correct earlier visits.
        """
        shares,_=self.feedback_allocation(root_gain)
        rows=[{**row} for row in self.cycle_outcomes]
        rows.append({'key':self.pending_cycle['key'],'cycle_id':self.pending_cycle['cycle_id'],
                     'attributed_gain':0.0})
        for owner,share in shares.items():
            if owner<1 or owner>len(rows) or rows[owner-1]['cycle_id']!=owner:
                raise ValueError('Learning share refers to an unknown acquisition cycle')
            rows[owner-1]['attributed_gain']+=share
        stats={}
        for row in rows:
            item=stats.setdefault(row['key'],{'visits':0,'credit':0.0,'expected_yield':YIELD_PRIOR})
            reward=row['attributed_gain']
            item['expected_yield']=(1-YIELD_UPDATE)*item['expected_yield']+YIELD_UPDATE*reward
            item['credit']+=reward
            item['visits']+=1
        return stats

    def choose_action(self):
        self.cycle_initial_facts=set(self.state.facts)
        self.pending_credit_events=[]
        total=len(self.cycle_outcomes)
        options={}
        for key in (ROOT,*self.state.data):
            stats=self.action_stats.get(key,{'visits':0,'credit':0.0})
            forecast=self.expected_yield(key)
            exploration=UCB_EXPLORATION*math.sqrt(math.log(total+2)/(stats['visits']+1))
            options[key]={'expected_yield':forecast,'exploration':exploration,
                          'score':forecast+exploration,'visits':stats['visits'],
                          'yield_is_prior':stats['visits']==0}
        selected=max(options,key=lambda k:options[k]['score'])
        self.pending_cycle={'cycle_id':total+1,'key':selected,'mode':'discover' if selected==ROOT else 'refine',
            'options':options,'prior_candidates':list(self.state.data),
            'initial_fact_count':len(self.state.facts),
            'model_attempts_before':sum(self.operation_attempts.values())}
        self.record({'event':'acquisition_scheduled',**self.pending_cycle})
        return self.pending_cycle

    def choose_inquiry(self,preferred):
        if preferred!=ROOT:
            eligible={preferred:self.state.data[preferred]}
        else:
            prior=set(self.pending_cycle['prior_candidates']) if self.pending_cycle else set()
            fresh=set(self.state.facts)-self.cycle_initial_facts
            eligible={key:row for key,row in self.state.data.items()
                      if key not in prior or fresh.intersection(row['fact_ids'])}
        if not eligible:
            self.record({'event':'reasoning_scheduled','case':None,'reason':'no source-grounded candidates'})
            return None
        options={}
        for key,row in eligible.items():
            reviews=len(row['review_attempts'])
            archive=self.state.archives.get(key)
            attained=AttainmentArchive.from_record(archive).score if archive else 0.0
            options[key]=1/(1+reviews)+(1-attained)*0.1+(0.05 if key==preferred else 0)
        selected=max(options,key=options.get)
        if self.pending_cycle is not None:self.pending_cycle['reviewed_inquiry']=selected
        self.record({'event':'reasoning_scheduled','case':selected,'scores':options,
            'attribution':'selected inquiry' if preferred!=ROOT else 'new inquiry or newly attached evidence in this cycle'})
        return selected

    def on_unit(self,mode,key,source,item,contribution,view):
        control={'stop':view.controller_step.stop,'gain':view.controller_step.gain,
            'predicted_next_gain':view.controller_step.predicted_next_gain,
            'reason':view.controller_step.reason}
        self.record({'event':'episode_unit','mode':mode,'case':key,'unit':view.unit_label,
            'credit':view.controller_input.as_record(),'controller':control})
        if mode=='discovery' and self.pending_cycle is not None:
            work=self.pending_cycle
            update=contribution.episode_update
            if update is None:
                raise ValueError('A discovery cycle must return its child update')
            delta=update.report.as_record()['delta']
            gain=view.controller_step.gain
            shares,audit=self.feedback_allocation(view.controller_step.gain)
            projected=self.project_feedback(view.controller_step.gain)
            for owner,share in shares.items():
                if owner<=len(self.cycle_outcomes):
                    self.cycle_outcomes[owner-1]['attributed_gain']+=share
            outcome={**work,'method_credit':gain,'root_credit':view.controller_step.gain,
                'attributed_gain':shares.get(work['cycle_id'],0.0),
                'credit_allocation':audit,
                'model_attempts':sum(self.operation_attempts.values())-work['model_attempts_before'],
                'new_candidates':[k for k in self.state.data if k not in work['prior_candidates']],
                'new_candidate_count':delta['new_inquiries'],
                'new_claim_count':delta['new_source_claims'],
                'new_reading_count':delta['new_source_readings'],
                'new_reasoning_attempts':delta['new_reasoning_attempts'],
                'controller':control}
            self.cycle_outcomes.append(outcome)
            self.action_stats=projected
            self.record({'event':'learning_feedback_allocated','cycle_id':work['cycle_id'],
                'method_gain':gain,'attributed_shares':shares,'allocations':audit,
                'action_forecasts':projected})
            self.record({'event':'discovery_cycle_completed',**outcome})
            self.pending_cycle=None
            self.pending_credit_events=[]
            self.save_state()

    def record(self,event):
        self.sequence+=1
        event={'sequence':self.sequence,'time':datetime.now(timezone.utc).isoformat(),**event}
        self.events.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
        self.events.flush()
        if self.progress:
            route=event.get('route',{})
            print(f"{self.sequence}: {event['event']} {event.get('case','')} {route.get('model','')} {event.get('reason','')}",file=sys.stderr,flush=True)

    def ask(self,operation,key,tier=ModelTier.REASONING,**request):
        self.operation_attempts[operation]=self.operation_attempts.get(operation,0)+1
        try:
            payload={'operation':operation,'research_question':self.question,**request}
            if operation=='measure_answer_quality':
                payload.update(rubric=RUBRIC,quality_rubric=QUALITY_RUBRIC)
            if operation=='measure_theory_quality':
                payload.update(rubric=THEORY_RUBRIC,quality_rubric=QUALITY_RUBRIC)
            response=self.providers['agent'](payload,tier=tier)
            self.operation_successes[operation]=self.operation_successes.get(operation,0)+1
            self.failure_streaks[operation]=0
            return response
        except Exception as exc:
            message=str(exc)
            self.failure_streaks[operation]=self.failure_streaks.get(operation,0)+1
            if 'invalid_json_schema' in message:
                self.unavailable={'scope':'response_schema_contract','error':message}
            if ('Codex login is unavailable' in message or 'Codex returned HTTP 401' in message
                    or 'Codex returned HTTP 403' in message or 'monthly usage limit exceeded' in message):
                self.unavailable={'scope':'provider_transport','error':message}
            if self.failure_streaks[operation]>=WINDOW:
                self.unavailable={'scope':'operation_contract','operation':operation,
                    'attempts':self.operation_attempts[operation],
                    'error':'Four consecutive operation responses failed; this is an execution failure, not scientific saturation.'}
            raise

    def store_source(self,document,raw):
        if document.sha256 not in self.sources:
            value={'source':document.source,'title':document.title,'version':document.sha256,
                   'text':document.text,'retrieval':raw.get('retrieval',{}) if isinstance(raw,dict) else {}}
            (self.output/'sources'/f'{document.sha256}.json').write_text(json.dumps(value,ensure_ascii=False))
            self.sources[document.sha256]=value
        self.record({'event':'source_retrieved','source':document.source,'version':document.sha256,
                     'characters':len(document.text)})

    def save_state(self):
        path=self.output/'state.json'
        temporary=self.output/'state.pending'
        temporary.write_text(json.dumps(self.state.as_record(),ensure_ascii=False,indent=2))
        temporary.replace(path)

    def leaf(self,key,event,label):
        return self.action(key,lambda:event,label)

    def action(self,key,execute,label):
        def acquire(_):
            try:
                event=execute()
            except Exception as exc:
                event={'kind':'failure','stage':label,'error':str(exc)}
                self.record({'event':'operation_failed','case':key,'stage':label,'error':str(exc)})
            return GoalProposal({'key':key,'event':event,'scope_keys':self.scope_keys(key)})
        return Leaf(None,extract=acquire,result=lambda _,value:value,label=label)

    def episode(self,mode,key,goal,contract=None,**parts):
        source=Source(self,mode,key,**parts)
        self.episode_number+=1
        unique=f'{key}:{self.episode_number}'
        def result(record):
            cases=self.state.catalog() if key==ROOT else [self.state.case(key)]
            return Record({'cases':cases,'completion':{'ended_by':record.ended_by,'units':record.units_consumed},
                'cross_case_account':(self.state.theory['current']['answer']
                                      if key==ROOT and self.state.theory['current'] else None),
                'reading_coverage':[e for e in self.state.events.values() if e.get('kind')=='source_reading'
                                    and (key==ROOT or e['inquiry']==key)],
                'work':{'action':source.parts.get('action'),
                        'reviewed_inquiry':source.parts.get('reviewed_inquiry')}})
        message=Record({'scope':'general discovery' if key==ROOT else 'focused inquiry'})
        return Episode(self.grains[mode],unique,source,
            EpisodeRequest(goal,message,contract),build_result=result,
            on_unit=lambda item,contribution,view:self.on_unit(mode,key,source,item,contribution,view))

    def child(self,view,mode,key,question,**parts):
        contract=ReportContract('Improve discovery or the source-grounded comparison of causal frames',
            {'candidate_fit':RUBRIC,'theory_fit':THEORY_RUBRIC,
             'epistemic_quality':QUALITY_RUBRIC},
            {'delta':'Counts of new inquiry, claim, and reasoning records in this visit',
             'completion':'Actual child termination and work units consumed'})
        goal=EpisodeGoal.child(view.goal,objective=question,
            result_contract={'mode':mode,'candidate_fit':RUBRIC,'theory_fit':THEORY_RUBRIC,
                             'epistemic_quality':QUALITY_RUBRIC})
        before_cases=len(self.state.data)
        before_facts=len(self.state.facts)
        before_reviews=(sum(len(row['review_attempts']) for row in self.state.data.values())
                        +len(self.state.theory['review_attempts']))
        reading_count=sum(event.get('kind')=='source_reading' for event in self.state.events.values())
        child=self.episode(mode,key,goal,contract,**parts)
        keys=None if view.grain.name in ('discovery','cycle') else self.scope_keys(key)
        def report(result,completion,request):
            return {'delta':{
                'new_inquiries':len(self.state.data)-before_cases,
                'new_source_claims':len(self.state.facts)-before_facts,
                'new_reasoning_attempts':(sum(len(row['review_attempts'])
                                             for row in self.state.data.values())
                                          +len(self.state.theory['review_attempts'])-before_reviews),
                'new_source_readings':sum(event.get('kind')=='source_reading'
                                           for event in self.state.events.values())-reading_count},
                'completion':{'ended_by':completion.ended_by,'units':completion.units_consumed}}
        return ChildEpisodeUnit(child,
            receive_result=lambda result,completion,request:Record(self.state.snapshot(keys)),
            synthesize_report=report)

    def normalize_candidates(self,proposals,references,reference_field):
        candidates=[];labels=set();errors=[]
        for candidate in proposals:
            try:
                label=string(candidate['label'],'candidate label')
                if label in labels or label in ('new','unresolved',*self.state.data):
                    raise ValueError('Candidate labels must be unique and distinct from identity keys')
                refs=candidate[reference_field]
                if not refs or any(ref not in references for ref in refs):
                    raise ValueError('Every candidate must resolve to exact acquired source records')
                pair=normalize_inference_pair(candidate.get('inference_pair'),references)
                string(candidate.get('same_case_as'),'proposed same_case_as')
                identity='case_'+digest({name:pair[name] for name in
                    ('generalizable_rule','proposed_domain')})[:20]
                for name in ('title','topic','causal_structure','generalization_defect',
                             'identity_rationale','frame_one_question','frame_one_claim',
                             'frame_two_question','frame_two_claim'):
                    string(candidate.get(name),name)
                labels.add(label)
                candidates.append({**candidate,'case_id':identity,'inference_pair':pair,
                    'evidence_refs':list(dict.fromkeys([*(references[ref] for ref in refs),
                        *pair['frame_one_warrant_refs'],*pair['frame_two_warrant_refs']]))})
            except (KeyError,TypeError,ValueError) as exc:
                errors.append('Candidate provenance/identity contract: '+str(exc))
        if candidates:
            candidates=self.resolve_candidate_identities(candidates)
        return candidates,errors

    def resolve_candidate_identities(self,candidates):
        """Resolve biological cases and their conceptual themes separately.

        Semantic equivalence requires a model; no score or eligibility decision
        enters this operation. Every source-linked occurrence remains in state.
        """
        catalog=self.state.identity_catalog()
        candidate_chars=len(json.dumps(candidates,ensure_ascii=False))
        budget=min(IDENTITY_CATALOG_CHARS,
                   max(0,MAX_ITERATION_CHARS-candidate_chars-6000))
        selected,omitted=select_identity_catalog(catalog,candidates,budget)
        visible_ids=[row['key'] for row in selected]
        contracts={}
        for candidate in candidates:
            contracts[candidate['label']]=schema({
                'same_case_as':{'type':'string','enum':['new',*visible_ids,*contracts]},
                'same_theme_as':{'type':'string','enum':['new',*visible_ids,*contracts]},
                'identity_rationale':TEXT})
        response=self.ask('resolve_candidate_identities',ROOT,
            known_conceptual_inquiries=selected,
            unshown_inquiry_count=len(omitted),
            candidate_occurrences=candidates,
            response_contract=schema({'identities':schema(contracts)}))
        resolved=[];aliases={};themes={}
        for candidate in candidates:
            item=response['identities'][candidate['label']]
            match=item['same_case_as']
            if match=='new':
                identity='case_'+digest({name:candidate['inference_pair'][name] for name in
                    ('generalizable_rule','proposed_domain')})[:20]
            elif match in self.state.data:identity=match
            elif match in aliases:identity=aliases[match]
            else:raise ValueError('Resolved identity must name a known inquiry or earlier occurrence')
            theme_match=item['same_theme_as']
            if match!='new':
                theme=(self.state.data[match]['theme_id'] if match in self.state.data
                       else themes[match])
            elif theme_match=='new':
                theme='theme_'+digest({name:candidate[name] for name in
                    ('causal_structure','generalization_defect')})[:20]
            elif theme_match in self.state.data:
                theme=self.state.data[theme_match]['theme_id']
            elif theme_match in themes:
                theme=themes[theme_match]
            else:
                raise ValueError('Resolved theme must name a known inquiry or earlier occurrence')
            aliases[candidate['label']]=identity
            themes[candidate['label']]=theme
            resolved.append({**candidate,'case_id':identity,'theme_id':theme,
                'same_case_as':match,'same_theme_as':theme_match,
                'proposed_identity':{'same_case_as':candidate['same_case_as'],
                    'identity_rationale':candidate['identity_rationale']},
                'identity_rationale':string(item['identity_rationale'],'identity rationale')})
        self.record({'event':'candidate_identities_resolved',
            'known_inquiry_count':len(self.state.data),
            'shown_inquiry_ids':visible_ids,'omitted_inquiry_ids':omitted,
            'resolutions':[{'label':c['label'],'case_id':c['case_id'],
                'theme_id':c['theme_id'],'same_case_as':c['same_case_as'],
                'same_theme_as':c['same_theme_as'],'identity_rationale':c['identity_rationale'],
                'proposed_identity':c['proposed_identity']} for c in resolved]})
        return resolved

    def synthesize(self):
        """Infer comparisons and search hypotheses across acquired records.

        A reasoning call is required: joining assertion identifiers cannot infer
        a contextual causal comparison or its unresolved experimental contrast.
        """
        facts=self.state.facts
        if not facts:
            return {'kind':'discovery_reasoning','source':'cross-source reasoning',
                'version':digest([]),'candidates':[],'learning':'No acquired evidence is available yet.',
                'search_hypotheses':[],'shown_fact_ids':[],'unshown_fact_ids':[]}
        fresh=[key for key in facts if key not in self.cycle_initial_facts]
        focus=self.pending_cycle['key'] if self.pending_cycle else ROOT
        focus_refs=self.state.data[focus]['fact_ids'] if focus!=ROOT else []
        priority=list(dict.fromkeys([*fresh,*focus_refs]))
        query=' '.join([self.question,*[facts[key]['claim'] for key in priority]])
        older=[Span(key,0,len(f['claim']+' '+f['quote']),f['claim']+' '+f['quote'])
               for key,f in facts.items() if key not in priority]
        ordering=[*priority,*[span.key for span in rank_spans(older,query)]]
        selected={};size=0
        for key in ordering:
            length=len(json.dumps(facts[key],ensure_ascii=False))
            if size+length<=SYNTHESIS_CONTEXT_CHARS:
                selected[key]=facts[key];size+=length
        catalog=self.state.identity_catalog()
        inquiry_view,omitted_inquiries=select_identity_catalog(catalog,[{
            'title':query,'topic':query,'causal_structure':query,
            'generalization_defect':query,
            'frame_one_question':query,'frame_two_question':query,
            'same_case_as':focus if focus!=ROOT else 'new'}],SYNTHESIS_CATALOG_CHARS)
        refs={'type':'string','enum':list(selected)}
        response=self.ask('synthesize_discovery',ROOT,source_claims=list(selected.values()),
            new_fact_ids=[key for key in fresh if key in selected],
            unshown_fact_count=len(facts)-len(selected),known_inquiries=inquiry_view,
            unshown_inquiry_count=len(omitted_inquiries),
            cross_case_account=(self.state.theory['current']['answer']
                                if self.state.theory['current'] else None),
            response_contract=schema({'candidates':{'type':'array','items':candidate_contract('evidence_refs',refs)},
                'learning':TEXT,'search_hypotheses':{'type':'array','items':schema({
                    'frame_hypothesis':TEXT,'possible_translation_break':TEXT,
                    'discriminating_evidence':TEXT,'query_vocabulary':{'type':'array','items':TEXT}})}}))
        candidates,errors=self.normalize_candidates(response['candidates'],{key:key for key in selected},'evidence_refs')
        self.record({'event':'discovery_reasoning_completed','shown_fact_ids':list(selected),
            'shown_inquiry_ids':[row['key'] for row in inquiry_view],
            'omitted_inquiry_ids':omitted_inquiries,
            'source_versions':sorted({f['source_version'] for f in selected.values()}),
            'context_characters':size,'unshown_fact_count':len(facts)-len(selected),
            'candidate_ids':[c['case_id'] for c in candidates],'contract_errors':errors,
            'learning':response['learning'],'search_hypotheses':response['search_hypotheses']})
        return {'kind':'discovery_reasoning','source':'cross-source reasoning',
            'version':digest(selected),'candidates':candidates,'learning':response['learning'],
            'search_hypotheses':response['search_hypotheses'],'shown_fact_ids':list(selected),
            'unshown_fact_ids':[key for key in facts if key not in selected],
            'contract_errors':errors}

    def read(self,key,parts,spans,question):
        document=parts['document']
        available={s.key:s for s in spans}
        memory=parts['memory']
        fact_schema=schema({'label':TEXT,'claim':TEXT,'quote':TEXT,'span':{'type':'string','enum':list(available)},
                            'side':{'type':'string','enum':list(SIDES)},
                            'source_kind':{'type':'string','enum':['primary','review','unknown']}})
        candidate_schema=candidate_contract('evidence_labels')
        answer=self.ask('extract_claims_and_discover',key,question=question,source=document.source,
            source_title=document.title,source_spans={k:s.as_record() for k,s in available.items()},
            retained_context=memory,
            response_contract=schema({'facts':{'type':'array','items':fact_schema},
                'candidates':{'type':'array','items':candidate_schema},
                'unresolved_questions':{'type':'array','items':TEXT},
                'memory_anchors':{'type':'array','items':schema({'span':{'type':'string','enum':list(available)},'quote':TEXT})}}))
        facts=[]
        labels={}
        errors=[]
        if not isinstance(answer.get('facts'),list):
            raise ValueError('facts must be an array')
        for proposed in answer['facts']:
            try:
                label=string(proposed['label'],'fact label')
                if label in labels:raise ValueError('Fact labels must be unique')
                span=available[proposed['span']]
                quote=string(proposed['quote'],'quote')
                if quote not in span.text:
                    raise ValueError('Quotation is not an exact substring of its supplied source span')
                if proposed['side'] not in SIDES or proposed['source_kind'] not in ('primary','review','unknown'):
                    raise ValueError('Undeclared evidence classification')
                start=span.start+span.text.index(quote)
                fact={'source':document.source,'source_version':document.sha256,
                      'title':document.title,'claim':string(proposed['claim'],'claim'),
                      'quote':quote,'start':start,'end':start+len(quote),
                      'side':proposed['side'],'source_kind':proposed['source_kind']}
                fact['id']='evidence:'+digest({k:fact[k] for k in
                    ('source','source_version','quote','start','end')})[:24]
                facts.append(fact)
                labels[label]=fact['id']
            except (KeyError,TypeError,ValueError) as exc:
                errors.append(str(exc))
        candidates,candidate_errors=self.normalize_candidates(answer.get('candidates',[]),labels,'evidence_labels')
        errors.extend(candidate_errors)
        anchors=[]
        for item in answer.get('memory_anchors',[]):
            if isinstance(item,dict) and item.get('span') in available and isinstance(item.get('quote'),str) and item['quote'] in available[item['span']].text:
                anchors.append({'source_version':document.sha256,**item})
        retained=[]
        seen=set()
        capacity=0
        for anchor in reversed([*memory['anchors'],*anchors]):
            signature=(anchor['span'],anchor['quote'])
            if signature not in seen and capacity+len(anchor['quote'])<=WINDOW_CHARS:
                retained.append(anchor)
                seen.add(signature)
                capacity+=len(anchor['quote'])
        parts['memory']={'anchors':list(reversed(retained)),'unresolved':answer.get('unresolved_questions',[])}
        self.reading_memory[document.sha256]=parts['memory']
        ranges=[[s.start,s.end] for s in spans]
        document.examined_ranges.extend(ranges)
        self.record({'event':'reading_completed','case':key,'source':document.source,
                     'source_version':document.sha256,'ranges':ranges,'facts':len(facts),
                     'candidate_proposals':len(candidates),'candidate_ids':[c['case_id'] for c in candidates],
                     'rejected_quotes':errors,'unexamined_characters':sum(len(s.text) for s in document.remaining())})
        return {'kind':'source_reading','facts':facts,'candidates':candidates,'source':document.source,
                'version':document.sha256,'ranges':ranges,'rejected_quotes':errors,
                'unresolved_questions':answer.get('unresolved_questions',[])}

    def review(self,key,role,version):
        case=self.state.case(key)
        if version!=case['evidence_version']:
            raise ValueError('The reasoning request must use the declared evidence version')
        options=[f['id'] for f in case['facts']]
        refs={'type':'array','uniqueItems':True,'items':{'type':'string','enum':options}}
        answer_schema=schema({'conclusion':TEXT,'comparison_notes':TEXT,'explanation':TEXT,
            'inference_pair':inference_pair_contract({'type':'string','enum':options}),
            'argument':argument_contract({'type':'string','enum':options}),
            'evidence_refs':refs,'uncertainties':{'type':'array','items':schema({
                'question':TEXT,'consequence':TEXT,'evidence_refs':refs})},
            'next_search_need':{'type':'string'}})
        answer=self.ask('assess_case_'+role,key,source_claims=case['facts'],
            current_answer=case['current_answer'],
            discovered_hypothesis={name:case[name] for name in ('title','causal_structure',
                'generalization_defect','frame_one_question','frame_one_claim',
                'frame_two_question','frame_two_claim','inference_pair','occurrences')},
            response_contract=answer_schema)
        return self.measure_answer(key,answer,role,version)

    def theorize(self):
        """Revise one cross-case account from acquired claims and case reasoning."""
        facts=self.state.facts
        if not facts:
            return {'kind':'theory_unavailable','reason':'No acquired source claims'}
        fresh=[ref for ref in facts if ref not in self.cycle_initial_facts]
        reviewed=[self.state.case(key) for key,row in self.state.data.items()
                  if (row['current'] and row['current']['argument_progress']['support_breadth']>0)]
        if not reviewed:
            return {'kind':'theory_unavailable',
                    'reason':'No reviewed argument has a source-grounded supported generalization'}
        reviewed.sort(key=lambda case:(case['argument_progress']['score'],
                                        quality_utility(case['epistemic_quality'])),reverse=True)
        chosen=[];themes=set()
        for case in reviewed:
            if case['theme_id'] not in themes and len(chosen)<2:
                chosen.append(case);themes.add(case['theme_id'])
        for case in reviewed:
            if len(chosen)>=2:break
            if case not in chosen:chosen.append(case)
        selected_cases=[{
            name:case[name] for name in ('key','title','topic','theme_id')}
            | {'current_answer':{
                'conclusion':case['current_answer']['conclusion'],
                'argument':argument_brief(case['current_answer']['argument'])},
                'argument_progress':case['argument_progress']}
            for case in chosen]
        current=self.state.theory['current']
        current_refs=current['answer']['evidence_refs'] if current else []
        argument_evidence=[ref for case in chosen for ref in
                           argument_refs(case['current_answer']['argument'])]
        priority=list(dict.fromkeys([*argument_evidence,*current_refs,*fresh,
            *[ref for case in selected_cases for ref in
              self.state.data[case['key']]['fact_ids']]]))
        ordering=[*priority,*[ref for ref in facts if ref not in priority]]
        selected={};size=0
        for ref in ordering:
            length=len(json.dumps(facts[ref],ensure_ascii=False))
            if size+length<=THEORY_CONTEXT_CHARS:
                selected[ref]=facts[ref];size+=length
        evidence_options=list(selected)
        case_options=[case['key'] for case in selected_cases]
        refs={'type':'array','uniqueItems':True,
              'items':{'type':'string','enum':evidence_options}}
        cases={'type':'array','uniqueItems':True,
               'items':{'type':'string','enum':case_options}}
        answer_schema=schema({'conclusion':TEXT,'cross_case_mapping':TEXT,'explanation':TEXT,
            'case_refs':cases,'evidence_refs':refs,
            'uncertainties':{'type':'array','items':schema({
                'question':TEXT,'consequence':TEXT,'evidence_refs':refs})},
            'next_frame_gap':{'type':'string','minLength':1,'maxLength':500},
            'next_evidence_need':{'type':'string','minLength':1,'maxLength':500},
            'avoid_pattern':{'type':'string','minLength':1,'maxLength':500}})
        answer=self.ask('develop_theory',ROOT,source_claims=list(selected.values()),
            reviewed_cases=selected_cases,current_answer=current['answer'] if current else None,
            unshown_fact_count=len(facts)-len(selected),response_contract=answer_schema)
        for field in ('conclusion','cross_case_mapping','explanation',
                      'next_frame_gap','next_evidence_need','avoid_pattern'):
            string(answer.get(field),field)
        if (not isinstance(answer.get('evidence_refs'),list)
                or any(ref not in selected for ref in answer['evidence_refs'])
                or not isinstance(answer.get('case_refs'),list)
                or any(ref not in case_options for ref in answer['case_refs'])):
            raise ValueError('Cross-case account references must name supplied evidence and cases')
        for issue in answer['uncertainties']:
            if any(ref not in selected for ref in issue['evidence_refs']):
                raise ValueError('Theory uncertainty cites an unavailable source claim')
        measurement=schema({'score':{'type':'number','minimum':0,'maximum':1},
            'evidence_refs':refs,'rationale':TEXT})
        fit=schema({name:measurement for name in THEORY_RUBRIC})
        quality=schema({name:measurement for name in QUALITY_RUBRIC})
        previous=(schema({'theory_fit':fit,'epistemic_quality':quality}) if current else {'type':'null'})
        known=self.state.theory['materials']
        measured=self.ask('measure_theory_quality',ROOT,source_claims=list(selected.values()),
            reviewed_cases=selected_cases,answer=answer,
            previous_answer=current['answer'] if current else None,
            known_materials={identity:{name:row['answer'][name] for name in
                ('conclusion','cross_case_mapping','case_refs')}
                for identity,row in known.items()},
            response_contract=schema({'equivalent_to':{'type':'string','enum':['new',*known]},
                'identity_rationale':TEXT,'theory_fit':fit,'epistemic_quality':quality,
                'previous_measurements':previous,'revision_summary':TEXT}))
        self.validate_measurements(measured['theory_fit'],THEORY_RUBRIC,evidence_options)
        self.validate_measurements(measured['epistemic_quality'],QUALITY_RUBRIC,evidence_options)
        prior=measured['previous_measurements']
        if current:
            self.validate_measurements(prior['theory_fit'],THEORY_RUBRIC,evidence_options)
            self.validate_measurements(prior['epistemic_quality'],QUALITY_RUBRIC,evidence_options)
        elif prior is not None:
            raise ValueError('No previous cross-case account exists')
        match=measured['equivalent_to']
        if match not in ('new',*known):
            raise ValueError('Unknown cross-case material identity')
        identity=match if match!='new' else 'answer:'+digest({
            'theory':answer,'rubric':CREDIT_VERSION})[:24]
        case_material_refs={case_id:self.state.data[case_id]['current']['material_id']
                            for case_id in answer['case_refs']}
        review={'material_id':identity,'answer':answer,
            'theory_fit':measured['theory_fit'],
            'epistemic_quality':measured['epistemic_quality'],
            'theory_progress':derive_theory_progress(measured['theory_fit'],{
                case_id:self.state.data[case_id]['current']['argument_progress']
                for case_id in case_material_refs}),
            'identity_rationale':string(measured['identity_rationale'],'identity rationale')}
        return {'kind':'theory_review_attempt','source':'cross-case reasoning',
            'version':digest(selected),'theory_review':review,
            'case_material_refs':case_material_refs,
            'previous_material_id':current['material_id'] if current else None,
            'previous_measurements':prior,
            'revision_summary':string(measured['revision_summary'],'revision summary')}

    def measure_answer(self,key,answer,role,version):
        """Interpret warrant and semantic identity; arithmetic alone cannot do so.

        This is a REASONING call site. A fresh score of unchanged material is an
        audit observation; the archive keeps that material's first measurement.
        """
        case=self.state.case(key)
        if version!=case['evidence_version']:
            raise ValueError('Measurement must use the current declared evidence version')
        options=[f['id'] for f in case['facts']]
        for field in ('conclusion','comparison_notes','explanation'):
            string(answer.get(field),field)
        refs=answer.get('evidence_refs')
        if not isinstance(refs,list) or any(ref not in options for ref in refs):
            raise ValueError('Answer references must name existing source records')
        pair=normalize_inference_pair(answer.get('inference_pair'),{ref:ref for ref in options})
        argument=normalize_argument(answer.get('argument'),options)
        answer['inference_pair']=pair
        answer['argument']=argument
        answer['evidence_refs']=list(dict.fromkeys([
            *refs,*pair['frame_one_warrant_refs'],*pair['frame_two_warrant_refs'],
            *argument_refs(argument)]))
        known=case['materials']
        measurement_schema=schema({'score':{'type':'number','minimum':0,'maximum':1},
            'evidence_refs':{'type':'array','items':{'type':'string'}},'rationale':TEXT})
        fit_schema=schema({name:measurement_schema for name in RUBRIC})
        quality_schema=schema({name:measurement_schema for name in QUALITY_RUBRIC})
        step_schema=schema({name:schema({identity:measurement_schema for identity in identities})
                            for name,identities in step_ids(argument).items()})
        support_ids=[item['id'] for item in argument['generalization']['supports']]
        repair_ids=[item['id'] for item in argument['generalization']['repair_tests']]
        support_measurement_schema=schema({**measurement_schema['properties'],
            'reviewed_relation':{'type':'string','enum':[
                'application','component','restatement','speculative']},
            'evidence_basis':{'type':'string','enum':[
                'direct_result','secondary_summary','citation_title','unsupported']}})
        support_schema=schema({identity:support_measurement_schema for identity in support_ids})
        repair_schema=schema({identity:measurement_schema for identity in repair_ids})
        previous_schema=(schema({'candidate_fit':fit_schema,'epistemic_quality':quality_schema})
                         if case['current_answer'] else {'type':'null'})
        measured=self.ask('measure_answer_quality',key,source_claims=case['facts'],answer=answer,
            inquiry_generalization={name:case['inference_pair'][name] for name in
                ('generalizable_rule','proposed_domain','domain_rationale')},
            previous_answer=case['current_answer'],
            known_materials={identity:row['answer'] for identity,row in known.items()},
            response_contract=schema({'equivalent_to':{'type':'string','enum':['new',*known]},
                'identity_rationale':TEXT,'candidate_fit':fit_schema,
                'epistemic_quality':quality_schema,'step_warrants':step_schema,
                'support_warrants':support_schema,'repair_warrants':repair_schema,
                'previous_measurements':previous_schema,
                'revision_summary':TEXT}))
        match=measured.get('equivalent_to')
        if match not in ('new',*known):
            raise ValueError('Measurement must use a declared material identity')
        self.validate_measurements(measured.get('candidate_fit'),RUBRIC,options)
        self.validate_measurements(measured.get('epistemic_quality'),QUALITY_RUBRIC,options)
        steps=measured.get('step_warrants')
        if not isinstance(steps,dict) or set(steps)!=set(step_ids(argument)):
            raise ValueError('Step warrant measurements require both causal frames')
        for frame_name,identities in step_ids(argument).items():
            self.validate_measurements(steps[frame_name],identities,options)
        self.validate_measurements(measured.get('support_warrants'),support_ids,options)
        for identity,row in measured['support_warrants'].items():
            if row.get('reviewed_relation') not in (
                    'application','component','restatement','speculative'):
                raise ValueError(f'Support {identity} needs an independent relation verdict')
            if row.get('evidence_basis') not in (
                    'direct_result','secondary_summary','citation_title','unsupported'):
                raise ValueError(f'Support {identity} needs an evidence-basis verdict')
        self.validate_measurements(measured.get('repair_warrants'),repair_ids,options)
        previous=measured.get('previous_measurements')
        if case['current_answer']:
            if not isinstance(previous,dict):
                raise ValueError('Paired assessment requires previous-answer measurements')
            self.validate_measurements(previous.get('candidate_fit'),RUBRIC,options)
            self.validate_measurements(previous.get('epistemic_quality'),QUALITY_RUBRIC,options)
        elif previous is not None:
            raise ValueError('There is no previous answer to assess')
        identity=(match if match!='new' else 'answer:'+digest({'case':key,
                   'answer':answer,'rubric':CREDIT_VERSION})[:24])
        review={'material_id':identity,'answer':answer,'candidate_fit':measured['candidate_fit'],
                'epistemic_quality':measured['epistemic_quality'],
                'step_warrants':steps,'support_warrants':measured['support_warrants'],
                'repair_warrants':measured['repair_warrants'],
                'argument_progress':argument_progress(argument,measured['candidate_fit'],steps,
                    measured['support_warrants'],measured['repair_warrants']),
                'identity_rationale':string(measured.get('identity_rationale'),'identity_rationale')}
        return {'kind':'review_attempt','attempt_id':f'{self.output.name}:{self.sequence}',
                'role':role,'version':version,'review':review,
                'previous_material_id':case['material_id'],'previous_measurements':previous,
                'revision_summary':string(measured.get('revision_summary'),'revision_summary')}

    @staticmethod
    def validate_measurements(values,columns,options):
        if not isinstance(values,dict) or set(values)!=set(columns):
            raise ValueError('Measurements must contain each declared continuous dimension')
        for name,row in values.items():
            score=row.get('score')
            if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:
                raise ValueError('Quality measurements must be finite numbers in [0,1]')
            refs=row.get('evidence_refs')
            if not isinstance(refs,list) or any(ref not in options for ref in refs):
                raise ValueError('Measurement references must name existing evidence')
            string(row.get('rationale'),name+' rationale')

    def execute(self):
        names=('reversal_task.py','reversal_prompts.py','causal_argument.py','cli.py','configured_providers.py',
               'credit_assignment/attainment.py','method_loop/episode_binding/web_search_binding.py',
               'task_executors/evidence_extraction/argo.py',
               'task_executors/evidence_extraction/codex.py')
        registration={'experiment':CREDIT_VERSION,'question':self.question,'target':self.target,
            'initial_candidates':list(self.state.data),'preset_cases':[],
            'candidate_fit_rubric':RUBRIC,'theory_fit_rubric':THEORY_RUBRIC,
            'quality_rubric':QUALITY_RUBRIC,'case_credit_columns':COLUMNS,
            'theory_credit_columns':THEORY_COLUMNS,
            'evaluator_policy':self.state.evaluator_policy,
            'case_identity':'one proposed generalizable causal inference and its changing support and challenge graph per inquiry; separately assign a conceptual translation theme',
            'candidate_origin':'exact retrieved source spans plus explicitly provisional inference',
            'paired_inference_record':'each review records a warranted rule and domain, distinct theories it supports through explicit additions and derivations, an independently warranted target theory, a transported prediction, and concrete repair tests',
            'credit_formula':'case archives use three epistemic coordinates and derived argument_progress; the collection account archive uses three epistemic coordinates and derived theory_progress, bounded by a diminishing combination of cited case progress. Both progress scores use 0.05 warranted rule + 0.20 distinct support breadth + 0.75 bottlenecked challenge; theory interpretation and nontransport ratings may lower these components but cannot create reversal credit absent a cited case challenge',
            'learning_feedback':'each positive method gain is allocated once across distinct first evidence-origin cycles for newly used citations and the current reasoning cycle; visit-order forecasts are recomputed after delayed shares; this allocation does not add method credit',
            'root_attention':{'rule':'smoothed expected attributed gain per action visit, recomputed after delayed feedback, plus exploration*sqrt(log(cycles+2)/(visits+1))',
                              'exploration':UCB_EXPLORATION,'initial_expected_yield':YIELD_PRIOR,
                              'yield_update_fraction':YIELD_UPDATE},
            'review_attribution':'focused acquisition reviews that inquiry; general acquisition reviews only newly created inquiries or those receiving new evidence in the cycle',
            'controller_window':WINDOW,'root_stop_threshold':ROOT_STOP_YIELD,
            'root_stop_statistic':'maximum one-step expected attributed gain across general discovery and every known inquiry after delayed feedback is assigned; literal zero only',
            'child_stop_threshold':STOP_EPSILON,'root_observation':'one integrated acquisition, frame synthesis, case inquiry and cross-case reasoning cycle',
            'synthesis_context_characters':SYNTHESIS_CONTEXT_CHARS,
            'theory_source_context_characters':THEORY_CONTEXT_CHARS,
            'review_handoff':'bounded citation-first JSON view, with text truncation and unshown-claim counts; compact measurement schema validates reference identities after response; full reviewed argument retained in state',
            'scientific_gates':False,'source_provenance_validation':True,
            'report_preference':'same-evidence epistemic utility; first canonical credit measurement persists; every material alternative is offered',
            'search_limit':SEARCH_LIMIT,'reading_window_chars':WINDOW_CHARS,
            'post_collection_selection':'only after stopping: all task-fit scores >=0.70 and a source-grounded challenge; retain every inquiry in complete report',
            'failure_rule':'four consecutive failed structured operations or unavailable provider ends as execution failure',
            'bounds':'ten is a reporting milestone, not a credit capacity or stop; no run-wide iteration, source, time or spend cap; numerical continuation governs',
            'model_tiers':self.providers['model_tiers'],
            'page_relevance_model':self.providers['assessor'].model,
            'implementation_hashes':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                     for name in names}}
        (self.output/'registration.json').write_text(json.dumps(registration,indent=2))
        snapshot=self.output/'implementation';snapshot.mkdir()
        for name in names:
            path=snapshot/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((Path(__file__).parent/name).read_bytes())
        self.record({'event':'experiment_registered',**registration})
        goal=EpisodeGoal.root(objective=self.question,result_contract={
            'target':self.target,'result':'source-discovered causal generalizations, warranted uses, and tested limits',
            'scientific_gates':False})
        try:
            root=self.episode('discovery',ROOT,goal)
            record=root.run(self.context)
            (self.output/'episodes.json').write_text(json.dumps(record.as_record(),ensure_ascii=False,indent=2))
            cases=[self.state.case(key) for key in self.state.data]
            def fit(case):
                return (math.prod(row['score'] for row in case['measurements'].values())
                        **(1/len(case['measurements'])) if case['measurements'] else 0.0)
            ranked=sorted(cases,key=fit,reverse=True)
            selected=[];selected_themes=set()
            for case in ranked:
                if (case['theme_id'] not in selected_themes and case['measurements']
                        and case['argument_progress']['challenge_strength']>0
                        and all(row['score']>=0.70 for row in case['measurements'].values())):
                    selected.append(case['key']);selected_themes.add(case['theme_id'])
            report={'run_complete':self.unavailable is None,'question':self.question,'target':self.target,
                'candidate_count':len(cases),'reviewed_count':sum(bool(c['current_answer']) for c in cases),
                'conceptual_theme_count':len({case['theme_id'] for case in cases}),
                'provisional_selected':selected,'provisional_target_met':len(selected)>=self.target,
                'selection_interpretation':'post-run numeric screen, not independent scientific validation',
                'inquiries':ranked,'credit':self.state.snapshot(),
                'cross_case_account':self.state.theory['current'],
                'cross_case_materials':self.state.theory['materials'],
                'queries':self.queries,'cycle_outcomes':self.cycle_outcomes,
                'output_directory':str(self.output),'provider_failure':self.unavailable,
                'operation_attempts':self.operation_attempts,'operation_successes':self.operation_successes,
                'post_collection_filtering':'all task-fit scores >=0.70 and positive challenged-generalization strength, at most one representative per conceptual theme, after Episode closure only; all accounts retained',
                'completion':{'ended_by':record.ended_by,'units':record.units_consumed},
                'interpretation':'Source-driven discovery of pragmatically useful causal frames with continuous provisional judgments. Numerical saturation is observed lack of improvement, not truth or literature exhaustion.'}
            (self.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
            self.write_markdown(report)
            return {k:v for k,v in report.items() if k not in ('inquiries','queries','cycle_outcomes')}
        finally:
            self.save_state()
            self.events.close()

    def write_markdown(self,report):
        lines=['# Source-driven discovery of biological causal frames','',
            'Question: '+report['question'],'',
            f"Discovered inquiries: {report['candidate_count']}; reviewed: {report['reviewed_count']}; post-run provisional selection: {len(report['provisional_selected'])}/{report['target']}.",'',
            'All candidates and negative accounts remain available. Selection is applied only after the loop stops. Numerical saturation does not establish truth or exhaustiveness.','']
        for case in report['inquiries']:
            pair=((case['current_answer'] or {}).get('inference_pair')
                  or case['inference_pair'])
            lines.extend(['## '+case['title'],'',
                'First question: '+case['frame_one_question'],'',
                'First causal claim: '+case['frame_one_claim'],'',
                'Second question: '+case['frame_two_question'],'',
                'Second causal claim: '+case['frame_two_claim'],'',
                'First linked causal inference: '+pair['frame_one_causal_inference'],'',
                'First warrant refs: '+', '.join(pair['frame_one_warrant_refs']),'',
                'Second linked causal inference: '+pair['frame_two_causal_inference'],'',
                'Second warrant refs: '+', '.join(pair['frame_two_warrant_refs']),'',
                'Proposed translation: '+pair['proposed_translation'],'',
                'Causal tension: '+pair['causal_tension'],'',
                'Category and relation comparison: '+case['causal_structure'],'',
                'Inference translation to assess: '+case['generalization_defect'],'',
                'Identity rationale: '+case['identity_rationale'],''])
            answer=case['current_answer']
            if answer:
                lines.extend([answer['conclusion'],'',answer['comparison_notes'],'',answer['explanation'],''])
                argument=answer['argument']
                for frame_name,title in (('frame_one','First causal construction'),
                                         ('frame_two','Second causal construction')):
                    frame=argument[frame_name]
                    lines.extend(['### '+title,'',
                        'Question: '+frame['question'],'',
                        'Actors and categories: '+frame['actors_and_categories'],'',
                        'Boundary: '+frame['boundary'],'',
                        'Intervention: '+frame['intervention'],'',
                        'Outcome: '+frame['outcome'],'',
                        'Scope: '+frame['scope'],'',
                        'Pragmatic use: '+frame['pragmatic_use'],''])
                    for step in frame['steps']:
                        lines.append(f"- **{step['id']} ({step['role']})** {step['claim']} "
                            f"Premises: {', '.join(step['premise_ids']) or 'none'}. "
                            f"Reasoning: {step['reasoning']} "
                            f"Source refs: {', '.join(step['warrant_refs']) or 'none'}.")
                    lines.extend(['','Conclusion step: '+(frame['conclusion_step_id'] or 'open'),''])
                bridge=argument['translation']
                lines.extend(['### Attempted translation and failure','',
                    'Source step: '+(bridge['source_step_id'] or 'open'),'',
                    'Target step: '+(bridge['target_step_id'] or 'open'),'',
                    'Actor map: '+bridge['actor_map'],'',
                    'Intervention map: '+bridge['intervention_map'],'',
                    'Outcome map: '+bridge['outcome_map'],'',
                    'Relation map: '+bridge['relation_map'],'',
                    'Bridge premise: '+bridge['bridge_premise'],'',
                    'Transported inference: '+bridge['transported_inference'],'',
                    'Target inference: '+bridge['target_inference'],'',
                    'Failure mechanism: '+bridge['failure_mechanism'],'',
                    'Consequence: '+bridge['consequence'],''])
                alternative=argument['fixed_category_alternative']
                generalization=argument['generalization']
                lines.extend(['### Proposed generalizable inference','',
                    'Rule: '+generalization['rule'],'',
                    'Proposed domain: '+generalization['proposed_domain'],'',
                    'Domain warrant: '+generalization['domain_warrant'],'',
                    'Target domain status: '+generalization['target_domain_status'],'',
                    'Target theory: '+generalization['target_theory'],''])
                for item in generalization['supports']:
                    lines.extend(['#### Supported theory: '+item['id'],'',
                        item['theory'],'','Question: '+item['question'],'',
                        'Consequence: '+item['consequence'],'',
                        'Relation to rule: '+item['relation_to_rule'],'',
                        'Status: '+item['status'],'',
                        'Derivation: '+item['derivation'],'',
                        'Formal addition: '+item['addition']['formal_change'],'',
                        'Graph edits: '+json.dumps(item['addition']['edits'],ensure_ascii=False),''])
                for item in generalization['repair_tests']:
                    lines.extend(['#### Tested repair: '+item['id'],'',
                        'Status: '+item['status'],'',
                        'Formal change: '+item['addition']['formal_change'],'',
                        'Graph edits: '+json.dumps(item['addition']['edits'],ensure_ascii=False),'',
                        'Target prediction: '+item['target_prediction'],'',
                        'Tested earlier theories: '+', '.join(item['tested_support_ids']),'',
                        'Effect on prior uses: '+item['prior_uses_after_repair'],'',
                        'Independent constraint: '+item['independent_constraint'],''])
                lines.extend(['### Fixed-category alternative','',
                    'Status: '+alternative['status'],'',
                    'Shared vocabulary: '+alternative['fixed_vocabulary'],'',
                    'Added context or dependencies: '+alternative['added_context_or_dependencies'],'',
                    'Reconciliation: '+alternative['reconciliation'],'',
                    f"Argument progress: {case['argument_progress']['score']:.3f}",''])
                for title,measurements in (('Epistemic quality',case['epistemic_quality']),
                                           ('Task fit',case['measurements'])):
                    lines.extend(['### '+title,''])
                    for name,row in measurements.items():
                        lines.append(f"- **{name}: {row['score']:.3f}** — {row['rationale']}")
                    lines.append('')
                lines.extend(['### Remaining uncertainty',''])
                for issue in answer['uncertainties']:
                    lines.append('- '+issue['question']+' '+issue['consequence'])
                lines.extend(['','Next information need: '+answer['next_search_need'],''])
            else:lines.extend(['This source-grounded candidate has not yet received a completed comparison.',''])
            lines.extend(['### Sources and provenance',''])
            seen=set()
            for fact in case['facts']:
                marker=(fact['source'],fact['source_version'])
                if marker in seen:continue
                seen.add(marker)
                lines.append(f"- [{fact['title'] or fact['source']}]({fact['source']}) — source version {fact['source_version'][:12]}; exact evidence spans and quotations are retained in state.json.")
            lines.append('')
        theory=report['cross_case_account']
        lines.extend(['## Cross-case causal-generalization account',''])
        if theory:
            account=theory['answer']
            lines.extend([account['conclusion'],'',account['cross_case_mapping'],'',
                account['explanation'],'',
                'Next frame gap: '+account['next_frame_gap'],'',
                'Next evidence need: '+account['next_evidence_need'],'',
                'Avoid repeating: '+account['avoid_pattern'],'',
                f"Derived theory progress: {theory['theory_progress']['score']:.3f}",
                'Progress grounded in cases: '+', '.join(theory['theory_progress']['source_case_ids']),''])
            for title,measurements in (('Epistemic quality',theory['epistemic_quality']),
                                       ('Theory fit',theory['theory_fit'])):
                lines.extend(['### '+title,''])
                for name,row in measurements.items():
                    lines.append(f"- **{name}: {row['score']:.3f}** — {row['rationale']}")
                lines.append('')
            lines.extend(['Source references: '+', '.join(account['evidence_refs']),
                          'Case references: '+', '.join(account['case_refs']),''])
        else:
            lines.extend(['No source-grounded cross-case account was completed.',''])
        lines.extend(['## Limits','',
            'Source transcription, semantic identity, frame comparisons and scientific judgment are distinct. Identity and quality are model assessments requiring calibration. An optimistic first canonical measurement can suppress future credit. Summed local volumes assume distinct inquiries. This report preserves provisional and negative accounts; the post-run selection does not certify the requested collection.',''])
        (self.output/'report.md').write_text('\n'.join(lines))


def run_discovery(*,config,output=None,progress=False,question=QUESTION,target=10):
    if output is None:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output=Path.home()/'think-and-find-runs'/f'discovery-{stamp}'
    return Run(config,output,progress,question=question,target=target).execute()
