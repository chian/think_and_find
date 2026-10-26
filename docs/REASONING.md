# Reasoning that guides further research

## Design target

A generic research system needs a cycle in which evidence informs reasoning and
reasoning informs acquisition. It must be able to investigate an inference,
not only search for missing factual fields. This document specifies the intended
relationship; it does not claim the copied runtime already implements it.

The copied system already proposes hypotheses, reviews evidence and assesses
contextual relationships. Those operations should be evaluated for reuse. The
transition is broader than replacing an arithmetic function with one model call:
state, parent returns, search objectives, dependency tracking and progress must
all understand the reasoning result.

## A reasoning result

A useful structured result records the question addressed, a provisional
conclusion, the premises used, the inference connecting them, relevant alternatives
or objections, and what remains unresolved. References should identify actual
source records or prior reasoning results. Source statements and interpretations
must be distinguishable even when they occur in the same result.

At the design level, an operation can be described as:

```text
reason(question, evidence, existing_argument, task_criteria)
    -> conclusion, premise_references, rationale,
       alternatives, unresolved_questions, proposed_information_needs
```

This is a conceptual contract, not a new implemented function or an instruction
to add another mandatory review to every extraction. Reasoning is substantive
work chosen to advance the task. Its format and criteria should fit that work.

## From an unresolved inference to a search

For each proposed information need, retain the inference it bears on and how a
finding could change the current conclusion. Search strategies turn that need
into literal queries. Source selection and reading inherit the same question;
they should not revert to broad topical relevance after the parent has identified
a specific uncertainty.

For example, an account of individual units and an account of the collective
they form may both be useful, while an inferred causal role fails to translate
between them. The next useful search may seek how interactions construct the
collective actor, whether each account's categories match its biological
question, or evidence that distinguishes a changed causal relation from a
routine modifier under fixed categories. More measurements of one effect alone
would not resolve that inferential uncertainty.

A source may explicitly discuss the comparison, but that is not required. The
system can infer a relationship across sources if it records the premises and
reasoning. Requiring every inferred conclusion to be an exact source quotation
would prevent the reasoning this design is intended to support.

## Revisions and dependencies

A result should record which evidence and prior judgments it used. New relevant
evidence or a changed interpretation can make its reasoning require reconsideration.
Do not confuse that with a source changing, a provider failing, or an unchanged
calculation becoming numerically stale.

Preserve completed work when a later attempt fails. A failed search does not
refute a claim; an unread document tail does not establish absence; an invalid
model response does not establish that the underlying question is unanswerable.

## Task criteria and uncertainty

The task supplies continuous measures of the developing answer. The active
operator policy has no scientific gates: unresolved comparisons, warrant and
uncertainty affect quality rather than admission. Separate candidate suitability
from epistemic progress so a supported correction or negative conclusion can
improve the answer. Apply collection filters downstream of research. Criteria
can include comparability and diversity where needed, without claiming universal
requirements for every research question.

The current discovery binding measures source fidelity, uncertainty discipline,
explanatory usefulness and four task-fit dimensions for question-specific
causal-frame comparisons. They occupy separate continuous coordinates in one
local hypervolume. A zero target-fit coordinate yields zero volume continuously;
the research record still retains negative and unresolved accounts. A paired assessment compares prior
and proposed accounts against the same current evidence. Historical attainment
pays only once for identified material; another rating or cosmetic rewrite cannot
mint progress. Sources create the candidate collection inside the loop. See
[the discovery policy and its limits](DISCOVERY_PLAN.md).

No finite checklist in this design guarantees that biological context is sufficient.
The system must be able to expose uncertainty about the categories or comparison
itself. Reporting a limit or rejecting a tempting example can advance reasoning
without being counted as a requested positive example.

## Work still required

Define task-neutral record types and operation contracts; determine what existing
hypothesis and review machinery can be reused; carry unresolved inferences into
acquisition objectives; define numerical progress over those results; and verify
that subsequent searches actually respond to the reasoning. These are future
implementation steps. The later task-specific prototype verifies selected
reasoning-and-credit behaviors without completing that generic interface.
