# Task authority and episode ownership

The user's original scientific instructions, including their later corrections,
define this one generic research task type: finding and assessing examples of
pragmatic causal pluralism across natural phenomena. Recover and read those messages when resolving
its causal-inference definitions, theory relationships, generalizability bounds
and useful reversals. This file does not replace or abbreviate that authority.

The application must investigate through the existing Episode structure. Every
Episode revisits a scoped decision using its assignment, actual outcomes and
requested child returns. Research order emerges from those choices. The parent
can reconsider an interpretation, commission relevant evidence work or select
another useful operation as its information changes.

The deliverable requires reasoning with traceable evidence and explicit limits.
An inventory of sources, filled fields, operational success or numerical yielding
alone does not establish that the scientific request has been answered.

The existing Episode hierarchy is wired to the causal scoring sheets. Candidate
inferences have explicit identities and their own application, comparison and
repair assessments. The framework can discover and revisit multiple candidates.

Instructions to the coding agent to study methodological literature specify
how to develop the formalism and scoring criteria. They are not research targets
for the application. The runtime objective is defined in
`case_reports/research/task.py`; it prescribes no author, paper, or example.
The earlier runs directed at methodological literature pursued the wrong task
and are not evidence of successful scientific execution.

## Original scientific instructions

The following complete user messages were recovered from the original session
logs. They are quoted verbatim. Later corrections govern earlier formulations.
The [full recovered discussion](/home/chia/recovered-instructions/think-and-find/full-relevant-user-transcript.md)
retains the surrounding user instructions with exact source locations.

**2026-10-23T16:23:58.597Z** — session `01a16051-c42a-7c40-ad2c-8ad2ac4565ac`, original log line 8965.

> The prompts look good, but I want to plan with you some ideas for improving them. The idea of vaguely finding contradictions is rough. At best, it is a very very vague set of instructions. SO it could use some few-shot examples and perhaps it could use some formalisms. For example, it might be good to break down each hypothesis into parts - candidate inferences is what Gentner would call them. Another way of thinking of them would be linked pairs in an SCM. If you can break a hypothesis down atomically and show that under one framing, scale, etc.., you have one candidate inference and at another scale, you have the opposite one, that is what we are looking for formulaically. It's not 100% locked down but this should be more defining.

**2026-10-27T15:22:42.087Z** — session `01a179b5-73fa-79b2-ba5b-07959985a86d`, original log line 5587.

> I think I know what you mean by bridge. You mean generalizable causal inference. So to identify what you call the bridge, we must first identify which "facts" we expect, based on the analysis or system we are solving, to be generalizable rules. By defining those explicitly, we can target searching for when that generalization breaks and why. Adding another link in the causal structure is a trivial break and easily fixed. A fix that breaks the generalizability on other systems is a problematic break and the thing we are looking for. Does that make sense to you? So for that to work, we need two things. We need to identify the generalizable causal inference being proposed, and we need to know what overall hypothesis or theory that generalizable causal inference supports. We then need a credit assignment that adds credit when finding more hypotheses or theories that this same generalizable causal inference supports or can be trivially added to to support (and we should make that formal addition). And then finally, we need a big credit add when we find the opposite. A theory or hypothesis that this generalizable causal inference is not compatible with for a non-trivial reason. I.e., we need to find something where the origianl generalizable causal inference does not fit the hypothesis or theory and in addition, any fixes to that are not useful. I.e., cannot further be used by that theory. This is why I suggested scale and course-graining. B/c often when looking at larger systems, a microscopic fact is not very useful and in fact gets in the way of identifying the larger pattern. You don't look at where all the atoms are to figure out the bending mode of the spoon. It's not only implausible and unwieldy, the fundamental physics do not support it without a lot of fitting assumptions which then render the theory really not very predictive anyhow.

**2026-10-27T15:27:59.230Z** — session `01a179b5-73fa-79b2-ba5b-07959985a86d`, original log line 5647.

> I think you should make it a goal to implement and test this. In addition, you should not hard-prompt no new organisms and so on. It should depend on how the generalizable causal inference should generalize. If it's an ecological rule about herbivores, it absolutely can involve new animals. Does this make sense? You should prompt this to rely on explicitly naming the nature of the generalizability

**2026-10-27T22:24:29.774Z** — session `01a179b5-73fa-79b2-ba5b-07959985a86d`, original log line 8235.

> Perhaps we need better language to distinguish these two things. The consequence should apply to a hypothesis or theory. A hypothesis or theory should involve many causal links or candidate inferences or mechanisms. The individual inference being generalized needs to have a name that is distinguished easily. As you can see, there is a lot of typing to make the point precise.

**2026-10-27T22:26:18.863Z** — session `01a179b5-73fa-79b2-ba5b-07959985a86d`, original log line 8251.

> This is evidence of G's scope and that should get credit. That's what we said and designed. A reversal should get even more credit, but a similarity still get's credits. Freeze that language somewhere in a readme so I can refer back to it as needed. The real problem here is btw not in the credit unless it counted this as a reversal for some reason, the real issue here is that it said an algorithm is causal. Do you understand why this is not what we are after?

**2026-10-27T22:28:19.949Z** — session `01a179b5-73fa-79b2-ba5b-07959985a86d`, original log line 8313.

> Try to restrict the search to natural phenomena. A calculator (which is ultimately what this is) is causal of predicting some calculation result and always is. But that's not helpful per se. That's a mechanistic operation where you are providing the mechanism. So that's kind of tautological.


## Current design and precedence

The later discussion in the current conversation makes the scientific requirements
into a scoring rubric: a global sheet records the state of a generalized causal
inference and children supply evidence or reasoning for scoped subsheets. The
scientific parent reasons about that state and chooses further work within the
existing iterative Episodes. This is the organizing design, rather than the
older list of accounting proposals.

[The consolidated scoring specification](CAUSAL_SCORING_SHEET.md) describes that
one design, its subsheets, communication and credit meanings, and the remaining
implementation gaps. It distinguishes explicit requirements from proposed
formalization and does not claim that the current implementation satisfies them.

The recovered transcript linked above ends before the current conversation's
scoring-sheet discussion. It is historical context, not a complete specification.
In particular, the older attained-volume and fractional-feedback proposals do
not by themselves define the later rubric's scientific operands. Their prior
acceptance is retained; the later discussion does not explicitly cancel them.
Their integration must be explained against the rubric before calling the
result a completed credit design. Neither restoring old code nor counting
completed sheet fields accomplishes that integration.

Halpern and Lakatos were assigned to the coding agent as methodological reading
for constructing the criteria. They were not assigned to the application as a
research objective. The runtime investigates natural phenomena generally, with
no prescribed author, case or second task binding.

The latest clarification makes expected generalizability bounds provisional,
revisable expectations whose purpose is to reveal and explain where expectations
break. Literature and reasoning across literature ground them where available;
provisional judgment is explicitly permitted for these bounds when unavailable.
Bounds may expand or shrink as hypotheses are investigated. Preserve the change
and its reason, distinguishing better definitions/missing dependencies within
general rules from special exceptions or magic fudge factors. This permission
for judgment does not extend to subjective numerical scoring.

Expected generalizability bounds now have structured versioned expectation,
break and support records. Revisions retain exact prior expectations, triggering
scientific snapshots, proposed rules and their retained/lost consequences.
Bounded inspect and commission paths expose historical public records; G changes
invalidate current expectations without erasing the original claim. Later
support can address that original claim and expected-bounds version explicitly.
The offline synthetic demonstration is `scripts/show_expected_bounds.py`;
it validates state/link behavior, not scientific proof or empirical discovery.
