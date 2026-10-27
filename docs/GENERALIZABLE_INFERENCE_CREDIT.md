# Inference-centered discovery target

The unit of discovery is a proposed generalizable causal inference `G`, not a
pair of papers or organisms. `G` states its premise, causal conclusion,
intervention and outcome meanings, abstraction or scale, proposed domain, and
the source-backed reason that this domain is plausible. A claim that works in
one source does not itself warrant unrestricted generalization.

The research record contains a support graph for `G`. Each distinct hypothesis
or theory `H_i` answers a causal question beyond the source proof of `G` and
records a prediction or causal conclusion that `G` helps derive,
an explicit derivation, its source warrant, and any addition `Delta_i` needed to
make the derivation. The addition records typed graph edits (`add_variable`,
`add_edge`, `change_mapping`, or `constrain_parameter`) with named terms and
an expression, rather than saying only that context matters. It must preserve the
core inference and previously supported predictions. New wording of the same
consequence does not count as new support.

A challenge names an independently warranted theory `H_star`, a prediction
obtained by transporting `G` into it, and the conflict with `H_star`. The
challenge tests concrete candidate repairs. For each repair it records the
formal graph edits, whether independently available evidence constrains them,
what they predict in `H_star`, and the IDs of earlier `H_i` whose predictions
are tested again. The
strongest negative result is bounded: no *tested, biologically plausible*
repair retains useful generalization. It is not a proof that no larger causal
graph could describe both systems.

The proposed domain of `G` decides which organisms and scales are relevant.
An ecological rule about herbivores may reasonably be tested in different
animals. Merely switching organism or context does not itself establish a
structural failure. Conversely, an unsupported leap outside the declared and
warranted domain does not earn reversal credit. A grain or scale change is
interesting when the inferential role of `G` fails despite a plausible domain
claim and ordinary repairs lose reusable predictive or explanatory value.

## Credit objective

Use source-backed, distinct `H_i` to build bounded, diminishing positive
credit for the *same* `G`. Each proposed support records whether it is an
application, a component of the original rule, a restatement, or speculation.
Only applications can earn reuse credit, and the reviewer must independently
confirm that relation in a structured verdict. A numerical warrant score alone
cannot turn a component of the source proof into reuse credit. Each also has a
warranted, provisional, or unsupported status. A merely plausible mechanism or
suggested experiment stays provisional and earns no support credit. A warranted
derivation earns credit only when the inference, addition, prediction, and
source support are explicit. A new `H_i` adds credit only for a genuinely new
consequence, not a second citation or paraphrase.

Give a substantially larger bounded bonus for a challenge only when all of
these are strong: the local warrant and domain of `G`; prior support breadth;
independent warrant for `H_star`; a specific transported prediction and
conflict; and a repair audit showing loss of independent constraint,
predictive discrimination, or prior reusable support. A simple variable or
edge addition that remains useful earns support credit as a formal extension,
not the challenge bonus. Failed repairs are evidence about the limits of `G`,
not license to declare a universal impossibility.

Credit is assigned when a coherent, reviewed argument improves the stored
support graph or challenge witness. The search controller receives marginal
credit from those changes and may choose which proof obligation to pursue.
The representation and score do not impose a fixed sequence of research
operations. Log each score component and its cited source and inference IDs so
the behavior can be audited after a long run.

The full argument stays in durable state. The quality-review handoff uses a
bounded JSON view: it retains every support and repair identity, includes
abbreviated cited source claims and a small catalog of other claims, and labels
truncated text and omitted claim counts. Previous answers are summarized by
their rule, frame conclusions, supports, repairs, and citations. The reviewer
is instructed to treat missing detail as uncertainty. The measurement response
schema uses compact string references; the runtime checks every returned
reference against the acquired claims after the model responds. This keeps a
large argument from silently overflowing the Episode message and schema limits,
but a highly expanded argument may still need paged source and argument review.

The implemented task coordinate is

```text
g = min(source_derivation_strength, generalization_warrant)
s_i = min(g, support_warrant_i), with zero unless both the argument and the
      independent reviewer classify the support as an application; also zero
      for speculative, provisional, unsupported, uncited, or repeated theories
      or consequences
B = 1 - exp(-sum_i s_i)
C = min(g, B, target_theory_strength, target_domain_warrant,
        repair_audit_strength, transport_failure_strength)
argument_progress = 0.05*g + 0.20*B + 0.75*C
```

`C` is zero when the target is outside the justified domain, no repair is
tested, a repair works or remains unresolved, the target lacks an independent
citation, or the fixed-category alternative reconciles the findings or remains
unresolved. Each
repair and support has a separate cited warrant rating. The `0.75` challenge
share is large relative to local and reuse credit, but a weak prerequisite
limits it. The formula is one coordinate of the existing coherent-material
attainment archive, so duplicate materials do not earn repeated credit.

## Adversarial checks

- A warranted inference with one supported theory earns useful partial credit.
- A second distinct, independently warranted theory using the same inference
  earns more credit; a paraphrase or extra citation does not.
- A formal context-variable or edge addition that predicts the challenge and
  preserves earlier uses earns no large failure bonus.
- An unsupported target theory or a target outside the warranted domain earns
  no large failure bonus.
- A scale transition with two useful theories, a specific failed transported
  prediction, and independently examined repairs that lose predictive value
  earns the large bonus.
- Changing organisms alone earns no failure bonus, but a rule whose warranted
  domain spans organisms can use evidence from different organisms.
