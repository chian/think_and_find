# Theory reconstruction

The scientific parent commissions reconstruction of a theory's connected
explanatory structure. The initial request supplies scientific scope and any
known entities, property definitions and quantity roles. Empty component maps
allow the theory's vocabulary to be discovered from sources. Focused questions
come from dependencies found during reconstruction.

`theory_language.py` defines one closed vocabulary for entities, quantities,
equations, causal relations, mechanisms, assumptions, derivations, approximations
and gaps. Quantities distinguish coordinates, states, parameters and literal
constants; field/function applications bind their declared arguments. Source
references resolve exact evidence record versions. Every component uses that
same vocabulary across Page, Search, Search Strategy and Theory Reconstruction.

A mechanism names at least one declared entity, quantity or statement as its
output: what the process produces or explains. Research gaps remain references,
not mechanism outputs. Qualitative outputs can be represented explicitly; that
structural condition does not establish source fidelity or verify equations.

The shared `theory_fragment` is a versioned scientific-sheet record. Additions
cannot replace an existing component. An explicit new component may supersede
an older one while retaining its definition and provenance. The fragment can be
supplied as a scientific premise to the parent or Reasoning.

Reconstruction repeatedly chooses integration, literature search, source reread,
Reasoning or a substantive-gap child. A substantive-gap child investigates an
actual recorded question and can commission further searches or rereads. A
separate theory-format repair child corrects rejected typed additions using
validator errors and the same evidence. Report and Lexical readers have an
extraction-repair child for malformed source extraction. All use the common
Episode loop and numerical yield estimator.

Extraction repair changes only host-identified rejected targets; accepted fields
remain fixed. A whole response can be replaced only after its original parsing
or top-level shape failed. Failed corrective calls retain the current response
and outstanding targets, while valid partial corrections become fixed.

Search Strategy and Search translate evidence into the fragment and return an
exact key/version reference, added component IDs and unresolved requirements.
The current typed fragment is supplied once in the parent's decision context;
child returns do not copy its component definitions. Full versions remain in
the scientific sheet and can be retrieved through ordinary `recall` decisions.
Raw source content is supplied for commissioned or selected records and evidence
cited by rejected additions during format repair. Other acquired evidence is
listed by exact key/version, source and locator, so the episode can select what
to recall. It is not automatically copied into every subsequent prompt.
Normalization reports give form identities, matching statement groups and
unresolved operations; they do not serialize recursively expanded dependencies.
These projections use no character, item or token quota. Final unreviewed
outcomes remain part of the ordinary episode return. Request-relative credit is
new source-grounded component identities within the commissioned reconstruction,
requested component kinds and focus. Page and processing readers retain local
extraction credit. Child scores are not summed; each parent recomputes its own
request projection. Global G credit does not drive literature continuation.

Exact rational normalization deduplicates supported algebraic forms on the same
declared quantities and retained domains. Re-citing a relationship does not
create another law. Unknown symbols, invalid signatures, silent overwrites and
identically zero denominators produce validator errors. Unsupported analytic
equivalence and prose equivalence remain unresolved. Structural acceptance does
not verify a source interpretation or establish a causal claim.

`scripts/run_theory_reconstruction.py` is a focused live experiment of this
production subtree. It uses configured providers and the ordinary controller;
only its question and output directory are supplied. Run it through the
per-window machine-resource launcher. Trace, parent return, theory graph,
controller state and complete audit remain under the launch's run directory.
