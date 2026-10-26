# Checkpoint ownership and continuation

## Preserved mechanism

`checkpoint.py` and `method_loop/reconstruction.py` retain the original local
checkpoint implementation. The design adapts ordered reconstruction mechanisms
from OpenChia into repository-owned code; it has no sibling repository runtime
dependency. Relevant original attribution and license notices remain in source.

The CLI records external requests and replies and Episode lifecycle frames.
Continuation reconstructs state from a matching saved history before permitting
new work. Ordinary resume requires compatible code, Python/runtime identity,
inputs and non-secret provider settings. Credentials are supplied afresh.

The copied CLI also retains an explicit compatible-fork interface for admitting
specified changed implementation files at a committed unit boundary. A fork is
a separate experiment, not silent migration of an existing result.

No checkpoints or historical run outputs were copied into this workspace. These
mechanisms are retained source capabilities, not operations performed during
the migration.

## State boundary

A generic reasoning workflow will need continuation to reconstruct evidence,
current arguments and their premises, revisions, unresolved questions, selected
searches, reading cursors, contracts and numerical controller history. The current
implementation reconstructs the existing domain records; generic reasoning state
must be added through the same boundary when implemented.

Changing a task definition, inference contract, model route or code must not
silently reinterpret a prior run as if it had executed the new design. Explicit
compatibility and provenance are required for any future continuation mechanism.

## Storage and external effects

The retained journal uses a manifest, ordered committed records, a pending request
marker and physical-attempt records. Its implementation includes hash checks,
atomic writes and an exclusive owner lock. These are implementation properties
to preserve, not a claim that this migration retested crash recovery.

Completed observational provider replies can be reconstructed locally. An
unanswered request is different from a committed result. The current mechanism
is serial and does not provide a generic reconciliation protocol for external
state-changing operations or concurrent Episode execution.

Provider credentials do not belong in recorded requests, outputs or checkpoints.
Source and model content can remain in research artifacts, so those artifacts
are separate from the source workspace and from the eventual Git repository.

## Existing CLI compatibility

The existing `--checkpoint-path`, `--resume` and explicit fork flags remain on the
preserved launcher. Its registered inputs are still domain-specific. A generic
CLI and generic task registration are pending work; this documentation does not
claim those interfaces already exist.
