"""Collator scaffolding for modality-specific batching.

The v0.1 toy workflows operate on individual processed samples, so full
batching/collation is intentionally deferred. This module exists as the stable
extension point listed in PLAN.md.
"""


class IdentityCollator:
    """Return samples unchanged for toy and custom single-sample workflows."""

    def __call__(self, samples):
        return samples
