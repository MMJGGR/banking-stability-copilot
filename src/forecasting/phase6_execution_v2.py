"""Phase 6 execution with registered rare-event-safe validation windows."""
from __future__ import annotations

from . import phase6_execution as base
from .phase6_replacement import estimator_factory, validation_config
from .phase6_validation import validate_candidates_registered


def _validate(replacement):
    return validate_candidates_registered(
        replacement,
        estimator_factory=estimator_factory,
        config=validation_config(),
    )


# phase6_execution.run resolves this module global at runtime. Replacing only
# the development validator leaves source parsing, score construction, output
# contracts and production firewalls unchanged.
base.validate_candidates = _validate


if __name__ == "__main__":
    base.main()
