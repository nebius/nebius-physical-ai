"""Inspect native Isaac training logs before promoting a checkpoint."""

from pathlib import Path
import re

_PHYSICS_FAILURE = re.compile(
    r"PhysX error|simulation will miss interactions", re.IGNORECASE
)


def inspect_training_log(path: Path) -> dict[str, bool | int]:
    """Report native physics failures even when the trainer exits successfully.

    Args:
        path: Complete combined stdout and stderr log from the trainer.

    Returns:
        Physics validity and the number of lines reporting native failures.

    Raises:
        OSError: The required training log cannot be read.
    """
    errors = 0
    with path.open(encoding="utf-8", errors="replace") as log:
        for line in log:
            if _PHYSICS_FAILURE.search(line):
                errors += 1
    return {"physics_valid": errors == 0, "physics_error_count": errors}
