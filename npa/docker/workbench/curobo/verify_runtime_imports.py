"""Verify the built cuRobo image crosses the real benchmark import boundary."""

from __future__ import annotations

import json
from pathlib import Path


_EXPECTED_DATASETS = {
    "motion_benchmaker": {"groups": 8, "rows": 800},
    "mpinets": {"groups": 12, "rows": 1800},
}
_RECEIPT_PATH = Path("/usr/share/doc/npa-curobo/runtime-import.json")


def _population(loader) -> dict[str, int]:
    groups = loader()
    return {
        "groups": len(groups),
        "rows": sum(len(problems) for problems in groups.values()),
    }


def _verify_imports() -> dict:
    from npa.workbench.curobo.runner import _benchmark_module

    _benchmark_module()
    from robometrics.datasets import motion_benchmaker_raw, mpinets_raw

    datasets = {
        "motion_benchmaker": _population(motion_benchmaker_raw),
        "mpinets": _population(mpinets_raw),
    }
    if datasets != _EXPECTED_DATASETS:
        raise RuntimeError("pinned cuRobo benchmark population changed")
    return {
        "schema_version": "npa.curobo.runtime-import.v1",
        "pinocchio_upstream_import": "passed",
        "datasets": datasets,
    }


def main() -> None:
    """Import the pinned benchmark and record its exact dataset population.

    Args:
        None.

    Returns:
        None.

    Raises:
        ImportError: Pinocchio or another benchmark runtime dependency is absent.
        RuntimeError: The pinned benchmark datasets have unexpected populations.
    """

    payload = json.dumps(
        _verify_imports(), sort_keys=True, separators=(",", ":")
    ).encode()
    _RECEIPT_PATH.write_bytes(payload + b"\n")


if __name__ == "__main__":
    main()
