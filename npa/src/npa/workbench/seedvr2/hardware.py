"""Validate the declared SeedVR2 hardware against observed device properties."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

# Allow reporting/driver overhead, while rejecting reduced-memory partitions.
GPU_CONTRACTS = {"H100": ("9.0", 75_000), "B200": ("10.0", 170_000)}
ARCHES_ROOT = Path("/usr/share/doc/npa-seedvr2/extension-arches")


def validate_gpu(gpu: dict[str, Any], expected: str) -> None:
    """Require one full-memory, non-MIG device of the explicitly declared type.

    Args:
        gpu: Observed nvidia-smi device inventory.
        expected: Explicit H100 or B200 validation contract.
    Returns:
        None.
    Raises:
        ValueError: Any required hardware predicate fails.
    """
    expected = getattr(expected, "value", expected)
    contract = GPU_CONTRACTS.get(expected)
    if contract is None:
        raise ValueError("unsupported SeedVR2 GPU contract")
    capability, memory_floor = contract
    if (
        gpu.get("status") != "available"
        or gpu.get("count") != "1"
        or gpu.get("compute_capability") != capability
        or re.search(rf"\b{re.escape(expected)}\b", str(gpu.get("name", ""))) is None
        or gpu.get("mig_mode") != "Disabled"
        or not str(gpu.get("memory_mib", "")).isdigit()
        or int(gpu["memory_mib"]) < memory_floor
    ):
        raise ValueError(
            f"SeedVR2 execution requires one full-memory verified {expected} GPU"
        )


def require_b200_build_inventory() -> None:
    """Reject a Hopper-only build before loading the model on B200.

    This checks baked inventory declarations. Immutable-image qualification
    separately measures every actual extension and executes the kernel probe.

    Args:
        None.
    Returns:
        None.
    Raises:
        ValueError: Required baked SM90/SM100 inventory is missing or invalid.
    """
    for name in ("flash-attn", "apex"):
        try:
            report = json.loads((ARCHES_ROOT / f"{name}.json").read_bytes())
        except (OSError, ValueError) as exc:
            raise ValueError(
                "B200 requires baked native extension inventories"
            ) from exc
        if not isinstance(report, dict) or not report:
            raise ValueError("B200 native extension inventory is empty or invalid")
        for entry in report.values():
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("sass"), list)
                or any(not isinstance(arch, str) for arch in entry["sass"])
                or not {"sm_90", "sm_100"}.issubset(entry["sass"])
                or entry.get("missing")
                or entry.get("missing_exact")
            ):
                raise ValueError("B200 requires native SM90 and SM100 extensions")
