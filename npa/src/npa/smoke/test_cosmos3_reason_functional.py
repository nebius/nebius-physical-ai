"""Cosmos3-Reason golden eval — a REAL capability test.

Runs an actual Cosmos-Reason VLM inference (the same ``run_cosmos_reason_vlm``
path the sim2real VLM-eval stage uses) over a couple of synthetic frames and a
short task, and asserts a structured judgment is returned. This exercises the
container's real job (vision-language judgment of a rollout), not a CUDA/import
probe.

GPU-gated and heavy: the gated Cosmos-Reason weights auto-download on first use
(HF_TOKEN + NVIDIA license). Import-safe on the default interpreter — torch /
transformers / PIL are imported lazily inside ``main``.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


class _WiringResult:
    def __init__(self, name: str, ok: bool, detail: str = "") -> None:
        self.name = name
        self.ok = ok
        self.detail = detail


def check_reason_cache_wiring() -> _WiringResult:
    """Infra-free precondition check that the Cosmos-Reason path is importable.

    The GPU capability run lives in ``main`` (real ``run_cosmos_reason_vlm``
    inference); this helper is the fast, import-only sanity check used by the
    standard unit suite.
    """

    try:
        from npa.cli.workbench import cosmos3 as cosmos3_cli
        from npa.workbench.cosmos.reason import run_cosmos_reason_vlm  # noqa: F401
    except Exception as exc:  # pragma: no cover - import failure path
        return _WiringResult("reason cli wiring", False, str(exc))
    if not hasattr(cosmos3_cli, "app"):
        return _WiringResult("reason cli wiring", False, "missing cosmos3 Typer app")
    return _WiringResult("reason cli wiring", True, "npa.workbench.cosmos.reason.run_cosmos_reason_vlm")


def _synthetic_frames(n: int = 2) -> list[Path]:
    from PIL import Image

    tmp = Path(tempfile.mkdtemp(prefix="cosmos3_reason_ge_"))
    paths: list[Path] = []
    for i in range(n):
        # Distinct solid-color frames so the model has real image input to attend to.
        color = (40 + 60 * i, 90, 160 - 40 * i)
        img = Image.new("RGB", (256, 256), color=color)
        p = tmp / f"frame-{i:03d}.png"
        img.save(p)
        paths.append(p)
    return paths


def main() -> int:
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - image without torch
        print(f"[FAIL] torch import: {exc}")
        return 1

    from npa.workbench.cosmos.reason import (
        CosmosReasonError,
        DEFAULT_REASON2_MODEL,
        run_cosmos_reason_vlm,
    )

    model_id = os.environ.get("COSMOS_REASON_MODEL", DEFAULT_REASON2_MODEL)
    frames = _synthetic_frames()
    actions = [{"step": i, "action": [0.0, 0.0, 0.0]} for i in range(len(frames))]

    try:
        payload = run_cosmos_reason_vlm(
            model_id=model_id,
            image_paths=frames,
            actions=actions,
            task_description="Assess whether the robot arm places the cube on the table.",
            rollout_id="cosmos3-reason-golden-eval",
            threshold=0.5,
        )
    except CosmosReasonError as exc:
        print(f"[FAIL] cosmos-reason inference: {exc}")
        return 1

    if not isinstance(payload, dict) or payload.get("component_source") != "cosmos_reason_vlm":
        print(f"[FAIL] unexpected reason payload: {payload!r}")
        return 1
    # A real judgment carries a score and a success verdict.
    if "score" not in payload or "success" not in payload:
        print(f"[FAIL] reason payload missing judgment fields: {sorted(payload)}")
        return 1
    print(
        f"[PASS] cosmos-reason judged rollout: model={payload.get('model')} "
        f"score={payload.get('score')} success={payload.get('success')} "
        f"frames={payload.get('frame_count')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
