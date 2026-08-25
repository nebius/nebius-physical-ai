"""Keep public Sim2Real development builds independent of local image tags."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORKBENCH = ROOT / "npa" / "docker" / "workbench"


def _default_base(relative: str) -> str:
    text = (WORKBENCH / relative).read_text(encoding="utf-8")
    match = re.search(r"^ARG BASE_IMAGE=(\S+)$", text, re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_sim2real_gpu_overlays_use_immutable_public_bases() -> None:
    for dockerfile in (
        "sim2real-envgen/Dockerfile",
        "cosmos3-reason/Dockerfile",
    ):
        base = _default_base(dockerfile)
        assert base.startswith(
            "ghcr.io/nebius/nebius-physical-ai/"
        ), dockerfile
        assert re.search(r"@sha256:[0-9a-f]{64}$", base), dockerfile
