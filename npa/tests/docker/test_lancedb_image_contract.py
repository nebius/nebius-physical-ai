"""Static guards for the LanceDB service image's real runtime path."""

from __future__ import annotations

from pathlib import Path

from npa.smoke.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa/docker/workbench/lancedb/Dockerfile"


def test_lancedb_nonroot_entrypoint_and_healthcheck_are_runnable() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY --chmod=0755 docker/workbench/lancedb/entrypoint.sh" in text
    assert "LANCEDB_SMOKE_ENTRYPOINT=/entrypoint.sh" in text
    assert "/readyz" in text
    assert text.index("USER ubuntu") < text.index("HEALTHCHECK")


def test_lancedb_dependency_overlay_is_consistent() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    install = text.index("pip install --no-cache-dir -r /app/requirements.txt")
    remove_unused_spin = text.index("python -m pip uninstall -y spin")
    verify = text.index("python -m pip check")

    assert install < remove_unused_spin < verify


def test_lancedb_golden_eval_describes_the_built_runtime() -> None:
    safety = load_manifest()["lancedb"].safety

    assert safety["runs_as"] == "ubuntu"
    assert "npa-detection-training@sha256:" in safety["base_image"]
    assert "/readyz" in safety["network"]
