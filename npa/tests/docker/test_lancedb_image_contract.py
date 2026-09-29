"""Static guards for the LanceDB service image's real runtime path."""

from __future__ import annotations

from pathlib import Path

from npa.smoke.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa/docker/workbench/lancedb/Dockerfile"


def test_lancedb_nonroot_entrypoint_and_healthcheck_are_runnable() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY --chmod=0755 docker/workbench/lancedb/entrypoint.sh" in text
    # Local-container deploy intentionally maps the runtime to the host uid/gid.
    # The flattened application must therefore be readable even when that uid is
    # not the image's named ubuntu user (uid 1000).
    runtime_sources = (
        "bdd100k_import.py",
        "bdd100k_udfs.py",
        "backfill.py",
        "views.py",
        "server.py",
        "test_lancedb_functional.py",
    )
    runtime_copy_lines = [
        line
        for line in text.splitlines()
        if line.startswith("COPY --chmod=0644 src/npa/")
    ]
    for source in runtime_sources:
        assert any(source in line for line in runtime_copy_lines)
    assert "LANCEDB_SMOKE_ENTRYPOINT=/entrypoint.sh" in text
    assert "LANCEDB_SMOKE_UDF_MODULE=npa_lancedb_bdd100k_udfs" in text
    assert "from npa_lancedb_bdd100k_udfs import _dhash_bytes" in text
    assert "_dhash_bytes(data.getvalue()) == 0" in text
    assert "/readyz" in text
    assert text.index("USER ubuntu") < text.index("HEALTHCHECK")


def test_lancedb_dependency_overlay_is_consistent() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    install = text.index("pip install --no-cache-dir -r /app/requirements.txt")
    remove_unused_spin = text.index("python -m pip uninstall -y spin")
    verify = text.index("python -m pip check")

    assert install < remove_unused_spin < verify

    requirements = (ROOT / "npa/docker/workbench/lancedb/requirements.txt").read_text(
        encoding="utf-8"
    )
    assert "npa-detection-training parent" in requirements
    assert "npa-workbench-cuda-base" not in requirements
    assert "pillow==12.3.0" in requirements


def test_lancedb_preserves_the_parent_header_security_refresh() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG UBUNTU_SNAPSHOT=20260910T000000Z" in text
    assert "ARG LINUX_LIBC_DEV_VERSION=6.8.0-139.139" in text
    assert '"${UBUNTU_SNAPSHOT}" "${LINUX_LIBC_DEV_VERSION}"' in text
    assert "6.8.0-138.138" not in text


def test_lancedb_golden_eval_describes_the_built_runtime() -> None:
    safety = load_manifest()["lancedb"].safety

    assert safety["runs_as"] == "ubuntu"
    assert "npa-detection-training@sha256:" in safety["base_image"]
    assert "/readyz" in safety["network"]
