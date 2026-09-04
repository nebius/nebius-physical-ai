from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa/docker/workbench/paidf-anomalygen-sky/Dockerfile"
ENTRYPOINT = ROOT / "npa/docker/workbench/paidf-anomalygen-sky/entrypoint.sh"


def test_paidf_anomalygen_sky_parent_and_runtime_are_fixed() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "nvcr.io" not in text
    assert "dbaf7d7d9003f048230f9026da5969e9e5931785" in text
    assert "nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04@sha256:" in text
    assert "nvidia/cuda:13.2.1-base-ubuntu24.04@sha256:" in text
    assert "USER ubuntu" in text
    assert "ENTRYPOINT [\"/usr/local/bin/npa-sky-entrypoint\"]" in text
    assert "org.nebius.npa.skypilot-bootstrap-contract" not in text
    assert "git -C /src fetch -q --depth 1 origin" in text
    assert "bash /tmp/build_wheels.sh --in-container" in text


def test_paidf_anomalygen_sky_bootstrap_source_is_complete() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    for dependency in ("openssh-server", "rsync", "netcat-openbsd", "sudo"):
        assert dependency in dockerfile
    assert "rm -f /etc/ssh/ssh_host_*" in dockerfile
    assert "ubuntu ALL=(ALL) NOPASSWD:ALL" in dockerfile
    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")
    assert 'exec "$@"' in entrypoint
    assert "exec /bin/bash" in entrypoint
