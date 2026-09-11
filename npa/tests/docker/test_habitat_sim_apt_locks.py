"""Check signed-snapshot package locks against the dedicated Dockerfile."""

from __future__ import annotations

from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "npa/docker/workbench/habitat-sim"


def _lock(name: str) -> dict[str, object]:
    return yaml.safe_load((PACKAGE / name).read_text(encoding="utf-8"))


def test_locks_share_exact_base_snapshot_signature_and_components() -> None:
    build = _lock("apt-build.lock")
    runtime = _lock("apt-runtime.lock")
    for lock in (build, runtime):
        assert lock["schema_version"] == "npa.habitat-sim.apt-lock.v1"
        assert lock["snapshot"] == "20260903T121500Z"
        assert lock["components"] == ["main", "universe"]
        assert lock["signed_by"] == "/usr/share/keyrings/ubuntu-archive-keyring.gpg"
        assert "@sha256:" in lock["ubuntu_base"]


def test_every_direct_package_has_exact_version_source_and_hash() -> None:
    for filename in ("apt-build.lock", "apt-runtime.lock"):
        packages = _lock(filename)["packages"]
        assert packages and len({row["binary"] for row in packages}) == len(packages)
        for row in packages:
            assert row["version"] and row["source"]
            assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"])


def test_dockerfile_direct_installs_equal_lock_versions() -> None:
    dockerfile = (PACKAGE / "Dockerfile").read_text(encoding="utf-8")
    build_stage, runtime_stage = dockerfile.split("FROM ${BASE_IMAGE} AS runtime", 1)
    for lock, stage in (
        (_lock("apt-build.lock"), build_stage),
        (_lock("apt-runtime.lock"), runtime_stage),
    ):
        for row in lock["packages"]:
            assert f"{row['binary']}={row['version']}" in stage


def test_ca_bootstrap_is_bound_to_same_snapshot_and_exact_hash() -> None:
    build = _lock("apt-build.lock")
    ca = build["ca_bootstrap"]
    dockerfile = (PACKAGE / "Dockerfile").read_text(encoding="utf-8")
    assert ca == {
        "package": "ca-certificates",
        "version": "20260601~22.04.1",
        "bytes": 140666,
        "sha256": "6e8cdcc8c86103acd4fc14649eac62ff2037108389074a7b167567af33c32245",
        "source": "ca-certificates",
    }
    assert f"ARG CA_DEB_SHA256={ca['sha256']}" in dockerfile
    assert "${APT_SNAPSHOT}/pool/main/c/ca-certificates/" in dockerfile


def test_no_floating_upgrade_or_unverified_apt_transport() -> None:
    dockerfile = (PACKAGE / "Dockerfile").read_text(encoding="utf-8")
    assert "apt-get upgrade" not in dockerfile
    assert "--allow-unauthenticated" not in dockerfile
    assert "Acquire::https::Verify" not in dockerfile
    assert "trusted=yes" not in dockerfile
