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
        "certificate_count": 121,
        "config_bytes": 4930,
        "config_sha256": "bd46a6383240ac4c0904cd896d0be22c5862c130435c795c893ff56bd141c38d",
        "bundle_bytes": 182140,
        "bundle_sha256": "9481fcd95f41b221f02f14d896535fe500bec539bc563c4cdca1acee483a8bdd",
    }
    assert f"ARG CA_DEB_SHA256={ca['sha256']}" in dockerfile
    assert f"ARG CA_CERT_COUNT={ca['certificate_count']}" in dockerfile
    assert f"ARG CA_CONFIG_BYTES={ca['config_bytes']}" in dockerfile
    assert f"ARG CA_CONFIG_SHA256={ca['config_sha256']}" in dockerfile
    assert f"ARG CA_BUNDLE_BYTES={ca['bundle_bytes']}" in dockerfile
    assert f"ARG CA_BUNDLE_SHA256={ca['bundle_sha256']}" in dockerfile
    assert "${APT_SNAPSHOT}/pool/main/c/ca-certificates/" in dockerfile
    assert dockerfile.count('stat -c %s "${ca_config}"') == 1
    assert dockerfile.count("stat -c %s /etc/ca-certificates.conf") == 2


def test_rsync_corresponding_source_is_exact_and_accompanies_the_binary() -> None:
    runtime = _lock("apt-runtime.lock")
    [source] = runtime["corresponding_sources"]
    assert source == {
        "binary": "rsync",
        "source": "rsync",
        "version": "3.2.7-0ubuntu0.22.04.7",
        "delivery": "accompanying-exact-source",
        "directory": "pool/main/r/rsync",
        "signed_index": {
            "suite": "jammy-updates",
            "path": "dists/jammy-updates/main/source/Sources.xz",
            "bytes": 520028,
            "sha256": "e2d0a07fd09ee8134bc10e23b1183270473cb6b45703803878d0273564cb54a0",
        },
        "artifacts": [
            {
                "filename": "rsync_3.2.7.orig.tar.gz",
                "bytes": 1149787,
                "sha256": "4e7d9d3f6ed10878c58c5fb724a67dacf4b6aac7340b13e488fb2dc41346f2bb",
                "locator": "https://snapshot.ubuntu.com/ubuntu/20260903T121500Z/pool/main/r/rsync/rsync_3.2.7.orig.tar.gz",
            },
            {
                "filename": "rsync_3.2.7.orig.tar.gz.asc",
                "bytes": 195,
                "sha256": "8e054b8e852f371fbcb757de51f1a07de5621ae959ea766d3c3e5439d7b5f4ae",
                "locator": "https://snapshot.ubuntu.com/ubuntu/20260903T121500Z/pool/main/r/rsync/rsync_3.2.7.orig.tar.gz.asc",
            },
            {
                "filename": "rsync_3.2.7-0ubuntu0.22.04.7.debian.tar.xz",
                "bytes": 117112,
                "sha256": "05db0546477de617be806c2abd8d16f8aed4aa2cfe8d2273b3a030750c6cb811",
                "locator": "https://snapshot.ubuntu.com/ubuntu/20260903T121500Z/pool/main/r/rsync/rsync_3.2.7-0ubuntu0.22.04.7.debian.tar.xz",
            },
            {
                "filename": "rsync_3.2.7-0ubuntu0.22.04.7.dsc",
                "bytes": 2402,
                "sha256": "5e6e79695b709ce1606ddbb63f5c28c7cdc304f857f42e317bd53e753b8ca15d",
                "locator": "https://snapshot.ubuntu.com/ubuntu/20260903T121500Z/pool/main/r/rsync/rsync_3.2.7-0ubuntu0.22.04.7.dsc",
            },
        ],
    }
    dockerfile = (PACKAGE / "Dockerfile").read_text(encoding="utf-8")
    build, runtime_stage = dockerfile.split("FROM ${BASE_IMAGE} AS runtime", 1)
    assert "Types: deb deb-src" in build
    assert "Types: deb deb-src" not in runtime_stage
    assert "apt-get source --download-only rsync=3.2.7-0ubuntu0.22.04.7" in build
    for artifact in source["artifacts"]:
        assert artifact["filename"] in build
        assert str(artifact["bytes"]) in build
        assert artifact["sha256"] in build
        assert artifact["locator"].endswith("/" + artifact["filename"])
    assert "/opt/notices/ubuntu-sources/rsync" in build


def test_no_floating_upgrade_or_unverified_apt_transport() -> None:
    dockerfile = (PACKAGE / "Dockerfile").read_text(encoding="utf-8")
    assert "apt-get upgrade" not in dockerfile
    assert "--allow-unauthenticated" not in dockerfile
    assert "Acquire::https::Verify" not in dockerfile
    assert "trusted=yes" not in dockerfile
    assert "update-ca-certificates" not in dockerfile
