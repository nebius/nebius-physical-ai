"""Check signed-snapshot package locks against the dedicated Dockerfile."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "npa/docker/workbench/habitat-sim"
APT_VERIFIER = PACKAGE / "verify_apt_artifacts.sh"


def _lock(name: str) -> dict[str, object]:
    return yaml.safe_load((PACKAGE / name).read_text(encoding="utf-8"))


def _run_apt_verifier(command: str, *args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(APT_VERIFIER), command, *(str(arg) for arg in args)],
        check=False,
        capture_output=True,
        text=True,
    )


def _write_direct_lock(path: Path, rows: list[tuple[str, str, str]]) -> None:
    packages = "\n".join(
        f"  - {{binary: {name}, version: {version}, source: {name}, sha256: {digest}}}"
        for name, version, digest in rows
    )
    path.write_text(
        "schema_version: npa.habitat-sim.apt-lock.v1\n"
        f"packages:\n{packages}\n"
        "transitive_contract: signed snapshot\n",
        encoding="utf-8",
    )


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


def test_dockerfile_direct_installs_use_all_lock_records(tmp_path: Path) -> None:
    dockerfile = (PACKAGE / "Dockerfile").read_text(encoding="utf-8")
    build_stage, runtime_stage = dockerfile.split("FROM ${BASE_IMAGE} AS runtime", 1)
    for filename, stage, records_name in (
        ("apt-build.lock", build_stage, "npa-build-direct-packages"),
        ("apt-runtime.lock", runtime_stage, "npa-runtime-direct-packages"),
    ):
        records = tmp_path / records_name
        result = _run_apt_verifier("direct-records", PACKAGE / filename, records)
        assert result.returncode == 0, result.stdout + result.stderr
        expected = {
            f"{row['binary']}|{row['version']}|{row['sha256']}"
            for row in _lock(filename)["packages"]
        }
        assert set(records.read_text(encoding="utf-8").splitlines()) == expected
        invocation = stage.rindex("npa-habitat-verify-apt verify-direct")
        install = stage.index("apt-get install -y --no-install-recommends")
        assert invocation < install
        assert str(Path("/", "tmp", records_name)) in stage

    assert dockerfile.count("npa-habitat-verify-apt direct-records") == 2
    assert dockerfile.count("npa-habitat-verify-apt verify-direct") == 2


def _assert_verifier_mount_in_consuming_run(dockerfile: str) -> None:
    stages = dockerfile.split("FROM ${BASE_IMAGE} AS runtime", 1)
    sources = (
        "from=npa-source-provenance,source=/inputs/docker/workbench/habitat-sim/",
        "from=build,source=/opt/npa-source-provenance/inputs/docker/workbench/habitat-sim/",
    )
    for stage, source in zip(stages, sources, strict=True):
        consumers = [
            line
            for line in stage.replace("\\\n", "").splitlines()
            if line.startswith("RUN ")
            and "bash /usr/local/libexec/npa-habitat-verify-apt " in line
        ]
        assert len(consumers) == 1
        mount = (
            f"--mount={source}verify_apt_artifacts.sh,"
            "target=/usr/local/libexec/npa-habitat-verify-apt,ro"
        )
        assert consumers[0].startswith(f"RUN {mount} "), (
            "APT verifier must be mounted in its consuming RUN"
        )


def test_apt_verifier_is_mounted_in_each_consuming_run() -> None:
    _assert_verifier_mount_in_consuming_run((PACKAGE / "Dockerfile").read_text())


@pytest.mark.parametrize("stage_index", [0, 1])
def test_apt_verifier_mount_in_previous_run_is_rejected(stage_index: int) -> None:
    stages = (
        (PACKAGE / "Dockerfile").read_text().split("FROM ${BASE_IMAGE} AS runtime", 1)
    )
    stage = stages[stage_index]
    mounted = re.search(r"RUN (--mount=[^\n]+npa-habitat-verify-apt,ro) \\\n", stage)
    assert mounted is not None
    mount = mounted.group(1)
    # A successful earlier RUN cannot make a BuildKit bind mount persistent.
    stages[stage_index] = stage.replace(
        f"RUN {mount} \\\n", f"RUN {mount} true\nRUN \\\n", 1
    )
    with pytest.raises(AssertionError, match="mounted in its consuming RUN"):
        _assert_verifier_mount_in_consuming_run(
            "FROM ${BASE_IMAGE} AS runtime".join(stages)
        )


def test_direct_package_verifier_rejects_a_lock_hash_mutation(tmp_path: Path) -> None:
    archives = tmp_path / "archives"
    archives.mkdir()
    first = _build_fixture_deb(archives, "first-package", "1.0-1")
    second = _build_fixture_deb(archives, "second-package", "2.0-1")
    rows = [
        ("first-package", "1.0-1", hashlib.sha256(first.read_bytes()).hexdigest()),
        (
            "second-package",
            "2.0-1",
            hashlib.sha256(second.read_bytes()).hexdigest(),
        ),
    ]
    lock = tmp_path / "apt.lock"
    records = tmp_path / "records"
    _write_direct_lock(lock, rows)

    emitted = _run_apt_verifier("direct-records", lock, records)
    assert emitted.returncode == 0, emitted.stdout + emitted.stderr
    accepted = _run_apt_verifier("verify-direct", records, archives)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr

    hostile_digest = "0" * 64
    if rows[0][2] == hostile_digest:
        hostile_digest = "1" * 64
    _write_direct_lock(lock, [(rows[0][0], rows[0][1], hostile_digest), rows[1]])
    hostile_emitted = _run_apt_verifier("direct-records", lock, records)
    assert hostile_emitted.returncode == 0
    rejected = _run_apt_verifier("verify-direct", records, archives)
    assert rejected.returncode != 0


def test_signed_source_index_verifier_rejects_hash_and_size_mutations(
    tmp_path: Path,
) -> None:
    payload = b"signed compressed source index fixture"
    source_index = tmp_path / "Sources.xz"
    source_index.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    lock = tmp_path / "apt-runtime.lock"
    record = tmp_path / "source-index-record"

    def write_lock(*, declared_bytes: int, declared_digest: str) -> None:
        lock.write_text(
            "corresponding_sources:\n"
            "  - binary: rsync\n"
            "    signed_index:\n"
            "      suite: jammy-updates\n"
            "      path: dists/jammy-updates/main/source/Sources.xz\n"
            f"      bytes: {declared_bytes}\n"
            f"      sha256: {declared_digest}\n"
            "    artifacts:\n"
            "      - {filename: fixture}\n",
            encoding="utf-8",
        )

    def verify() -> subprocess.CompletedProcess[str]:
        emitted = _run_apt_verifier("source-index-record", lock, record)
        assert emitted.returncode == 0, emitted.stdout + emitted.stderr
        _suite, _path, bytes_value, digest_value = (
            record.read_text(encoding="utf-8").strip().split("|")
        )
        return _run_apt_verifier("verify-file", source_index, bytes_value, digest_value)

    write_lock(declared_bytes=len(payload), declared_digest=digest)
    accepted = verify()
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr

    write_lock(declared_bytes=len(payload) + 1, declared_digest=digest)
    wrong_size = verify()
    assert wrong_size.returncode != 0

    wrong_digest = "0" * 64 if digest != "0" * 64 else "1" * 64
    write_lock(declared_bytes=len(payload), declared_digest=wrong_digest)
    wrong_hash = verify()
    assert wrong_hash.returncode != 0


def test_source_index_verification_precedes_source_artifact_acceptance() -> None:
    dockerfile = (PACKAGE / "Dockerfile").read_text(encoding="utf-8")
    build_stage = dockerfile.split("FROM ${BASE_IMAGE} AS runtime", 1)[0]
    verify = build_stage.rindex("npa-habitat-verify-apt verify-file")
    source_download = build_stage.index(
        "apt-get source --download-only rsync=3.2.7-0ubuntu0.22.04.7"
    )
    assert verify < source_download


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
        "config_bytes": 4809,
        "config_sha256": "fe407f6205ff90c56d4f073ffaa2a8abde7835558919c19367a7cd4fd6312dad",
        "bundle_bytes": 182140,
        "bundle_sha256": "9481fcd95f41b221f02f14d896535fe500bec539bc563c4cdca1acee483a8bdd",
        "x509_parser": {
            "command": "/usr/bin/openssl",
            "packages": [
                {
                    "binary": "openssl",
                    "version": "3.0.2-0ubuntu1.29",
                    "bytes": 1184520,
                    "sha256": "bf5e804a0533d55ea9aa90320bcbce42981996dd41e8d4bcd5d15e41f91c2817",
                    "source": "openssl",
                },
                {
                    "binary": "libssl3",
                    "version": "3.0.2-0ubuntu1.29",
                    "bytes": 1906444,
                    "sha256": "c12bbf0074c44019bdf4b035b058c24eb904fc772c94707b1df801a4d130e146",
                    "source": "openssl",
                },
            ],
        },
    }
    assert f"ARG CA_DEB_SHA256={ca['sha256']}" in dockerfile
    assert f"ARG CA_CERT_COUNT={ca['certificate_count']}" in dockerfile
    assert f"ARG CA_CONFIG_BYTES={ca['config_bytes']}" in dockerfile
    assert f"ARG CA_CONFIG_SHA256={ca['config_sha256']}" in dockerfile
    assert f"ARG CA_BUNDLE_BYTES={ca['bundle_bytes']}" in dockerfile
    assert f"ARG CA_BUNDLE_SHA256={ca['bundle_sha256']}" in dockerfile
    for package in ca["x509_parser"]["packages"]:
        assert (
            f"ARG {package['binary'].upper()}_DEB_SHA256={package['sha256']}"
            in dockerfile
        )
        assert f"{package['binary']}_{package['version']}_amd64.deb" in dockerfile
        assert f"source=/{package['binary']}.deb" in dockerfile
    assert "${APT_SNAPSHOT}/pool/main/c/ca-certificates/" in dockerfile
    assert dockerfile.count('stat -c %s "${ca_config}"') == 1
    assert dockerfile.count("stat -c %s /etc/ca-certificates.conf") == 2
    assert dockerfile.count("/usr/bin/openssl x509") == 1
    assert dockerfile.count("openssl x509") == 1


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
    assert "URIs: https://snapshot.ubuntu.com/ubuntu/${APT_SNAPSHOT}/" in build
    assert "Suites: jammy jammy-updates jammy-security" in build
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
    assert (
        dockerfile.count("URIs: https://snapshot.ubuntu.com/ubuntu/${APT_SNAPSHOT}/")
        == 2
    )
    assert dockerfile.count("Suites: jammy jammy-updates jammy-security") == 2
    assert "Suites: ${APT_SNAPSHOT}" not in dockerfile


def test_build_python_family_is_fully_bound_to_signed_snapshot() -> None:
    build = _lock("apt-build.lock")
    expected = "3.10.6-1~22.04.1"
    required_candidates = {
        "candidate_python3": expected,
        "candidate_libpython3_dev": expected,
        "candidate_python3_dev": expected,
        "candidate_python3_venv": expected,
    }
    assert build["compatibility"].items() >= required_candidates.items()

    packages = {row["binary"]: row for row in build["packages"]}
    assert {
        name: (packages[name]["version"], packages[name]["sha256"])
        for name in ("python3", "libpython3-dev", "python3-dev", "python3-venv")
    } == {
        "python3": (
            expected,
            "43dbcc12d790ed9360be4a1dc690ac0a7464a5e7c3170785886df84811a3f5a5",
        ),
        "libpython3-dev": (
            expected,
            "7692175a21453d3cf15b66ed5a9a661b160113a99cb24eeb6a3b9acd43afa642",
        ),
        "python3-dev": (
            expected,
            "f7d6f5a0eb31d36f7fcba156b16da91035128c0edd1e643f07e7bbbecf5c6c7f",
        ),
        "python3-venv": (
            expected,
            "ee8d678c61b0721aff196ab1320e41cf4f5ebe3fcd22363b82e22c733b105866",
        ),
    }


def _build_fixture_deb(
    root: Path, name: str, version: str, *, depends: str = ""
) -> Path:
    package_root = root / f"{name}-{version}"
    control_dir = package_root / "DEBIAN"
    control_dir.mkdir(parents=True)
    control_dir.chmod(0o755)
    control = [
        f"Package: {name}",
        f"Version: {version}",
        "Architecture: amd64",
        "Maintainer: NPA test <noreply@example.invalid>",
        "Description: hermetic Python-family resolver fixture",
    ]
    if depends:
        control.append(f"Depends: {depends}")
    (control_dir / "control").write_text("\n".join(control) + "\n", encoding="utf-8")
    output = root / f"{name}_{version}_amd64.deb"
    result = subprocess.run(
        ["dpkg-deb", "--build", str(package_root), str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return output


def _simulate_python_family_resolution(
    tmp_path: Path, *, python_dev_version: str, family_version: str
) -> subprocess.CompletedProcess[str]:
    packages = [
        _build_fixture_deb(tmp_path, "python3", family_version),
        _build_fixture_deb(tmp_path, "libpython3-dev", family_version),
        _build_fixture_deb(
            tmp_path,
            "python3-dev",
            python_dev_version,
            depends=(
                f"python3 (= {python_dev_version}), "
                f"libpython3-dev (= {python_dev_version})"
            ),
        ),
        _build_fixture_deb(
            tmp_path,
            "python3-venv",
            family_version,
            depends=f"python3 (= {family_version})",
        ),
    ]
    state = tmp_path / "state"
    (state / "lists/partial").mkdir(parents=True)
    cache = tmp_path / "cache"
    (cache / "archives/partial").mkdir(parents=True)
    sources = tmp_path / "sources.list"
    sources.write_text("", encoding="utf-8")
    source_parts = tmp_path / "source-parts"
    source_parts.mkdir()
    return subprocess.run(
        [
            "apt-get",
            "-o",
            f"Dir::State={state}",
            "-o",
            f"Dir::Cache={cache}",
            "-o",
            "Dir::State::status=/dev/null",
            "-o",
            f"Dir::Etc::sourcelist={sources}",
            "-o",
            f"Dir::Etc::sourceparts={source_parts}",
            "-o",
            "Debug::NoLocking=1",
            "--simulate",
            "--no-install-recommends",
            "install",
            *(str(package) for package in packages),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_apt_resolver_accepts_exact_family_and_rejects_old_mixed_family(
    tmp_path: Path,
) -> None:
    expected = "3.10.6-1~22.04.1"
    exact = _simulate_python_family_resolution(
        tmp_path / "exact",
        python_dev_version=expected,
        family_version=expected,
    )
    assert exact.returncode == 0, exact.stdout + exact.stderr

    mixed = _simulate_python_family_resolution(
        tmp_path / "mixed",
        python_dev_version="3.10.6-1~22.04",
        family_version=expected,
    )
    output = mixed.stdout + mixed.stderr
    assert mixed.returncode != 0
    assert "python3-dev : Depends: python3 (= 3.10.6-1~22.04)" in output
    assert "Unable to correct problems" in output


def test_executed_candidate_gate_rejects_mixed_or_drifting_family() -> None:
    dockerfile = (PACKAGE / "Dockerfile").read_text(encoding="utf-8")
    start = dockerfile.index("npa_require_python_family()")
    end = dockerfile.index("    npa_require_python_family '3.10.6-1~22.04.1'", start)
    function = dockerfile[start:end].replace("\\\n", "\n")

    def run(candidates: dict[str, str]) -> subprocess.CompletedProcess[str]:
        cases = "\n".join(
            f"{name}) printf '%s\\n' '  Candidate: {version}' ;;"
            for name, version in candidates.items()
        )
        fake_apt_cache = (
            "apt-cache() {\n"
            '  test "$1" = policy\n'
            '  case "$2" in\n'
            f"{cases}\n"
            "  esac\n"
            "}\n"
        )
        return subprocess.run(
            [
                "bash",
                "-ceu",
                fake_apt_cache
                + function
                + "\nnpa_require_python_family '3.10.6-1~22.04.1' "
                "python3 libpython3-dev python3-dev python3-venv",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    exact = {
        name: "3.10.6-1~22.04.1"
        for name in ("python3", "libpython3-dev", "python3-dev", "python3-venv")
    }
    assert run(exact).returncode == 0
    assert run(exact | {"python3-dev": "3.10.6-1~22.04"}).returncode != 0
    assert run({name: "3.10.6-1~22.04.2" for name in exact}).returncode != 0
