from __future__ import annotations

import json
from pathlib import Path
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/gymnasium-robotics"


def test_neutral_dockerfile_is_pinned_non_root_and_gated_before_network() -> None:
    text = (IMAGE / "Dockerfile").read_text(encoding="utf-8")
    assert (
        "ubuntu:noble-20260905@sha256:a61567bd31828687156d735ea8eb01ba4e37636e225dd6a48ba94136a70d9d61"
        in text
    )
    assert text.index("./build.sh verify-bootstrap-locks") < text.index(
        "./build.sh install-bootstrap"
    )
    assert "runtime-bootstrap.py" in text
    assert "USER ubuntu" in text
    assert 'ENTRYPOINT ["/usr/local/bin/npa-gymnasium-entrypoint"]' in text
    assert "COPY --from=" not in text
    assert "/opt/venv" not in text
    assert "pip install" not in text
    assert "nvidia/cuda" not in text.lower()
    assert "nvcr.io" not in text.lower()


def test_candidate_copies_only_neutral_code_metadata_and_notices() -> None:
    text = (IMAGE / "Dockerfile").read_text(encoding="utf-8")
    copied = [line for line in text.splitlines() if line.startswith("COPY ")]
    assert copied
    for forbidden in (
        "gymnasium_robotics/",
        ".whl",
        ".tar.gz",
        "runtime-cache",
        "requirements.in",
    ):
        assert all(forbidden not in line for line in copied)
    for required in (
        "runtime-bootstrap.py",
        "capability_smoke.py",
        "asset-lock.json",
        "source-lock.json",
        "THIRD_PARTY_NOTICES.md",
    ):
        assert any(required in line for line in copied)


def test_repository_locks_are_complete_exact_and_machine_readable() -> None:
    for name in (
        "source-lock.json",
        "apt-runtime.lock.json",
        "corresponding-source.lock.json",
    ):
        payload = json.loads((IMAGE / name).read_text(encoding="utf-8"))
        assert payload["status"] == "complete"
        assert payload["reason"]
    requirements = (IMAGE / "requirements.lock").read_text()
    assert "# status: complete" in requirements
    assert requirements.count("--hash=sha256:") == 19


def test_image_build_accepts_only_the_exact_complete_pre_network_locks() -> None:
    completed = subprocess.run(
        [str(IMAGE / "build.sh"), "verify-bootstrap-locks"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    script = (IMAGE / "build.sh").read_text(encoding="utf-8")
    assert script.index("missing ephemeral TLS trust input") < script.index(
        "apt-get update"
    )
    assert "EXPECTED_FINAL_PACKAGE_MANIFEST_SHA256" in script
    assert "https://snapshot.ubuntu.com/ubuntu/20260905T000000Z" in script


def test_packaging_contract_does_not_claim_a_built_or_supported_image() -> None:
    contract = yaml.safe_load(
        (ROOT / "npa/docker/workbench/packaging-contract.yaml").read_text()
    )
    entry = contract["images"]["gymnasium-robotics"]
    assert entry["redistribution"] == "public"
    assert entry["phase"] == "pre-registration-quarantine"
    assert "No image" in entry["notes"]
    assert "neutral" in entry["notes"].lower()
    assert "accepted_manifest" not in entry


def test_notices_keep_runtime_provenance_without_claiming_a_grant() -> None:
    text = (IMAGE / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for token in (
        "MIT",
        "GPL-2.0-only",
        "Apache-2.0",
        "LICENSE.md",
        "59d6bdf35bd9cf53185a20eb63413fdfe57fe77c",
        "not a license grant",
        "not present in the neutral candidate layers",
    ):
        assert token in text


def test_six_delivery_boundaries_are_classified_independently() -> None:
    lock = json.loads((IMAGE / "source-lock.json").read_text(encoding="utf-8"))
    assert lock["delivery"] == {
        "source": "operator-owned-runtime-cache",
        "baked_runtime": "neutral-bootstrap-only",
        "weights": "none",
        "data_assets": "runtime-cache-only",
        "runtime_cache": "operator-owned-and-external",
        "outputs": "operator-owned-run-artifacts",
    }
    assert lock["decision_sha256"] == (
        "758a29a6fae55075dc4ba879907e81f949b7a4e23fa726b790fd4361241697a3"
    )
