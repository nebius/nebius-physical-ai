from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/gymnasium-robotics"


def test_phase_a_dockerfile_is_exactly_pinned_and_fail_closed() -> None:
    text = (IMAGE / "Dockerfile").read_text(encoding="utf-8")
    assert (
        "ubuntu:noble-20260905@sha256:a61567bd31828687156d735ea8eb01ba4e37636e225dd6a48ba94136a70d9d61"
        in text
    )
    assert text.index("./build.sh verify-locks") < text.index("./build.sh install")
    assert "USER ubuntu" in text
    assert "COPY --chmod=0555" in text
    assert "COPY --chmod=0444" in text
    assert 'ENTRYPOINT ["/usr/local/bin/npa-gymnasium-entrypoint"]' in text
    assert "nvidia/cuda" not in text.lower()
    assert "nvcr.io" not in text.lower()


def test_every_incomplete_lock_is_explicit_and_machine_readable() -> None:
    for name in (
        "source-lock.json",
        "apt-runtime.lock.json",
        "corresponding-source.lock.json",
    ):
        payload = json.loads((IMAGE / name).read_text(encoding="utf-8"))
        assert payload["status"] == "phase-a-incomplete"
        assert payload["reason"]
    assert "# status: phase-a-incomplete" in (IMAGE / "requirements.lock").read_text()


def test_packaging_contract_does_not_claim_a_built_image() -> None:
    contract = yaml.safe_load(
        (ROOT / "npa/docker/workbench/packaging-contract.yaml").read_text()
    )
    entry = contract["images"]["gymnasium-robotics"]
    assert entry["redistribution"] == "public"
    assert entry["phase"] == "pre-registration-quarantine"
    assert "No image has been built" in entry["notes"]
    assert "accepted_manifest" not in entry


def test_notices_preserve_all_three_license_boundaries() -> None:
    text = (IMAGE / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for token in (
        "MIT",
        "GPL-2.0-only",
        "Apache-2.0",
        "LICENSE.md",
        "59d6bdf35bd9cf53185a20eb63413fdfe57fe77c",
    ):
        assert token in text
