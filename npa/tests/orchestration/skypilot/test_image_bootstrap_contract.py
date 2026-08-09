from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from npa.orchestration.skypilot.image_bootstrap_contract import (
    ATTESTATION_LABEL,
    CONTRACT_VERSION,
    ImageBootstrapContractError,
    ImageContractEvidence,
    cache_key,
    immutable_image_reference,
    load_cached_evidence,
    probe_image_capabilities,
    store_cached_evidence,
    verify_attestation,
)


DIGEST = "sha256:" + "a" * 64
IMAGE = "registry.example/npa-fiftyone:validation"


def test_root_and_compliant_non_root_probe_share_exact_contract() -> None:
    calls: list[list[str]] = []

    def runner(argv, _env):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    evidence = probe_image_capabilities(
        image=IMAGE, digest=DIGEST, context="ctx-exact", runner=runner
    )
    assert evidence.ok
    assert evidence.cleanup == "verified"
    assert "sudo -n true" in calls[0][-1]
    assert "command -v sshd || test -x /usr/sbin/sshd" in calls[0][-1]
    assert "command -v rsync" in calls[0][-1]
    assert "command -v service" in calls[0][-1]
    assert "test -w /tmp" in calls[0][-1]
    assert "--command" not in calls[0]  # entrypoint must forward pod args
    assert calls[-1][-5:] == [
        "delete",
        "pod",
        calls[0][4],
        "--ignore-not-found=true",
        "--wait=true",
    ]


@pytest.mark.parametrize(
    "failure",
    [
        "sudo: a password is required",
        "sudo: not found",
        "sshd: not found",
        "rsync: not found",
        "service: not found",
        "entrypoint rejected arguments",
    ],
)
def test_missing_capability_or_bad_entrypoint_fails_closed(failure: str) -> None:
    sequence = iter(
        [
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 1, "", failure),
            subprocess.CompletedProcess([], 0, "", ""),
        ]
    )
    evidence = probe_image_capabilities(
        image=IMAGE,
        digest=DIGEST,
        context="ctx-exact",
        runner=lambda _argv, _env: next(sequence),
    )
    assert evidence.state == "incompatible"
    assert evidence.cleanup == "verified"


def test_probe_transport_and_cleanup_failures_are_indeterminate() -> None:
    create_failure = probe_image_capabilities(
        image=IMAGE,
        digest=DIGEST,
        context="ctx",
        runner=lambda argv, _env: subprocess.CompletedProcess(
            argv, 1, "", "connection refused"
        ),
    )
    assert create_failure.state == "indeterminate"

    sequence = iter(
        [
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 1, "", "RBAC denied"),
        ]
    )
    cleanup_failure = probe_image_capabilities(
        image=IMAGE,
        digest=DIGEST,
        context="ctx",
        runner=lambda _argv, _env: next(sequence),
    )
    assert cleanup_failure.state == "indeterminate"
    assert cleanup_failure.cleanup == "failed"


def test_attestation_is_digest_and_version_bound() -> None:
    ok = verify_attestation(
        image=IMAGE,
        digest=DIGEST,
        labels={ATTESTATION_LABEL: CONTRACT_VERSION},
    )
    assert ok.ok
    assert ok.image.endswith("@" + DIGEST)

    mismatch = verify_attestation(
        image=IMAGE,
        digest=DIGEST,
        labels={ATTESTATION_LABEL: "old-contract"},
    )
    assert mismatch.state == "incompatible"
    assert "version mismatch" in mismatch.detail

    with pytest.raises(ImageBootstrapContractError):
        immutable_image_reference(IMAGE, "")


def test_cache_isolated_by_digest_and_contract_version(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    evidence = ImageContractEvidence(
        image=IMAGE + "@" + DIGEST,
        digest=DIGEST,
        contract_version=CONTRACT_VERSION,
        state="compatible",
        source="oci_attestation",
    )
    store_cached_evidence(path, evidence)
    assert load_cached_evidence(path, DIGEST) == evidence
    assert load_cached_evidence(path, "sha256:" + "b" * 64) is None
    payload = json.loads(path.read_text())
    assert list(payload) == [cache_key(DIGEST)]
    assert path.stat().st_mode & 0o077 == 0
