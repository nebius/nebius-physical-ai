from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading

import pytest

from npa.orchestration.skypilot.image_bootstrap_contract import (
    ATTESTATION_LABEL,
    CONTRACT_VERSION,
    ImageBootstrapContractError,
    ImageContractEvidence,
    cache_key,
    immutable_image_reference,
    is_first_party_image,
    load_cached_evidence,
    parse_oci_reference,
    probe_image_capabilities,
    store_cached_evidence,
    verify_attestation,
)


DIGEST = "sha256:" + "a" * 64
IMAGE = "registry.example/npa-fiftyone:validation"


def _pod_payload(name: str, probe_id: str, *, uid: str = "uid-owned", image: str = "") -> str:
    return json.dumps(
        {
            "metadata": {
                "name": name,
                "uid": uid,
                "labels": {
                    "npa.nebius.com/owned": "true",
                    "npa.nebius.com/purpose": "sky-image-probe",
                    "npa.nebius.com/probe-id": probe_id,
                },
            },
            "spec": {"containers": [{"image": image or f"registry.example/npa-fiftyone@{DIGEST}"}]},
        }
    )


def _successful_runner(calls: list[list[str]], *, wait_error: str = "", delete_error: str = ""):
    def runner(argv, _env):
        calls.append(argv)
        action = argv[3]
        if action == "run":
            return subprocess.CompletedProcess(argv, 0, "", "")
        if action == "get":
            name = argv[5]
            probe_id = name.rsplit("-", 1)[-1]
            return subprocess.CompletedProcess(argv, 0, _pod_payload(name, probe_id), "")
        if action == "wait":
            return subprocess.CompletedProcess(argv, int(bool(wait_error)), "", wait_error)
        if action == "delete":
            return subprocess.CompletedProcess(argv, int(bool(delete_error)), "", delete_error)
        raise AssertionError(argv)

    return runner


def test_root_and_compliant_non_root_probe_share_exact_contract() -> None:
    calls: list[list[str]] = []

    evidence = probe_image_capabilities(
        image=IMAGE,
        digest=DIGEST,
        context="ctx-exact",
        runner=_successful_runner(calls),
        nonce_factory=lambda: "a" * 16,
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
    calls: list[list[str]] = []
    evidence = probe_image_capabilities(
        image=IMAGE,
        digest=DIGEST,
        context="ctx-exact",
        runner=_successful_runner(calls, wait_error=failure),
        nonce_factory=lambda: "b" * 16,
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
        nonce_factory=lambda: "c" * 16,
    )
    assert create_failure.state == "indeterminate"

    calls: list[list[str]] = []
    cleanup_failure = probe_image_capabilities(
        image=IMAGE,
        digest=DIGEST,
        context="ctx",
        runner=_successful_runner(calls, delete_error="RBAC denied"),
        nonce_factory=lambda: "d" * 16,
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


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("registry.example:5000/team/npa-tool:tag", "registry.example:5000/team/npa-tool"),
        ("[2001:db8::1]:5000/team/npa-tool:tag", "[2001:db8::1]:5000/team/npa-tool"),
        (f"registry.example:5000/team/npa-tool:tag@{DIGEST}", "registry.example:5000/team/npa-tool"),
        ("registry.example:5000/team/npa-tool", "registry.example:5000/team/npa-tool"),
    ],
)
def test_shared_oci_parser_handles_registry_ports_and_tags(reference: str, expected: str) -> None:
    parsed = parse_oci_reference(reference)
    assert parsed.name == expected
    assert immutable_image_reference(reference, DIGEST) == f"{expected}@{DIGEST}"
    assert is_first_party_image(reference)


@pytest.mark.parametrize(
    "reference",
    [
        "npa-tool:latest",
        "https://registry.example/npa-tool:tag",
        "registry.example:bad/npa-tool:tag",
        "registry.example/team/npa-tool:",
        "registry.example/team/npa-tool@not-a-digest",
        "registry.example/team/npa-tool@sha256:abc",
        "registry.example//npa-tool:tag",
        "registry.example/team/../npa-tool:tag",
    ],
)
def test_malformed_oci_references_fail_closed(reference: str) -> None:
    with pytest.raises(ImageBootstrapContractError):
        parse_oci_reference(reference)
    assert not is_first_party_image(reference)


def test_embedded_digest_must_match_resolved_digest() -> None:
    with pytest.raises(ImageBootstrapContractError, match="conflicts"):
        immutable_image_reference(
            f"registry.example/npa-tool:tag@{'sha256:' + 'b' * 64}", DIGEST
        )


def test_same_digest_concurrent_probes_have_unique_owned_pods() -> None:
    calls: list[list[str]] = []
    lock = threading.Lock()

    def runner(argv, env):
        with lock:
            return _successful_runner(calls)(argv, env)

    results = []

    def probe(nonce: str) -> None:
        results.append(
            probe_image_capabilities(
                image=IMAGE,
                digest=DIGEST,
                context="ctx",
                runner=runner,
                nonce_factory=lambda: nonce,
            )
        )

    threads = [
        threading.Thread(target=probe, args=("1" * 16,)),
        threading.Thread(target=probe, args=("2" * 16,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 2 and all(item.ok for item in results)
    created = {argv[4] for argv in calls if argv[3] == "run"}
    deleted = {argv[5] for argv in calls if argv[3] == "delete"}
    assert len(created) == 2
    assert deleted == created


def test_probe_retries_already_exists_with_a_new_nonce() -> None:
    calls: list[list[str]] = []
    nonces = iter(("3" * 16, "4" * 16))

    def runner(argv, env):
        if argv[3] == "run" and argv[4].endswith("3" * 16):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 1, "", "Error from server (AlreadyExists)")
        return _successful_runner(calls)(argv, env)

    evidence = probe_image_capabilities(
        image=IMAGE,
        digest=DIGEST,
        context="ctx",
        runner=runner,
        nonce_factory=lambda: next(nonces),
    )
    assert evidence.ok
    assert len([argv for argv in calls if argv[3] == "run"]) == 2


def test_replacement_identity_is_refused_and_never_deleted() -> None:
    calls: list[list[str]] = []
    reads = 0

    def runner(argv, env):
        nonlocal reads
        calls.append(argv)
        if argv[3] == "run" or argv[3] == "wait":
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[3] == "get":
            reads += 1
            name = argv[5]
            probe_id = name.rsplit("-", 1)[-1]
            return subprocess.CompletedProcess(
                argv, 0, _pod_payload(name, probe_id, uid=f"uid-{reads}"), ""
            )
        raise AssertionError("replacement pod must not be deleted")

    evidence = probe_image_capabilities(
        image=IMAGE,
        digest=DIGEST,
        context="ctx",
        runner=runner,
        nonce_factory=lambda: "5" * 16,
    )
    assert evidence.state == "indeterminate"
    assert evidence.cleanup == "refused_identity_mismatch"
    assert not any(argv[3] == "delete" for argv in calls)


def test_operator_interruption_cleans_only_the_owned_probe() -> None:
    calls: list[list[str]] = []

    def runner(argv, env):
        if argv[3] == "wait":
            calls.append(argv)
            raise KeyboardInterrupt
        return _successful_runner(calls)(argv, env)

    with pytest.raises(KeyboardInterrupt):
        probe_image_capabilities(
            image=IMAGE,
            digest=DIGEST,
            context="ctx",
            runner=runner,
            nonce_factory=lambda: "6" * 16,
        )
    deletes = [argv for argv in calls if argv[3] == "delete"]
    assert len(deletes) == 1
    assert deletes[0][5].endswith("6" * 16)


def test_operator_interruption_preserves_primary_when_cleanup_fails() -> None:
    calls: list[list[str]] = []

    def runner(argv, env):
        if argv[3] == "wait":
            calls.append(argv)
            raise KeyboardInterrupt
        return _successful_runner(calls, delete_error="cleanup denied")(argv, env)

    with pytest.raises(KeyboardInterrupt) as caught:
        probe_image_capabilities(
            image=IMAGE,
            digest=DIGEST,
            context="ctx",
            runner=runner,
            nonce_factory=lambda: "7" * 16,
        )
    notes = getattr(caught.value, "__notes__", [])
    fallback = getattr(caught.value, "__npa_cleanup_note__", "")
    assert "cleanup denied" in " ".join([*notes, fallback])
