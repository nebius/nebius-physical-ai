"""Unit contracts for fail-closed RoboTwin submit authorization."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from npa.orchestration.npa_workflow import robotwin_preflight as preflight_module

from npa.orchestration.npa_workflow.robotwin_preflight import (
    CHILD_BUCKET_ENV,
    CHILD_CONFIG_PATH_ENV,
    CHILD_IMAGE_ENV,
    CHILD_OUTPUT_PREFIX_ENV,
    CHILD_OUTPUT_ROOT_ENV,
    CHILD_RUN_ID_ENV,
    CHILD_RUNTIME_AUTH_ENV,
    CONTEXT_ENV_NAMES,
    CUSTOMER_ENTITLEMENT_ENV,
    CUSTOMER_ENTITLEMENT_NOTICE,
    CUSTOMER_TERMS,
    CUSTOMER_USE_SCOPE,
    MAX_CONTEXT_BYTES,
    MAX_TRANSPORT_BYTES,
    PUBLIC_CONTEXT_ENV,
    RobotwinPreflightError,
    CustomerAuthorizationBoundaryRefusal,
    bind_submit_coordinates,
    decode_transport,
    encode_runtime_authorization,
    encode_transport,
    is_robotwin_request,
    load_runtime_authorization,
    materialize_transport,
    prepare_inner_submit,
    prepare_live_submit,
    read_customer_entitlement,
    read_owner_context,
    recognize_contract,
    validate_context_bytes,
    validate_control_plane_source,
)
from npa.orchestration.npa_workflow.spec import load_spec


ROOT = Path(__file__).resolve().parents[4]
ROBOTWIN_SPEC = ROOT / "workflows" / "testing" / "byof-robotwin.yaml"
_REAL_SOURCE_BYTE_GATE = preflight_module._require_verified_control_plane_source_bytes
SOURCE_FINGERPRINT = "f" * 64
SOURCE_URI = f"s3://control-source-bucket/npa-src/npa/{SOURCE_FINGERPRINT}/"


def _source_contract(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "control_plane_source_uri": SOURCE_URI,
        "control_plane_source_origin": "environment",
        "control_plane_source_fingerprint": SOURCE_FINGERPRINT,
        "source_proof_provider": lambda _uri, _fingerprint: None,
    }
    values.update(updates)
    return values


def _config_files(
    tmp_path: Path, context_name: str = "robotwin-context"
) -> tuple[Path, Path]:
    kubeconfig = tmp_path / "kubeconfig.yaml"
    kubeconfig.write_text(
        "\n".join(
            (
                "apiVersion: v1",
                "kind: Config",
                f"current-context: {context_name}",
                "clusters:",
                "  - name: robotwin-cluster",
                "    cluster:",
                "      server: https://cluster.example.invalid",
                "      certificate-authority-data: Y2E=",
                "contexts:",
                f"  - name: {context_name}",
                "    context:",
                "      cluster: robotwin-cluster",
                "      user: robotwin-user",
                "users:",
                "  - name: robotwin-user",
                "    user:",
                "      token: portable-test-token",
                "",
            )
        ),
        encoding="utf-8",
    )
    skypilot = tmp_path / "skypilot.yaml"
    skypilot.write_text(
        "kubernetes:\n  allowed_contexts: [robotwin-context]\n",
        encoding="utf-8",
    )
    kubeconfig.chmod(0o600)
    skypilot.chmod(0o600)
    return kubeconfig, skypilot


def _context_payload(tmp_path: Path, **updates: object) -> dict[str, object]:
    kubeconfig, skypilot = _config_files(tmp_path)
    payload: dict[str, object] = {
        "solution": "robotwin",
        "ownership_provenance": "manager-issued",
        "customer_scope_id": "customer-scope-canary",
        "workflow_sha256": "718bb6ae47c8e5e7e761303ebda9e962afa446a6b84030dade7c224cd255ece3",
        "source_revision": "96c1feab536306b50c26af200044fcdf126e8904",
        "curobo_revision": "d64c4b005459db10c5dd867d8b30a87d5bda9bdb",
        "asset_revision": "785feb15aa4a4f532395ad2b1d2be5f28cb561ad",
        "runtime_lock_sha256": "dda9bfebe81250247d25259d655589f8f3b95af7d8629d31b49c59a6af3150ee",
        "bootstrap_image": "registry.example/robotwin-private/npa-robotwin@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "reservation": {
            "policy": "STRICT",
            "accelerator": "RTXPRO-6000-BLACKWELL-SERVER-EDITION",
            "count": 1,
        },
        "project": "robotwin-project-canary",
        "nebius_profile": "robotwin-profile",
        "kubeconfig": str(kubeconfig),
        "kubernetes_context": "robotwin-context",
        "skypilot_config_path": str(skypilot),
        "bucket": "robotwin-bucket-canary",
        "output_root": "s3://robotwin-bucket-canary/robotwin-output",
        "run_id": "robotwin-private-run-canary",
    }
    payload.update(updates)
    return payload


def _context_file(tmp_path: Path, **updates: object) -> tuple[Path, bytes]:
    raw = json.dumps(_context_payload(tmp_path, **updates), sort_keys=True).encode()
    path = tmp_path / "context.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    return path, raw


def _entitlement_payload(
    context: dict[str, object], **updates: object
) -> SimpleNamespace:
    payload: dict[str, object] = {
        "issuer": "https://customer-auth.example.invalid",
        "customer_scope_id": context["customer_scope_id"],
        "run_id": context["run_id"],
        "runtime_manifest_sha256": context["runtime_lock_sha256"],
        "issued_at": "2026-01-01T00:00:00Z",
        "expires_at": "2099-01-01T00:00:00Z",
        "decision": "accepted",
        "intended_activity": CUSTOMER_USE_SCOPE,
        "terms": list(CUSTOMER_TERMS),
        "assertion_id": "assertion-canary-0001",
        "nonce": "nonce-canary-00000001",
    }
    payload.update(updates)
    return SimpleNamespace(**payload)


class _AuthenticatedBoundary:
    def __init__(
        self,
        assertion: SimpleNamespace,
        *,
        refusal: str = "",
    ) -> None:
        self.assertion = assertion
        self.trusted_issuer = "https://customer-auth.example.invalid"
        self.refusal = refusal
        self.requests: list[object] = []
        self.consumed = False

    def consume_once(self, request: object) -> SimpleNamespace:
        self.requests.append(request)
        if self.refusal:
            raise CustomerAuthorizationBoundaryRefusal(self.refusal)
        if self.consumed:
            raise CustomerAuthorizationBoundaryRefusal("replayed")
        self.consumed = True
        return self.assertion


def _entitlement_file(
    tmp_path: Path,
    *,
    context: dict[str, object] | None = None,
    **updates: object,
) -> tuple[Path, bytes]:
    bound_context = context or _context_payload(tmp_path)
    unsigned = vars(_entitlement_payload(bound_context, **updates))
    unsigned["provenance"] = "customer-issued"
    raw = json.dumps(unsigned, sort_keys=True).encode()
    path = tmp_path / "customer-entitlement.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    return path, raw


def _authorization_environment(
    tmp_path: Path, **context_updates: object
) -> tuple[dict[str, str], Path, bytes, Path, bytes]:
    context = _context_payload(tmp_path, **context_updates)
    context_raw = json.dumps(context, sort_keys=True).encode()
    context_path = tmp_path / "context.json"
    context_path.write_bytes(context_raw)
    context_path.chmod(0o600)
    entitlement_path, entitlement_raw = _entitlement_file(tmp_path, context=context)
    return (
        {
            PUBLIC_CONTEXT_ENV: str(context_path),
        },
        context_path,
        context_raw,
        entitlement_path,
        entitlement_raw,
    )


def _validate_context(
    tmp_path: Path,
    raw: bytes,
    *,
    entitlement_updates: dict[str, object] | None = None,
):
    context = json.loads(raw)
    assertion = _entitlement_payload(context, **(entitlement_updates or {}))
    return validate_context_bytes(
        raw,
        customer_authorization_boundary=_AuthenticatedBoundary(assertion),
    )


def _load_authorization(environment: dict[str, str], raw: bytes):
    context = json.loads(raw)
    return load_runtime_authorization(
        environment,
        customer_authorization_boundary=_AuthenticatedBoundary(
            _entitlement_payload(context)
        ),
    )


def test_exact_contract_recognition_is_narrow_and_non_robotwin_is_inert() -> None:
    spec = load_spec(ROBOTWIN_SPEC)
    assert recognize_contract(spec)
    assert not recognize_contract(
        load_spec(ROOT / "workflows" / "testing" / "byof.yaml")
    )

    with pytest.raises(RobotwinPreflightError, match="workflow-config-repo_ref"):
        recognize_contract(replace(spec, config={**spec.config, "repo_ref": "main"}))
    with pytest.raises(RobotwinPreflightError, match="workflow-config-schema"):
        recognize_contract(replace(spec, config={**spec.config, "extra": "unreviewed"}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("solution_name", "RoBoTwIn"),
        ("repo_url", "https://github.com/RoboTwin-Platform/RoboTwin.git"),
        ("base_image", "tool://robotwin"),
        ("image", "registry.invalid/private/npa-robotwin:mutable"),
        ("smoke_command", "/opt/npa/robotwin/robotwin-runtime run"),
        (
            "capability_name",
            "beat_block_hammer_successful_seed_replay_collection",
        ),
        (
            "resource_profile_yaml",
            "byof-solution-smoke-robotwin-rtxpro-gpu",
        ),
    ],
)
def test_relabelled_workflow_still_enters_exact_contract_refusal(
    field: str, value: str
) -> None:
    spec = load_spec(ROBOTWIN_SPEC)
    config = {
        **spec.config,
        "solution_name": "generic-solution",
        "repo_url": "https://example.invalid/generic/repository.git",
        "repo_ref": "main",
        "runtime_context_env": "GENERIC_RUNTIME_CONTEXT",
        "base_image": "tool://generic",
        "smoke_command": "echo generic",
        "capability_name": "generic_capability",
        "resource_profile_yaml": "generic-profile",
        field: value,
    }

    with pytest.raises(RobotwinPreflightError, match="workflow-config"):
        recognize_contract(
            replace(
                spec,
                metadata={**spec.metadata, "name": "generic-workflow"},
                config=config,
            )
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("solution_name", "RoBoTwIn"),
        ("repo_url", "https://github.com/robotwin-platform/robotwin"),
        (
            "repo_url",
            "https://github.com:443/RoboTwin-Platform/RoboTwin.git/",
        ),
        ("repo_url", "http://github.com:80/RoboTwin-Platform/RoboTwin.git"),
        ("repo_url", "ssh://git@github.com:22/RoboTwin-Platform/RoboTwin.git"),
        ("repo_url", "git://github.com:9418/RoboTwin-Platform/RoboTwin.git"),
        ("repo_url", "https://github.com./RoboTwin-Platform/RoboTwin.git"),
        ("repo_url", "git@github.com:RoboTwin-Platform/RoboTwin.git"),
        ("base_image", "tool://robotwin"),
        ("base_image", "registry.invalid/private/npa-robotwin:mutable"),
        ("image", "registry.invalid/npa-robotwin@sha256:" + "a" * 64),
        ("image", "registry.invalid/private/npa-robotwin:mutable"),
        ("smoke_command", "/opt/npa/robotwin/robotwin-runtime run"),
        (
            "capability_name",
            "beat_block_hammer_successful_seed_replay_collection",
        ),
        ("yaml_path", "/private/byof-solution-smoke-robotwin-rtxpro-gpu.yaml"),
    ],
)
def test_robotwin_request_cannot_be_relabelled(field: str, value: str) -> None:
    assert is_robotwin_request(**{field: value})


def test_owner_context_read_is_byte_exact_bounded_owner_only_and_no_follow(
    tmp_path: Path,
) -> None:
    path, raw = _context_file(tmp_path)
    assert read_owner_context({PUBLIC_CONTEXT_ENV: str(path)}) == raw

    path.chmod(0o640)
    with pytest.raises(RobotwinPreflightError, match="context-not-owner-only"):
        read_owner_context({PUBLIC_CONTEXT_ENV: str(path)})
    path.chmod(0o600)
    link = tmp_path / "context-link.json"
    link.symlink_to(path)
    with pytest.raises(RobotwinPreflightError, match="context-unreadable"):
        read_owner_context({PUBLIC_CONTEXT_ENV: str(link)})

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * (MAX_CONTEXT_BYTES + 1))
    oversized.chmod(0o600)
    with pytest.raises(RobotwinPreflightError, match="context-too-large"):
        read_owner_context({PUBLIC_CONTEXT_ENV: str(oversized)})


def test_unsigned_customer_entitlement_file_is_never_an_authority(
    tmp_path: Path,
) -> None:
    path, _raw = _entitlement_file(tmp_path)
    for mode in (0o600, 0o640):
        path.chmod(mode)
        with pytest.raises(
            RobotwinPreflightError,
            match="customer-authorization-unsigned-local-file",
        ):
            read_customer_entitlement({CUSTOMER_ENTITLEMENT_ENV: str(path)})
    with pytest.raises(RobotwinPreflightError, match="needs_customer_acceptance"):
        read_customer_entitlement({})


def test_owner_file_fifo_refuses_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "runtime-context.fifo"
    os.mkfifo(fifo, mode=0o600)

    with pytest.raises(RobotwinPreflightError, match="context-not-regular"):
        read_owner_context({PUBLIC_CONTEXT_ENV: str(fifo)})


def test_control_plane_source_is_explicit_immutable_and_separate(
    tmp_path: Path,
) -> None:
    authorization = _validate_context(tmp_path, _context_file(tmp_path)[1])
    fingerprint = "a" * 64
    source = f"s3://control-source-bucket/npa-src/npa/{fingerprint}/"

    assert (
        validate_control_plane_source(
            authorization,
            source_uri=source,
            source_origin="environment",
            local_fingerprint=fingerprint,
            proof_provider=lambda _uri, _fingerprint: None,
        )
        == source
    )
    invalid = (
        ("", "environment", fingerprint, "explicit-uri-required"),
        (source, "saved", fingerprint, "explicit-uri-required"),
        (
            f"s3://{authorization.bucket}/npa-src/npa/{fingerprint}/",
            "environment",
            fingerprint,
            "output-bucket-reuse",
        ),
        (
            "s3://control-source-bucket/npa-src/npa/current/",
            "environment",
            fingerprint,
            "not-immutable",
        ),
    )
    for value, origin, local, category in invalid:
        with pytest.raises(RobotwinPreflightError, match=category):
            validate_control_plane_source(
                authorization,
                source_uri=value,
                source_origin=origin,
                local_fingerprint=local,
                proof_provider=lambda _uri, _fingerprint: None,
            )
    with pytest.raises(RobotwinPreflightError, match="source-staging-forbidden"):
        validate_control_plane_source(
            authorization,
            source_uri=source,
            source_origin="environment",
            local_fingerprint=fingerprint,
            source_staging_requested=True,
            proof_provider=lambda _uri, _fingerprint: None,
        )


def test_phase_a_refuses_unproven_source_bytes() -> None:
    with pytest.raises(
        RobotwinPreflightError, match="control-plane-source-byte-proof-unavailable"
    ):
        _REAL_SOURCE_BYTE_GATE(
            "s3://control-source.invalid/npa-src/" + "a" * 64,
            "a" * 64,
        )


@pytest.mark.parametrize(
    ("updates", "category"),
    [
        ({"solution": "another"}, "context-wrong-solution"),
        ({"ownership_provenance": "self-issued"}, "ownership-provenance"),
        (
            {
                "reservation": {
                    "policy": "BEST_EFFORT",
                    "accelerator": "RTXPRO-6000-BLACKWELL-SERVER-EDITION",
                    "count": 1,
                }
            },
            "reservation-policy-not-strict",
        ),
        (
            {
                "reservation": {
                    "policy": "STRICT",
                    "accelerator": "B200",
                    "count": 1,
                }
            },
            "reservation-accelerator-mismatch",
        ),
        (
            {
                "reservation": {
                    "policy": "STRICT",
                    "accelerator": "RTXPRO-6000-BLACKWELL-SERVER-EDITION",
                    "count": True,
                }
            },
            "reservation-count-not-one",
        ),
        (
            {"bootstrap_image": "docker.io/public/example:latest"},
            "bootstrap-image-not-immutable",
        ),
        (
            {"bootstrap_image": "private-namespace/example"},
            "bootstrap-image-not-immutable",
        ),
        ({"output_root": "s3://other-bucket/output"}, "output-root-bucket"),
        ({"run_id": "unscoped-run"}, "run-id-invalid"),
    ],
)
def test_context_refuses_invalid_types_identity_and_destination(
    tmp_path: Path, updates: dict[str, object], category: str
) -> None:
    raw = json.dumps(_context_payload(tmp_path, **updates), sort_keys=True).encode()
    with pytest.raises(RobotwinPreflightError, match=category):
        _validate_context(tmp_path, raw)


@pytest.mark.parametrize(
    ("updates", "category"),
    [
        ({"issuer": ""}, "issuer-invalid"),
        ({"decision": "declined"}, "customer-authorization-declined"),
        ({"customer_scope_id": "another-customer"}, "wrong-customer"),
        ({"run_id": "robotwin-another-run"}, "wrong-run"),
        ({"runtime_manifest_sha256": "0" * 64}, "wrong-manifest"),
        ({"expires_at": "not-a-date"}, "expiry-invalid"),
        ({"expires_at": "2020-01-01T00:00:00Z"}, "customer-authorization-stale"),
        ({"issued_at": "2099-01-01T00:00:00Z"}, "not-yet-valid"),
        ({"intended_activity": "commercial-service"}, "activity-mismatch"),
        ({"terms": []}, "terms-mismatch"),
        ({"assertion_id": "short"}, "assertion-id-invalid"),
        ({"nonce": "short"}, "nonce-invalid"),
    ],
)
def test_customer_entitlement_is_exact_run_scoped_and_fail_closed(
    tmp_path: Path, updates: dict[str, object], category: str
) -> None:
    raw = json.dumps(_context_payload(tmp_path), sort_keys=True).encode()
    with pytest.raises(RobotwinPreflightError, match=category):
        _validate_context(tmp_path, raw, entitlement_updates=updates)


def test_customer_entitlement_notice_names_terms_responsibility_and_resume() -> None:
    assert "CUDA 12.8.1" in CUSTOMER_ENTITLEMENT_NOTICE
    assert "cuDNN 9.8.0" in CUSTOMER_ENTITLEMENT_NOTICE
    assert "CuRobo v0.7.8" in CUSTOMER_ENTITLEMENT_NOTICE
    assert "customer representative" in CUSTOMER_ENTITLEMENT_NOTICE
    assert "Decline" in CUSTOMER_ENTITLEMENT_NOTICE
    assert "accept and resume" in CUSTOMER_ENTITLEMENT_NOTICE
    assert CUSTOMER_ENTITLEMENT_ENV in CUSTOMER_ENTITLEMENT_NOTICE
    assert "technical artifact-lock" in CUSTOMER_ENTITLEMENT_NOTICE
    assert "aggregate" not in CUSTOMER_ENTITLEMENT_NOTICE


def test_authenticated_boundary_is_exact_bound_and_replay_safe(tmp_path: Path) -> None:
    environment, _path, raw, _unsigned_path, _unsigned_raw = _authorization_environment(
        tmp_path
    )
    assertion = _entitlement_payload(json.loads(raw))
    boundary = _AuthenticatedBoundary(assertion)

    authorization = load_runtime_authorization(
        environment, customer_authorization_boundary=boundary
    )

    assert (
        authorization.customer_authorization_sha256
        == hashlib.sha256(authorization.raw_customer_authorization).hexdigest()
    )
    assert len(boundary.requests) == 1
    request = boundary.requests[0]
    assert request.expected_issuer == "https://customer-auth.example.invalid"
    assert request.customer_scope_id == "customer-scope-canary"
    assert request.run_id == "robotwin-private-run-canary"
    assert request.runtime_manifest_sha256 == preflight_module.RUNTIME_LOCK_SHA256
    assert request.intended_activity == CUSTOMER_USE_SCOPE
    assert request.terms == tuple((term["id"], term["url"]) for term in CUSTOMER_TERMS)
    with pytest.raises(RobotwinPreflightError, match="customer-authorization-replayed"):
        load_runtime_authorization(
            environment, customer_authorization_boundary=boundary
        )


@pytest.mark.parametrize(
    ("issuer", "category"),
    (
        ("https://attacker-auth.example.invalid", "issuer-mismatch"),
        ("https://CUSTOMER-auth.example.invalid", "issuer-mismatch"),
        ("https://customer-auth.example.invalid/", "issuer-mismatch"),
        ("https://manager-auth.example.invalid", "issuer-mismatch"),
        ("https://customer-auth.exampıe.invalid", "issuer-invalid"),
    ),
)
def test_authenticated_boundary_requires_exact_trusted_issuer(
    tmp_path: Path, issuer: str, category: str
) -> None:
    environment, _path, raw, _unsigned_path, _unsigned_raw = _authorization_environment(
        tmp_path
    )
    boundary = _AuthenticatedBoundary(
        _entitlement_payload(json.loads(raw), issuer=issuer)
    )

    with pytest.raises(RobotwinPreflightError, match=category):
        load_runtime_authorization(
            environment, customer_authorization_boundary=boundary
        )

    assert len(boundary.requests) == 1
    assert boundary.requests[0].expected_issuer == boundary.trusted_issuer


def test_authenticated_boundary_requires_a_trusted_issuer_before_consumption(
    tmp_path: Path,
) -> None:
    environment, _path, raw, _unsigned_path, _unsigned_raw = _authorization_environment(
        tmp_path
    )
    assertion = _entitlement_payload(json.loads(raw))
    calls: list[object] = []
    boundary = SimpleNamespace(
        consume_once=lambda request: calls.append(request) or assertion
    )

    with pytest.raises(RobotwinPreflightError, match="trusted-issuer-unavailable"):
        load_runtime_authorization(
            environment, customer_authorization_boundary=boundary
        )
    assert calls == []


@pytest.mark.parametrize(
    "category",
    ("absent", "declined", "replayed", "unauthenticated", "malformed"),
)
def test_authenticated_boundary_refusals_are_fixed_and_redacted(
    tmp_path: Path, category: str
) -> None:
    environment, _path, raw, _unsigned_path, _unsigned_raw = _authorization_environment(
        tmp_path
    )
    private_assertion = _entitlement_payload(
        json.loads(raw), assertion_id="private-assertion-canary"
    )
    boundary = _AuthenticatedBoundary(private_assertion, refusal=category)

    with pytest.raises(
        RobotwinPreflightError, match=f"customer-authorization-{category}"
    ) as caught:
        load_runtime_authorization(
            environment, customer_authorization_boundary=boundary
        )

    assert "private-assertion-canary" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_customer_entitlement_parse_failure_discards_private_exception_graph(
    tmp_path: Path,
) -> None:
    raw = json.dumps(_context_payload(tmp_path), sort_keys=True).encode()
    with pytest.raises(RobotwinPreflightError) as caught:
        _validate_context(
            tmp_path,
            raw,
            entitlement_updates={"expires_at": "private-expiry-canary"},
        )

    assert "private-expiry-canary" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_portable_configs_reject_external_files_and_all_exec_plugins(
    tmp_path: Path,
) -> None:
    payload = _context_payload(tmp_path)
    Path(str(payload["skypilot_config_path"])).write_text(
        "include: /private/other.yaml\n", encoding="utf-8"
    )
    with pytest.raises(RobotwinPreflightError, match="external-reference"):
        _validate_context(tmp_path, json.dumps(payload, sort_keys=True).encode())

    payload = _context_payload(tmp_path)
    Path(str(payload["kubeconfig"])).write_text(
        "\n".join(
            (
                "apiVersion: v1",
                "kind: Config",
                "current-context: robotwin-context",
                "clusters:",
                "  - name: robotwin-cluster",
                "    cluster:",
                "      server: https://cluster.example.invalid",
                "      certificate-authority-data: Y2E=",
                "contexts:",
                "  - name: robotwin-context",
                "    context:",
                "      cluster: robotwin-cluster",
                "      user: robotwin-user",
                "users:",
                "  - name: robotwin-user",
                "    user: {exec: {command: nebius}}",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(RobotwinPreflightError, match="exec-plugin-refused"):
        _validate_context(tmp_path, json.dumps(payload, sort_keys=True).encode())

    payload = _context_payload(tmp_path)
    Path(str(payload["skypilot_config_path"])).write_text(
        "kubernetes:\n  allowed_contexts: [another-context]\n",
        encoding="utf-8",
    )
    with pytest.raises(RobotwinPreflightError, match="context-mismatch"):
        _validate_context(tmp_path, json.dumps(payload, sort_keys=True).encode())

    payload = _context_payload(tmp_path)
    Path(str(payload["kubeconfig"])).write_text(
        Path(str(payload["kubeconfig"]))
        .read_text(encoding="utf-8")
        .replace("token: portable-test-token", "tokenFile: relative-token"),
        encoding="utf-8",
    )
    with pytest.raises(RobotwinPreflightError, match="user-not-portable"):
        _validate_context(tmp_path, json.dumps(payload, sort_keys=True).encode())

    payload = _context_payload(tmp_path)
    Path(str(payload["kubeconfig"])).write_text(
        Path(str(payload["kubeconfig"]))
        .read_text(encoding="utf-8")
        .replace(
            "token: portable-test-token",
            "token: portable-test-token\n      tokenFile: relative-token",
        ),
        encoding="utf-8",
    )
    with pytest.raises(RobotwinPreflightError, match="user-not-portable"):
        _validate_context(tmp_path, json.dumps(payload, sort_keys=True).encode())


@pytest.mark.parametrize(
    "extra",
    (
        "jobs:\n  controller:\n    resources:\n      accelerators: B200:1\n",
        "jobs:\n  controller:\n    resources:\n      image_id: docker:example.invalid/controller\n",
        "nebius:\n  project_id: unreviewed\n",
        "kubernetes:\n  allowed_contexts: [robotwin-context]\n  pod_config: {}\n",
    ),
)
def test_skypilot_config_rejects_topology_and_provider_extensions(
    tmp_path: Path, extra: str
) -> None:
    payload = _context_payload(tmp_path)
    Path(str(payload["skypilot_config_path"])).write_text(
        "kubernetes:\n  allowed_contexts: [robotwin-context]\n" + extra,
        encoding="utf-8",
    )

    with pytest.raises(RobotwinPreflightError, match="skypilot-config"):
        _validate_context(tmp_path, json.dumps(payload, sort_keys=True).encode())


def test_transport_round_trip_is_digest_bound_and_repr_redacted(tmp_path: Path) -> None:
    environment, _path, raw, _entitlement_path, entitlement_raw = (
        _authorization_environment(tmp_path)
    )
    authorization = _load_authorization(environment, raw)
    transport = encode_transport(authorization)
    with pytest.raises(
        RobotwinPreflightError,
        match="customer-authorization-independent-proof-unavailable",
    ):
        decode_transport(transport)

    assert entitlement_raw not in transport.encode()

    payload = json.loads(transport)
    payload["files"]["kubeconfig"]["sha256"] = "0" * 64
    with pytest.raises(RobotwinPreflightError, match="digest-mismatch"):
        decode_transport(json.dumps(payload))

    payload = json.loads(transport)
    payload["files"]["kubeconfig"] = {
        "base64": "",
        "sha256": hashlib.sha256(b"").hexdigest(),
    }
    with pytest.raises(
        RobotwinPreflightError,
        match="customer-authorization-independent-proof-unavailable",
    ):
        decode_transport(json.dumps(payload))

    with pytest.raises(RobotwinPreflightError, match="transport-size-invalid"):
        decode_transport("x" * (MAX_TRANSPORT_BYTES + 1))


def test_worker_materialization_preserves_bytes_modes_and_paths(tmp_path: Path) -> None:
    environment, _path, raw, _entitlement_path, entitlement_raw = (
        _authorization_environment(tmp_path)
    )
    authorization = _load_authorization(environment, raw)
    directory = tmp_path / "materialized"
    directory.mkdir(mode=0o700)

    with pytest.raises(
        RobotwinPreflightError,
        match="customer-authorization-independent-proof-unavailable",
    ):
        materialize_transport(encode_transport(authorization), directory)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert list(directory.iterdir()) == []


def test_worker_materialization_rolls_back_partial_private_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment, _path, raw, _entitlement_path, _entitlement_raw = (
        _authorization_environment(tmp_path)
    )
    authorization = _load_authorization(environment, raw)
    directory = tmp_path / "materialized"
    directory.mkdir(mode=0o700)
    real_write = preflight_module.write_owner_file
    calls = 0

    def fail_second(path: Path, payload: bytes, *, directory_fd=None) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        real_write(path, payload, directory_fd=directory_fd)

    monkeypatch.setattr(preflight_module, "write_owner_file", fail_second)
    with pytest.raises(
        RobotwinPreflightError,
        match="customer-authorization-independent-proof-unavailable",
    ):
        materialize_transport(encode_transport(authorization), directory)
    assert list(directory.iterdir()) == []


def test_worker_materialization_removes_the_file_whose_fsync_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment, _path, raw, _entitlement_path, _entitlement_raw = (
        _authorization_environment(tmp_path)
    )
    authorization = _load_authorization(environment, raw)
    directory = tmp_path / "materialized"
    directory.mkdir(mode=0o700)

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(preflight_module.os, "fsync", fail_fsync)
    with pytest.raises(
        RobotwinPreflightError,
        match="customer-authorization-independent-proof-unavailable",
    ):
        materialize_transport(encode_transport(authorization), directory)
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize(
    "updates",
    (
        {"issuer": "https://attacker-auth.example.invalid"},
        {"assertion_id": "replayed-assertion-0001"},
    ),
)
def test_transported_customer_receipts_require_independent_proof(
    tmp_path: Path, updates: dict[str, object]
) -> None:
    environment, _path, raw, _entitlement_path, _entitlement_raw = (
        _authorization_environment(tmp_path)
    )
    authorization = _load_authorization(environment, raw)
    payload = json.loads(authorization.raw_customer_authorization)
    payload.update(updates)
    receipt = json.dumps(payload, sort_keys=True).encode()
    with pytest.raises(
        RobotwinPreflightError,
        match="customer-authorization-independent-proof-unavailable",
    ):
        preflight_module._parse_transported_customer_authorization(
            receipt,
            customer_scope_id=payload["customer_scope_id"],
            run_id=payload["run_id"],
            context=raw,
        )


@pytest.mark.parametrize("reference", ["~definitely-no-such-user/context", "bad\0path"])
def test_owner_file_reference_failures_are_fixed_category(reference: str) -> None:
    with pytest.raises(RobotwinPreflightError, match="context-unreadable") as caught:
        read_owner_context({PUBLIC_CONTEXT_ENV: reference})
    assert caught.value.__cause__ is None


def test_malformed_kube_server_is_sanitized_and_does_not_consume(
    tmp_path: Path,
) -> None:
    payload = _context_payload(tmp_path)
    kubeconfig = Path(str(payload["kubeconfig"]))
    kubeconfig.write_text(
        kubeconfig.read_text(encoding="utf-8").replace(
            "https://cluster.example.invalid", "https://[invalid-host"
        ),
        encoding="utf-8",
    )
    raw = json.dumps(payload, sort_keys=True).encode()
    boundary = _AuthenticatedBoundary(_entitlement_payload(payload))

    with pytest.raises(RobotwinPreflightError, match="kubeconfig-cluster-not-portable"):
        validate_context_bytes(raw, customer_authorization_boundary=boundary)
    assert boundary.consumed is False


def test_invalid_config_does_not_consume_assertion_before_corrected_retry(
    tmp_path: Path,
) -> None:
    payload = _context_payload(tmp_path)
    raw = json.dumps(payload, sort_keys=True).encode()
    boundary = _AuthenticatedBoundary(_entitlement_payload(payload))
    kubeconfig = Path(str(payload["kubeconfig"]))
    valid = kubeconfig.read_bytes()
    kubeconfig.write_text("not: [valid", encoding="utf-8")

    with pytest.raises(RobotwinPreflightError, match="kubeconfig-invalid"):
        validate_context_bytes(raw, customer_authorization_boundary=boundary)
    assert boundary.consumed is False

    kubeconfig.write_bytes(valid)
    authorization = validate_context_bytes(
        raw, customer_authorization_boundary=boundary
    )
    assert boundary.consumed is True
    assert authorization.run_id == payload["run_id"]


def test_inner_submit_reuses_the_validated_authorization_without_a_consent_proxy(
    tmp_path: Path,
) -> None:
    environment, _context_path, raw, _entitlement_path, _entitlement_raw = (
        _authorization_environment(tmp_path)
    )
    authorization = _load_authorization(environment, raw)
    image = authorization.bootstrap_image
    runtime_capability = encode_runtime_authorization(authorization)
    capability_payload = json.loads(runtime_capability)
    environment = {
        "KUBECONFIG": authorization.kubeconfig_source,
        "KUBECONTEXT": authorization.kubernetes_context,
        "NPA_BYOF_K8S_CONTEXT": authorization.kubernetes_context,
        "NPA_BYOF_PROJECT": authorization.project,
        "NPA_NEBIUS_PROFILE": authorization.profile,
        "NEBIUS_PROFILE": authorization.profile,
        CHILD_BUCKET_ENV: authorization.bucket,
        CHILD_CONFIG_PATH_ENV: authorization.skypilot_config_source,
        CHILD_IMAGE_ENV: image,
        CHILD_OUTPUT_PREFIX_ENV: (
            f"{authorization.output_root}/{authorization.run_id}/"
        ),
        CHILD_OUTPUT_ROOT_ENV: authorization.output_root,
        CHILD_RUN_ID_ENV: authorization.run_id,
        CHILD_RUNTIME_AUTH_ENV: runtime_capability,
        "NPA_BYOF_ROBOTWIN_RESERVATION_EVIDENCE_SHA256": (authorization.context_sha256),
        "NPA_BYOF_ROBOTWIN_IMAGE_SCAN_SHA256": "b" * 64,
        "NPA_BYOF_ROBOTWIN_IMAGE_SCAN_ARCHIVES": "2",
        "AWS_ENDPOINT_URL": "https://storage.eu-north1.nebius.cloud",
        "NEBIUS_S3_ENDPOINT": "https://storage.eu-north1.nebius.cloud",
        "AWS_ACCESS_KEY_ID": "storage-access-canary",
        "AWS_SECRET_ACCESS_KEY": "storage-secret-canary",
    }

    context = prepare_inner_submit(authorization, environment)

    assert context.layer == "inner"
    assert context.transport_value == ""
    assert image not in repr(context)
    assert set(capability_payload) == {
        "schema_version",
        "capability_id",
        "runtime_manifest_sha256",
        "workflow_sha256",
        "source_revision",
        "curobo_revision",
        "asset_revision",
        "bootstrap_image_sha256",
        "runtime_binding_sha256",
        "expires_at",
    }
    for forbidden in (
        authorization.raw_context.decode(),
        authorization.raw_customer_authorization.decode(),
        authorization.project,
        authorization.profile,
        authorization.kubernetes_context,
        authorization.bucket,
        authorization.output_root,
        "assertion-canary-0001",
        "nonce-canary-00000001",
    ):
        assert forbidden not in runtime_capability
    with pytest.raises(RobotwinPreflightError, match="inner-environment-mismatch"):
        prepare_inner_submit(
            authorization,
            {**environment, CHILD_OUTPUT_PREFIX_ENV: "s3://wrong/output/"},
        )


def test_live_submit_requires_public_secret_name_and_binds_coordinates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.robotwin_preflight.RUNTIME_LOCK_STATUS",
        "complete",
    )
    environment, _context_path, _raw, _entitlement_path, _entitlement_raw = (
        _authorization_environment(tmp_path)
    )
    spec = load_spec(ROBOTWIN_SPEC)

    with pytest.raises(RobotwinPreflightError, match="secret-not-requested"):
        prepare_live_submit(
            spec,
            requested_secret_envs=(),
            environ=environment,
            **_source_contract(),
        )
    with pytest.raises(
        RobotwinPreflightError, match="needs_customer_acceptance"
    ) as caught:
        prepare_live_submit(
            spec,
            requested_secret_envs=(PUBLIC_CONTEXT_ENV,),
            environ=environment,
            **_source_contract(),
        )
    assert caught.value.notice == preflight_module.customer_acceptance_notice()
    assert caught.value.notice["side_effects_started"] is False
    with pytest.raises(
        RobotwinPreflightError, match="customer-authorization-unsigned-local-file"
    ):
        prepare_live_submit(
            spec,
            requested_secret_envs=(PUBLIC_CONTEXT_ENV, CUSTOMER_ENTITLEMENT_ENV),
            environ=environment,
            **_source_contract(),
        )
    boundary = _AuthenticatedBoundary(_entitlement_payload(json.loads(_raw)))
    context = prepare_live_submit(
        spec,
        requested_secret_envs=(PUBLIC_CONTEXT_ENV,),
        environ=environment,
        customer_authorization_boundary=boundary,
        **_source_contract(),
    )
    assert context is not None
    project, infra, config_path = bind_submit_coordinates(
        context,
        project="",
        infra="",
        config_path=None,
        s3_bucket="",
        resume_requested=False,
        alternate_output_requested=False,
        outer_override_requested=False,
        runtime_requested=False,
    )
    assert project == context.authorization.project
    assert infra == f"k8s/{context.authorization.kubernetes_context}"
    assert config_path == Path(context.authorization.skypilot_config_source)

    with pytest.raises(RobotwinPreflightError, match="submit-resume-refused"):
        bind_submit_coordinates(
            context,
            project=project,
            infra=infra,
            config_path=config_path,
            s3_bucket=context.authorization.bucket,
            resume_requested=True,
            alternate_output_requested=False,
            outer_override_requested=False,
            runtime_requested=False,
        )


def test_live_submit_proves_exact_source_once_before_consuming_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(preflight_module, "RUNTIME_LOCK_STATUS", "complete")
    environment, _context_path, raw, _entitlement_path, _entitlement_raw = (
        _authorization_environment(tmp_path)
    )
    events: list[tuple[str, ...]] = []

    def prove_source(uri: str, fingerprint: str) -> None:
        events.append(("source-proof", uri, fingerprint))

    class OrderedBoundary(_AuthenticatedBoundary):
        def consume_once(self, request: object) -> SimpleNamespace:
            events.append(("consume-once",))
            return super().consume_once(request)

    boundary = OrderedBoundary(_entitlement_payload(json.loads(raw)))
    context = prepare_live_submit(
        load_spec(ROBOTWIN_SPEC),
        requested_secret_envs=(PUBLIC_CONTEXT_ENV,),
        environ=environment,
        customer_authorization_boundary=boundary,
        **_source_contract(source_proof_provider=prove_source),
    )

    assert context is not None
    assert events == [
        ("source-proof", SOURCE_URI, SOURCE_FINGERPRINT),
        ("consume-once",),
    ]
    assert len(boundary.requests) == 1


def test_live_submit_real_source_proof_refuses_before_authority_consumption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(preflight_module, "RUNTIME_LOCK_STATUS", "complete")
    environment, _context_path, raw, _entitlement_path, _entitlement_raw = (
        _authorization_environment(tmp_path)
    )
    boundary = _AuthenticatedBoundary(_entitlement_payload(json.loads(raw)))

    with pytest.raises(
        RobotwinPreflightError,
        match="control-plane-source-byte-proof-unavailable",
    ):
        prepare_live_submit(
            load_spec(ROBOTWIN_SPEC),
            requested_secret_envs=(PUBLIC_CONTEXT_ENV,),
            environ=environment,
            customer_authorization_boundary=boundary,
            **_source_contract(source_proof_provider=None),
        )

    assert boundary.requests == []
    assert boundary.consumed is False


@pytest.mark.parametrize(
    "internal_name",
    tuple(
        name
        for name in CONTEXT_ENV_NAMES
        if name not in {PUBLIC_CONTEXT_ENV, CUSTOMER_ENTITLEMENT_ENV}
    ),
)
def test_live_submit_rejects_every_internal_context_channel_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, internal_name: str
) -> None:
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.robotwin_preflight.RUNTIME_LOCK_STATUS",
        "complete",
    )
    environment, _context_path, _raw, _entitlement_path, _entitlement_raw = (
        _authorization_environment(tmp_path)
    )
    environment[internal_name] = "private-internal-channel-canary"

    with pytest.raises(
        RobotwinPreflightError, match="internal-context-channel-forbidden"
    ) as caught:
        prepare_live_submit(
            load_spec(ROBOTWIN_SPEC),
            requested_secret_envs=(PUBLIC_CONTEXT_ENV,),
            environ=environment,
            customer_authorization_boundary=_AuthenticatedBoundary(
                _entitlement_payload(json.loads(_raw))
            ),
            **_source_contract(),
        )

    assert "private-internal-channel-canary" not in str(caught.value)


def test_invalid_config_refuses_before_assertion_is_consumed(tmp_path: Path) -> None:
    environment, _context_path, raw, _entitlement_path, _entitlement_raw = (
        _authorization_environment(tmp_path)
    )
    context = json.loads(raw)
    Path(str(context["kubeconfig"])).unlink()
    Path(str(context["skypilot_config_path"])).unlink()

    boundary = _AuthenticatedBoundary(
        _entitlement_payload(context, expires_at="2020-01-01T00:00:00Z")
    )
    with pytest.raises(RobotwinPreflightError, match="kubeconfig-unreadable"):
        load_runtime_authorization(
            environment,
            customer_authorization_boundary=boundary,
        )
    assert boundary.consumed is False


def test_live_submit_refuses_disabled_runtime_after_context_validation(
    tmp_path: Path,
) -> None:
    environment, _context_path, raw, _entitlement_path, _entitlement_raw = (
        _authorization_environment(tmp_path)
    )
    with pytest.raises(
        RobotwinPreflightError,
        match="runtime-delivery-technical-gates-incomplete",
    ):
        prepare_live_submit(
            load_spec(ROBOTWIN_SPEC),
            requested_secret_envs=(PUBLIC_CONTEXT_ENV,),
            environ=environment,
            customer_authorization_boundary=_AuthenticatedBoundary(
                _entitlement_payload(json.loads(raw))
            ),
            **_source_contract(),
        )


def test_parser_refusal_has_no_cause_or_context_graph(tmp_path: Path) -> None:
    context = tmp_path / "private-context-canary.json"
    context.write_bytes(b"{private-document-canary")
    context.chmod(0o600)

    with pytest.raises(RobotwinPreflightError) as caught:
        load_runtime_authorization({PUBLIC_CONTEXT_ENV: str(context)})

    error: BaseException | None = caught.value
    while error is not None:
        rendered = str(error)
        assert str(context) not in rendered
        assert "private-document-canary" not in rendered
        assert error.__cause__ is None
        error = error.__context__


def test_all_confidential_parser_refusals_discard_private_exception_graphs(
    tmp_path: Path,
) -> None:
    def assert_clean(error: BaseException, *private_values: str) -> None:
        current: BaseException | None = error
        while current is not None:
            rendered = str(current)
            for value in private_values:
                assert value not in rendered
            assert current.__cause__ is None
            current = current.__context__

    missing = tmp_path / "private-missing-context-canary.json"
    with pytest.raises(RobotwinPreflightError) as caught:
        load_runtime_authorization({PUBLIC_CONTEXT_ENV: str(missing)})
    assert_clean(caught.value, str(missing))

    context, _ = _context_file(tmp_path)
    link = tmp_path / "private-context-link-canary.json"
    link.symlink_to(context)
    with pytest.raises(RobotwinPreflightError) as caught:
        load_runtime_authorization({PUBLIC_CONTEXT_ENV: str(link)})
    assert_clean(caught.value, str(link), str(context))

    for field, private_document in (
        ("kubeconfig", "{private-kubeconfig-document-canary"),
        ("skypilot_config_path", "{private-skypilot-document-canary"),
    ):
        payload = _context_payload(tmp_path)
        private_path = Path(str(payload[field]))
        private_path.write_text(private_document, encoding="utf-8")
        raw = json.dumps(payload, sort_keys=True).encode()
        with pytest.raises(RobotwinPreflightError) as caught:
            _validate_context(tmp_path, raw)
        assert_clean(
            caught.value,
            str(private_path),
            private_document,
            "robotwin-project-canary",
            "portable-test-token",
        )

    environment, _context_path, _raw, entitlement_path, _entitlement_raw = (
        _authorization_environment(tmp_path)
    )
    environment[CUSTOMER_ENTITLEMENT_ENV] = str(entitlement_path)
    with pytest.raises(RobotwinPreflightError) as caught:
        load_runtime_authorization(environment)
    assert_clean(caught.value, str(entitlement_path), "customer-scope-canary")

    private_transport = "{private-transport-document-canary"
    with pytest.raises(RobotwinPreflightError) as caught:
        decode_transport(private_transport)
    assert_clean(caught.value, private_transport)
