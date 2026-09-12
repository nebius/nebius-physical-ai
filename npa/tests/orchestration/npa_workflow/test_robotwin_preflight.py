"""Unit contracts for fail-closed RoboTwin submit authorization."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import stat

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
    MAX_CONTEXT_BYTES,
    MAX_TRANSPORT_BYTES,
    PUBLIC_CONTEXT_ENV,
    RobotwinPreflightError,
    bind_submit_coordinates,
    decode_transport,
    encode_runtime_authorization,
    encode_transport,
    is_robotwin_request,
    load_runtime_authorization,
    materialize_transport,
    prepare_inner_submit,
    prepare_live_submit,
    read_owner_context,
    recognize_contract,
    validate_context_bytes,
    validate_control_plane_source,
)
from npa.orchestration.npa_workflow.spec import load_spec


ROOT = Path(__file__).resolve().parents[4]
ROBOTWIN_SPEC = ROOT / "workflows" / "testing" / "byof-robotwin.yaml"
_REAL_RUNTIME_USE_RECEIPT_GATE = preflight_module._require_genuine_runtime_use_receipt
_REAL_SOURCE_BYTE_GATE = preflight_module._require_verified_control_plane_source_bytes


@pytest.fixture(autouse=True)
def _supply_phase_b_proofs_only_inside_unit_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let structural tests reach post-proof checks without weakening production."""

    monkeypatch.setattr(
        preflight_module, "_require_genuine_runtime_use_receipt", lambda _raw: None
    )
    monkeypatch.setattr(
        preflight_module,
        "_require_verified_control_plane_source_bytes",
        lambda _uri, _fingerprint: None,
    )


def _config_files(tmp_path: Path, context_name: str = "robotwin-context") -> tuple[Path, Path]:
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
        "workflow_sha256": "718bb6ae47c8e5e7e761303ebda9e962afa446a6b84030dade7c224cd255ece3",
        "source_revision": "96c1feab536306b50c26af200044fcdf126e8904",
        "curobo_revision": "d64c4b005459db10c5dd867d8b30a87d5bda9bdb",
        "asset_revision": "785feb15aa4a4f532395ad2b1d2be5f28cb561ad",
        "runtime_lock_sha256": "c42c4037392f51ad6c2473eb3f07843738a4c5147328ace1686ddb9cf553b4ef",
        "bootstrap_image": "registry.example/robotwin-private/npa-robotwin@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "reservation": {
            "policy": "STRICT",
            "accelerator": "RTXPRO-6000-BLACKWELL-SERVER-EDITION",
            "count": 1,
        },
        "license_acceptance": {
            "nvidia_cuda_eula": True,
            "nvidia_cudnn_sla": True,
            "curobo_noncommercial_research_or_evaluation": True,
            "robotwin2_aggregate_asset_and_output_terms": True,
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


def test_exact_contract_recognition_is_narrow_and_non_robotwin_is_inert() -> None:
    spec = load_spec(ROBOTWIN_SPEC)
    assert recognize_contract(spec)
    assert not recognize_contract(
        load_spec(ROOT / "workflows" / "testing" / "byof.yaml")
    )

    with pytest.raises(RobotwinPreflightError, match="workflow-config-repo_ref"):
        recognize_contract(
            replace(spec, config={**spec.config, "repo_ref": "main"})
        )
    with pytest.raises(RobotwinPreflightError, match="workflow-config-schema"):
        recognize_contract(
            replace(spec, config={**spec.config, "extra": "unreviewed"})
        )


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


def test_control_plane_source_is_explicit_immutable_and_separate(tmp_path: Path) -> None:
    authorization = validate_context_bytes(_context_file(tmp_path)[1])
    fingerprint = "a" * 64
    source = f"s3://control-source-bucket/npa-src/npa/{fingerprint}/"

    assert (
        validate_control_plane_source(
            authorization,
            source_uri=source,
            source_origin="environment",
            local_fingerprint=fingerprint,
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
            )
    with pytest.raises(RobotwinPreflightError, match="source-staging-forbidden"):
        validate_control_plane_source(
            authorization,
            source_uri=source,
            source_origin="environment",
            local_fingerprint=fingerprint,
            source_staging_requested=True,
        )


def test_phase_a_refuses_self_certified_decisions_and_unproven_source_bytes() -> None:
    private_context = b'{"license_acceptance":{"self_certified":true}}'
    with pytest.raises(
        RobotwinPreflightError, match="manager-runtime-use-receipt-unavailable"
    ):
        _REAL_RUNTIME_USE_RECEIPT_GATE(private_context)
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
        ({"bootstrap_image": "docker.io/public/example:latest"}, "bootstrap-image-not-immutable"),
        ({"bootstrap_image": "private-namespace/example"}, "bootstrap-image-not-immutable"),
        ({"output_root": "s3://other-bucket/output"}, "output-root-bucket"),
        ({"run_id": "unscoped-run"}, "run-id-invalid"),
    ],
)
def test_context_refuses_invalid_types_identity_and_destination(
    tmp_path: Path, updates: dict[str, object], category: str
) -> None:
    raw = json.dumps(_context_payload(tmp_path, **updates), sort_keys=True).encode()
    with pytest.raises(RobotwinPreflightError, match=category):
        validate_context_bytes(raw)


def test_each_runtime_use_decision_is_independently_required(tmp_path: Path) -> None:
    for decision in (
        "nvidia_cuda_eula",
        "nvidia_cudnn_sla",
        "curobo_noncommercial_research_or_evaluation",
        "robotwin2_aggregate_asset_and_output_terms",
    ):
        decisions = dict(_context_payload(tmp_path)["license_acceptance"])
        decisions[decision] = False
        raw = json.dumps(
            _context_payload(tmp_path, license_acceptance=decisions), sort_keys=True
        ).encode()
        with pytest.raises(RobotwinPreflightError, match=decision):
            validate_context_bytes(raw)


def test_portable_configs_reject_external_files_and_all_exec_plugins(
    tmp_path: Path,
) -> None:
    payload = _context_payload(tmp_path)
    Path(str(payload["skypilot_config_path"])).write_text(
        "include: /private/other.yaml\n", encoding="utf-8"
    )
    with pytest.raises(RobotwinPreflightError, match="external-reference"):
        validate_context_bytes(json.dumps(payload, sort_keys=True).encode())

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
        validate_context_bytes(json.dumps(payload, sort_keys=True).encode())

    payload = _context_payload(tmp_path)
    Path(str(payload["skypilot_config_path"])).write_text(
        "kubernetes:\n  allowed_contexts: [another-context]\n",
        encoding="utf-8",
    )
    with pytest.raises(RobotwinPreflightError, match="context-mismatch"):
        validate_context_bytes(json.dumps(payload, sort_keys=True).encode())

    payload = _context_payload(tmp_path)
    Path(str(payload["kubeconfig"])).write_text(
        Path(str(payload["kubeconfig"])).read_text(encoding="utf-8").replace(
            "token: portable-test-token", "tokenFile: relative-token"
        ),
        encoding="utf-8",
    )
    with pytest.raises(RobotwinPreflightError, match="user-not-portable"):
        validate_context_bytes(json.dumps(payload, sort_keys=True).encode())

    payload = _context_payload(tmp_path)
    Path(str(payload["kubeconfig"])).write_text(
        Path(str(payload["kubeconfig"])).read_text(encoding="utf-8").replace(
            "token: portable-test-token",
            "token: portable-test-token\n      tokenFile: relative-token",
        ),
        encoding="utf-8",
    )
    with pytest.raises(RobotwinPreflightError, match="user-not-portable"):
        validate_context_bytes(json.dumps(payload, sort_keys=True).encode())


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
        validate_context_bytes(json.dumps(payload, sort_keys=True).encode())


def test_transport_round_trip_is_digest_bound_and_repr_redacted(tmp_path: Path) -> None:
    path, raw = _context_file(tmp_path)
    authorization = load_runtime_authorization({PUBLIC_CONTEXT_ENV: str(path)})
    transport = encode_transport(authorization)
    decoded = decode_transport(transport)

    assert decoded.raw_context == raw
    assert decoded.context_sha256 == hashlib.sha256(raw).hexdigest()
    assert decoded.kubeconfig_bytes == authorization.kubeconfig_bytes
    assert "robotwin-project-canary" not in repr(decoded)
    assert raw.decode() not in repr(decoded)
    assert "portable-test-token" in decoded.redactions

    payload = json.loads(transport)
    payload["files"]["kubeconfig"]["sha256"] = "0" * 64
    with pytest.raises(RobotwinPreflightError, match="digest-mismatch"):
        decode_transport(json.dumps(payload))

    payload = json.loads(transport)
    payload["files"]["kubeconfig"] = {
        "base64": "",
        "sha256": hashlib.sha256(b"").hexdigest(),
    }
    with pytest.raises(RobotwinPreflightError, match="kubeconfig-invalid"):
        decode_transport(json.dumps(payload))

    with pytest.raises(RobotwinPreflightError, match="transport-size-invalid"):
        decode_transport("x" * (MAX_TRANSPORT_BYTES + 1))


def test_worker_materialization_preserves_bytes_modes_and_paths(tmp_path: Path) -> None:
    path, raw = _context_file(tmp_path)
    authorization = load_runtime_authorization({PUBLIC_CONTEXT_ENV: str(path)})
    directory = tmp_path / "materialized"
    directory.mkdir(mode=0o755)

    result = materialize_transport(encode_transport(authorization), directory)

    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert result.context_path.read_bytes() == raw
    assert result.kubeconfig_path.read_bytes() == authorization.kubeconfig_bytes
    assert result.skypilot_config_path.read_bytes() == authorization.skypilot_config_bytes
    for child in directory.iterdir():
        assert child.is_file() and not child.is_symlink()
        assert stat.S_IMODE(child.stat().st_mode) == 0o600


def test_inner_submit_reuses_the_validated_authorization_without_a_consent_proxy(
    tmp_path: Path,
) -> None:
    context_path, _ = _context_file(tmp_path)
    authorization = load_runtime_authorization({PUBLIC_CONTEXT_ENV: str(context_path)})
    image = authorization.bootstrap_image
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
        CHILD_RUNTIME_AUTH_ENV: encode_runtime_authorization(authorization),
        "NPA_BYOF_ROBOTWIN_RESERVATION_EVIDENCE_SHA256": (
            authorization.context_sha256
        ),
        "NPA_BYOF_ROBOTWIN_IMAGE_SCAN_SHA256": "b" * 64,
        "NPA_BYOF_ROBOTWIN_IMAGE_SCAN_ARCHIVES": "2",
        "AWS_ENDPOINT_URL": "https://storage.eu-north1.nebius.cloud",
        "NEBIUS_S3_ENDPOINT": "https://storage.eu-north1.nebius.cloud",
    }

    context = prepare_inner_submit(authorization, environment)

    assert context.layer == "inner"
    assert context.transport_value == ""
    assert image not in repr(context)
    with pytest.raises(RobotwinPreflightError, match="inner-environment-mismatch"):
        prepare_inner_submit(
            authorization,
            {**environment, CHILD_OUTPUT_PREFIX_ENV: "s3://wrong/output/"},
        )


def test_live_submit_requires_public_secret_name_and_binds_coordinates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.robotwin_preflight.RUNTIME_LOCK_STATUS",
        "complete",
    )
    context_path, _ = _context_file(tmp_path)
    spec = load_spec(ROBOTWIN_SPEC)
    environment = {PUBLIC_CONTEXT_ENV: str(context_path)}

    with pytest.raises(RobotwinPreflightError, match="secret-not-requested"):
        prepare_live_submit(spec, requested_secret_envs=(), environ=environment)
    context = prepare_live_submit(
        spec,
        requested_secret_envs=(PUBLIC_CONTEXT_ENV,),
        environ=environment,
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


def test_live_submit_refuses_incomplete_lock_after_context_validation(
    tmp_path: Path,
) -> None:
    context_path, _ = _context_file(tmp_path)
    with pytest.raises(RobotwinPreflightError, match="runtime-lock-incomplete"):
        prepare_live_submit(
            load_spec(ROBOTWIN_SPEC),
            requested_secret_envs=(PUBLIC_CONTEXT_ENV,),
            environ={PUBLIC_CONTEXT_ENV: str(context_path)},
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
            validate_context_bytes(raw)
        assert_clean(
            caught.value,
            str(private_path),
            private_document,
            "robotwin-project-canary",
            "portable-test-token",
        )

    private_transport = "{private-transport-document-canary"
    with pytest.raises(RobotwinPreflightError) as caught:
        decode_transport(private_transport)
    assert_clean(caught.value, private_transport)
