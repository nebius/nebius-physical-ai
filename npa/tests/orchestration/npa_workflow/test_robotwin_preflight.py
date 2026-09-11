"""Unit contracts for fail-closed RoboTwin submit authorization."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import stat

import pytest

from npa.orchestration.npa_workflow.robotwin_preflight import (
    MAX_CONTEXT_BYTES,
    MAX_TRANSPORT_BYTES,
    PUBLIC_CONTEXT_ENV,
    RobotwinPreflightError,
    bind_submit_coordinates,
    decode_transport,
    encode_transport,
    load_runtime_authorization,
    materialize_transport,
    prepare_live_submit,
    read_owner_context,
    recognize_contract,
    validate_context_bytes,
)
from npa.orchestration.npa_workflow.spec import load_spec


ROOT = Path(__file__).resolve().parents[4]
ROBOTWIN_SPEC = ROOT / "workflows" / "testing" / "byof-robotwin.yaml"


def _config_files(tmp_path: Path, context_name: str = "robotwin-context") -> tuple[Path, Path]:
    kubeconfig = tmp_path / "kubeconfig.yaml"
    kubeconfig.write_text(
        "\n".join(
            (
                "apiVersion: v1",
                "kind: Config",
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
        "registry": "registry.example/robotwin-private",
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
        ({"registry": "docker.io/public/example"}, "registry-not-private"),
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


def test_live_submit_requires_public_secret_name_and_binds_coordinates(
    tmp_path: Path,
) -> None:
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
