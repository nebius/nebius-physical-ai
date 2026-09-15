"""Actual boto3 profile resolution with all S3 requests prohibited."""

import os

import pytest
from botocore.client import BaseClient

from npa.orchestration.skypilot.storage_preflight import (
    _fingerprint, _profile_probe, nebius_mount_destinations,
)


@pytest.fixture
def profile(monkeypatch, tmp_path):
    for name in tuple(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    aws = tmp_path / ".aws"
    aws.mkdir()
    (aws / "credentials").write_text("[nebius]\naws_access_key_id = selected-access\naws_secret_access_key = selected-secret\n")
    (aws / "config").write_text("[profile nebius]\nregion = eu-west1\nendpoint_url = https://storage.eu-west1.nebius.cloud\n")
    calls = []

    def no_request(*args, **kwargs):
        calls.append(args)
        raise AssertionError("preflight must not make S3 requests")

    monkeypatch.setattr(BaseClient, "_make_api_call", no_request)
    yield aws
    assert not calls


def request():
    return {"fingerprint": _fingerprint("selected-access", "selected-secret"),
            "endpoint": "https://storage.eu-west1.nebius.cloud"}


def test_real_named_profile_matches_selected_pair_and_endpoint(profile, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "unrelated-environment-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "unrelated-environment-secret")
    assert _profile_probe(request()) == {"status": "pass"}


def test_real_named_profile_ignores_matching_environment_keys_when_saved_pair_differs(profile, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "selected-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "selected-secret")
    (profile / "credentials").write_text("[nebius]\naws_access_key_id = other-access\naws_secret_access_key = other-secret\n")
    assert _profile_probe(request()) == {"status": "fail", "reason": "principal"}


def test_actual_s3_client_endpoint_must_match(profile):
    (profile / "config").write_text("[profile nebius]\nregion = eu-north1\nendpoint_url = https://storage.eu-north1.nebius.cloud\n")
    assert _profile_probe(request()) == {"status": "fail", "reason": "endpoint"}


def test_actual_s3_client_endpoint_environment_precedence(profile, monkeypatch):
    (profile / "config").write_text("[profile nebius]\nendpoint_url = https://storage.eu-north1.nebius.cloud\n")
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", request()["endpoint"])
    assert _profile_probe(request()) == {"status": "pass"}


@pytest.mark.parametrize("variable,name", [("AWS_SHARED_CREDENTIALS_FILE", "credentials"), ("AWS_CONFIG_FILE", "config")])
def test_nondefault_credential_files_cannot_be_propagated_to_controller(profile, monkeypatch, variable, name):
    override = profile.parent / ("custom-" + name)
    override.write_bytes((profile / name).read_bytes())
    monkeypatch.setenv(variable, str(override))
    assert _profile_probe(request()) == {"status": "fail", "reason": "file_override"}


@pytest.mark.parametrize("variable,name", [("AWS_SHARED_CREDENTIALS_FILE", "credentials"), ("AWS_CONFIG_FILE", "config")])
def test_explicit_default_credential_paths_preserve_propagation(profile, monkeypatch, variable, name):
    monkeypatch.setenv(variable, str(profile / name))
    assert _profile_probe(request()) == {"status": "pass"}


@pytest.mark.parametrize("missing", ["credentials", "config"])
def test_missing_profile_is_unknown(profile, missing):
    (profile / missing).unlink()
    assert _profile_probe(request()) == {"status": "unknown", "reason": "missing_profile"}


def test_dynamic_profile_is_rejected_without_executing_credential_process(profile):
    with (profile / "config").open("a") as handle:
        handle.write("credential_process = deliberately-invalid-command\n")
    assert _profile_probe(request()) == {"status": "unknown", "reason": "dynamic_profile"}


def test_session_token_profile_cannot_change_signing_identity(profile):
    with (profile / "credentials").open("a") as handle:
        handle.write("aws_session_token = unexpected-session\n")
    assert _profile_probe(request()) == {"status": "fail", "reason": "principal"}


def test_raw_durable_mount_probes_its_instrumented_prefix():
    from npa.execution_preflight import skypilot_output_destinations
    from npa.orchestration.skypilot.workflow_state import WorkflowS3Config, _instrument_stage_doc

    document = {"run": "true"}
    _instrument_stage_doc(document, run_id="unit", stage="stage", manifest_json="{}", mount_path="/mnt/state",
                          state=WorkflowS3Config("unit-output", "task/run", request()["endpoint"]))
    assert nebius_mount_destinations([document]) == {"s3://unit-output/task/run": "directory"}
    assert skypilot_output_destinations([document]) == {"s3://unit-output/task/run": "directory"}


def test_raw_readonly_or_copy_mount_does_not_authorize_write_probes():
    docs = [{"file_mounts": {"/mnt/input": {"source": "nebius://public-input/data", **option}}}
            for option in ({"read_only": True}, {"mode": "COPY"})]
    docs.append({"file_mounts": {"/mnt/input": "nebius://public-input/data"}})
    assert nebius_mount_destinations(docs) == {}


@pytest.mark.parametrize("override", [
    {"envs": {"AWS_CONFIG_FILE": "/custom/config"}},
    {"resources": {"kubernetes": {"pod_config": {"spec": {"containers": [
        {"env": [{"name": "AWS_SHARED_CREDENTIALS_FILE", "valueFrom": {"secretKeyRef": {"name": "external", "key": "path"}}}]}]}}}}},
    {"file_mounts": {"~/.aws/credentials": "/custom/credentials"}},
])
def test_worker_profile_replacements_fail_before_profile_or_storage_access(monkeypatch, override):
    from unittest.mock import Mock
    from npa.execution_preflight import ExecutionPreflightError
    from npa.orchestration.skypilot.storage_preflight import verify_nebius_mount_principal

    probe = Mock()
    monkeypatch.setattr("npa.orchestration.skypilot.storage_preflight.subprocess.run", probe)
    docs = [{"file_mounts": {"/mnt/data": {"source": "nebius://unit-output/task", "store": "NEBIUS"}}}, override]
    with pytest.raises(ExecutionPreflightError, match="AWS"):
        verify_nebius_mount_principal(docs, target=None, sky_bin="", environment={}, cwd=None)
    probe.assert_not_called()
