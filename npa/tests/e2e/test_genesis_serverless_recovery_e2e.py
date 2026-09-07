"""Real training with a deliberately lost create response and a fresh client.

Opt in with NPA_E2E_GENESIS_RECOVERY_CONFIG pointing to an owner-only JSON file:
argv (the complete Genesis serverless train-teacher CLI arguments), project_id,
image, source_revision (the changed client's commit), output_uri, and evidence_dir.
The payload image must be pinned by digest. Configure scope, worker credentials and
capacity isolation through the supported NPA runtime configuration before this
test. This test does not provision capacity or alter quotas. Evidence is private;
pytest output never contains the CLI result or provider resource identifiers.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlparse

import pytest
from typer.testing import CliRunner


def _resume_validation_record(path: Path, initial: dict) -> dict:
    if not path.exists():
        return initial
    if path.stat().st_mode & 0o077:
        raise ValueError("existing recovery evidence must be owner-only")
    saved = json.loads(path.read_text())
    keys = ("schema_version", "source_revision", "source_modules", "image_digest", "project_id", "request_sha256")
    if not isinstance(saved, dict) or any(saved.get(key) != initial[key] for key in keys):
        raise ValueError("existing recovery evidence belongs to a different source or workload; preserve it")
    return saved


def _capture_cli_output(path: Path, content: str, secrets: tuple[str, ...]) -> None:
    from npa.verification import sanitize_reason

    for value in secrets:
        if value:
            content = content.replace(value, "<redacted>")
    content = "\n".join(sanitize_reason(line, limit=max(1, len(line) * 2)) for line in content.splitlines())
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(content + "\n")


@pytest.mark.e2e_serverless
@pytest.mark.public_inputs
def test_genesis_serverless_lost_response_and_fresh_client(monkeypatch):
    config_path = os.environ.get("NPA_E2E_GENESIS_RECOVERY_CONFIG", "")
    if not config_path:
        pytest.skip("requires an explicitly configured real Genesis recovery workload")
    config_file = Path(config_path)
    assert config_file.stat().st_mode & 0o077 == 0, "live config must be owner-only"
    config = json.loads(config_file.read_text())
    arguments = config["argv"]
    assert arguments[:3] == ["workbench", "genesis", "train-teacher"]
    assert "--submit-only" not in arguments
    assert "--job-name" in arguments
    assert arguments[arguments.index("--runtime") + 1] == "serverless"
    assert arguments[arguments.index("--output-format") + 1] == "json"
    assert arguments[arguments.index("--image") + 1] == config["image"]
    assert arguments[arguments.index("--output-path") + 1].rstrip("/") == config["output_uri"].rstrip("/")
    assert "@sha256:" in config["image"], "real payload image must be immutable"

    from npa.cli.main import app
    from npa.clients.config import default_project_name
    from npa.clients.serverless import ServerlessClient
    from npa.clients.storage import StorageClient
    from npa.orchestration.skypilot.registry_preflight import fetch_image_config_metadata
    from npa.orchestration.npa_workflow.submit_credentials import resolve_submit_credentials

    digest, labels = fetch_image_config_metadata(config["image"])
    assert config["image"].endswith("@" + digest)
    if config.get("image_source_revision"):
        assert labels.get("org.opencontainers.image.revision") == config["image_source_revision"]
    evidence_dir = Path(config["evidence_dir"])
    evidence_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    evidence_dir.chmod(0o700)
    record = {
        "schema_version": "npa.serverless.recovery-validation.v1",
        "completed": False,
        "source_revision": config["source_revision"],
        "image_source_revision": labels.get("org.opencontainers.image.revision", ""),
        "image_digest": digest,
        "source_modules": {},
        "create_calls": 0,
        "controlled_response_loss": False,
        "project_id": config["project_id"],
        "request_sha256": hashlib.sha256(json.dumps(arguments, separators=(",", ":")).encode()).hexdigest(),
    }
    for name in (
        "npa.clients.serverless", "npa.serverless_common.launch",
        "npa.serverless_common.supervision", "npa.orchestration.npa_workflow.supervisor",
        "npa.cli.genesis",
    ):
        source = Path(inspect.getfile(importlib.import_module(name)))
        record["source_modules"][name] = hashlib.sha256(source.read_bytes()).hexdigest()
    checkout = Path(inspect.getfile(importlib.import_module("npa.cli.genesis"))).resolve().parents[5]
    actual_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, text=True, capture_output=True, check=True).stdout.strip()
    assert actual_revision == config["source_revision"], "client source revision differs from requested validation"
    record = _resume_validation_record(evidence_dir / "recovery-validation.json", record)
    storage = resolve_submit_credentials(
        project=config.get("project") or default_project_name(), requested=("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
    )
    secrets = tuple(value for value in (
        storage.access_key_id, storage.secret_access_key, *storage.secret_values.values(),
        *(value for key, value in os.environ.items() if any(marker in key.upper() for marker in ("TOKEN", "SECRET", "PASSWORD", "ACCESS_KEY"))),
    ) if value)

    def save():
        temporary = evidence_dir / ".recovery-validation.json.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(evidence_dir / "recovery-validation.json")

    def invoke(phase):
        invocations = record.setdefault("cli_invocations", [])
        entry = {"phase": phase, "sequence": len(invocations) + 1, "completed": False}
        invocations.append(entry)
        save()
        result = CliRunner().invoke(app, arguments)
        for stream in ("stdout", "stderr"):
            path = evidence_dir / f"cli-{entry['sequence']}-{phase}-{stream}.txt"
            _capture_cli_output(path, getattr(result, stream), secrets)
            entry[stream + "_path"] = str(path)
        entry.update(exit_code=result.exit_code, completed=True)
        save()
        return result

    def lose_create_response(args, **kwargs):
        is_create = any(args[index:index + 3] == ["ai", "job", "create"] for index in range(len(args) - 2))
        if is_create:
            record["create_calls"] += 1
            save()
        try:
            response = subprocess.run(args, **kwargs)
        except subprocess.TimeoutExpired:
            if is_create:
                record["create_response_timeout"] = True
                save()
            raise
        if is_create and response.returncode == 0:
            # The real backend accepted the job. Drop only the client response:
            # the changed create-timeout path must recover it by exact identity.
            record["controlled_response_loss"] = True
            save()
            raise subprocess.TimeoutExpired(["nebius", "ai", "job", "create"], kwargs["timeout"])
        return response

    monkeypatch.setattr("npa.cli.genesis.ServerlessClient", lambda: ServerlessClient(subprocess_runner=lose_create_response))
    save()
    first = invoke("initial-or-resumed")
    record.setdefault("first_cli_exit_code", first.exit_code)
    save()
    assert first.exit_code == 0, "real Genesis CLI failed; inspect private provider and supervisor evidence"
    first_payload = json.loads(first.stdout)
    if record.get("job_id") and record["job_id"] != first_payload["job_id"]:
        pytest.fail("reconnected CLI returned a different backend identity", pytrace=False)
    record["job_id"] = first_payload["job_id"]
    record["job_name"] = first_payload["job_name"]
    save()
    assert record["create_calls"] == 1
    assert record["controlled_response_loss"] or record.get("create_response_timeout"), "no create-response loss was exercised"
    assert first_payload.get("job_status", first_payload["status"]) == "succeeded"

    # A second CLI invocation constructs a new client and reads durable state;
    # it must reuse the completed workload rather than repeat training.
    reconnected = invoke("fresh-client")
    record["reconnect_cli_exit_code"] = reconnected.exit_code
    save()
    assert reconnected.exit_code == 0, "fresh-client recovery failed"
    second_payload = json.loads(reconnected.stdout)
    assert second_payload["job_id"] == first_payload["job_id"]
    assert second_payload["job_status"] == "succeeded"
    assert record["create_calls"] == 1

    # Independent provider observation bypasses the intentionally lossy client.
    job = ServerlessClient().get_job(first_payload["job_id"], config["project_id"])
    assert job.id == first_payload["job_id"] and job.status == "succeeded"
    record["provider_state"] = job.provider_state
    record["compute_instance_ids"] = [
        instance["compute_instance_id"] for instance in job.raw.get("status", {}).get("instances", [])
        if instance.get("compute_instance_id")
    ]
    assert record["compute_instance_ids"], "provider must expose actual allocation identities"

    import torch

    s3 = StorageClient.from_environment(
        endpoint_url=storage.endpoint_url,
        aws_access_key_id=storage.access_key_id,
        aws_secret_access_key=storage.secret_access_key,
    ).s3
    parsed = urlparse(config["output_uri"])
    prefix = parsed.path.lstrip("/").rstrip("/")
    record["artifacts"] = []
    for name in ("model.pt", "train_teacher_summary.json", "npa_genesis_checkpoint_manifest.json"):
        response = s3.get_object(Bucket=parsed.netloc, Key=f"{prefix}/{name}")
        body = response["Body"].read()
        assert body and len(body) == response["ContentLength"]
        if name == "model.pt":
            checkpoint = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
            assert isinstance(checkpoint, dict) and checkpoint.get("model_state_dict")
            detail = {"checkpoint_keys": sorted(checkpoint)}
        else:
            summary = json.loads(body)
            assert isinstance(summary, dict)
            detail = {"status": summary.get("status"), "max_iterations": summary.get("max_iterations"), "n_envs": summary.get("n_envs")}
        record["artifacts"].append({
            "role": "checkpoint" if name == "model.pt" else "training_summary",
            "name": name, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
            **detail,
        })
    record["completed"] = True
    record["same_backend_identity"] = True
    save()


def test_recovery_evidence_resume_preserves_prior_create_and_completion(tmp_path):
    initial = {
        "schema_version": "unit", "source_revision": "source", "source_modules": {"module": "hash"},
        "image_digest": "image", "project_id": "project-unit", "request_sha256": "request",
        "create_calls": 0, "controlled_response_loss": False, "completed": False,
    }
    saved = {**initial, "create_calls": 1, "controlled_response_loss": True, "completed": True, "job_id": "provider-unit"}
    path = tmp_path / "recovery.json"
    path.write_text(json.dumps(saved))
    path.chmod(0o600)
    assert _resume_validation_record(path, initial) == saved
    with pytest.raises(ValueError, match="different source or workload"):
        _resume_validation_record(path, {**initial, "request_sha256": "different"})
    assert json.loads(path.read_text()) == saved


def test_private_cli_capture_redacts_secrets_and_does_not_overwrite(tmp_path):
    path = tmp_path / "cli.txt"
    _capture_cli_output(path, "provider-unit\npassword=other-private-value\nunit-secret", ("unit-secret",))
    captured = path.read_text()
    assert "provider-unit" in captured
    assert "unit-secret" not in captured
    assert "other-private-value" not in captured
    assert path.stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        _capture_cli_output(path, "replacement", ())


def test_public_inputs_still_require_live_opt_in_but_not_unrelated_hf_entitlement(monkeypatch):
    import importlib.util
    from types import SimpleNamespace

    module_spec = importlib.util.spec_from_file_location("runtime_e2e_gates", Path(__file__).with_name("conftest.py"))
    gates = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(gates)
    monkeypatch.setattr(gates, "_hf_token_configured", lambda: False)

    def item(public):
        marks = []
        return SimpleNamespace(
            get_closest_marker=lambda name: name == "e2e_serverless" or (public and name == "public_inputs"),
            add_marker=marks.append, marks=marks,
        )

    public, gated = item(True), item(False)
    monkeypatch.setenv("NPA_INTEGRATION_E2E", "1")
    gates.pytest_collection_modifyitems(None, [public, gated])
    assert not public.marks
    assert len(gated.marks) == 1
    monkeypatch.delenv("NPA_INTEGRATION_E2E")
    gates.pytest_collection_modifyitems(None, [public])
    assert len(public.marks) == 1
