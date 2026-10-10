"""Verify native GPU training logs, checkpoints, and their validity classifications."""

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pytest

from npa.clients.project_credential_store import project_credential_record
from npa.workbench.isaac_lab import training_validity

pytestmark = pytest.mark.e2e


def _storage_client(project_id):
    record = project_credential_record(project_id, migrate_legacy=False)
    assert record and record["project_id"] == project_id
    storage = record["storage"]
    bucket = urlparse(storage["bucket"]).netloc or storage["bucket"].strip("/")
    client = boto3.client(
        "s3",
        endpoint_url=storage["endpoint_url"],
        aws_access_key_id=storage["aws_access_key_id"],
        aws_secret_access_key=storage["aws_secret_access_key"],
    )
    return client, bucket


def _read(client, bucket, artifact):
    parsed = urlparse(artifact["uri"])
    assert parsed.scheme == "s3" and parsed.netloc == bucket
    with client.get_object(Bucket=bucket, Key=parsed.path.lstrip("/"))["Body"] as body:
        payload = body.read()
    assert payload
    if "sha256" in artifact:
        assert hashlib.sha256(payload).hexdigest() == artifact["sha256"]
    return payload


def test_current_validator_classifies_native_zero_exit_runs(tmp_path):
    config_path = os.environ.get("NPA_ISAAC_TRAINING_VALIDITY_VERIFY_CONFIG")
    if not config_path:
        pytest.skip("supply owner-private native training evidence references")
    config = json.loads(Path(config_path).read_text())
    client, bucket = _storage_client(config["project_id"])
    proof = json.loads(_read(client, bucket, {"uri": config["provenance_uri"]}))
    source_hash = hashlib.sha256(
        Path(training_validity.__file__).read_bytes()
    ).hexdigest()
    assert proof["validator_sha256"] == source_hash
    assert {case["physics_valid"] for case in proof["cases"]} == {False, True}
    for index, case in enumerate(proof["cases"]):
        completion = json.loads(_read(client, bucket, case["completion"]))
        assert completion["returncode"] == 0
        _read(client, bucket, case["checkpoint"])
        log = tmp_path / f"case-{index}.log"
        log.write_bytes(_read(client, bucket, case["log"]))
        assert training_validity.inspect_training_log(log) == {
            "physics_valid": case["physics_valid"],
            "physics_error_count": case["physics_error_count"],
        }
