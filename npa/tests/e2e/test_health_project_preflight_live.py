"""Validate selected-project health against real S3 with private config copies.

Run with NPA_INTEGRATION_E2E=1 and NPA_E2E_PROJECT set to an explicitly
configured test project. Only the prefix test writes objects; it removes its
unique objects in a finally block. No cloud identity or credential is printed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid

import pytest
import yaml

from npa.clients.config import CONFIG_PATH, resolve_project_storage
from npa.clients.credentials import CREDENTIALS_PATH
from npa.clients.storage import StorageClient
from npa.lifecycle_intent import OperationIntent, operation_intent

pytestmark = pytest.mark.e2e


@pytest.fixture
def live_project(tmp_path):
    project = os.environ.get("NPA_E2E_PROJECT", "").strip()
    if not project:
        pytest.skip("Set NPA_E2E_PROJECT to an explicitly configured test project")
    with operation_intent(OperationIntent.OBSERVE):
        storage = resolve_project_storage(
            project, include_shared_credentials=False, include_environment=False,
        )
    if not all(
        (
            storage.checkpoint_bucket,
            storage.endpoint_url,
            storage.aws_access_key_id,
            storage.aws_secret_access_key,
        )
    ):
        pytest.fail("Selected live project needs complete object storage configuration")
    directory = tmp_path / "config"
    directory.mkdir(mode=0o700)
    for source in (CONFIG_PATH, CREDENTIALS_PATH):
        destination = directory / source.name
        destination.write_bytes(source.read_bytes())
        destination.chmod(0o600)
    return project, storage, directory


def _preflight(project, directory, **overrides):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npa",
            "workbench",
            "health",
            "preflight",
            "--checks",
            "s3",
            "--json",
            *(["--project", project] if project else []),
        ],
        env={**os.environ, "NPA_CONFIG_DIR": str(directory), **overrides},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        pytest.fail("Storage preflight did not return one JSON document")
    statuses = [check["status"] for check in payload.get("checks", [])]
    return result.returncode, payload.get("ok"), statuses


def test_live_project_overrides_unrelated_host_storage_without_changing_config(
    live_project,
):
    project, _storage, directory = live_project
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    overrides = {
        "AWS_ACCESS_KEY_ID": "synthetic-invalid-access",
        "AWS_SECRET_ACCESS_KEY": "synthetic-invalid-secret",
        "NEBIUS_S3_BUCKET": "s3://synthetic-unrelated-bucket",
        "NPA_CHECKPOINT_BUCKET": "s3://synthetic-unrelated-bucket",
    }
    assert _preflight(None, directory, **overrides) == (1, False, ["FAIL"])
    assert _preflight(project, directory, **overrides) == (0, True, ["PASS"])
    unchanged = before == {path.name: path.read_bytes() for path in directory.iterdir()}
    assert unchanged, "Readiness checks changed the private configuration"


def test_live_unknown_project_does_not_report_host_readiness(live_project):
    _project, _storage, directory = live_project
    assert _preflight("missing-" + uuid.uuid4().hex, directory) == (1, False, ["FAIL"])


def test_live_bad_project_signature_is_a_failure(live_project):
    project, _storage, directory = live_project
    config = yaml.safe_load((directory / "config.yaml").read_text())
    project_id = config["projects"][project]["project_id"]
    path = directory / "credentials.yaml"
    document = yaml.safe_load(path.read_text())
    record = document["project_credentials"]["projects"][project_id]["storage"]
    record["aws_secret_access_key"] = "synthetic-invalid-signing-key"
    path.write_text(yaml.safe_dump(document))
    assert _preflight(project, directory) == (1, False, ["FAIL"])


@pytest.mark.parametrize("missing", [
    "bucket", "endpoint_url", "aws_access_key_id", "aws_secret_access_key", "record",
])
def test_live_incomplete_project_does_not_borrow_valid_host_credentials(live_project, missing):
    project, storage, directory = live_project
    overrides = {
        "AWS_ACCESS_KEY_ID": storage.aws_access_key_id,
        "AWS_SECRET_ACCESS_KEY": storage.aws_secret_access_key,
        "AWS_ENDPOINT_URL": storage.endpoint_url,
        "NPA_CHECKPOINT_BUCKET": "s3://" + storage.checkpoint_bucket.removeprefix("s3://"),
    }
    config_path = directory / "config.yaml"
    document = yaml.safe_load(config_path.read_text())
    project_config = document["projects"][project]
    project_id = project_config["project_id"]
    for section in ("object-storage", "object_storage", "storage", "terraform_state"):
        project_config.pop(section, None)
    config_path.write_text(yaml.safe_dump(document))
    path = directory / "credentials.yaml"
    document = yaml.safe_load(path.read_text())
    projects = document["project_credentials"]["projects"]
    if missing == "record":
        projects.pop(project_id)
    else:
        projects[project_id]["storage"].pop(missing)
    path.write_text(yaml.safe_dump(document))
    before = {item.name: item.read_bytes() for item in directory.iterdir()}
    assert _preflight(None, directory, **overrides) == (0, True, ["PASS"])
    assert _preflight(project, directory, **overrides) == (1, False, ["FAIL"])
    assert before == {item.name: item.read_bytes() for item in directory.iterdir()}


def test_live_probe_checks_empty_and_populated_prefix_with_one_request(live_project):
    _project, storage, _directory = live_project
    client = StorageClient(
        endpoint_url=storage.endpoint_url,
        aws_access_key_id=storage.aws_access_key_id,
        aws_secret_access_key=storage.aws_secret_access_key,
    )
    bucket = storage.checkpoint_bucket.removeprefix("s3://").split("/", 1)[0]
    prefix = "health-preflight-test/" + uuid.uuid4().hex + "/"
    keys = [prefix + "first", prefix + "second"]
    requests = []
    truncated = []
    client.s3.meta.events.register(
        "before-parameter-build.s3.ListObjectsV2",
        lambda params, **_: requests.append(dict(params)),
    )
    client.s3.meta.events.register(
        "after-call.s3.ListObjectsV2",
        lambda parsed, **_: truncated.append(parsed.get("IsTruncated", False)),
    )
    try:
        client.probe_list_access(f"s3://{bucket}/{prefix}")
        for key in keys:
            client.s3.put_object(
                Bucket=bucket, Key=key, Body=b"health preflight fixture"
            )
        client.probe_list_access(f"s3://{bucket}/{prefix}")
        assert len(requests) == 2
        assert truncated == [False, True]
        assert all(request.get("MaxKeys") == 1 for request in requests)
        assert all("ContinuationToken" not in request for request in requests)
    finally:
        for key in keys:
            client.s3.delete_object(Bucket=bucket, Key=key)
