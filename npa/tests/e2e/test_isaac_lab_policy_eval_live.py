"""Read back a real checkpoint evaluation produced by the current evaluator."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pytest

from npa.cli.isaac_lab import eval_runner
from npa.clients.project_credential_store import project_credential_record

pytestmark = pytest.mark.e2e


def test_current_evaluator_loaded_the_retained_checkpoint() -> None:
    if not os.environ.get("NPA_ISAAC_EVAL_VERIFY_CONFIG"):
        pytest.skip("supply owner-private project and evaluation artifact references")
    config = json.loads(Path(os.environ["NPA_ISAAC_EVAL_VERIFY_CONFIG"]).read_text())
    record = project_credential_record(config["project_id"], migrate_legacy=False)
    assert record and record["project_id"] == config["project_id"]
    storage = record["storage"]
    bucket = urlparse(storage["bucket"]).netloc or storage["bucket"].strip("/")
    client = boto3.client(
        "s3",
        endpoint_url=storage["endpoint_url"],
        aws_access_key_id=storage["aws_access_key_id"],
        aws_secret_access_key=storage["aws_secret_access_key"],
    )

    def read(uri: str) -> bytes:
        parsed = urlparse(uri)
        assert parsed.scheme == "s3" and parsed.netloc == bucket
        with client.get_object(Bucket=bucket, Key=parsed.path.lstrip("/"))[
            "Body"
        ] as body:
            payload = body.read()
        assert payload
        return payload

    proof = json.loads(read(config["provenance_uri"]))
    source_sha = hashlib.sha256(Path(eval_runner.__file__).read_bytes()).hexdigest()
    assert proof["evaluator_sha256"] == source_sha
    report_bytes = read(proof["report_uri"])
    assert hashlib.sha256(report_bytes).hexdigest() == proof["report_sha256"]
    assert (
        hashlib.sha256(read(proof["checkpoint_uri"])).hexdigest()
        == proof["checkpoint_sha256"]
    )
    report = json.loads(report_bytes)
    assert report["format"] == eval_runner.EVAL_FORMAT
    assert report["status"] == "success" and report["policy_loaded"] is True
    assert report["num_episodes"] == len(report["episodes"]) > 0
    assert report["task"] == config["task"] and report["seed"] == config["seed"]
    assert report["num_episodes"] == config["num_episodes"]
    assert report["success_metric"] == config["success_metric"]
    assert all(episode["steps"] > 0 for episode in report["episodes"])
