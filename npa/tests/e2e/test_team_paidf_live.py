"""Audit completed team PAIDF batches through personal API access, without GPU submits."""

import io
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from npa.workbench.team.client import TeamClient, load_bearer_token
from npa.workbench.team.errors import TeamError

from .test_npa_workflow_submit_live_e2e import _assert_paidf_live_artifacts

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(0)]


@pytest.fixture
def completed_team_batch():
    path = os.environ.get("NPA_TEAM_PAIDF_LIVE_CONFIG", "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not path:
        pytest.skip("Select a private completed team PAIDF batch")
    selected = Path(path)
    assert selected.is_absolute() and not selected.stat().st_mode & 0o077
    configuration = json.loads(selected.read_text())
    assert len(configuration["accounts"]) >= 4
    return configuration


class _RunArtifacts:
    """Adapt canonical artifact checks to an authenticated, run-scoped API transport."""

    def __init__(self, client, record):
        self.client, self.run_id = client, record["id"]
        parsed = urlsplit(record["run_uri"])
        assert parsed.scheme == "s3" and parsed.netloc
        self.bucket, self.prefix = parsed.netloc, parsed.path.strip("/") + "/"
        self.index = client.run(self.run_id, "artifacts")["artifacts"]

    def _relative(self, bucket, key):
        assert bucket == self.bucket and key.startswith(self.prefix)
        return key[len(self.prefix) :]

    def get_object(self, *, Bucket, Key):
        body = io.BytesIO()
        self.client.download(self.run_id, self._relative(Bucket, Key), body)
        body.seek(0)
        return {"Body": body}

    def head_object(self, *, Bucket, Key):
        relative = self._relative(Bucket, Key)
        matches = [item for item in self.index if item["name"] == relative]
        assert len(matches) == 1, "Expected run artifact is absent"
        return {"ContentLength": matches[0]["size"]}

    def list_objects_v2(self, *, Bucket, Prefix):
        relative = self._relative(Bucket, Prefix)
        return {
            "Contents": [
                {"Key": self.prefix + item["name"], "Size": item["size"]}
                for item in self.index
                if item["name"].startswith(relative)
            ]
        }

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return self

    def paginate(self, **kwargs):
        yield self.list_objects_v2(**kwargs)

    def download_file(self, bucket, key, filename):
        with Path(filename).open("xb") as destination:
            self.client.download(self.run_id, self._relative(bucket, key), destination)


def _connect(configuration, account):
    return TeamClient(
        configuration["endpoint"],
        load_bearer_token("NPA_TEAM_TOKEN", Path(account["token_file"])),
        ca_file=configuration.get("ca_file"),
    )


def _read_report(objects, relative):
    with objects.get_object(Bucket=objects.bucket, Key=objects.prefix + relative)[
        "Body"
    ] as body:
        return json.load(body)


def _audit_run(client, record, monkeypatch):
    import npa.clients.project_credentials as credentials

    assert client.run(record["id"], "status")["status"] == "succeeded"
    objects = _RunArtifacts(client, record)
    monkeypatch.setattr(credentials, "s3_client_for_project", lambda *a, **k: objects)
    provenance = _read_report(objects, "input/provenance.json")
    assert provenance["source_kind"] == "lerobot_dataset"
    assert provenance["episode"] == record["episode"]
    assert provenance["camera"] == record["camera"]
    runtime = _read_report(objects, "npa-workflow/runtime.json")
    assert runtime["run_id"] == record["id"]
    assert runtime["status"] == "succeeded" and runtime["waves"]
    for wave in runtime["waves"]:
        assert wave["job_id"] and wave["job_name"]
        assert wave["status"] == "succeeded" and wave["sky_status"] == "SUCCEEDED"
        assert wave["replayed"] is False and wave["adopted"] is False
    _assert_paidf_live_artifacts(
        spec="paidf-cosmos3.yaml",
        run_prefix_uri=record["run_uri"],
        waves=runtime["waves"],
        bucket=objects.bucket,
        run_id=record["id"],
        e2e_project=None,
    )


def test_twenty_episode_batches_have_real_outputs_and_separate_owners(
    completed_team_batch, monkeypatch
):
    """Verify two groups' complete episodes, native outputs, and cross-user denial."""
    configuration = completed_team_batch
    groups, owners = {}, set()
    foreign_run = configuration["accounts"][1]["runs"][0]["id"]
    for index, account in enumerate(configuration["accounts"]):
        client = _connect(configuration, account)
        try:
            identity = client.whoami()
            assert identity["subject"] not in owners, (
                "Accounts must have distinct owners"
            )
            owners.add(identity["subject"])
            assert len(identity["groups"]) >= 1
            episodes = groups.setdefault(account["dataset"], set())
            if index == 0:
                with pytest.raises(TeamError):
                    client.run(foreign_run, "status")
            for record in account["runs"]:
                assert record["episode"] not in episodes, "Episode assigned twice"
                episodes.add(record["episode"])
                _audit_run(client, record, monkeypatch)
        finally:
            client.close()
    assert len(groups) >= 2, "Select ALOHA and another LeRobot dataset"
    assert all(episodes == set(range(20)) for episodes in groups.values())
