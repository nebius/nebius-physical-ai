"""Check native bundle containment and measured evaluation translation."""

from __future__ import annotations

import io
import hashlib
import json
import shutil
import tarfile

import pytest

from npa.workflows.field_failure.native_artifacts import _archive, _extract
from npa.workflows.field_failure.native_policy import _episode_metrics


@pytest.mark.parametrize(
    "name", ["../outside", "/outside", "a/../../outside", "a\\outside"]
)
def test_capture_archive_cannot_escape_its_directory(tmp_path, name):
    archive = tmp_path / "capture.tar"
    with tarfile.open(archive, "w") as stream:
        member = tarfile.TarInfo(name)
        member.size = 3
        stream.addfile(member, io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="contained regular"):
        _extract(archive, tmp_path / "decoded")
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize(
    "kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.DIRTYPE]
)
def test_capture_archive_rejects_non_regular_members(tmp_path, kind):
    archive = tmp_path / "capture.tar"
    with tarfile.open(archive, "w") as stream:
        member = tarfile.TarInfo("capture.json")
        member.type = kind
        member.linkname = "../outside"
        stream.addfile(member)
    with pytest.raises(ValueError, match="contained regular"):
        _extract(archive, tmp_path / "decoded")


def test_capture_archive_rejects_duplicate_member_names(tmp_path):
    archive = tmp_path / "capture.tar"
    with tarfile.open(archive, "w") as stream:
        for content in (b"first", b"later"):
            member = tarfile.TarInfo("capture.json")
            member.size = len(content)
            stream.addfile(member, io.BytesIO(content))
    with pytest.raises(ValueError, match="unique contained"):
        _extract(archive, tmp_path / "decoded")


def test_native_bundle_preserves_nested_measured_bytes(tmp_path):
    source = tmp_path / "source"
    (source / "depth").mkdir(parents=True)
    (source / "depth/frame.png").write_bytes(b"measured depth bytes")
    (source / "capture.json").write_text('{"calibration":"pinned"}')
    archive = tmp_path / "capture.tar"
    _archive(source, archive)
    target = tmp_path / "decoded"
    _extract(archive, target)
    assert (target / "depth/frame.png").read_bytes() == b"measured depth bytes"
    assert (target / "capture.json").read_bytes() == (
        source / "capture.json"
    ).read_bytes()


@pytest.mark.parametrize(
    "success,contacts,physical_failures,termination",
    [
        (True, 0, 0, "success"),
        (False, 4, 0, "failure"),
        (False, 0, 1, "failure"),
        (False, 0, 0, "timeout"),
    ],
)
def test_measured_episode_keeps_failures_in_the_comparison(
    success, contacts, physical_failures, termination
):
    scene = {"scenario_id": "held-room", "asset": {"sha256": "a" * 64}}
    row = {
        "seed": 29,
        "steps": 180,
        "success": success,
        "collision_steps": contacts,
        "peer_collision_steps": 0,
        "physical_failure_steps": physical_failures,
        "path_length_m": 5.25,
        "goal_distance_m": 0.2 if success else 1.4,
    }
    record = _episode_metrics(
        scene,
        row,
        [
            {"name": "success"},
            {"name": "path_length_m"},
            {"name": "physical_failure_steps"},
        ],
    )
    assert record["termination"] == termination
    assert record["status"] == "completed"
    assert record["metrics"] == {
        "success": float(success),
        "path_length_m": 5.25,
        "physical_failure_steps": float(physical_failures),
    }
    assert record["scene_sha256"] == "a" * 64
    assert record["seed"] == 29


def test_native_success_cannot_mask_measured_physical_failure():
    scene = {"scenario_id": "held-room", "asset": {"sha256": "a" * 64}}
    row = dict(
        seed=29,
        steps=2,
        success=True,
        collision_steps=0,
        peer_collision_steps=0,
        physical_failure_steps=1,
        path_length_m=1.0,
        goal_distance_m=0.0,
    )
    with pytest.raises(ValueError, match="success after physical failure"):
        _episode_metrics(scene, row, [{"name": "success"}])


def test_native_evaluation_keeps_rendered_evidence_after_worker_cleanup(
    tmp_path, monkeypatch
):
    from npa.workflows.field_failure import native_artifacts, native_policy

    uploaded = {}

    class Storage:
        def put_bytes_conditional(self, data, uri, *, if_none_match):
            assert if_none_match and uri not in uploaded
            uploaded[uri] = data

    monkeypatch.setattr(native_artifacts, "_storage", lambda: Storage())
    output = tmp_path / "native-output"
    (output / "rendered-rollout").mkdir(parents=True)
    # Deliberately synthetic bytes: this checks retention, not renderer validity.
    (output / "rendered-rollout/00001.png").write_bytes(b"synthetic frame")
    (output / "trajectory.json").write_text('[{"physical_failure":[1]}]')
    request = {
        "bundle_sha256": "a" * 64,
        "adapter": {"entrypoint": "example:evaluate"},
        "attempt_id": "unit-attempt",
        "inputs_sha256": "b" * 64,
        "output_prefix": "s3://example-bucket/unit-evaluation/",
        "policy": {"policy_id": "candidate", "checkpoint": {"sha256": "c" * 64}},
        "protocol": {"sha256": "d" * 64},
    }
    report = {"checkpoint_sha256": "c" * 64, "success_rate": 0.0}
    native_policy._retain_evaluation(
        request, {"scenario_id": "untouched-scene"}, output, report, tmp_path, 0
    )
    shutil.rmtree(output)
    prefix = request["output_prefix"]
    record = json.loads(uploaded[prefix + "native-evaluation-0.json"])
    archive = uploaded[record["artifacts"]["uri"]]
    assert hashlib.sha256(archive).hexdigest() == record["artifacts"]["sha256"]
    assert record["policy"] == request["policy"]
    assert record["protocol_sha256"] == request["protocol"]["sha256"]
    assert record["native"] == report
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        assert (
            stream.extractfile("rendered-rollout/00001.png").read()
            == b"synthetic frame"
        )
        assert json.load(stream.extractfile("trajectory.json")) == [
            {"physical_failure": [1]}
        ]
