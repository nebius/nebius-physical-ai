from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from botocore.exceptions import ClientError
from PIL import Image

from npa.workflows.sim2real.checkpoint_selection import resolve_selected_checkpoint
from npa.workflows.sim2real.models import Sim2RealLoopConfig
from npa.workflows.sim2real.stage14_finalize import download_plan
from npa.workflows.sim2real_rerun_regen import (
    Sim2RealRerunRegenError,
    _current_checkpoint_sources,
    _download_if_exists,
    _download_render_tree,
    _ensure_policy_access_metadata,
    _heldout_render_source,
    _latest_completed_inner_evidence_rel,
    _renders_dir_for_report,
    download_rrd_from_s3,
    publish_regen_outputs,
    regen_sim2real_rrd,
    sync_heldout_renders,
)
from npa.workflows.sim2real_viz import (
    Sim2RealVizError,
    _heldout_pointcloud_frames,
    _heldout_render_episodes,
    _heldout_renders_root,
)


def _config() -> Sim2RealLoopConfig:
    return Sim2RealLoopConfig(
        run_id="run-a",
        s3_bucket="demo-bucket",
        s3_prefix="sim2real-b",
        s3_endpoint="https://storage.example",
        outer_iterations=3,
    )


def _root_render_lineage() -> dict[str, object]:
    return {
        "evaluation_split": "gold_heldout",
        "outer_iteration": 1,
        "render_lineage": {
            "evaluation_split": "gold_heldout",
            "canonical_s3_uri": (
                "s3://demo-bucket/sim2real-b/run-a/eval/gold-heldout/outer-01/renders/"
            ),
            "local_relative_dir": ".",
        },
    }


def test_render_destination_cannot_equal_regeneration_root(tmp_path: Path) -> None:
    with pytest.raises(Sim2RealRerunRegenError):
        _renders_dir_for_report(_config(), tmp_path, _root_render_lineage())
    with pytest.raises(Sim2RealVizError):
        _heldout_renders_root(tmp_path, _root_render_lineage())
    with pytest.raises(RuntimeError):
        download_plan(
            root="s3://demo-bucket/sim2real-b/run-a",
            outer_iteration=1,
            evidence={"iterations": []},
            gold=_root_render_lineage(),
        )


def test_failed_root_render_transfer_preserves_run_tree(tmp_path: Path) -> None:
    class FailingStorage:
        def download_directory(self, _uri: str, _destination: str) -> None:
            raise OSError("forced transfer failure")

    marker = tmp_path / "must-survive.txt"
    marker.write_text("sentinel", encoding="utf-8")
    with pytest.raises(Sim2RealRerunRegenError):
        sync_heldout_renders(
            _config(),
            tmp_path,
            heldout_report=_root_render_lineage(),
            client=FailingStorage(),
        )
    assert marker.read_text(encoding="utf-8") == "sentinel"


def test_destructive_render_sink_rejects_run_root(tmp_path: Path) -> None:
    class FailingStorage:
        def download_directory(self, _uri: str, _destination: str) -> None:
            raise OSError("forced transfer failure")

    marker = tmp_path / "must-survive.txt"
    marker.write_text("sentinel", encoding="utf-8")
    with pytest.raises(Sim2RealRerunRegenError):
        _download_render_tree(
            FailingStorage(),
            "s3://demo-bucket/run/renders/",
            tmp_path,
            containment_root=tmp_path,
        )
    assert marker.read_text(encoding="utf-8") == "sentinel"


def test_remote_completed_pair_requires_exact_nonempty_objects() -> None:
    run_prefix = "runs/run-a/"
    objects = {
        run_prefix + "inner_loop/outer-01/evidence.json",
        run_prefix + "inner_loop/outer-02/evidence.json",
        run_prefix + "eval/gold-heldout/outer-01/report.json",
        run_prefix + "eval/gold-heldout/outer-02/renders/env/camera-000.png",
    }

    class Paginator:
        def paginate(self, *, Bucket: str, Prefix: str, Delimiter: str):
            assert Bucket == "demo-bucket"
            children = sorted(
                {
                    Prefix + key[len(Prefix) :].split(Delimiter, 1)[0] + Delimiter
                    for key in objects
                    if key.startswith(Prefix) and Delimiter in key[len(Prefix) :]
                }
            )
            return [{"CommonPrefixes": [{"Prefix": item} for item in children]}]

    class S3:
        def get_paginator(self, name: str) -> Paginator:
            assert name == "list_objects_v2"
            return Paginator()

        def head_object(self, *, Bucket: str, Key: str):
            if Bucket == "demo-bucket" and Key in objects:
                return {"ContentLength": 2}
            raise ClientError(
                {"Error": {"Code": "404", "Message": "not found"}},
                "HeadObject",
            )

    class Storage:
        _s3 = S3()

    assert (
        _latest_completed_inner_evidence_rel(
            Storage(), "s3://demo-bucket/" + run_prefix
        )
        == "inner_loop/outer-01/evidence.json"
    )


def test_authoritative_heldout_load_failure_suppresses_access(
    tmp_path: Path,
) -> None:
    checkpoint = b"checkpoint"
    digest = hashlib.sha256(checkpoint).hexdigest()
    uri = "s3://demo-bucket/sim2real-b/run-a/model.pt"
    heldout = {
        "policy_checkpoint_uri": uri,
        "policy_checkpoint_sha256": digest,
        "policy_checkpoint_size_bytes": len(checkpoint),
        "policy_inference_provenance": {
            "checkpoint_uri": uri,
            "checkpoint_sha256": digest,
            "generator_policy_sha256": digest,
            "checkpoint_size_bytes": len(checkpoint),
            "loaded_for_inference": False,
            "stock_or_scripted_policy": False,
            "actor_is_learned": True,
            "scripted_post_actor_controller": False,
            "policy_composition": "learned_actor_only",
            "post_actor_controller": None,
        },
    }
    candidate = tmp_path / "checkpoints/candidate/candidate.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(
        json.dumps(
            {
                "deployable_policy": True,
                "policy_bytes_available": True,
                "policy_checkpoint_uri": uri,
            }
        ),
        encoding="utf-8",
    )

    class Storage:
        def download_file(self, _uri: str, destination: str) -> None:
            Path(destination).write_bytes(checkpoint)

    access = _ensure_policy_access_metadata(
        _config(),
        tmp_path,
        storage=Storage(),
        report={},
        heldout_report=heldout,
    )
    assert access["deployable_policy"] is False
    assert access["authenticated_download_command"] == ""
    assert access["ui_action"] == ""


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("checkpoint_uri", ""),
        ("checkpoint_sha256", ""),
        ("checkpoint_size_bytes", 0),
        ("generator_policy_sha256", ""),
    ],
)
def test_selected_checkpoint_requires_complete_identity(
    field: str,
    invalid: object,
) -> None:
    uri = "s3://demo-bucket/sim2real-b/run-a/model.pt"
    incomplete = {
        "checkpoint_uri": uri,
        "checkpoint_sha256": "a" * 64,
        "checkpoint_size_bytes": 128,
        "generator_policy_sha256": "a" * 64,
    }
    incomplete[field] = invalid
    selected_uri = str(incomplete["checkpoint_uri"])
    with pytest.raises(ValueError):
        resolve_selected_checkpoint(
            {
                "selected_checkpoint_uri": selected_uri,
                "final_checkpoint_uri": selected_uri,
                "checkpoint_selection": dict(incomplete),
                "checkpoint_candidates": [dict(incomplete)],
            }
        )


def test_wholly_empty_checkpoint_reference_is_absent() -> None:
    assert _current_checkpoint_sources(
        {
            "selected_checkpoint_uri": "",
            "final_checkpoint_uri": "",
            "checkpoint_selection": {},
        }
    ) == ({}, {})


def test_viewer_uses_canonical_lineage_without_local_alias(tmp_path: Path) -> None:
    report = {
        "evaluation_split": "gold_heldout",
        "outer_iteration": 3,
        "render_lineage": {
            "evaluation_split": "gold_heldout",
            "canonical_s3_uri": (
                "s3://demo-bucket/sim2real-b/run-a/eval/gold-heldout/outer-03/renders/"
            ),
        },
    }
    expected = tmp_path / "eval/gold-heldout/outer-03/renders"
    assert _renders_dir_for_report(_config(), tmp_path, report) == expected
    assert _heldout_renders_root(tmp_path, report) == expected


def test_explicit_validation_report_cannot_use_legacy_render_fallback() -> None:
    with pytest.raises(Sim2RealRerunRegenError):
        _heldout_render_source(_config(), {"evaluation_split": "validation"})


def test_cross_run_canonical_render_uri_is_rejected() -> None:
    report = {
        "evaluation_split": "gold_heldout",
        "outer_iteration": 1,
        "render_lineage": {
            "evaluation_split": "gold_heldout",
            "canonical_s3_uri": (
                "s3://demo-bucket/sim2real-b/run-b/eval/gold-heldout/outer-01/renders/"
            ),
            "local_relative_dir": "eval/gold-heldout/outer-01/renders",
        },
    }
    with pytest.raises(Sim2RealRerunRegenError):
        _heldout_render_source(_config(), report)


@pytest.mark.parametrize("kind", ["absolute", "traversal", "symlink"])
def test_manifest_environment_path_cannot_escape_render_root(
    tmp_path: Path,
    kind: str,
) -> None:
    renders = tmp_path / "eval/gold-heldout/outer-01/renders"
    renders.mkdir(parents=True)
    outside = tmp_path / "outside-env"
    outside.mkdir()
    Image.new("RGB", (2, 2), (17, 99, 201)).save(outside / "camera-000.png")
    if kind == "absolute":
        env_id = str(outside.resolve())
    elif kind == "traversal":
        env_id = os.path.relpath(outside, renders)
    else:
        real_env = renders / "real-env"
        real_env.mkdir()
        (renders / "linked-env").symlink_to(real_env, target_is_directory=True)
        env_id = "linked-env"
    report = {
        "local_renders_dir": str(renders),
        "render_manifest": {
            "episodes": [{"env_id": env_id, "frames": ["camera-000.png"]}]
        },
    }
    with pytest.raises(Sim2RealVizError):
        _heldout_render_episodes(tmp_path, report)


def test_manifest_frame_cannot_be_a_symlink(tmp_path: Path) -> None:
    renders = tmp_path / "eval/gold-heldout/outer-01/renders"
    env = renders / "env-0001"
    env.mkdir(parents=True)
    target = env / "real-camera.png"
    Image.new("RGB", (2, 2), (17, 99, 201)).save(target)
    (env / "camera-000.png").symlink_to(target)
    report = {
        "local_renders_dir": str(renders),
        "render_manifest": {
            "episodes": [{"env_id": "env-0001", "frames": ["camera-000.png"]}]
        },
    }
    with pytest.raises(Sim2RealVizError):
        _heldout_render_episodes(tmp_path, report)


def test_singleton_download_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    local = tmp_path / "run"
    local.mkdir()
    outside = tmp_path / "outside-eval"
    outside.mkdir()
    (local / "eval").symlink_to(outside, target_is_directory=True)

    class Storage:
        def download_path(self, _uri: str, destination: str) -> None:
            Path(destination).write_text('{"source":"remote"}', encoding="utf-8")

    with pytest.raises(Sim2RealRerunRegenError):
        _download_if_exists(
            Storage(),
            "s3://demo-bucket/run/report.json",
            local / "eval/gold-heldout/outer-01/report.json",
            containment_root=local,
        )
    assert not (outside / "gold-heldout/outer-01/report.json").exists()


def test_duplicate_candidate_parse_failure_persists_fail_closed_state(
    tmp_path: Path,
) -> None:
    inner = tmp_path / "inner_loop/outer-01/evidence.json"
    heldout = tmp_path / "eval/heldout/report.json"
    candidate = tmp_path / "checkpoints/candidate/candidate.json"
    inner.parent.mkdir(parents=True)
    heldout.parent.mkdir(parents=True)
    candidate.parent.mkdir(parents=True)
    inner.write_text("{}", encoding="utf-8")
    heldout.write_text("{}", encoding="utf-8")
    candidate.write_text(
        (
            '{"deployable_policy":false,"deployable_policy":true,'
            '"policy_bytes_available":true,'
            '"policy_download_command":"stale download",'
            '"policy_ui_action":"stale action"}'
        ),
        encoding="utf-8",
    )

    with pytest.raises(Sim2RealRerunRegenError):
        regen_sim2real_rrd(
            _config(),
            local_dir=tmp_path,
            sync_inputs=False,
            client=object(),
        )

    payload = json.loads(candidate.read_text(encoding="utf-8"))
    assert payload["deployable_policy"] is False
    assert payload["policy_bytes_available"] is False
    assert "policy_download_command" not in payload
    assert "policy_ui_action" not in payload


def test_explicit_validation_lineage_is_not_final_evidence(
    tmp_path: Path,
) -> None:
    report = {
        "evaluation_split": "validation",
        "outer_iteration": 1,
        "render_lineage": {
            "evaluation_split": "validation",
            "canonical_s3_uri": (
                "s3://demo-bucket/sim2real-b/run-a/eval/gold-heldout/outer-01/renders/"
            ),
            "local_relative_dir": "eval/gold-heldout/outer-01/renders",
        },
    }
    with pytest.raises(Sim2RealVizError):
        _heldout_renders_root(tmp_path, report)
    with pytest.raises(RuntimeError):
        download_plan(
            root="s3://demo-bucket/sim2real-b/run-a",
            outer_iteration=1,
            evidence={"iterations": []},
            gold=report,
        )


def test_present_empty_current_report_suppresses_compatibility_access(
    tmp_path: Path,
) -> None:
    uri = "s3://demo-bucket/sim2real-b/run-a/model.pt"
    candidate = tmp_path / "checkpoints/candidate/candidate.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(
        json.dumps({"deployable_policy": True, "policy_checkpoint_uri": uri}),
        encoding="utf-8",
    )
    downloads: list[str] = []

    class Storage:
        def download_file(self, source: str, destination: str) -> None:
            downloads.append(source)
            Path(destination).write_bytes(b"checkpoint")

    access = _ensure_policy_access_metadata(
        _config(),
        tmp_path,
        storage=Storage(),
        report={},
        heldout_report={},
    )
    assert access["deployable_policy"] is False
    assert access["policy_bytes_available"] is False
    assert downloads == []


def test_fallback_scan_rejects_symlinked_frame(tmp_path: Path) -> None:
    renders = tmp_path / "eval/gold-heldout/outer-01/renders"
    env = renders / "env-0001"
    env.mkdir(parents=True)
    outside = tmp_path / "outside.png"
    Image.new("RGB", (2, 2), (11, 22, 33)).save(outside)
    (env / "camera-000.png").symlink_to(outside)
    with pytest.raises(Sim2RealVizError):
        _heldout_render_episodes(
            tmp_path,
            {
                "local_renders_dir": str(renders),
                "render_manifest": {"episodes": []},
            },
        )


def test_complete_checkpoint_identity_rejects_empty_s3_authority() -> None:
    uri = "s3:///model.pt"
    identity = {
        "checkpoint_uri": uri,
        "checkpoint_sha256": "a" * 64,
        "checkpoint_size_bytes": 1,
        "generator_policy_sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="URI is malformed"):
        resolve_selected_checkpoint(
            {
                "selected_checkpoint_uri": uri,
                "final_checkpoint_uri": uri,
                "checkpoint_selection": dict(identity),
                "checkpoint_candidates": [dict(identity)],
            }
        )


def test_sealed_render_source_is_exact_run_iteration_prefix() -> None:
    base = {
        "evaluation_split": "gold_heldout",
        "outer_iteration": 1,
        "render_lineage": {
            "evaluation_split": "gold_heldout",
            "local_relative_dir": "eval/gold-heldout/outer-01/renders",
        },
    }
    for uri in (
        "s3://demo-bucket/sim2real-b/run-a/",
        "s3://demo-bucket/sim2real-b/run-a/eval/gold-heldout/outer-03/renders/",
    ):
        report = json.loads(json.dumps(base))
        report["render_lineage"]["canonical_s3_uri"] = uri
        with pytest.raises(
            Sim2RealRerunRegenError,
            match="configured run and iteration",
        ):
            _heldout_render_source(_config(), report)


def test_stage14_rejects_cross_iteration_render_lineage() -> None:
    report = {
        "evaluation_split": "gold_heldout",
        "outer_iteration": 1,
        "render_lineage": {
            "evaluation_split": "gold_heldout",
            "canonical_s3_uri": (
                "s3://demo-bucket/sim2real-b/run-a/eval/gold-heldout/outer-03/renders/"
            ),
            "local_relative_dir": "eval/gold-heldout/outer-01/renders",
        },
    }
    with pytest.raises(RuntimeError, match="safe canonical render lineage"):
        download_plan(
            root="s3://demo-bucket/sim2real-b/run-a",
            outer_iteration=1,
            evidence={"iterations": []},
            gold=report,
        )


def test_regeneration_does_not_guess_sealed_outer_iteration(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        Sim2RealRerunRegenError,
        match="invalid outer iteration",
    ):
        _renders_dir_for_report(
            _config(),
            tmp_path,
            {
                "evaluation_split": "gold_heldout",
                "render_lineage": {"evaluation_split": "gold_heldout"},
            },
        )


def test_operator_symlink_above_containment_root_is_allowed(
    tmp_path: Path,
) -> None:
    physical = tmp_path / "physical"
    local = physical / "run"
    local.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(physical, target_is_directory=True)
    destination = alias / "run/input.json"

    class Storage:
        def download_path(self, _uri: str, target: str) -> None:
            Path(target).write_text('{"ok":true}', encoding="utf-8")

    assert _download_if_exists(
        Storage(),
        "s3://demo-bucket/run/input.json",
        destination,
        containment_root=alias / "run",
    )
    assert destination.read_text(encoding="utf-8") == '{"ok":true}'


def test_current_loaded_report_requires_complete_matching_identity(
    tmp_path: Path,
) -> None:
    checkpoint = b"checkpoint"
    uri = "s3://demo-bucket/sim2real-b/run-a/model.pt"
    candidate = tmp_path / "checkpoints/candidate/candidate.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(
        json.dumps(
            {
                "deployable_policy": True,
                "policy_checkpoint_uri": uri,
            }
        ),
        encoding="utf-8",
    )
    downloads: list[str] = []

    class Storage:
        def download_file(self, source: str, destination: str) -> None:
            downloads.append(source)
            Path(destination).write_bytes(checkpoint)

    access = _ensure_policy_access_metadata(
        _config(),
        tmp_path,
        storage=Storage(),
        report={},
        heldout_report={"policy_inference_provenance": {"loaded_for_inference": True}},
    )
    assert access["deployable_policy"] is False
    assert access["policy_bytes_available"] is False
    assert access["authenticated_download_command"] == ""
    assert access["ui_action"] == ""
    assert downloads == []


@pytest.mark.parametrize("symlink_level", ["pointcloud", "env", "view", "sample"])
def test_pointcloud_loader_rejects_symlinked_source(
    tmp_path: Path,
    symlink_level: str,
) -> None:
    renders = tmp_path / "eval/gold-heldout/outer-01/renders"
    outside = tmp_path / "outside"
    outside.mkdir()
    root = renders / "_pointcloud"
    env = root / "env-0001"
    view = env / "primary"
    if symlink_level == "pointcloud":
        renders.mkdir(parents=True)
        root.symlink_to(outside, target_is_directory=True)
    elif symlink_level == "env":
        root.mkdir(parents=True)
        env.symlink_to(outside, target_is_directory=True)
    elif symlink_level == "view":
        env.mkdir(parents=True)
        view.symlink_to(outside, target_is_directory=True)
    else:
        view.mkdir(parents=True)
        target = outside / "cloud-000000.npz"
        np.savez(target, xyz=np.zeros((1, 3)), rgb=np.zeros((1, 3)))
        (view / target.name).symlink_to(target)

    with pytest.raises(Sim2RealVizError, match="symlink"):
        _heldout_pointcloud_frames(
            tmp_path,
            {"local_renders_dir": str(renders)},
        )


def test_relative_regeneration_directory_publishes_absolute_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    local_dir = Path("regen")
    report = local_dir / "eval/heldout/report.json"
    frame = local_dir / "eval/heldout/renders/env-0001/camera-000.png"
    recording = local_dir / "reports/sim2real.rrd"
    report.parent.mkdir(parents=True)
    frame.parent.mkdir(parents=True)
    recording.parent.mkdir(parents=True)
    report.write_text("{}", encoding="utf-8")
    frame.write_bytes(b"frame")
    recording.write_bytes(b"rrd")
    uploads: list[str] = []

    class Storage:
        def upload_file(self, source: str, destination: str) -> str:
            uploads.append(source)
            return destination

        def upload_directory(self, source: str, destination: str) -> str:
            uploads.append(source)
            return destination

    publish_regen_outputs(_config(), local_dir, client=Storage())
    assert uploads
    assert all(Path(source).is_absolute() for source in uploads)


def test_exact_legacy_gold_render_uri_remains_supported(tmp_path: Path) -> None:
    uri = (
        "s3://demo-bucket/sim2real-b/run-a/component-io/heldout-eval/"
        "gold_heldout-outer-01/output/renders/"
    )
    downloads: list[str] = []

    class Storage:
        def download_directory(self, source: str, destination: str) -> None:
            downloads.append(source)
            frame = Path(destination) / "env-0001/camera-000.png"
            frame.parent.mkdir(parents=True)
            frame.write_bytes(b"frame")

    report = {
        "evaluation_split": "gold_heldout",
        "outer_iteration": 1,
        "render_lineage": {
            "evaluation_split": "gold_heldout",
            "renders_s3_uri": uri,
            "local_relative_dir": "eval/gold-heldout/outer-01/renders",
        },
    }
    assert sync_heldout_renders(
        _config(),
        tmp_path,
        heldout_report=report,
        client=Storage(),
    )
    assert downloads == [uri]


@pytest.mark.parametrize("symlink_level", ["run", "reports"])
def test_default_rrd_download_checks_internal_symlinked_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symlink_level: str,
) -> None:
    default_root = tmp_path / "default"
    run_root = default_root / "run-a"
    outside = tmp_path / "outside"
    default_root.mkdir()
    outside.mkdir()
    if symlink_level == "run":
        run_root.symlink_to(outside, target_is_directory=True)
        escaped = outside / "reports/sim2real.rrd"
    else:
        run_root.mkdir()
        (run_root / "reports").symlink_to(outside, target_is_directory=True)
        escaped = outside / "sim2real.rrd"
    monkeypatch.setattr(
        "npa.workflows.sim2real_rerun_regen.DEFAULT_REGEN_ROOT",
        default_root,
    )

    class Storage:
        def download_path(self, _uri: str, destination: str) -> None:
            Path(destination).write_bytes(b"rrd")

    with pytest.raises(Sim2RealRerunRegenError, match="symlinked ancestor"):
        download_rrd_from_s3(
            _config(),
            dest_path=run_root / "reports/sim2real.rrd",
            client=Storage(),
        )
    assert not escaped.exists()
