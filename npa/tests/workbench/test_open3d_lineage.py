"""Persisted-input mutations must fail before native rendering or publication."""

from __future__ import annotations

import json
import subprocess

import pytest

from npa.workbench.open3d import runtime
from npa.workbench.open3d.artifacts import Open3dError, canonical, sha256_bytes
from npa.workbench.open3d.schemas import (
    RegistrationManifest,
    RunRequest,
    StageDemoRequest,
)

PREFIX = "s3://example-bucket"
IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _documents():
    manifest = RegistrationManifest.model_validate(
        {
            "fragments": [
                {
                    "id": key,
                    "uri": f"{PREFIX}/prepared/{key}.ply",
                    "sha256": sha256_bytes(b"cloud"),
                    "bytes": 5,
                }
                for key in ("a", "b")
            ],
            "voxel_size": 0.05,
        }
    ).model_dump(mode="json")
    graph = {
        "nodes": [{"fragment_id": key, "pose": IDENTITY} for key in ("a", "b")],
        "edges": [
            {
                "source_node_id": 0,
                "target_node_id": 1,
                "transformation": IDENTITY,
                "uncertain": False,
            }
        ],
        "fused_sha256": sha256_bytes(b"fused"),
    }
    registration = {
        "kind": "multiway",
        "input_path": f"{PREFIX}/prepared",
        "manifest_sha256": sha256_bytes(canonical(manifest)),
        "voxel_size": 0.05,
        "pose_graph": graph,
    }
    reconstruction = {
        "schema_version": "npa.open3d.reconstruction.v1",
        "registration_path": f"{PREFIX}/graph",
        "registration_result_sha256": sha256_bytes(canonical(registration)),
        "support_distance_factor": 0,
        "unsupported_vertices_removed": 0,
        "mesh": {"sha256": sha256_bytes(b"mesh")},
        "mesh_uncropped": {"sha256": sha256_bytes(b"uncropped")},
    }
    return {
        "mesh/result.json": canonical(reconstruction),
        "graph/result.json": canonical(registration),
        "graph/pose_graph.json": canonical(graph),
        "prepared/manifest.json": canonical(manifest),
        "graph/fused.ply": b"fused",
        "mesh/mesh.ply": b"mesh",
        "mesh/mesh_uncropped.ply": b"uncropped",
        "prepared/a.ply": b"cloud",
        "prepared/b.ply": b"cloud",
    }


def _bind_storage(monkeypatch, tmp_path, documents):
    monkeypatch.setattr(runtime, "_tool_paths", lambda _: None)
    monkeypatch.setattr(
        runtime, "read_bytes_uri", lambda uri: documents[uri.removeprefix(PREFIX + "/")]
    )
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(runtime, "_work_dir", lambda _: work)
    return RunRequest(
        input_path=f"{PREFIX}/mesh", output_path=f"{PREFIX}/reports", run_id="test"
    )


@pytest.mark.parametrize(
    "changed",
    [
        "graph/fused.ply",
        "mesh/mesh.ply",
        "mesh/mesh_uncropped.ply",
        "graph/pose_graph.json",
        "prepared/manifest.json",
        "graph/result.json",
    ],
)
def test_changed_recording_input_is_rejected_before_render(
    monkeypatch, tmp_path, changed
):
    documents = _documents()
    if changed.endswith(".json"):
        value = json.loads(documents[changed])
        if changed == "graph/pose_graph.json":
            value["nodes"][1]["pose"][0][3] = 10
        elif changed == "prepared/manifest.json":
            value["voxel_size"] = 0.5
        else:
            value["run_id"] = "replacement-run"
        documents[changed] = canonical(value)
    else:
        documents[changed] += b"replacement"
    request = _bind_storage(monkeypatch, tmp_path, documents)
    monkeypatch.setattr(
        runtime,
        "_run_runner",
        lambda *_a: pytest.fail("unverified input reached native rendering"),
    )
    monkeypatch.setattr(
        runtime, "_publish", lambda *_a: pytest.fail("unverified input was published")
    )
    with pytest.raises(Open3dError, match="does not match"):
        runtime.visualize(request)


def test_legacy_reconstruction_requires_regeneration(monkeypatch, tmp_path):
    documents = _documents()
    reconstruction = json.loads(documents["mesh/result.json"])
    del reconstruction["registration_result_sha256"]
    documents["mesh/result.json"] = canonical(reconstruction)
    request = _bind_storage(monkeypatch, tmp_path, documents)
    monkeypatch.setattr(
        runtime,
        "_run_runner",
        lambda *_a: pytest.fail("missing identity reached rendering"),
    )
    with pytest.raises(Open3dError, match="regenerate the upstream artifact"):
        runtime.visualize(request)


def test_verified_recording_carries_exact_input_hashes(monkeypatch, tmp_path):
    documents = _documents()
    request = _bind_storage(monkeypatch, tmp_path, documents)
    captured = {}
    published = {}

    def run(_kind, payload, root, _run_id):
        captured.update(payload)
        output = root / "output"
        output.mkdir()
        (output / "point_cloud.rrd").write_bytes(b"recording-fixture")
        (output / "runner.json").write_bytes(
            canonical(
                {
                    "sha256": sha256_bytes(b"recording-fixture"),
                    "unsupported_triangles_shown": 0,
                    "input_provenance": payload["input_provenance"],
                }
            )
        )
        return output

    monkeypatch.setattr(runtime, "_run_runner", run)
    # Native RRD decoding has independent functional coverage; this fixture
    # isolates the public runtime's input/output identity boundary.
    monkeypatch.setattr(runtime, "verify_rerun_recording", lambda _: None)
    monkeypatch.setattr(
        runtime, "_publish", lambda uri, data: published.update({uri: data})
    )
    result = runtime.visualize(request)
    hashes = result["input_provenance"]
    for key, path in {
        "fused": "graph/fused.ply",
        "mesh": "mesh/mesh.ply",
        "mesh_uncropped": "mesh/mesh_uncropped.ply",
        "registration_result": "graph/result.json",
        "reconstruction_result": "mesh/result.json",
    }.items():
        assert hashes[f"{key}_sha256"] == sha256_bytes(documents[path])
    assert hashes == captured["input_provenance"]
    assert len(published) == 2
    assert not (tmp_path / "work").exists()


def test_native_failure_retains_the_advertised_private_log(monkeypatch, tmp_path):
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    monkeypatch.setattr(runtime, "_work_dir", lambda _: work)
    for name in ("validate_write_path", "authorize_uri"):
        monkeypatch.setattr(runtime, name, lambda *_a, **_k: None)

    def failed_run(command, **kwargs):
        kwargs["stdout"].write(b"native diagnostic\n")
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(runtime.subprocess, "run", failed_run)
    with pytest.raises(Open3dError, match="retained local log"):
        runtime.stage_demo(StageDemoRequest(output_path=f"{PREFIX}/staged"))
    assert (work / "runtime.log").read_bytes() == b"native diagnostic\n"
    assert work.stat().st_mode & 0o777 == 0o700


def test_reconstruction_binds_the_registration_it_actually_read(monkeypatch, tmp_path):
    documents = _documents()
    _bind_storage(monkeypatch, tmp_path, documents)
    published = {}

    def run(_kind, _payload, root, _run_id):
        output = root / "output"
        output.mkdir()
        mesh = {
            "vertex_count": 3,
            "triangle_count": 1,
            "surface_area": 1.0,
            "is_edge_manifold": True,
            "is_vertex_manifold": True,
            "is_watertight": False,
            "extent": [1.0, 1.0, 1.0],
            "sha256": sha256_bytes(b"mesh"),
        }
        for name in ("mesh.ply", "mesh_uncropped.ply"):
            (output / name).write_bytes(b"mesh")
        (output / "runner.json").write_bytes(
            canonical(
                {
                    "mesh": mesh,
                    "mesh_uncropped": mesh,
                    "support_distance_factor": 0,
                    "support_before_crop": {"unsupported_area_fraction": 0},
                    "support": {"unsupported_area_fraction": 0},
                    "coverage": {
                        "fraction_within_voxel": 1.0,
                        "sample_to_surface_rmse": 0.0,
                        "samples": 1,
                    },
                }
            )
        )
        return output

    monkeypatch.setattr(runtime, "_run_runner", run)
    monkeypatch.setattr(
        runtime, "_publish", lambda uri, data: published.update({uri: data})
    )
    result = runtime.reconstruct(
        runtime.ReconstructRequest(
            input_path=f"{PREFIX}/graph",
            output_path=f"{PREFIX}/mesh",
            run_id="test",
        )
    )
    expected = sha256_bytes(documents["graph/result.json"])
    assert result["registration_result_sha256"] == expected
    persisted = json.loads(published[f"{PREFIX}/mesh/result.json"])
    assert persisted["registration_result_sha256"] == expected
