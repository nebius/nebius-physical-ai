"""All reconstruction consumers verify converted inputs before reading them."""

from __future__ import annotations

import hashlib
import json
import subprocess

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.nurec import colmap, nurec


@pytest.fixture
def converted_sequence(tmp_path):
    root = tmp_path / "sequence"
    root.mkdir()
    meta = root / "sequence.json"
    meta.write_text(json.dumps({
        "version": "v4",
        "component_stores": [{"path": "data.zarr.itar"}],
    }))
    (root / "data.zarr.itar").write_bytes(b"opaque shard bytes")
    (root / "npa-rig.json").write_text('{"reference_camera":"camera1"}')
    members = [
        {"path": path.name, "bytes": path.stat().st_size,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(root.iterdir())
    ]
    report = root / colmap.CONVERSION_REPORT
    report.write_text(json.dumps({
        "schema_version": 1, "status": "ok", "engine": "nvidia-ncore-colmap",
        "ncore_meta": meta.name, "poses_component_group": "npa_rig",
        "members": members,
        "publication": {"mode": "immutable-prefix-v1", "claim": colmap.PUBLICATION_CLAIM},
    }))
    (root / colmap.PUBLICATION_CLAIM).write_text(json.dumps({
        "schema_version": 1,
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }))
    colmap.verify_conversion_inventory(root)
    return meta


@pytest.mark.parametrize("changed", ["data.zarr.itar", "npa-rig.json", colmap.PUBLICATION_CLAIM])
def test_direct_reconstruction_rejects_changed_conversion_before_runner(
    tmp_path, converted_sequence, changed
):
    (converted_sequence.parent / changed).write_bytes(b"changed")
    calls = []
    with pytest.raises(nurec.NurecError, match="inventory"):
        nurec.reconstruct_scene(
            nurec.NurecConfig(out_dir=tmp_path / "out"),
            ncore_json=str(converted_sequence),
            runner=lambda *args, **kwargs: calls.append(args) or subprocess.CompletedProcess([], 1),
        )
    assert not calls
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("changed", ["data.zarr.itar", "npa-rig.json", colmap.PUBLICATION_CLAIM])
def test_local_cli_rejects_changed_conversion_before_sensor_selection(
    monkeypatch, converted_sequence, changed
):
    import npa.cli.nurec as cli

    (converted_sequence.parent / changed).write_bytes(b"changed")
    calls = []
    monkeypatch.setattr(nurec, "ncore_sensor_ids", lambda *_: calls.append("sensors") or ((), ()))
    monkeypatch.setattr(nurec, "read_rig_sidecar", lambda *_: calls.append("rig") or {})
    monkeypatch.setattr(cli, "reconstruct_scene", lambda *a, **k: calls.append("runner"))
    result = CliRunner().invoke(app, [
        "workbench", "nurec", "reconstruct", "--ncore-json", str(converted_sequence),
        "--output", "json",
    ])
    assert result.exit_code != 0
    assert not calls
    assert json.loads(result.stdout)["status"] == "failed"
    assert "inventory" in result.stdout


def test_valid_local_conversion_and_legacy_dry_run_still_work(tmp_path, converted_sequence):
    for meta in (converted_sequence, tmp_path / "legacy.json"):
        result = nurec.reconstruct_scene(
            nurec.NurecConfig(out_dir=tmp_path / "out"),
            ncore_json=str(meta), dry_run=True,
        )
        assert result.ok
