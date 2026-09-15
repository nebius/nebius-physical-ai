"""Thin CLI/SDK clients of the shared NCore conversion implementation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.nurec import colmap


@pytest.mark.parametrize("has_unrelated_file", [False, True])
def test_explicit_ncore_source_without_metadata_never_falls_back(
    tmp_path, has_unrelated_file
):
    from npa.cli.nurec import _materialize_ncore
    from npa.workbench.nurec.nurec import NurecConfig, NurecError

    source = tmp_path / "explicit-source"
    source.mkdir()
    if has_unrelated_file:
        (source / "other.json").write_text("{}")
    with pytest.raises(NurecError, match="NCore.*metadata"):
        _materialize_ncore(NurecConfig(cache_dir=tmp_path / "cache"), str(source))


def test_colmap_help():
    result = CliRunner().invoke(app, ["workbench", "nurec", "convert-colmap", "--help"])
    assert result.exit_code == 0, result.output
    for flag in (
        "--input-path",
        "--output-path",
        "--dataset-root",
        "--colmap-dir",
        "--images-dir",
        "--rig-mode",
        "--scratch-dir",
    ):
        assert flag in result.output


def test_cli_json_contract_and_shared_request(monkeypatch):
    seen = []

    def run(request):
        seen.append(request)
        print("vendor progress")
        return {"status": "ok", "counts": {"images": 2}}

    monkeypatch.setattr(colmap, "convert_colmap", run)
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "nurec",
            "convert-colmap",
            "--input-path",
            "s3://test-bucket/input.zip",
            "--output-path",
            "s3://test-bucket/output/",
            "--rig-mode",
            "preserve",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["counts"]["images"] == 2
    assert seen[0].rig_mode == "preserve"


def test_cli_validation_errors_do_not_echo_input():
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "nurec",
            "convert-colmap",
            "--input-path",
            "https://private-value.invalid/data",
            "--output-path",
            "s3://test-bucket/output/",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "failed"
    assert "private-value" not in result.output


def test_cli_domain_error_is_json(monkeypatch):
    def fail(request):
        raise colmap.NcoreConversionError("official converter unavailable")

    monkeypatch.setattr(colmap, "convert_colmap", fail)
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "nurec",
            "convert-colmap",
            "--input-path",
            "s3://test-bucket/input/",
            "--output-path",
            "s3://test-bucket/output/",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"] == "official converter unavailable"


def test_sdk_calls_module_directly(monkeypatch):
    from npa.sdk.workbench.nurec import convert_colmap

    seen = []
    monkeypatch.setattr(
        colmap,
        "convert_colmap",
        lambda request: seen.append(request) or {"status": "ok"},
    )
    assert (
        convert_colmap(
            "s3://test-bucket/input/", "s3://test-bucket/output/", rig_mode="preserve"
        )["status"]
        == "ok"
    )
    assert isinstance(seen[0], colmap.ColmapConversionRequest)


def test_toolref_is_real_configurable_cli_without_nre_gate():
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    entry = TOOL_CATALOG["workbench.nurec.convert_colmap"]
    assert entry.argv_template[:4] == ["npa", "workbench", "nurec", "convert-colmap"]
    assert "{{config.colmap_input_uri}}" in entry.argv_template
    assert "{{config.ncore_sequence_uri}}" in entry.argv_template
    assert "{{config.rig_mode}}" in entry.argv_template
    assert not entry.access_capabilities


@pytest.mark.parametrize("override", [False, True])
def test_model_cli_sdk_and_rendered_workflow_agree_on_staging(
    monkeypatch, tmp_path, override
):
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec
    from npa.sdk.workbench.nurec import convert_colmap

    seen = []
    monkeypatch.setattr(
        colmap,
        "convert_colmap",
        lambda request: seen.append(request) or {"status": "ok"},
    )
    paths = (
        {
            "cache_dir": tmp_path / "custom cache",
            "scratch_dir": tmp_path / "custom scratch",
        }
        if override
        else {}
    )
    inputs = {
        "input_path": "s3://test-bucket/input/",
        "output_path": "s3://test-bucket/output/",
    }
    model = colmap.ColmapConversionRequest(**inputs, **paths)
    convert_colmap(**inputs, **paths)
    cli = ["workbench", "nurec", "convert-colmap"]
    for name, value in {**inputs, **paths}.items():
        cli.extend(["--" + name.replace("_", "-"), str(value)])
    result = CliRunner().invoke(app, cli + ["--output-format", "json"])
    assert result.exit_code == 0, result.output

    # Remove shipped overrides to also exercise the toolRef's own defaults.
    spec = load_spec(
        Path(__file__).resolve().parents[3]
        / "workflows/testing/nurec-colmap-reconstruct.yaml"
    )
    for name in ("cache_dir", "scratch_dir"):
        spec.config.pop(name)
    spec.config.update({name: str(value) for name, value in paths.items()})
    convert = build_plan(spec, run_id="staging-contract").steps[0]
    result = CliRunner().invoke(app, convert.argv[1:])
    assert result.exit_code == 0, result.output
    assert len(seen) == 3
    for request in seen:
        assert request.cache_dir == model.cache_dir
        assert request.scratch_dir == model.scratch_dir
    if not override:
        assert model.cache_dir.parts[0] == model.scratch_dir.parts[0] == "~"
