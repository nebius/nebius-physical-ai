from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def image_lineage_assertion(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    module = importlib.import_module("tests.e2e.test_npa_workflow_submit_live_e2e")
    return module._assert_paidf_stage_image_lineage


@pytest.fixture
def native_stage_spec(tmp_path: Path) -> Path:
    path = tmp_path / "workflow.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "npa.workflow/v0.0.1",
                "kind": "Workflow",
                "metadata": {"name": "image-lineage-fixture"},
                "config": {
                    "image": "registry.example.test/label@sha256:" + "1" * 64,
                    "final_dataset_uri": "s3://fixture/dataset.json",
                    "terminal_validation_uri": "s3://fixture/reports/label.json",
                },
                "resources": {"label": {"cpus": 1, "image": "{{config.image}}"}},
                "initial": "label",
                "states": {
                    "label": {
                        "resources": "label",
                        "toolRef": "workflow.paidf.validate_dataset",
                        "terminal": True,
                        "outputs": [
                            {
                                "uri": "s3://fixture/reports/label.json",
                                "schema": "npa.paidf.native.iaa-terminal-validation.v1",
                            }
                        ],
                    }
                },
            }
        )
    )
    return path


@pytest.mark.parametrize("override_kind", ["config", "global-image", "tool-image"])
def test_native_live_evidence_follows_submitted_image_overrides(
    native_stage_spec: Path, image_lineage_assertion, override_kind: str
) -> None:
    image = "registry.example.test/override@sha256:" + "2" * 64
    config_vars = (("image", image),) if override_kind == "config" else ()
    image_args = {
        "config": [],
        "global-image": ["--image", image],
        "tool-image": ["--image-override", f"workflow.paidf={image}"],
    }[override_kind]
    report = {
        "schema": "npa.paidf.native.iaa-terminal-validation.v1",
        "run_id": "fixture-run",
        "runtime_image": image,
    }
    assert (
        image_lineage_assertion(
            spec_path=native_stage_spec,
            config_vars=config_vars,
            image_args=image_args,
            registry="",
            run_id="fixture-run",
            read_artifact=lambda _uri: json.dumps(report).encode(),
        )
        == 1
    )


@pytest.mark.parametrize(
    "recorded_image",
    [
        None,
        "registry.example.test/vendor-parent@sha256:" + "1" * 64,
        "registry.example.test/label@sha256:" + "2" * 64,
    ],
    ids=["missing", "parent-instead-of-wrapper", "different-wrapper-digest"],
)
def test_native_live_evidence_rejects_missing_or_incorrect_runtime_image(
    native_stage_spec: Path, image_lineage_assertion, recorded_image: str | None
) -> None:
    report = {
        "schema": "npa.paidf.native.iaa-terminal-validation.v1",
        "run_id": "fixture-run",
        "runtime_image": recorded_image,
    }
    with pytest.raises(AssertionError, match="runtime image provenance mismatch"):
        image_lineage_assertion(
            spec_path=native_stage_spec,
            config_vars=(),
            image_args=[],
            registry="",
            run_id="fixture-run",
            read_artifact=lambda _uri: json.dumps(report).encode(),
        )
