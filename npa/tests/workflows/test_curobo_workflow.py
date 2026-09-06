"""Full benchmark workflow stages, handoffs and actual tool argv."""

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX
from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions
from npa.orchestration.npa_workflow.submit import prepare_npa_workflow_for_submit


def test_complete_benchmark_and_factual_rrd_are_reachable():
    path = (
        Path(__file__).resolve().parents[2]
        / "workflows/workbench/npa-workflows/curobo-benchmark.yaml"
    )
    spec = yaml.safe_load(path.read_text())
    assert spec["config"]["curobo_mode"] == "both"
    assert len(spec["states"]) == 4
    for state in spec["states"].values():
        argv = TOOL_CATALOG[state["toolRef"]].argv_template
        assert argv[:3] == ["npa", "workbench", "curobo"]
        assert "--output-path" in argv
    assert any(
        x["schema"] == "application/vnd.rerun.rrd"
        for x in spec["states"]["visualize"]["outputs"]
    )
    case = next(c for c in SUBMIT_LIVE_MATRIX if c.spec == path.name)
    assert not case.plan_only and case.image_tool == "curobo"


@pytest.mark.parametrize("source_matches", [True, False])
def test_full_workflow_baked_setup_uses_real_import_and_image_identity(tmp_path, source_matches):
    path = Path(__file__).resolve().parents[2] / "workflows/workbench/npa-workflows/curobo-benchmark.yaml"
    sha = "a" * 40
    image = "ghcr.io/nebius/nebius-physical-ai/npa-curobo@sha256:" + "b" * 64
    prepared = prepare_npa_workflow_for_submit(
        path,
        run_id="curobo-baked-contract",
        config_overrides={"require_baked_npa": "1", "baked_npa_import": "npa.cli.workbench.curobo", "source_sha": sha},
        render_options=SkypilotRenderOptions(image_overrides={"*": image}, materialize_registry_secrets=False),
    )
    try:
        documents = list(yaml.safe_load_all(prepared.skypilot_yaml_path.read_text()))
        tasks = [doc for doc in documents if doc and "setup" in doc]
        assert len(tasks) == 4
        for index, task in enumerate(tasks):
            assert task["resources"]["image_id"] == "docker:" + image
            assert task["envs"]["NPA_SIM2REAL_SOURCE_SHA"] == sha
            record = tmp_path / f"interpreter-{index}"
            setup = task["setup"].replace("/tmp/npa-python", str(record))
            assert "pip install" not in setup
            result = subprocess.run(
                ["/bin/bash", "-c", setup],
                env={"PATH": "/usr/bin:/bin", "NPA_BAKED_PYTHON": sys.executable,
                     "NPA_IMAGE_SOURCE_SHA": sha if source_matches else "c" * 40,
                     "NPA_SIM2REAL_SOURCE_SHA": sha},
                capture_output=True, text=True, check=False,
            )
            if source_matches:
                assert result.returncode == 0, result.stderr
                assert record.read_text().strip() == sys.executable
            else:
                assert result.returncode != 0
                assert "attestation does not match" in result.stderr
                assert not record.exists()
    finally:
        prepared.temp_dir.cleanup()
