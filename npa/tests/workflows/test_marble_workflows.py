"""Verify Marble stage contracts and honest live-matrix source selection."""

from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import yaml

from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.orchestration.npa_workflow.skypilot_render import TOOL_REF_IMAGE_TOOL
from npa.orchestration.npa_workflow.skypilot_render import _marble_runtime_setup
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "name,consumer",
    [("marble-world-capture", "capture"), ("marble-spatial-scan", "scan")],
)
def test_native_workflow_source_gpu_and_report(name, consumer):
    data = yaml.safe_load((ROOT / "workflows/testing" / f"{name}.yaml").read_text())
    assert data["config"]["world_source"] == "generate"
    assert data["config"]["source_overlay"] is True
    assert data["states"][consumer]["resources"] == "gpu"
    assert data["resources"]["gpu"]["accelerators"] == "RTXPRO6000:1"
    assert data["states"]["report"]["terminal"]
    for state in data["states"].values():
        assert state["toolRef"] in TOOL_CATALOG
        assert not TOOL_CATALOG[state["toolRef"]].stub
    assert TOOL_REF_IMAGE_TOOL[f"workbench.marble.{consumer}"] == "envgen"
    case = next(case for case in SUBMIT_LIVE_MATRIX if case.spec == f"{name}.yaml")
    assert not case.plan_only
    assert ("world_source", "sample-hobbit") in case.config_vars


def test_writable_marble_environment_retains_baked_libraries(tmp_path):
    base = tmp_path / "baked"
    writable = tmp_path / "writable"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(base)], check=True
    )
    interpreter = str(base / "bin/python")
    site = subprocess.check_output(
        [interpreter, "-c", "import site; print(site.getsitepackages()[0])"], text=True
    ).strip()
    marker = Path(site) / "baked_cuda_marker.py"
    marker.write_text("value = 73\n")
    script = _marble_runtime_setup().replace("/opt/npa/venv", str(base))
    venv_command = next(line for line in script.splitlines() if "-m venv " in line)
    script = script.replace(shlex.split(venv_command)[-1], str(writable))
    script += (
        "python - <<'PY'\n"
        "import baked_cuda_marker, pathlib, sys\n"
        "assert baked_cuda_marker.value == 73\n"
        f"assert pathlib.Path(sys.prefix) == pathlib.Path({str(writable)!r})\n"
        "PY\n"
    )
    subprocess.run(["/bin/bash", "-c", script], check=True)
    assert marker.read_text() == "value = 73\n"


def test_manufacturing_requires_generated_world_and_routes_gpu_consumers(monkeypatch):
    from npa.orchestration.npa_workflow.spec import load_spec
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.marble_credentials import marble_secret_names
    from npa.orchestration.npa_workflow.skypilot_render import (
        SkypilotRenderOptions,
        render_skypilot_yaml,
    )

    spec = load_spec(
        ROOT / "workflows/testing/marble-manufacturing-pallet-detection.yaml"
    )
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source")
    spec.config["world_source"] = "sample-hobbit"
    assert marble_secret_names(spec) == ("WLT_API_KEY",)
    plan = build_plan(spec, run_id="unit-manufacturing")
    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="unit-manufacturing",
        options=SkypilotRenderOptions(
            registry="registry.example", materialize_registry_secrets=False
        ),
    )
    docs = [doc for doc in yaml.safe_load_all(rendered) if doc]
    jobs = [doc for doc in docs if "run" in doc]
    assert len(jobs) == 5
    assert "--source generate" in jobs[1]["run"]
    assert "accelerators" not in jobs[0]["resources"]
    assert jobs[2]["resources"]["accelerators"] == "RTXPRO6000:1"
    assert "npa-envgen" in jobs[2]["resources"]["image_id"]
    assert "npa-detection-training" in jobs[3]["resources"]["image_id"]
    assert "accelerators" not in jobs[4]["resources"]


def test_sample_workflow_does_not_require_provider_key():
    from npa.orchestration.npa_workflow.spec import load_spec
    from npa.orchestration.npa_workflow.marble_credentials import marble_secret_names

    spec = load_spec(ROOT / "workflows/testing/marble-world-capture.yaml")
    assert marble_secret_names(spec) == ("WLT_API_KEY",)
    spec.config["world_source"] = "sample-hobbit"
    assert marble_secret_names(spec) == ()
