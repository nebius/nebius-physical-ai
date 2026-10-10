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


def test_quadruped_reuses_world_and_keeps_gpu_work_in_native_worker(monkeypatch):
    from npa.orchestration.npa_workflow.spec import load_spec
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.marble_credentials import marble_secret_names
    from npa.orchestration.npa_workflow.skypilot_render import (
        SkypilotRenderOptions,
        render_skypilot_yaml,
    )

    spec = load_spec(ROOT / "workflows/testing/marble-warehouse-quadruped.yaml")
    spec.config["world_uri"] = "s3://example-bucket/world"
    assert marble_secret_names(spec) == ()
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source")
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="test-go1"),
        run_id="test-go1",
        options=SkypilotRenderOptions(
            registry="registry.example", materialize_registry_secrets=False
        ),
    )
    jobs = [doc for doc in yaml.safe_load_all(rendered) if doc and "run" in doc]
    assert len(jobs) == 2
    assert jobs[0]["resources"]["accelerators"] == "RTXPRO6000:1"
    assert "quadruped-collect" in jobs[0]["run"]
    assert "--sensor-hz 25" in jobs[0]["run"] and "--samples 32" in jobs[0]["run"]
    assert "onnxruntime==1.23.2" in jobs[0]["setup"]
    assert "pycollada==0.9.3" in jobs[0]["setup"]
    assert "accelerators" not in jobs[1]["resources"]
    assert "imageio-ffmpeg==0.6.0" in jobs[1]["setup"]


def test_rover_workflow_routes_both_sensors_to_one_real_gpu_stage(monkeypatch):
    from npa.orchestration.npa_workflow.spec import load_spec
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.marble_credentials import marble_secret_names
    from npa.orchestration.npa_workflow.skypilot_render import (
        SkypilotRenderOptions,
        render_skypilot_yaml,
    )

    spec = load_spec(ROOT / "workflows/testing/marble-warehouse-rover.yaml")
    assert marble_secret_names(spec) == ("WLT_API_KEY",)
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source")
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="test-rover"),
        run_id="test-rover",
        options=SkypilotRenderOptions(
            registry="registry.example", materialize_registry_secrets=False
        ),
    )
    jobs = [doc for doc in yaml.safe_load_all(rendered) if doc and "run" in doc]
    assert len(jobs) == 3
    assert "accelerators" not in jobs[0]["resources"]
    assert jobs[1]["resources"]["accelerators"] == "RTXPRO6000:1"
    assert "rover-collect" in jobs[1]["run"] and "--sensor-hz 12" in jobs[1]["run"]
    assert "pybullet==3.2.7" in jobs[1]["setup"]
    assert "warp-lang==1.17.0" in jobs[1]["setup"]
    assert "gsplat==1.5.3" in jobs[1]["setup"]
    assert "accelerators" not in jobs[2]["resources"]


def test_navigation_workflow_preserves_native_runtime_and_checkpoint_handoff(
    monkeypatch,
):
    from npa.orchestration.npa_workflow.spec import load_spec
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.skypilot_render import (
        SkypilotRenderOptions,
        render_skypilot_yaml,
    )

    image = "registry.example.invalid/npa-isaac-lab@sha256:" + "a" * 64
    spec = load_spec(ROOT / "workflows/testing/marble-navigation-rl.yaml")
    spec.config.update(world_uri="s3://example-bucket/world", isaac_image=image)
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source")
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="navigation-test"),
        run_id="navigation-test",
        options=SkypilotRenderOptions(
            registry="registry.example", materialize_registry_secrets=False
        ),
    )
    jobs = [doc for doc in yaml.safe_load_all(rendered) if doc and "run" in doc]
    assert len(jobs) == 3
    assert "accelerators" not in jobs[0]["resources"]
    assert "navigation-prepare" in jobs[0]["run"]
    assert "pybullet==3.2.7" in jobs[0]["setup"]
    assert "usd-core==26.8" in jobs[0]["setup"]
    for job in jobs[1:]:
        assert job["resources"]["accelerators"] == "RTXPRO6000:1"
        assert job["resources"]["image_id"] == "docker:" + image
        assert job["envs"]["NPA_TASK_IMAGE"] == image
        assert "npa.workflows.navigation.stages import run_stage" in job["run"]
    assert "/training/" in jobs[2]["run"]
    assert "/evaluation/" in jobs[2]["run"]
