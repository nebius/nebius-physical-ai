from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
SINGLE_YAML = (
    ROOT
    / "npa"
    / "src"
    / "npa"
    / "workflows"
    / "byof"
    / "profiles"
    / "isaac-lab-rl-train.yaml"
)
# The raw isaac-lab-rl-sweep template is retired; its npa.workflow spec is the surface
# (live-verified on four GPUs in two batches plus a barrier — EVIDENCE §R3).
SWEEP_SPEC = ROOT / "workflows" / "testing" / "isaac-lab-rl-sweep.yaml"
WRAPPER_PATH = ROOT / "npa" / "scripts" / "run_isaac_lab_rl.py"
FUNCTIONAL_SMOKE = (
    ROOT / "npa" / "docker" / "workbench" / "isaac-lab" / "smoke_functional.py"
)


def _docs(path: Path) -> list[dict]:
    return [
        doc
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if doc is not None
    ]


def _load_wrapper_module():
    spec = importlib.util.spec_from_file_location("run_isaac_lab_rl", WRAPPER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_functional_smoke_is_explicitly_an_environment_step_probe() -> None:
    text = FUNCTIONAL_SMOKE.read_text(encoding="utf-8")

    assert "npa_isaac_lab_environment_step_trace_v1" in text
    assert "action_source" in text
    assert "run short training session" not in text
    assert "checkpoint" not in text.lower()


def test_isaac_lab_single_job_yaml_uses_rt_core_gpu_and_rsl_rl_entrypoint() -> None:
    docs = _docs(SINGLE_YAML)

    assert docs[0] == {"name": "isaac-lab-rl-train", "execution": "serial"}
    task = docs[1]
    assert task["resources"]["cloud"] == "kubernetes"
    assert task["resources"]["accelerators"] == "L40S:1"
    assert task["resources"]["cpus"] == 16
    assert task["resources"]["memory"] == 64
    assert task["resources"]["image_id"] == "tool://isaac-lab"
    assert "scripts/reinforcement_learning/rsl_rl/train.py" in task["run"]
    assert "--num_envs" in task["run"]
    assert "--max_iterations" in task["run"]
    assert "--visualizer none" in task["run"]
    assert "agent.save_interval=1" in task["envs"]["ISAAC_LAB_HYDRA_OVERRIDES"]
    # SkyPilot does not interpolate ${VAR} in envs; ship a concrete endpoint.
    assert task["envs"]["AWS_ENDPOINT_URL"] == "https://storage.eu-north1.nebius.cloud"
    assert "${AWS_ENDPOINT_URL}" not in task["envs"]["AWS_ENDPOINT_URL"]


def test_isaac_lab_yaml_files_have_no_literal_aws_endpoint_placeholders() -> None:
    yaml_paths = [
        SINGLE_YAML,
        ROOT
        / "npa"
        / "src"
        / "npa"
        / "workflows"
        / "byof"
        / "profiles"
        / "isaac-lab-rl-train-rtxpro.yaml",
        ROOT
        / "npa"
        / "src"
        / "npa"
        / "workflows"
        / "byof"
        / "profiles"
        / "isaac-lab-rl-train-rtxpro-smoke.yaml",
        ROOT
        / "npa"
        / "src"
        / "npa"
        / "workflows"
        / "byof"
        / "profiles"
        / "byof-datagen-rtxpro-smoke.yaml",
        ROOT
        / "npa"
        / "src"
        / "npa"
        / "workflows"
        / "byof"
        / "profiles"
        / "byof-container-smoke-rtxpro.yaml",
    ]
    for path in yaml_paths:
        text = path.read_text(encoding="utf-8")
        assert 'AWS_ENDPOINT_URL: "${AWS_ENDPOINT_URL}"' not in text, path
        docs = _docs(path)
        for doc in docs[1:]:
            envs = doc.get("envs") or {}
            if "AWS_ENDPOINT_URL" in envs:
                assert (
                    envs["AWS_ENDPOINT_URL"] == "https://storage.eu-north1.nebius.cloud"
                )


def test_isaac_lab_sweep_spec_uses_parallel_group_and_distinct_variants() -> None:
    """Same contract as the retired `execution: parallel` template, on the spec.

    The template declared four sibling SkyPilot tasks with per-variant ``RUN_VARIANT``
    and ``S3_OUTPUT_PREFIX`` envs; the spec declares one ``parallel:`` group whose
    members differ only by their ``params:`` overlay.
    """

    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec

    spec = load_spec(SWEEP_SPEC)
    groups = [state for state in spec.states.values() if state.parallel]

    assert len(groups) == 1
    members = groups[0].parallel
    assert [spec.states[name].params["variant"] for name in members] == [
        "lr-1e-3",
        "lr-3e-4",
        "entropy-0",
        "entropy-0-01",
    ]
    for name in members:
        member = spec.states[name]
        assert (
            spec.resources[member.resources]["accelerators"]
            == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
        )
        # Each variant writes under its own prefix, as the template's envs did.
        assert (
            member.params["variant_uri"].rstrip("/").endswith(member.params["variant"])
        )

    # The real RSL-RL entrypoint still runs in-pod, via the sweep stage module.
    plan = build_plan(spec, run_id="probe")
    sweep_steps = [step for step in plan.steps if step.group]
    assert len(sweep_steps) == 4
    assert all("npa.workflows.rl_sweep" in step.shell for step in sweep_steps)


def test_isaac_lab_runner_renders_and_submits(monkeypatch, tmp_path, capsys) -> None:
    wrapper = _load_wrapper_module()
    sky_bin = tmp_path / "sky"
    sky_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    sky_bin.chmod(sky_bin.stat().st_mode | stat.S_IXUSR)
    captured = {}

    def fake_submit_workflow(yaml_path, run_id, **kwargs):
        captured["run_id"] = run_id
        captured["kwargs"] = kwargs
        captured["docs"] = [
            doc
            for doc in yaml.safe_load_all(Path(yaml_path).read_text(encoding="utf-8"))
            if doc is not None
        ]
        return wrapper.WorkflowResult(
            status="SUBMITTED",
            job_id="42",
            returncode=0,
            log_paths={"config": str(tmp_path / "config.yaml")},
        )

    def fake_workflow_status(job_id, **kwargs):
        return wrapper.WorkflowResult(status="SUCCEEDED", job_id=job_id, returncode=0)

    monkeypatch.setattr(wrapper, "submit_workflow", fake_submit_workflow)
    monkeypatch.setattr(wrapper, "workflow_status", fake_workflow_status)

    rc = wrapper.main(
        [
            "--yaml",
            str(SINGLE_YAML),
            "--run-id",
            "isaac-test-run",
            "--task",
            "Isaac-Cartpole-v0",
            "--iterations",
            "3",
            "--project",
            "test-project",
            "--context",
            "test-context",
            "--output-root",
            "s3://bucket/isaac-lab-rl",
            "--image",
            "registry.example/npa-isaac-lab:test",
            "--sky-bin",
            str(sky_bin),
            "--poll-interval",
            "0",
            "--wait-timeout",
            "-1",
        ]
    )

    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert (
        output["outputs"]["checkpoint"]
        == "s3://bucket/isaac-lab-rl/isaac-test-run/npa_isaac_lab_checkpoint.pt"
    )
    assert captured["run_id"] == "isaac-test-run"
    assert captured["kwargs"]["project"] == "test-project"
    assert captured["kwargs"]["infra"] == "k8s/test-context"
    rendered_task = captured["docs"][1]
    assert rendered_task["envs"]["ISAAC_LAB_ITERATIONS"] == "3"
    assert (
        rendered_task["envs"]["S3_OUTPUT_PREFIX"]
        == "s3://bucket/isaac-lab-rl/isaac-test-run/"
    )
    assert json.loads(rendered_task["envs"]["NPA_EXECUTION_OUTPUTS"]) == [
        {
            "uri": "s3://bucket/isaac-lab-rl/isaac-test-run/",
            "kind": "directory",
        }
    ]
    assert (
        rendered_task["resources"]["image_id"]
        == "docker:registry.example/npa-isaac-lab:test"
    )
    assert (
        rendered_task["envs"]["AWS_ENDPOINT_URL"]
        == "https://storage.eu-north1.nebius.cloud"
    )
    assert (
        rendered_task["envs"]["NEBIUS_S3_ENDPOINT"]
        == "https://storage.eu-north1.nebius.cloud"
    )


def test_isaac_lab_runner_materializes_endpoint_from_env(monkeypatch, tmp_path) -> None:
    wrapper = _load_wrapper_module()
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.custom.example")
    docs = wrapper.render_workflow(
        SINGLE_YAML,
        run_id="endpoint-env",
        task="Isaac-Cartpole-v0",
        iterations=1,
        output_root="s3://bucket/isaac-lab-rl",
        image="registry.example/isaac-lab@sha256:" + "1" * 64,
    )
    envs = docs[1]["envs"]
    assert envs["AWS_ENDPOINT_URL"] == "https://storage.custom.example"
    assert envs["NEBIUS_S3_ENDPOINT"] == "https://storage.custom.example"
    assert envs["NPA_CHECKPOINT_S3_ENDPOINT_URL"] == "https://storage.custom.example"


def test_isaac_lab_runner_render_only_keeps_rendered_yaml(capsys) -> None:
    wrapper = _load_wrapper_module()

    rc = wrapper.main(
        [
            "--yaml",
            str(SINGLE_YAML),
            "--run-id",
            "isaac-render-only",
            "--task",
            "Isaac-Cartpole-v0",
            "--iterations",
            "2",
            "--render-only",
            "--image",
            "registry.example/isaac-lab@sha256:" + "1" * 64,
        ]
    )

    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    rendered = Path(output["rendered_yaml"])
    assert rendered.is_file()
    docs = [
        doc
        for doc in yaml.safe_load_all(rendered.read_text(encoding="utf-8"))
        if doc is not None
    ]
    assert docs[1]["envs"]["NPA_ISAAC_LAB_RUN_ID"] == "isaac-render-only"


def test_isaac_lab_runner_requires_live_context_before_teardown(
    monkeypatch, tmp_path
) -> None:
    wrapper = _load_wrapper_module()
    monkeypatch.setattr(wrapper, "resolve_byof_project", lambda: "test-project")
    monkeypatch.setattr(
        wrapper,
        "resolve_byof_kubernetes_target",
        lambda _project: type("Target", (), {"context": ""})(),
    )

    args = wrapper._parse_args(
        [
            "--yaml",
            str(SINGLE_YAML),
            "--run-id",
            "missing-context",
            "--isolated-config-dir",
            str(tmp_path / "sky"),
            "--image",
            "registry.example/isaac-lab@sha256:" + "1" * 64,
        ]
    )

    with pytest.raises(ValueError, match="live submission requires --context"):
        wrapper._submit_and_wait(args)

    assert not (tmp_path / "sky").exists()


ISAAC_PROFILES = sorted(SINGLE_YAML.parent.glob("isaac-lab-rl-train*.yaml"))


@pytest.mark.parametrize("profile", ISAAC_PROFILES, ids=lambda path: path.name)
def test_isaac_profiles_reject_withdrawn_default_in_real_renderer(profile) -> None:
    wrapper = _load_wrapper_module()
    assert _docs(profile)[1]["resources"]["image_id"] == "tool://isaac-lab"

    with pytest.raises(ValueError, match="has no consumable public release"):
        wrapper.render_workflow(
            profile, run_id="quarantined-image", task="Isaac-Cartpole-v0", iterations=1
        )


@pytest.mark.parametrize("profile", ISAAC_PROFILES, ids=lambda path: path.name)
@pytest.mark.parametrize("docker_prefix", ["", "docker:"])
def test_isaac_profiles_preserve_explicit_operator_image(
    profile, docker_prefix
) -> None:
    wrapper = _load_wrapper_module()
    image = "registry.example/isaac-lab@sha256:" + "1" * 64

    docs = wrapper.render_workflow(
        profile,
        run_id="operator-image",
        task="Isaac-Cartpole-v0",
        iterations=1,
        image=docker_prefix + image,
    )

    assert docs[1]["resources"]["image_id"] == "docker:" + image


def test_isaac_renderer_preserves_operator_profile_image(tmp_path) -> None:
    wrapper = _load_wrapper_module()
    docs = _docs(SINGLE_YAML)
    image = "docker:registry.example/isaac-lab@sha256:" + "2" * 64
    docs[1]["resources"]["image_id"] = image
    profile = tmp_path / "operator-profile.yaml"
    profile.write_text(yaml.safe_dump_all(docs))

    rendered = wrapper.render_workflow(
        profile, run_id="operator-profile", task="Isaac-Cartpole-v0", iterations=1
    )

    assert rendered[1]["resources"]["image_id"] == image


def test_isaac_renderer_consumes_the_governed_resolver_result(monkeypatch) -> None:
    wrapper = _load_wrapper_module()
    selected = "registry.example/isaac-lab@sha256:" + "3" * 64
    seen = []

    def resolve(tool):
        seen.append(tool)
        return selected

    monkeypatch.setattr(wrapper, "container_image_for_tool", resolve)
    docs = wrapper.render_workflow(
        SINGLE_YAML, run_id="governed-image", task="Isaac-Cartpole-v0", iterations=1
    )

    assert seen == ["isaac-lab"]
    assert docs[1]["resources"]["image_id"] == "docker:" + selected


def test_byof_cookbook_requires_an_explicit_base_image() -> None:
    cookbook = ROOT / "docs/workbench/cookbooks/byof-isaac-lab"
    dockerfile = (cookbook / "Dockerfile.example").read_text()

    assert "ARG NPA_ISAAC_LAB_BASE_IMAGE\n" in dockerfile
    assert "FROM ${NPA_ISAAC_LAB_BASE_IMAGE}" in dockerfile
    assert "NPA_ISAAC_LAB_BASE_IMAGE=" not in dockerfile
    assert (
        "NPA_ISAAC_LAB_BASE_IMAGE=$NPA_ISAAC_LAB_BASE_IMAGE"
        in (cookbook / "README.md").read_text()
    )


@pytest.mark.parametrize("render_only", [False, True])
def test_isaac_default_refusal_precedes_runtime_scope_and_submit(
    monkeypatch, capsys, render_only
) -> None:
    wrapper = _load_wrapper_module()

    def unexpected_runtime(*_args, **_kwargs):
        pytest.fail("quarantined image must fail before runtime scope or submission")

    for name in ("_execution_scope", "resolve_sky_bin", "submit_workflow"):
        monkeypatch.setattr(wrapper, name, unexpected_runtime)
    arguments = ["--yaml", str(SINGLE_YAML), "--run-id", "quarantine-boundary"]
    if render_only:
        arguments.append("--render-only")

    assert wrapper.main(arguments) == 2
    output = capsys.readouterr()
    assert not output.out
    assert "Error:" in output.err and "quarantined" in output.err
    assert "Supply --image" in output.err
    assert "Traceback" not in output.err
