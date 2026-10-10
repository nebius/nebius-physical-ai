"""Reject withdrawn workflow defaults before work and preserve exact operator images."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from npa.orchestration.npa_workflow import interpreter, runtime
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.readiness import load_readiness_record
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_skypilot_yaml,
    validate_immutable_image_override_bindings,
)
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit import merge_config_overrides

ROOT = Path(__file__).resolve().parents[4]
ISAAC = "registry.example.invalid/npa-isaac-lab@sha256:" + "a" * 64
CPU = "registry.example.invalid/npa-sonic@sha256:" + "b" * 64
EXACT_INPUT_SPECS = (
    "field-failure-reference-demo.yaml",
    "multicamera-rgbd-warehouse.yaml",
    "rgbd-scan-to-policy-demo.yaml",
    "rgbd-scan-to-isaac.yaml",
    "scan-to-isaac-navigation.yaml",
)
GOVERNED_SPECS = ("franka-rl-transfer.yaml",)


def _workflow_path(name):
    """Return the catalog location for an exact-image workflow fixture."""
    directory = "main" if name == "rgbd-scan-to-policy-demo.yaml" else "testing"
    return ROOT / "workflows" / directory / name


def _spec(tmp_path, config):
    path = tmp_path / "images.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "npa.workflow/v0.0.1",
                "kind": "Workflow",
                "metadata": {"name": "exact-images"},
                "config": config,
                "initial": "work",
                "states": {
                    "work": {"run": {"argv": ["echo", "ready"]}, "terminal": True}
                },
            }
        )
    )
    return load_spec(path)


@pytest.mark.parametrize(
    "image",
    [
        "",
        "tool://isaac-lab",
        "registry.example.invalid/runtime:latest",
        "namespace/runtime@sha256:" + "a" * 64,
        "registry.example.invalid/runtime:tag@sha256:" + "a" * 64,
        "registry.example.invalid/runtime@sha256:" + "a" * 63,
        "registry.example.invalid/runtime@sha256:" + "A" * 64,
        " " + ISAAC,
        "registry.example.invalid/team___simulation/runtime@sha256:" + "a" * 64,
        "registry.example.invalid/team/runtime..1@sha256:" + "a" * 64,
        "-registry.example.invalid/runtime@sha256:" + "a" * 64,
        "registry_.example.invalid/runtime@sha256:" + "a" * 64,
        "[2001:db8::1%zone]/runtime@sha256:" + "a" * 64,
        None,
        7,
    ],
)
def test_invalid_provenance_image_fails_before_state_expansion(
    tmp_path, monkeypatch, image
):
    spec = _spec(tmp_path, {"required_immutable_images": ["runtime"], "runtime": image})
    expand = Mock(side_effect=AssertionError("must not expand executable states"))
    monkeypatch.setattr(interpreter, "_expand_state", expand)
    with pytest.raises(NpaWorkflowError, match="--var runtime="):
        interpreter.build_plan(spec)
    expand.assert_not_called()


def test_all_invalid_required_images_are_reported_in_one_refusal(tmp_path):
    spec = _spec(
        tmp_path,
        {
            "required_immutable_images": ["cpu", "gpu"],
            "cpu": "",
            "gpu": "tool://isaac-lab",
        },
    )

    with pytest.raises(NpaWorkflowError) as raised:
        interpreter.build_plan(spec)

    message = str(raised.value)
    assert "Each of config.cpu, config.gpu requires" in message
    assert "--var cpu=" in message and "--var gpu=" in message


@pytest.mark.parametrize(
    "declaration", ["runtime", None, [7], [""], ["a/b"], ["runtime", "runtime"]]
)
def test_malformed_opt_in_contract_is_an_actionable_workflow_error(
    tmp_path, declaration
):
    spec = _spec(tmp_path, {"required_immutable_images": declaration, "runtime": ISAAC})
    with pytest.raises(NpaWorkflowError, match="required_immutable_images"):
        interpreter.build_plan(spec)


def test_overrides_and_config_tokens_resolve_before_exact_input_validation(tmp_path):
    spec = _spec(
        tmp_path,
        {
            "required_immutable_images": ["runtime"],
            "runtime": "{{config.operator_image}}",
            "operator_image": "",
        },
    )
    supplied = merge_config_overrides(spec, {"operator_image": ISAAC})
    assert interpreter.build_plan(supplied).steps[0].argv == ["echo", "ready"]


def test_workflows_without_opt_in_keep_existing_image_selection_contract(tmp_path):
    spec = _spec(tmp_path, {"runtime": "tool://isaac-lab"})
    assert interpreter.build_plan(spec).steps[0].argv == ["echo", "ready"]


@pytest.mark.parametrize(
    "repository",
    [
        "localhost/runtime",
        "registry.example.invalid:5000/team/runtime",
        "ghcr.io/team/project/runtime",
        "ghcr.io/team__simulation/runtime---1",
        "registry--mirror.example.invalid/team/runtime",
        "Registry--Mirror.example.invalid/team/runtime",
        "[2001:db8::1]/team/runtime",
        "[2001:db8::1]:5000/team/runtime",
    ],
)
def test_repository_qualified_immutable_inputs_are_accepted(tmp_path, repository):
    image = repository + "@sha256:" + "a" * 64
    spec = _spec(tmp_path, {"required_immutable_images": ["runtime"], "runtime": image})
    assert interpreter.build_plan(spec).steps[0].argv == ["echo", "ready"]


def test_missing_image_fails_before_runtime_store_or_executor(tmp_path, monkeypatch):
    spec = _spec(tmp_path, {"required_immutable_images": ["runtime"], "runtime": ""})
    store_factory, executor, store = Mock(), Mock(), Mock()
    monkeypatch.setattr(interpreter, "store_for_config", store_factory)
    for state_store in (None, store):
        with pytest.raises(NpaWorkflowError, match="--var runtime="):
            interpreter.run_workflow(
                spec,
                run_id="denied",
                execute=True,
                persist_state=True,
                state_store=state_store,
                step_executor=executor,
            )
    store_factory.assert_not_called()
    store.write_manifest.assert_not_called()
    executor.assert_not_called()


def test_missing_image_fails_before_runtime_tier_store_or_executor(
    tmp_path, monkeypatch
):
    spec = _spec(tmp_path, {"required_immutable_images": ["runtime"], "runtime": ""})
    store_factory = Mock(return_value=None)
    executor_factory = Mock()
    monkeypatch.setattr(runtime, "store_for_config", store_factory)
    monkeypatch.setattr(runtime, "SkyPilotWaveExecutor", executor_factory)

    with pytest.raises(NpaWorkflowError, match="--var runtime="):
        runtime.run_workflow_runtime(spec, run_id="runtime-denied")

    store_factory.assert_not_called()
    executor_factory.assert_not_called()


def test_tagged_digest_explains_the_required_canonical_form(tmp_path):
    tagged_digest = "registry.example.invalid/runtime:tag@sha256:" + "a" * 64
    spec = _spec(
        tmp_path,
        {"required_immutable_images": ["runtime"], "runtime": tagged_digest},
    )

    with pytest.raises(
        NpaWorkflowError, match=r"remove any :tag segment before @sha256"
    ):
        interpreter.build_plan(spec)


@pytest.mark.parametrize("name", EXACT_INPUT_SPECS)
def test_shipped_exact_image_consumers_require_operator_input_before_planning(name):
    spec = load_spec(_workflow_path(name))
    with pytest.raises(NpaWorkflowError, match="requires an explicit.*--var"):
        interpreter.build_plan(spec, run_id="missing-images")


def _supplied_spec(name):
    spec = load_spec(_workflow_path(name))
    images = {
        "isaac_image": ISAAC,
        "navigation_image": ISAAC,
        "assembly_image": CPU,
        "reconstruction_image": CPU,
    }
    return merge_config_overrides(
        spec, {key: images[key] for key in spec.config["required_immutable_images"]}
    )


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"workbench.open3d": ISAAC}, ISAAC),
        ({"*": ""}, "<SkyPilot default>"),
    ],
)
def test_required_image_override_binding_rejects_direct_renderer_bypass(
    overrides, expected
):
    spec = _supplied_spec("rgbd-scan-to-policy-demo.yaml")
    plan = interpreter.build_plan(spec, run_id="protected-image-override")
    step = replace(plan.steps[0], tool_ref="workbench.open3d.register")

    with pytest.raises(NpaWorkflowError, match="image selection changes") as raised:
        validate_immutable_image_override_bindings(
            spec,
            (step,),
            run_id="protected-image-override",
            options=SkypilotRenderOptions(image_overrides=overrides),
        )

    assert expected in str(raised.value)


def test_required_image_override_binding_allows_the_declared_digest():
    spec = _supplied_spec("rgbd-scan-to-policy-demo.yaml")
    plan = interpreter.build_plan(spec, run_id="protected-image-allow")
    step = replace(plan.steps[0], tool_ref="workbench.open3d.register")
    declared = str(step.resources_profile["image"])

    validate_immutable_image_override_bindings(
        spec,
        (step,),
        run_id="protected-image-allow",
        options=SkypilotRenderOptions(image_overrides={"workbench.open3d": declared}),
    )


def test_required_image_binding_rejects_a_digest_pin_replacement():
    spec = _supplied_spec("rgbd-scan-to-policy-demo.yaml")
    plan = interpreter.build_plan(spec, run_id="protected-image-pin")
    step = plan.steps[0]
    declared = str(step.resources_profile["image"])

    with pytest.raises(NpaWorkflowError, match="image selection changes"):
        validate_immutable_image_override_bindings(
            spec,
            (step,),
            run_id="protected-image-pin",
            options=SkypilotRenderOptions(image_digest_pins={declared: ISAAC}),
        )


@pytest.mark.parametrize("name", EXACT_INPUT_SPECS)
def test_exact_operator_images_reach_real_renderer_and_native_provenance(
    name, monkeypatch
):
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source-fixture")
    spec = _supplied_spec(name)
    plan = interpreter.build_plan(spec, run_id="explicit-images")
    text = render_skypilot_yaml(
        spec,
        plan,
        run_id="explicit-images",
        options=SkypilotRenderOptions(
            registry="registry.example.invalid",
            materialize_registry_secrets=False,
        ),
    )
    tasks = [task for task in yaml.safe_load_all(text) if task and "run" in task]
    gpu = [task for task in tasks if "accelerators" in task["resources"]]
    assert gpu and all(
        task["resources"]["image_id"] == "docker:" + ISAAC for task in gpu
    )
    native = [step for step in plan.steps if "--runtime-image" in step.argv]
    if native:
        assert all(
            step.argv[step.argv.index("--runtime-image") + 1] == ISAAC
            for step in native
        )
    assert "tool://isaac-lab" not in text


@pytest.mark.parametrize("name", GOVERNED_SPECS)
def test_governed_gpu_defaults_remain_quarantined_until_accepted(name, monkeypatch):
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source-fixture")
    spec = load_spec(ROOT / "workflows/testing" / name)
    with pytest.raises(NpaWorkflowError, match="isaac-lab.*quarantined"):
        render_skypilot_yaml(
            spec,
            interpreter.build_plan(spec),
            run_id="denied",
            options=SkypilotRenderOptions(
                materialize_registry_secrets=False,
            ),
        )


@pytest.mark.parametrize("name", GOVERNED_SPECS)
def test_governed_gpu_resource_accepts_exact_operator_var_without_changing_cpu(
    name, monkeypatch
):
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source-fixture")
    spec = merge_config_overrides(
        load_spec(ROOT / "workflows/testing" / name), {"isaac_image": ISAAC}
    )
    text = render_skypilot_yaml(
        spec,
        interpreter.build_plan(spec),
        run_id="explicit",
        options=SkypilotRenderOptions(
            materialize_registry_secrets=False,
        ),
    )
    tasks = [task for task in yaml.safe_load_all(text) if task and "run" in task]
    assert all(
        task["resources"]["image_id"] == "docker:" + ISAAC
        for task in tasks
        if "accelerators" in task["resources"]
    )
    assert all(
        task["resources"].get("image_id") != "docker:" + ISAAC
        for task in tasks
        if "accelerators" not in task["resources"]
    )


def test_no_shipped_workflow_automatically_selects_withdrawn_isaac_or_sonic_bytes():
    for path in (ROOT / "workflows").rglob("*.yaml"):
        document = yaml.safe_load(path.read_text())
        if not isinstance(document, dict):
            continue
        automatic = [*document.get("config", {}).values()]
        automatic.extend(
            item.get("image", "") for item in document.get("resources", {}).values()
        )
        for value in automatic:
            if isinstance(value, str):
                assert not any(
                    f"ghcr.io/nebius/nebius-physical-ai/npa-{tool}@sha256:" in value
                    for tool in ("isaac-lab", "sonic")
                ), path


@pytest.mark.parametrize("name", [*EXACT_INPUT_SPECS, *GOVERNED_SPECS])
def test_historical_records_remain_bound_and_current_records_await_workload_proof(name):
    archived = ROOT / "docs/workbench/evidence/workflow-defaults-807" / name
    old = load_readiness_record(archived.with_suffix(".readiness.json"))
    current = load_readiness_record(_workflow_path(name).with_suffix(".readiness.json"))
    assert old["workflow_sha256"] != current["workflow_sha256"]
    assert current["planning"]["task_fidelity"]["status"] == "unverified"
    assert current["prerequisites"]["source_image"]["status"] == "unverified"
    assert current["prerequisites"]["target_runtime"]["status"] == "unverified"
