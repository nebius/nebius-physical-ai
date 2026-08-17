"""PAIDF wiring for segmentation-conditioned augmentation and region masks.

The capability is only usable if an operator can select it at submit time, so
these tests drive the shipped blueprint through the same path a submit takes:
config overrides -> plan -> rendered SkyPilot task command.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_skypilot_yaml,
)
from npa.orchestration.npa_workflow.submit import merge_config_overrides

BLUEPRINT = (
    Path(__file__).resolve().parents[3]
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "physical-ai-data-factory.yaml"
)
RUNNER = CliRunner()


@pytest.fixture(autouse=True)
def _src_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/prefix/npa")


def _augment_task(spec) -> dict:
    plan = build_plan(spec, run_id="paidf", assume_decision="promote_checkpoint")
    text = render_skypilot_yaml(
        spec,
        plan,
        run_id="paidf",
        options=SkypilotRenderOptions(image_overrides={"*": ""}),
    )
    docs = [doc for doc in yaml.safe_load_all(text) if doc]
    return next(doc for doc in docs[1:] if doc["name"].startswith("augment"))


def _augment_flags(spec) -> dict[str, str]:
    """Parse the rendered augment command back into ``{flag: value}``.

    Parsing rather than substring matching is what makes an empty value
    meaningful: an unquoted ``--control-prompt`` followed by ``--mask-asset``
    would read as the prompt *being* ``--mask-asset``.
    """

    task = _augment_task(spec)
    line = next(
        statement
        for statement in str(task["run"]).splitlines()
        if "cosmos2 transfer" in statement
    )
    words = shlex.split(line)
    flags: dict[str, str] = {}
    for index, word in enumerate(words):
        if not word.startswith("--"):
            continue
        following = words[index + 1] if index + 1 < len(words) else ""
        flags[word] = "" if following.startswith("--") else following
    return flags


def test_the_shipped_default_still_conditions_on_edge() -> None:
    spec = load_spec(BLUEPRINT)
    validate_spec(spec)

    flags = _augment_flags(spec)

    assert flags["--control"] == "edge"
    assert flags["--control-weight"] == "1.0"
    # Unset asset/prompt flags render as empty strings and the CLI reads empty as
    # "not set", so the default run conditions exactly the way it did before.
    assert flags["--control-asset"] == ""
    assert flags["--control-prompt"] == ""
    assert flags["--mask-asset"] == ""
    assert flags["--mask-prompt"] == ""


def test_submit_can_switch_the_run_to_segmentation_conditioning() -> None:
    spec = merge_config_overrides(
        load_spec(BLUEPRINT),
        {
            "augment_control": "seg",
            "augment_control_weight": "0.8",
            "augment_control_prompt": "robot arm, conveyor, bin",
            "augment_mask_prompt": "robot arm",
        },
    )
    flags = _augment_flags(spec)

    assert flags["--control"] == "seg"
    assert flags["--control-weight"] == "0.8"
    assert flags["--control-prompt"] == "robot arm, conveyor, bin"
    assert flags["--mask-prompt"] == "robot arm"
    # On-the-fly segmentation needs no asset, which is the point.
    assert flags["--control-asset"] == ""


def test_submit_can_supply_a_precomputed_segmentation_map() -> None:
    spec = merge_config_overrides(
        load_spec(BLUEPRINT),
        {
            "augment_control": "seg",
            "augment_control_asset_uri": "s3://example-bucket/seg/robot_seg.mp4",
        },
    )
    assert _augment_flags(spec)["--control-asset"] == (
        "s3://example-bucket/seg/robot_seg.mp4"
    )


def test_depth_requires_precomputed_control_at_validation() -> None:
    with pytest.raises(NpaWorkflowError, match="depth control requires"):
        merge_config_overrides(
            load_spec(BLUEPRINT), {"augment_control": "depth"}
        )
    spec = merge_config_overrides(
        load_spec(BLUEPRINT),
        {
            "augment_control": "depth",
            "augment_control_asset_uri": "s3://operator-owned/depth/control.mp4",
        },
    )
    assert _augment_flags(spec)["--control"] == "depth"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"augment_control_weight": "1.1"}, "range 0.0-1.0"),
        ({"augment_control": "edge", "augment_control_prompt": "arm"}, "not text-driven"),
        (
            {
                "augment_mask_asset_uri": "s3://bucket/mask.mp4",
                "augment_mask_prompt": "arm",
            },
            "mutually exclusive",
        ),
        ({"n_augmentations": "0"}, "must be >= 1"),
        (
            {"n_augmentations": "2", "augment_nodes": "3"},
            "num_nodes=3 exceeds n_augmentations=2",
        ),
    ],
)
def test_semantic_contract_rejects_invalid_overrides_before_plan(
    overrides: dict[str, str], match: str
) -> None:
    with pytest.raises(NpaWorkflowError, match=match):
        merge_config_overrides(load_spec(BLUEPRINT), overrides)


def test_plan_and_submit_share_the_early_semantic_preflight() -> None:
    bad = [
        "--var",
        "augment_control=edge",
        "--var",
        "augment_control_prompt=robot arm",
    ]
    planned = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(BLUEPRINT),
            "--run-id",
            "bad-plan",
            "--assume-decision",
            "promote_checkpoint",
            *bad,
        ],
    )
    submitted = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(BLUEPRINT),
            "--run-id",
            "bad-submit",
            *bad,
        ],
    )
    assert planned.exit_code != 0
    assert submitted.exit_code != 0
    assert "not text-driven" in planned.output
    assert "not text-driven" in submitted.output


def test_submit_capacity_preflight_uses_resolved_paidf_gang_and_free_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli.workbench.workflow import _preflight_submit_gang_capacity
    from npa.orchestration.skypilot import k8s_gpu_catalog as gpu_catalog

    def node(name: str, *, free: int = 1) -> gpu_catalog.KubernetesGpuNode:
        return gpu_catalog.KubernetesGpuNode(
            name=name,
            ready=True,
            schedulable=True,
            products=("RTXPRO-6000-BLACKWELL-SERVER-EDITION",),
            capacity=1,
            allocatable=1,
            committed=1 - free,
            free=free,
            allocatable_cpu_millis=64_000,
            free_cpu_millis=64_000,
            allocatable_memory_bytes=256 * 1024**3,
            free_memory_bytes=256 * 1024**3,
            allocatable_pods=110,
            free_pod_slots=110,
        )

    inventory = gpu_catalog.KubernetesGpuInventory(
        context="task-scoped-context",
        ready_nodes=2,
        eligible_gpu_nodes=2,
        capacity=2,
        allocatable=2,
        products=("RTXPRO-6000-BLACKWELL-SERVER-EDITION",),
        node_labels={},
        nodes=(node("gpu-a"), node("gpu-b")),
    )
    monkeypatch.setattr(
        gpu_catalog,
        "discover_kubernetes_gpu_inventory",
        lambda *, context: inventory,
    )
    spec = merge_config_overrides(
        load_spec(BLUEPRINT),
        {"augment_nodes": "2", "n_augmentations": "2"},
    )
    checks = _preflight_submit_gang_capacity(
        spec,
        context="task-scoped-context",
        accelerator_overrides={
            "RTXPRO6000:1": "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
        },
    )
    assert checks == [
        {
            "context": "task-scoped-context",
            "accelerator": "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1",
            "node_count": 2,
            "compatible_free_nodes": 2,
            "selected_nodes": ["gpu-a", "gpu-b"],
            "cpus_per_node": 16.0,
            "memory_bytes_per_node": 128 * 1024**3,
            "allowed_nodes": [],
            "state": "augment",
            "profile": "gpu",
        }
    ]

    monkeypatch.setattr(
        gpu_catalog,
        "discover_kubernetes_gpu_inventory",
        lambda *, context: gpu_catalog.KubernetesGpuInventory(
            **{
                **inventory.__dict__,
                "nodes": (node("gpu-a"), node("gpu-b", free=0)),
            }
        ),
    )
    with pytest.raises(gpu_catalog.UnsatisfiableAcceleratorError, match="requires 2"):
        _preflight_submit_gang_capacity(spec, context="task-scoped-context")


def test_submit_capacity_preflight_reads_exact_skypilot_allowed_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import yaml

    from npa.cli.workbench.workflow import _skypilot_allowed_nodes
    from npa.orchestration.skypilot import _bin

    config = tmp_path / "sky.yaml"
    config.write_text(
        yaml.safe_dump({"kubernetes": {"allowed_nodes": ["gpu-b", "gpu-a"]}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        _bin,
        "resolve_config",
        lambda **_kwargs: SimpleNamespace(global_config_path=config),
    )
    assert _skypilot_allowed_nodes(
        sky_bin="pinned-sky",
        config_path=config,
        isolated_config_dir=tmp_path / "isolated",
    ) == ("gpu-b", "gpu-a")


def test_submit_capacity_preflight_does_not_resolve_sky_for_cpu_only_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import npa.cli.workbench.workflow as workflow_cli

    cpu_only = (
        Path(__file__).resolve().parents[3]
        / "workflows"
        / "workbench"
        / "npa-workflows"
        / "token-factory-parallel-fanout.yaml"
    )
    spec = load_spec(cpu_only)
    monkeypatch.setattr(
        workflow_cli,
        "_skypilot_allowed_nodes",
        lambda **_kwargs: pytest.fail("CPU-only specs must not resolve SkyPilot affinity"),
    )

    assert workflow_cli._preflight_submit_gang_capacity(
        spec,
        context="",
        allowed_nodes=None,
    ) == []


def test_checkpoint_preflight_uses_state_local_modality_overlay() -> None:
    from npa.cli.workbench.workflow import _transfer_control_modalities

    spec = load_spec(BLUEPRINT)
    spec.config["state_control"] = "depth"
    spec.states["augment"].params["augment_control"] = "{{config.state_control}}"

    assert spec.config["augment_control"] == "edge"
    assert _transfer_control_modalities(spec, run_id="submit-run") == {"depth"}


def test_validate_and_plan_resolve_state_local_control_tokens() -> None:
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import validate_spec

    spec = load_spec(BLUEPRINT)
    spec.config["state_control"] = "depth"
    spec.states["augment"].params.update(
        {
            "augment_control": "{{config.state_control}}",
            "augment_control_asset_uri": "s3://owner-controls/depth.mp4",
        }
    )

    validate_spec(spec)
    plan = build_plan(spec, run_id="state-local-depth")
    augment = next(step for step in plan.steps if step.state == "augment")
    assert "depth" in augment.argv

def test_the_control_prefix_is_a_sibling_of_the_augmented_clips() -> None:
    """Nesting it would make the evaluator read a control map as a variant."""

    spec = load_spec(BLUEPRINT)
    augment = str(spec.config["augment_uri"])
    control = str(spec.config["augment_control_uri"])

    assert control != augment
    assert not control.startswith(augment)
    assert not augment.startswith(control)
    published = _augment_flags(spec)["--control-output-uri"]
    assert published.endswith("/cosmos_control/")
    assert not published.startswith(
        "s3://example-bucket/physical-ai-data-factory/paidf/cosmos_augmented/"
    )


def test_exported_control_env_rides_along_to_the_augment_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The env fallbacks are only real if the renderer forwards them.

    The CLI reads NPA_COSMOS_* as a fallback for these flags, but that fallback
    is read inside the pod, so a variable exported next to the submit does
    nothing unless the rendered task carries it.
    """

    monkeypatch.setenv("NPA_COSMOS_CONTROL", "seg")
    monkeypatch.setenv("NPA_COSMOS_CONTROL_PROMPT", "robot arm, conveyor")
    monkeypatch.setenv("NPA_COSMOS_MASK_PROMPT", "robot arm")
    monkeypatch.setenv("NPA_COSMOS_MASK_ASSET", "s3://example-bucket/masks/arm.mp4")
    monkeypatch.setenv("NPA_COSMOS_CONTROL_ASSET", "s3://example-bucket/seg/arm.mp4")

    envs = _augment_task(load_spec(BLUEPRINT))["envs"]

    assert envs["NPA_COSMOS_CONTROL"] == "seg"
    assert envs["NPA_COSMOS_CONTROL_PROMPT"] == "robot arm, conveyor"
    assert envs["NPA_COSMOS_MASK_PROMPT"] == "robot arm"
    assert envs["NPA_COSMOS_MASK_ASSET"] == "s3://example-bucket/masks/arm.mp4"
    assert envs["NPA_COSMOS_CONTROL_ASSET"] == "s3://example-bucket/seg/arm.mp4"


def test_an_unexported_knob_leaves_the_pod_env_alone() -> None:
    envs = _augment_task(load_spec(BLUEPRINT))["envs"]

    assert "NPA_COSMOS_MASK_PROMPT" not in envs
    assert "NPA_COSMOS_CONTROL_ASSET" not in envs
