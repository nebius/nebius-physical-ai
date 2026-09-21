"""Verify declarative BEHAVIOR campaign fan-out and resume-stable bindings."""

from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from npa.orchestration.npa_workflow.spec import load_spec
from npa.workflows.behavior_challenge.campaign import (
    declare_panel,
    freeze_policy_identity,
    partition_panel,
)
from npa.workflows.behavior_challenge.campaign_workflow import (
    build_campaign_workflow,
)

IMAGE = "registry.example.invalid/behavior@sha256:" + "1" * 64


def _panel_and_partition(worker_count: int = 2):
    policy = freeze_policy_identity(
        "released-rlc-checkpoint-2",
        {
            "checkpoint": {"sha256": "2" * 64, "bytes": 100},
            "serving": {"sha256": "3" * 64, "bytes": 200},
        },
    )
    registry = [f"task_{index}" for index in range(100)]
    panel = declare_panel(policy, registry, ["task_1"], "development")
    return panel, partition_panel(panel, worker_count)


def _runtime(**extra):
    return {
        "upstream_root": "/opt/BEHAVIOR-1K",
        "evaluator_python": "/opt/behavior/bin/python",
        "data_root": "/data/behavior",
        "host": "127.0.0.1",
        "port": 8000,
        "policy_kind": "rlc",
        "policy_root": "/opt/rlc",
        "policy_python": "/opt/rlc/.venv/bin/python",
        "policy_checkpoint": "/models/checkpoint_2",
        "policy_archive": "/models/checkpoint_2.tar.gz",
        "policy_execution_variant": "adaptive-short-chunk-transition-refresh",
        **extra,
    }


def _slot(index: int, role: str):
    return {
        "worker_index": index,
        "resource": {
            "cloud": "kubernetes",
            "accelerators": "RTXPRO6000:1",
            "cpus": 16,
            "memory": "128Gi",
        },
        "workspace": f"/campaign/{role}/worker-{index}",
        "pvc": {"claim_name": f"behavior-{role}", "mount_path": "/campaign"},
    }


def _workflow(*, aggregate=True, slots=None, runtime=None):
    panel, partition = _panel_and_partition()
    aggregate_slot = None
    if aggregate:
        aggregate_slot = {
            "resource": {"cloud": "kubernetes", "cpus": 4, "memory": "16Gi"},
            "workspace": "/campaign/aggregate",
            "receipt_uri": "s3://bucket/campaign/panel-aggregate.json",
            "pvc": {"claim_name": "behavior-aggregate", "mount_path": "/campaign"},
        }
    return build_campaign_workflow(
        name="behavior-campaign-panel",
        panel=panel,
        partition=partition,
        panel_uri="s3://bucket/campaign/panel.json",
        partition_uri="s3://bucket/campaign/partition.json",
        state_prefix="s3://bucket/campaign/state",
        worker_receipts_prefix="s3://bucket/campaign/{{run.id}}/worker-receipts",
        runtime_image=IMAGE,
        runtime=runtime or _runtime(),
        worker_slots=slots or [_slot(0, "a"), _slot(1, "b")],
        aggregate=aggregate_slot,
    )


def test_generator_builds_valid_parallel_runtime_workflow(tmp_path):
    document = _workflow()
    workflow = tmp_path / "campaign.yaml"
    workflow.write_text(yaml.safe_dump(document, sort_keys=False))

    spec = load_spec(workflow)

    group = spec.states["campaign-workers"]
    assert spec.metadata["executionMode"] == "runtime"
    assert spec.config["source_overlay"] is True
    assert group.parallel == ["campaign-worker-0", "campaign-worker-1"]
    assert group.parallel_count == 2
    assert group.max_concurrency == 2
    assert group.next == "campaign-aggregate"
    assert spec.states["campaign-aggregate"].terminal is True
    assert "{{run." not in spec.config["state_prefix"]
    assert "{{run." not in spec.config["panel_uri"]
    assert "{{run." not in spec.config["partition_uri"]
    assert "{{run.id}}" in spec.config["worker_receipts_prefix"]


def test_comet_task_name_reaches_each_campaign_worker():
    document = _workflow(
        runtime=_runtime(
            policy_kind="comet12",
            policy_execution_variant="native",
            policy_task_name="task_1",
        )
    )

    assert document["config"]["policy_task_name"] == "task_1"
    for name in document["states"]["campaign-workers"]["parallel"]:
        argv = document["states"][name]["run"]["argv"]
        index = argv.index("--policy-task-name")
        assert argv[index + 1] == "{{config.policy_task_name}}"


def test_each_partition_worker_appears_once_with_stable_receipt_and_state_prefix():
    document = _workflow()
    members = document["states"]["campaign-workers"]["parallel"]

    assert members == ["campaign-worker-0", "campaign-worker-1"]
    for index, name in enumerate(members):
        state = document["states"][name]
        argv = state["run"]["argv"]
        assert argv.count("--worker-index") == 1
        assert argv[argv.index("--worker-index") + 1] == str(index)
        assert argv[argv.index("--output-path") + 1] == "{{config.state_prefix}}"
        receipt = f"{{{{config.worker_receipts_prefix}}}}/worker-{index}.json"
        assert argv[argv.index("--worker-receipt-uri") + 1] == receipt
        assert state["outputs"] == [{"uri": receipt}]


def test_receipts_are_invocation_scoped_while_case_state_remains_resume_stable():
    document = _workflow()

    assert document["config"]["state_prefix"] == "s3://bucket/campaign/state"
    assert document["config"]["worker_receipts_prefix"] == (
        "s3://bucket/campaign/{{run.id}}/worker-receipts"
    )


def test_worker_resource_profiles_bind_distinct_writable_pvcs():
    document = _workflow()

    for index, claim in enumerate(("behavior-a", "behavior-b")):
        resource = document["resources"][f"campaign-worker-{index}"]
        assert resource["image"] == "{{config.runtime_image}}"
        pod = resource["kubernetes"]["pod_config"]["spec"]
        assert pod["volumes"][-1]["persistentVolumeClaim"]["claimName"] == claim
        mount = pod["containers"][-1]["volumeMounts"][-1]
        assert mount == {"name": "campaign-workspace", "mountPath": "/campaign"}


def test_worker_argv_includes_operator_runtime_and_only_supplied_optional_flags():
    runtime = _runtime(
        policy_stock_correlation_asset="/models/correlation.bin",
        policy_stock_correlation_sha256="4" * 64,
    )
    argv = _workflow(runtime=runtime)["states"]["campaign-worker-0"]["run"]["argv"]

    for flag in (
        "--upstream-root",
        "--evaluator-python",
        "--data-root",
        "--host",
        "--port",
        "--policy-root",
        "--policy-python",
        "--policy-checkpoint",
        "--policy-archive",
        "--policy-execution-variant",
        "--policy-stock-correlation-asset",
        "--policy-stock-correlation-sha256",
    ):
        assert argv.count(flag) == 1
    assert "--policy-selected-export-receipt" not in argv


def test_optional_aggregate_is_a_real_barrier_and_can_be_omitted():
    with_aggregate = _workflow()
    aggregate = with_aggregate["states"]["campaign-aggregate"]
    argv = aggregate["run"]["argv"]
    assert aggregate["needs"] == ["campaign-workers"]
    assert argv[:4] == [
        "python3",
        "-m",
        "npa.workflows.behavior_challenge",
        "campaign-aggregate",
    ]
    assert argv[argv.index("--output-path") + 1] == "{{config.state_prefix}}"
    assert aggregate["outputs"][0]["schema"] == "npa.behavior.verified-panel.v1"

    without = _workflow(aggregate=False)
    assert "campaign-aggregate" not in without["states"]
    assert without["states"]["campaign-workers"]["terminal"] is True


def test_generator_does_not_mutate_frozen_inputs():
    panel, partition = _panel_and_partition()
    original_panel = deepcopy(panel)
    original_partition = deepcopy(partition)

    build_campaign_workflow(
        name="behavior-campaign-panel",
        panel=panel,
        partition=partition,
        panel_uri="s3://bucket/panel.json",
        partition_uri="s3://bucket/partition.json",
        state_prefix="s3://bucket/stable-state",
        worker_receipts_prefix="s3://bucket/receipts",
        runtime_image=IMAGE,
        runtime=_runtime(),
        worker_slots=[_slot(0, "a"), _slot(1, "b")],
    )

    assert panel == original_panel
    assert partition == original_partition


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("mutable-image", "immutable full SHA-256"),
        ("run-state", "stable across workflow resumes"),
        ("local-state", "scoped S3 location"),
        ("unsupported-receipt-token", "unsupported workflow token"),
        ("duplicate-worker", "exactly one slot"),
        ("workspace-outside-pvc", "inside its writable PVC"),
        ("tampered-partition", "deterministic case ownership"),
    ],
)
def test_generator_rejects_mutable_or_ambiguous_execution(change, message):
    panel, partition = _panel_and_partition()
    image = IMAGE
    state_prefix = "s3://bucket/stable-state"
    slots = [_slot(0, "a"), _slot(1, "b")]
    if change == "mutable-image":
        image = "registry.example.invalid/behavior:latest"
    elif change == "run-state":
        state_prefix = "s3://bucket/{{run.id}}/state"
    elif change == "local-state":
        state_prefix = "/campaign/stable-state"
    elif change == "unsupported-receipt-token":
        pass
    elif change == "duplicate-worker":
        slots[1]["worker_index"] = 0
    elif change == "workspace-outside-pvc":
        slots[0]["workspace"] = "/tmp/worker-0"
    else:
        partition["workers"][0]["case_ids"] = []

    with pytest.raises(ValueError, match=message):
        build_campaign_workflow(
            name="behavior-campaign-panel",
            panel=panel,
            partition=partition,
            panel_uri="s3://bucket/panel.json",
            partition_uri="s3://bucket/partition.json",
            state_prefix=state_prefix,
            worker_receipts_prefix=(
                "s3://bucket/{{state.id}}/receipts"
                if change == "unsupported-receipt-token"
                else "s3://bucket/{{run.id}}/receipts"
            ),
            runtime_image=image,
            runtime=_runtime(),
            worker_slots=slots,
        )
