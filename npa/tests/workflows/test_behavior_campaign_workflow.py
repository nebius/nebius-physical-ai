"""Verify declarative BEHAVIOR campaign fan-out and resume-stable bindings."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

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
from npa.workflows.behavior_challenge import campaign_runner

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


def test_trained_comet_provider_inputs_reach_each_campaign_worker():
    document = _workflow(
        runtime=_runtime(
            policy_kind="comet-trained",
            policy_execution_variant="native",
            policy_task_name="task_1",
            policy_trained_input_root="/evidence/selected-parity",
        )
    )

    assert document["config"]["policy_trained_input_root"] == (
        "/evidence/selected-parity"
    )
    for name in document["states"]["campaign-workers"]["parallel"]:
        argv = document["states"][name]["run"]["argv"]
        index = argv.index("--policy-trained-input-root")
        assert argv[index + 1] == "{{config.policy_trained_input_root}}"


def test_trained_comet_requires_provider_input_root_during_workflow_build():
    runtime = _runtime(
        policy_kind="comet-trained",
        policy_execution_variant="native",
        policy_task_name="task_1",
    )
    with pytest.raises(ValueError, match="requires policy_trained_input_root"):
        _workflow(runtime=runtime)


def test_other_policy_kind_rejects_trained_provider_input_root():
    runtime = _runtime(policy_trained_input_root="/evidence/selected-parity")
    with pytest.raises(ValueError, match="requires comet-trained"):
        _workflow(runtime=runtime)


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


def test_worker_runs_source_bound_simulator_startup_inline():
    slots = [_slot(0, "a"), _slot(1, "b")]
    for slot in slots:
        slot["workspace"] = "/campaign/worker"
    document = _workflow(
        runtime=_runtime(simulator_startup_spec="/campaign/startup-spec.json"),
        slots=slots,
    )

    assert document["config"]["simulator_startup_spec"] == (
        "/campaign/startup-spec.json"
    )
    for name in document["states"]["campaign-workers"]["parallel"]:
        argv = document["states"][name]["run"]["argv"]
        assert argv[argv.index("--simulator-startup-spec") + 1] == (
            "{{config.simulator_startup_spec}}"
        )


def test_global_startup_spec_rejects_distinct_worker_workspaces():
    with pytest.raises(ValueError, match="one identical worker workspace"):
        _workflow(runtime=_runtime(simulator_startup_spec="/campaign/spec.json"))


def test_global_startup_spec_path_must_be_absolute():
    slots = [_slot(0, "a"), _slot(1, "b")]
    for slot in slots:
        slot["workspace"] = "/campaign/worker"
    with pytest.raises(ValueError, match="path must be absolute"):
        _workflow(
            slots=slots,
            runtime=_runtime(simulator_startup_spec="startup-spec.json"),
        )


def test_per_worker_startup_bindings_render_mutually_exclusive_flags():
    slots = [_slot(0, "a"), _slot(1, "b")]
    slots[0]["simulator_startup"] = {"receipt": "/campaign/a/worker-0/startup.json"}
    slots[1]["simulator_startup"] = {"spec": "/campaign/b/worker-1/startup-spec.json"}

    document = _workflow(slots=slots)
    first = document["states"]["campaign-worker-0"]["run"]["argv"]
    second = document["states"]["campaign-worker-1"]["run"]["argv"]

    assert first[first.index("--simulator-startup-receipt") + 1] == (
        "/campaign/a/worker-0/startup.json"
    )
    assert "--simulator-startup-spec" not in first
    assert second[second.index("--simulator-startup-spec") + 1] == (
        "/campaign/b/worker-1/startup-spec.json"
    )
    assert "--simulator-startup-receipt" not in second


def test_generated_cross_workspace_specs_match_each_worker_before_startup(
    tmp_path: Path,
):
    slots = [_slot(0, "a"), _slot(1, "b")]
    for slot in slots:
        workspace = tmp_path / f"worker-{slot['worker_index']}"
        workspace.mkdir()
        slot.pop("pvc")
        slot["workspace"] = str(workspace)
        spec = tmp_path / f"startup-{slot['worker_index']}.json"
        spec.write_text(
            json.dumps(
                {
                    "schema": "npa.workbench.simulator-startup-spec.v1",
                    "apps": {
                        "isaac_root": str(tmp_path / "isaac"),
                        "owner_root": str(workspace),
                        "view_root": str(workspace / "view"),
                        "version_file": "VERSION",
                        "version": {"bytes": 0, "sha256": "0" * 64},
                        "applications": {},
                        "linked_directories": [],
                        "absent_directories": [],
                    },
                    "appdata_root": str(workspace / "appdata"),
                    "command": ["/bin/true"],
                    "marker_path": str(workspace / "marker.json"),
                    "shutdown_request_path": str(workspace / "shutdown.json"),
                    "log_path": str(workspace / "startup.log"),
                    "environment": {},
                    "evaluation_context": {
                        "upstream_root": str(tmp_path / "upstream"),
                        "evaluator_python": "/bin/true",
                        "data_root": str(tmp_path / "data"),
                    },
                }
            )
        )
        slot["simulator_startup"] = {"spec": str(spec)}

    document = _workflow(slots=slots)
    rendered = []
    for index in range(2):
        argv = document["states"][f"campaign-worker-{index}"]["run"]["argv"]
        workspace = Path(argv[argv.index("--workspace") + 1])
        spec = Path(argv[argv.index("--simulator-startup-spec") + 1])
        campaign_runner._validate_worker_startup_binding(
            SimpleNamespace(
                simulator_startup_spec=spec,
                simulator_startup_receipt=None,
            ),
            workspace,
        )
        rendered.append((workspace, spec))

    with pytest.raises(ValueError, match="owner_root must equal"):
        campaign_runner._validate_worker_startup_binding(
            SimpleNamespace(
                simulator_startup_spec=rendered[0][1],
                simulator_startup_receipt=None,
            ),
            rendered[1][0],
        )


@pytest.mark.parametrize(
    "binding, message",
    [
        (
            {"spec": "/campaign/spec.json", "receipt": "/campaign/receipt.json"},
            "exactly one",
        ),
        ({"receipt": "relative.json"}, "must be absolute"),
    ],
)
def test_per_worker_startup_binding_rejects_ambiguous_or_relative_paths(
    binding, message
):
    slots = [_slot(0, "a"), _slot(1, "b")]
    slots[0]["simulator_startup"] = binding
    with pytest.raises(ValueError, match=message):
        _workflow(slots=slots)


def test_global_and_per_worker_startup_bindings_conflict():
    slots = [_slot(0, "a"), _slot(1, "b")]
    for slot in slots:
        slot["workspace"] = "/campaign/worker"
    slots[0]["simulator_startup"] = {"receipt": "/campaign/worker/startup.json"}
    with pytest.raises(ValueError, match="conflicts"):
        _workflow(
            slots=slots,
            runtime=_runtime(simulator_startup_spec="/campaign/startup-spec.json"),
        )


def test_specialist_report_receipts_and_hashes_reach_every_worker():
    runtime = _runtime(
        policy_kind="rlc-specialist",
        policy_execution_variant="native",
        policy_specialist_equivalence_receipt="/evidence/equivalence.json",
        policy_specialist_equivalence_sha256="4" * 64,
        policy_specialist_report_admission="/evidence/admission.json",
        policy_specialist_report_admission_sha256="5" * 64,
    )
    document = _workflow(runtime=runtime)

    for name in document["states"]["campaign-workers"]["parallel"]:
        argv = document["states"][name]["run"]["argv"]
        for flag in (
            "--policy-specialist-equivalence-receipt",
            "--policy-specialist-equivalence-sha256",
            "--policy-specialist-report-admission",
            "--policy-specialist-report-admission-sha256",
        ):
            assert argv.count(flag) == 1


def test_specialist_report_runtime_requires_all_receipt_bindings():
    runtime = _runtime(
        policy_kind="rlc-specialist",
        policy_execution_variant="native",
        policy_specialist_equivalence_receipt="/evidence/equivalence.json",
    )

    with pytest.raises(ValueError, match="receipts and SHA-256 values"):
        _workflow(runtime=runtime)


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
        slots[0]["workspace"] = "/outside-pvc/worker-0"
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
