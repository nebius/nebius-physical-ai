"""Exercise TRAIN panels through the existing durable campaign lifecycle."""

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest

from npa.clients.storage import StoragePreconditionFailed
from npa.workflows.behavior_challenge import campaign, campaign_runner
from npa.workflows.behavior_challenge import nonreporting_train as train
from npa.workflows.behavior_challenge.campaign_status import panel_status
from npa.workflows.behavior_challenge.case_store import CaseStore


class MemoryStorage:
    def __init__(self):
        self.objects = {}
        self.revision = 0

    def read_bytes_with_etag(self, uri):
        return self.objects.get(uri)

    def put_bytes_conditional(
        self, payload, uri, *, if_match="", if_none_match=False, **_
    ):
        current = self.objects.get(uri)
        if (if_none_match and current) or (
            if_match and (not current or current[1] != if_match)
        ):
            raise StoragePreconditionFailed("conditional conflict")
        self.revision += 1
        etag = str(self.revision)
        self.objects[uri] = payload, etag
        return etag

    def download_file(self, uri, destination):
        Path(destination).write_bytes(self.objects[uri][0])


def _artifact(marker):
    return {"sha256": marker * 64, "bytes": 10}


def _panel():
    task = "fixture_task"
    protocol = train.declare_train_protocol(
        task,
        {
            "schema": "npa.behavior.nonreporting-train-task-mapping.v1",
            "split": "train",
            "task": task,
            "data_namespace": "fixture_dataset",
            "data_task_id": 7,
            "source_manifest": _artifact("1"),
            "split_manifest": _artifact("2"),
            "mapping_artifact": _artifact("3"),
        },
        [
            {"instance_id": 0, "rollout_id": 0},
            {"instance_id": 2, "rollout_id": 0},
        ],
        {
            "behavior_upstream_commit": "6cbf70b075816096e9be53958780769f3264d25d",
            "task_registry": _artifact("4"),
            "dataset": _artifact("5"),
            "dataset_view": _artifact("6"),
            "normalization": _artifact("7"),
            "tokenizer": _artifact("8"),
            "action_semantics": _artifact("9"),
        },
        {
            "argv_contract": _artifact("a"),
            "evaluator_source": _artifact("b"),
            "controller_source": _artifact("c"),
            "robot_config": _artifact("d"),
            "rng_contract": _artifact("e"),
            "wrapper": "fixture.Wrapper",
            "mode": "train",
            "num_envs": 1,
            "num_rollouts": 1,
            "write_video": True,
            "max_steps_argument": None,
            "model_prediction_horizon": 32,
            "executed_prefix": 32,
            "fresh_policy_process_per_case": True,
            "qualification_process_discarded": True,
        },
    )
    policy = campaign.freeze_policy_identity(
        "fixture", {"checkpoint": _artifact("f"), "serving": _artifact("0")}
    )
    return train.declare_train_panel(protocol, policy)


def _write_rollout(output, case):
    stem = f"{case['task']}_{case['instance_id']}_0"
    (output / "json").mkdir(exist_ok=True)
    (output / "videos").mkdir(exist_ok=True)
    distances = {part: 1.0 for part in ("base", "left", "right")}
    metrics = {
        **{key: case[key] for key in ("task", "instance_id", "rollout_id")},
        "steps": 2,
        "success": case["instance_id"] == 0,
        "q_score": {"final": 0.75 if case["instance_id"] == 0 else 0.25},
        "agent_distance": distances,
        "normalized_agent_distance": distances,
        "time": {
            "simulator_steps": 2,
            "simulator_time": 2 / 30,
            "normalized_time": 10.0,
        },
    }
    (output / f"json/{stem}.json").write_text(json.dumps(metrics) + "\n")
    with av.open(str(output / f"videos/{stem}.mp4"), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for value in (0, 128):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), value, dtype=np.uint8), format="rgb24"
            )
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


def test_train_panel_reuses_case_store_fresh_process_and_file_backed_aggregate(
    tmp_path,
):
    panel = _panel()
    partition = train.partition_train_panel(panel, 1)
    storage = MemoryStorage()
    store = CaseStore(storage, "s3://example-bucket/train", panel["panel_id"])
    events = []

    @contextmanager
    def prepare(case, _output):
        events.append(("start", case["instance_id"]))
        yield
        events.append(("stop", case["instance_id"]))

    def execute(case, output):
        events.append(("evaluate", case["instance_id"]))
        _write_rollout(output, case)

    records = campaign_runner.run_partition(
        panel,
        partition,
        0,
        store,
        tmp_path / "worker",
        execute,
        prepare_case=prepare,
    )
    verified = campaign_runner.aggregate_stored_panel(
        panel, store, tmp_path / "aggregate"
    )

    assert [record["instance_id"] for record in records] == [0, 2]
    assert events == [
        ("start", 0),
        ("evaluate", 0),
        ("stop", 0),
        ("start", 2),
        ("evaluate", 2),
        ("stop", 2),
    ]
    assert verified["aggregate"]["mean_q"] == 0.5
    assert verified["aggregate"]["artifact_bytes_verified_by_aggregator"] is True
    assert verified["aggregate"]["artifact_validation_contract"] == (
        "npa.behavior.inspect-rollout.v1"
    )
    assert verified["verification"] == {
        "all_original_bytes_downloaded_and_hashed": True,
        "all_original_videos_fully_decoded": True,
        "case_count": 2,
    }

    resumed = campaign_runner.run_partition(
        panel,
        partition,
        0,
        store,
        tmp_path / "resumed",
        lambda *_: pytest.fail("completed TRAIN case reran"),
        prepare_case=lambda *_: pytest.fail("completed policy restarted"),
    )
    assert resumed == records


def test_train_status_and_partition_dispatch_reject_cross_schema():
    panel = _panel()
    storage = MemoryStorage()
    status = panel_status(storage, panel, "s3://example-bucket/train")

    assert status["counts"]["unclaimed"] == 2
    assert status["panel_id"] == panel["panel_id"]
    legacy_registry = ["fixture_task"] + [f"task_{index}" for index in range(99)]
    legacy_panel = campaign.declare_panel(
        panel["policy_binding"], legacy_registry, ["fixture_task"], "development"
    )
    legacy_partition = campaign.partition_panel(legacy_panel, 1)
    with pytest.raises(ValueError, match="TRAIN partition"):
        campaign_runner.run_partition(
            panel,
            legacy_partition,
            0,
            CaseStore(storage, "s3://example-bucket/train-2", panel["panel_id"]),
            Path("/tmp/not-used"),
            lambda *_: None,
        )


def test_worker_declarations_dispatch_train_without_legacy_registry(tmp_path):
    panel = _panel()
    partition = train.partition_train_panel(panel, 1)
    storage = MemoryStorage()
    storage.objects.update(
        {
            "s3://example-bucket/panel.json": (json.dumps(panel).encode(), "1"),
            "s3://example-bucket/partition.json": (
                json.dumps(partition).encode(),
                "2",
            ),
        }
    )
    args = SimpleNamespace(
        panel_uri="s3://example-bucket/panel.json",
        partition_uri="s3://example-bucket/partition.json",
        upstream_root=tmp_path / "no-registry-checkout-needed-for-declaration",
    )

    assert campaign_runner._worker_declarations(args, storage, tmp_path) == (
        panel,
        partition,
    )


def test_train_managed_plan_uses_declared_checkpoint_and_train_split():
    panel = _panel()
    plan = campaign_runner._managed_plan(panel, panel["cases"][0])

    assert plan["recipe"] == {
        "tasks": ["fixture_task"],
        "split": "train",
        "upstream_commit": "6cbf70b075816096e9be53958780769f3264d25d",
        "policy_checkpoint_sha256": "f" * 64,
    }
    assert plan["cases"] == [panel["cases"][0]]


@pytest.mark.parametrize(
    "policy_kind",
    ("official", "rlc", "rlc-selected", "rlc-specialist", "comet12", "comet50"),
)
def test_train_execution_rejects_cli_policy_kinds_before_any_worker_effect(
    tmp_path, monkeypatch, policy_kind
):
    panel = _panel()
    partition = train.partition_train_panel(panel, 1)
    storage = MemoryStorage()
    effects = []

    def unexpected(label):
        def fail(*_args, **_kwargs):
            effects.append(label)
            pytest.fail(f"TRAIN hold guard ran after {label}")

        return fail

    monkeypatch.setattr(
        campaign_runner, "_prepare_worker_startup", unexpected("startup")
    )
    monkeypatch.setattr(campaign_runner, "_specialist_preclaim", unexpected("preclaim"))
    monkeypatch.setattr(campaign_runner, "CaseStore", unexpected("case-store"))
    monkeypatch.setattr(campaign_runner, "_prepared_evaluator", unexpected("policy"))

    with pytest.raises(ValueError, match="reviewed comet-native adapter"):
        campaign_runner._execute_partition(
            SimpleNamespace(policy_kind=policy_kind),
            panel,
            partition,
            storage,
            tmp_path,
        )
    assert effects == []
    assert storage.objects == {}
