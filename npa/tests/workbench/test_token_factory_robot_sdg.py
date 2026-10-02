"""Verify robot scene validation, physics gates, publication, and public interfaces."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.token_factory import TokenFactoryClient, TokenFactoryConfig
from npa.sdk.workbench.token_factory import RobotSdgRequest, robot_sdg
from npa.workbench.token_factory import TokenFactoryToolError
from npa.workbench.token_factory import robot_sdg as pipeline
from npa.workbench.token_factory import robot_artifacts
from npa.workbench.token_factory.robot_scene import RobotScene, plan_robot_scene
from npa.workbench.token_factory.robot_sim import physics_checks
from npa.workbench.token_factory.sdg_protocol import FAST_MODEL, REASONING_MODEL


def _scene(**changes):
    return {
        "object_x": -0.10,
        "object_y": -0.08,
        "goal_x": 0.10,
        "goal_y": 0.08,
        "object_color": "red",
        "target_color": "green",
        "lighting": 1.0,
        "reason": "Synthetic scene fixture",
        **changes,
    }


@pytest.fixture
def provider():
    calls, state = [], {}

    def handle(request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": [{"id": model} for model in (FAST_MODEL, REASONING_MODEL)]
                },
            )
        body = json.loads(request.content)
        calls.append(body)
        schema = body["response_format"]["json_schema"]["name"]
        answer = (
            {"task_type": "reasoning", "reason": "Synthetic classification"}
            if schema == "Route"
            else _scene()
        )
        if state.get("invalid_model") == body["model"] and schema == "RobotScene":
            answer["object_x"] = 10.0
        return httpx.Response(
            200,
            json={
                "id": f"synthetic-{len(calls)}",
                "model": body["model"],
                "choices": [
                    {
                        "message": {"content": json.dumps(answer)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "prompt_cache_hit_tokens": 64,
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        client = TokenFactoryClient(
            TokenFactoryConfig(
                api_key="synthetic-key", base_url="https://provider.invalid/v1"
            ),
            http_client=transport,
        )
        yield client, calls, state


@pytest.mark.parametrize(
    "change",
    [
        {"object_x": 0.13},
        {"lighting": 0.1},
        {"goal_x": -0.1, "goal_y": -0.08},
        {"object_color": "green"},
        {"object_color": "purple"},
        {"command": "execute"},
        {"object_x": float("nan")},
    ],
)
def test_rejects_unsafe_or_unrepresentable_scenes(change):
    with pytest.raises(ValueError):
        RobotScene.model_validate(_scene(**change))


def test_routed_scene_is_validated_and_invalid_primary_tries_other_model(provider):
    client, calls, state = provider
    state["invalid_model"] = REASONING_MODEL
    record = plan_robot_scene(
        client,
        {"id": "seed", "prompt": "plan a move"},
        router="token_factory",
        jev_key="",
    )
    assert record["routing"]["selected_model"] == REASONING_MODEL
    assert record["served_model"] == FAST_MODEL
    assert record["scene"]["object_x"] == -0.1
    assert [call["status"] for call in record["calls"]] == [
        "completed",
        "error",
        "completed",
    ]
    assert all(call["response_format"]["json_schema"]["strict"] for call in calls)
    assert calls[1]["messages"][0] == calls[2]["messages"][0]


def _physical_trace():
    count = 20
    positions = np.tile([0.1, 0.1, 0.425], (count, 1))
    positions[5, 2] += 0.15
    states = np.zeros((count, 9))
    states[:, -2:] = 0.05
    return {
        "state": states.copy(),
        "next_state": states,
        "actions": np.zeros((count, 4)),
        "object_position": np.tile([-0.1, -0.1, 0.425], (count, 1)),
        "next_object_position": positions,
        "goal": np.array([0.1, 0.1, 0.425]),
        "next_gripper_position": np.tile([0.1, 0.1, 0.575], (count, 1)),
        "finger_contacts": np.full(count, 2),
        "environment_success": np.ones(count, dtype=bool),
    }


@pytest.mark.parametrize(
    "failure",
    ["grasp", "lift", "place", "release", "retreat", "settle", "finite", "environment"],
)
def test_physics_gate_rejects_each_failed_physical_requirement(failure):
    arrays = _physical_trace()
    assert physics_checks(arrays)["accepted"]
    if failure == "grasp":
        arrays["finger_contacts"][:] = 0
    if failure == "lift":
        arrays["next_object_position"][:, 2] = 0.425
    if failure == "place":
        arrays["goal"][0] = -0.1
    if failure == "release":
        arrays["next_state"][-1, -2:] = 0.024
    if failure == "retreat":
        arrays["next_gripper_position"][-1, 2] = 0.44
    if failure == "settle":
        arrays["next_object_position"][-2, 0] += 0.02
    if failure == "finite":
        arrays["actions"][0, 0] = np.nan
    if failure == "environment":
        arrays["environment_success"][-1] = False
    assert not physics_checks(arrays)["accepted"]


def _request(tmp_path, **changes):
    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text(
        json.dumps({"id": "robot", "prompt": "plan a tabletop move"}) + "\n"
    )
    return RobotSdgRequest(
        input_path=str(seeds), output_path=str(tmp_path / "out"), **changes
    )


def test_dry_run_needs_no_robot_runtime_provider_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline, "runtime_versions", lambda: pytest.fail("runtime used")
    )
    monkeypatch.setattr(
        pipeline, "TokenFactoryClient", lambda: pytest.fail("provider used")
    )
    report = robot_sdg(_request(tmp_path, dry_run=True))
    assert report["status"] == "planned"
    assert not report["inference_performed"]
    assert not (tmp_path / "out").exists()


def test_existing_local_destination_is_not_overwritten(tmp_path, monkeypatch):
    request = _request(tmp_path)
    Path(request.output_path).mkdir()
    monkeypatch.setattr(pipeline, "runtime_versions", lambda: {})
    monkeypatch.setattr(
        pipeline, "TokenFactoryClient", lambda: pytest.fail("provider used")
    )
    with pytest.raises(FileExistsError):
        robot_sdg(request)


def test_failed_scene_does_not_run_simulation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline, "plan_robot_scene", lambda *args, **kwargs: {"status": "error"}
    )
    monkeypatch.setattr(
        pipeline,
        "simulate_robot_episode",
        lambda *args, **kwargs: pytest.fail("simulation used"),
    )
    assert (
        pipeline._run_episode(tmp_path, None, {}, 0, _request(tmp_path), "")["status"]
        == "error"
    )


def test_physics_rejection_cannot_be_overridden_by_successful_model_plan(
    tmp_path, monkeypatch, provider
):
    monkeypatch.setattr(
        pipeline, "simulate_robot_episode", lambda *args, **kwargs: {"accepted": False}
    )
    record = pipeline._run_episode(
        tmp_path,
        provider[0],
        {"id": "robot", "prompt": "plan"},
        0,
        _request(tmp_path),
        "",
    )
    assert record["served_model"] == REASONING_MODEL
    assert record["status"] == "rejected"
    assert record["reason"] == "physics_checks_failed"


def test_no_accepted_episodes_produces_no_training_dataset(tmp_path):
    assert robot_artifacts.export_robot_dataset(tmp_path, [{"status": "rejected"}]) == 0
    assert not (tmp_path / "dataset").exists()


def test_export_includes_only_physics_accepted_episodes(tmp_path, monkeypatch):
    records = []
    for index, status in enumerate(("accepted", "rejected", "accepted")):
        relative = f"episodes/episode_{index:04d}"
        (tmp_path / relative).mkdir(parents=True)
        records.append(
            {
                "status": status,
                "episode_path": relative,
                "simulation": {"task": f"Recorded task {index}"},
            }
        )

    def convert(demos, destination, **kwargs):
        assert (demos / "episode_0000").resolve() == tmp_path / records[0][
            "episode_path"
        ]
        assert (demos / "episode_0001").resolve() == tmp_path / records[2][
            "episode_path"
        ]
        assert not (demos / "episode_0002").exists()
        tasks = json.loads((demos / "metadata.json").read_text())["episodes"]
        assert [row["task"] for row in tasks] == ["Recorded task 0", "Recorded task 2"]
        assert kwargs["task_from_metadata"] is True
        (destination / "meta").mkdir(parents=True)
        (destination / "meta/info.json").write_text(
            json.dumps({"features": {"observation.state": {}, "action": {}}})
        )

    monkeypatch.setattr(robot_artifacts, "convert", convert)
    assert robot_artifacts.export_robot_dataset(tmp_path, records) == 2
    assert records[0]["dataset_episode_index"] == 0
    assert "dataset_episode_index" not in records[1]
    assert records[2]["dataset_episode_index"] == 1


def test_s3_publication_requires_empty_prefix_and_writes_manifest_last(
    tmp_path, monkeypatch
):
    (tmp_path / "index.html").write_text("synthetic gallery")
    operations = []

    class Storage:
        def upload_directory(self, directory, target, **kwargs):
            assert not (Path(directory) / "report.json").exists()
            operations.append(("directory", target, kwargs))

        def upload_file(self, local, target):
            operations.append(("manifest", target, json.loads(Path(local).read_text())))

    monkeypatch.setattr(
        robot_artifacts.StorageClient, "from_environment", lambda: Storage()
    )
    robot_artifacts.publish_robot_run(
        tmp_path, "s3://example-bucket/robot-run", {"status": "completed"}
    )
    assert operations[0][2] == {"require_empty": True}
    assert operations[1][1].endswith("/report.json")


def test_cli_rejects_local_handoffs_as_one_json_document(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "token-factory",
            "robot-sdg",
            "--input-path",
            str(tmp_path / "seeds.jsonl"),
            "--output-path",
            "s3://example-bucket/robots",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["result"] == "error"


def test_cli_calls_shared_robot_pipeline(monkeypatch):
    from npa.cli.workbench import token_factory as commands

    seen = []
    monkeypatch.setattr(
        commands,
        "run_robot_sdg",
        lambda request: seen.append(request) or {"status": "planned"},
    )
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "token-factory",
            "robot-sdg",
            "--input-path",
            "s3://example-bucket/seeds.jsonl",
            "--output-path",
            "s3://example-bucket/robots",
            "--seed",
            "42",
            "--dry-run",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["status"] == "planned"
    assert seen[0].seed == 42 and seen[0].dry_run


def test_missing_runtime_is_actionable(monkeypatch):
    from importlib.metadata import PackageNotFoundError
    from npa.workbench.token_factory import robot_sim

    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(robot_sim.metadata, "version", missing)
    with pytest.raises(TokenFactoryToolError, match=r"npa\[robot-sdg\]"):
        robot_sim.runtime_versions()
