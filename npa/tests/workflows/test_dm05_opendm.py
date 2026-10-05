"""Contract tests for the pinned OpenDM DM05/LIBERO workflow."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.workflows import dm05_opendm


ROOT = Path(__file__).parents[3]
SPEC = ROOT / "workflows" / "testing" / "dm05-opendm.yaml"
READINESS = SPEC.with_suffix(".readiness.json")


def test_normalization_contract_is_explicit_and_fixed() -> None:
    assert dm05_opendm.normalization_contract() == {
        "dataset": "libero_pi0_all",
        "robot_type": "Franka",
        "camera_keys": ["images_1", "images_2"],
        "camera_prompts_in_order": ["Head", "Left wrist"],
        "state": {"dimension": 8, "order": "six_joint_then_two_gripper"},
        "action": {"dimension": 7, "mode": "absolute"},
        "action_chunk": 10,
    }
    dm05_opendm.assert_normalization_contract(
        state=[0.0] * 8, image_count=2, action_dim=7
    )
    assert dm05_opendm.evaluator_observation_contract() == {
        "source": "Dexbotic LIBERO evaluator",
        "dimension": 8,
        "order": "eef_position_3_then_axis_angle_3_then_gripper_2",
        "camera_order": ["agentview_image", "robot0_eye_in_hand_image"],
        "http_api": "v1",
        "training_state_equivalent": False,
        "model_add_state": False,
    }


@pytest.mark.parametrize(
    ("state", "image_count", "action_dim", "message"),
    [
        ([0.0] * 7, 2, 7, "8-value Franka state"),
        ([0.0] * 8, 1, 7, "camera order"),
        ([0.0] * 8, 2, 8, "requires 7 action values"),
    ],
)
def test_normalization_contract_rejects_shape_drift(
    state: list[float], image_count: int, action_dim: int, message: str
) -> None:
    with pytest.raises(dm05_opendm.DM05WorkflowError, match=message):
        dm05_opendm.assert_normalization_contract(
            state=state, image_count=image_count, action_dim=action_dim
        )


def test_first_libero_observation_is_safe_and_uses_ordered_cameras(
    tmp_path: Path,
) -> None:
    data = tmp_path / "libero"
    image_root = data / "libero_pi0_all" / "image"
    jsonl = data / "libero_pi0_all" / "jsonl" / "episode.jsonl"
    jsonl.parent.mkdir(parents=True)
    (image_root / "episode").mkdir(parents=True)
    (image_root / "episode" / "head.jpg").write_bytes(b"head")
    (image_root / "episode" / "wrist.jpg").write_bytes(b"wrist")
    jsonl.write_text(
        json.dumps(
            {
                "state": [0.0] * 8,
                "prompt": "Pick up the object",
                "images_1": {"type": "image", "url": "episode/head.jpg"},
                "images_2": {"type": "image", "url": "episode/wrist.jpg"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    frame, images = dm05_opendm._first_libero_observation(data)
    payload = dm05_opendm._request_payload(frame, images)

    assert [path.name for path in images] == ["head.jpg", "wrist.jpg"]
    assert payload["observation"]["robot_type"] == "Franka"
    assert list(payload["observation"]["images"]) == ["1", "2"]


@pytest.mark.parametrize("escaped_url", ["../outside.jpg", "/tmp/outside.jpg"])
def test_first_libero_observation_rejects_parent_and_absolute_image_escapes(
    tmp_path: Path, escaped_url: str
) -> None:
    data = tmp_path / "libero"
    image_root = data / "libero_pi0_all" / "image"
    jsonl = data / "libero_pi0_all" / "jsonl" / "episode.jsonl"
    jsonl.parent.mkdir(parents=True)
    image_root.mkdir(parents=True)
    (image_root / "wrist.jpg").write_bytes(b"wrist")
    (image_root.parent / "outside.jpg").write_bytes(b"outside")
    jsonl.write_text(
        json.dumps(
            {
                "state": [0.0] * 8,
                "images_1": {"type": "image", "url": escaped_url},
                "images_2": {"type": "image", "url": "wrist.jpg"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(dm05_opendm.DM05WorkflowError, match="escapes"):
        dm05_opendm._first_libero_observation(data)


def test_first_libero_observation_rejects_a_symlink_image_escape(tmp_path: Path) -> None:
    data = tmp_path / "libero"
    image_root = data / "libero_pi0_all" / "image"
    jsonl = data / "libero_pi0_all" / "jsonl" / "episode.jsonl"
    jsonl.parent.mkdir(parents=True)
    image_root.mkdir(parents=True)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    (image_root / "head.jpg").symlink_to(outside)
    (image_root / "wrist.jpg").write_bytes(b"wrist")
    jsonl.write_text(
        json.dumps(
            {
                "state": [0.0] * 8,
                "images_1": {"type": "image", "url": "head.jpg"},
                "images_2": {"type": "image", "url": "wrist.jpg"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(dm05_opendm.DM05WorkflowError, match="escapes"):
        dm05_opendm._first_libero_observation(data)


def test_licensed_lerobot_conversion_preserves_camera_state_and_action_contract(
    tmp_path: Path,
) -> None:
    target = tmp_path / "libero"
    state = [float(index) for index in range(8)]
    action = [float(index) / 10 for index in range(7)]
    rows = [
        {
            "observation.images.image": {"bytes": b"head-bytes"},
            "observation.images.image2": {"bytes": b"wrist-bytes"},
            "observation.state": state,
            "action": action,
        }
    ]

    assert (
        dm05_opendm._write_lerobot_episode(
            target=target, episode_index=3, prompt="Place the object", rows=rows
        )
        == 1
    )
    frame, images = dm05_opendm._first_libero_observation(target)

    assert frame["prompt"] == "Place the object"
    assert frame["state"] == state
    assert frame["action"] == action
    assert [path.read_bytes() for path in images] == [b"head-bytes", b"wrist-bytes"]
    assert [path.parent.name for path in images] == ["episode_000003", "episode_000003"]


def test_licensed_lerobot_converter_uses_pinned_metadata_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    metadata = source / "meta"
    metadata.mkdir(parents=True)
    (metadata / "info.json").write_text(
        json.dumps(
            {
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "features": {
                    "observation.images.image": {"shape": [256, 256, 3]},
                    "observation.images.image2": {"shape": [256, 256, 3]},
                    "observation.state": {"shape": [8]},
                    "action": {"shape": [7]},
                },
            }
        ),
        encoding="utf-8",
    )
    (metadata / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["Open the drawer"]}) + "\n",
        encoding="utf-8",
    )
    parquet = source / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.touch()
    rows = [
        {
            "observation.images.image": {"bytes": b"head"},
            "observation.images.image2": {"bytes": b"wrist"},
            "observation.state": [0.0] * 8,
            "action": [0.0] * 7,
        }
    ]

    class Table:
        @staticmethod
        def to_pylist() -> list[dict[str, object]]:
            return rows

    parquet_module = types.ModuleType("pyarrow.parquet")
    parquet_module.read_table = lambda path: Table()  # type: ignore[attr-defined]
    pyarrow_module = types.ModuleType("pyarrow")
    pyarrow_module.__path__ = []  # type: ignore[attr-defined]
    pyarrow_module.parquet = parquet_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyarrow", pyarrow_module)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet_module)

    target = tmp_path / "converted"
    assert dm05_opendm._convert_licensed_lerobot_libero(
        source=source, target=target
    ) == {"episodes": 1, "frames": 1}
    frame, images = dm05_opendm._first_libero_observation(target)
    assert frame["prompt"] == "Open the drawer"
    assert frame["state"] == [0.0] * 8
    assert frame["action"] == [0.0] * 7
    assert [path.read_bytes() for path in images] == [b"head", b"wrist"]


def test_workflow_is_a_connected_five_stage_real_component_path() -> None:
    document = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    states = document["states"]
    assert list(states) == [
        "prepare_libero",
        "train_dm05",
        "serve_rollout",
        "evaluate_libero",
        "emit_reviewable_artifacts",
    ]
    assert all(
        "python3 -m npa.workflows.dm05_opendm" in state["run"]["shell"]
        for state in states.values()
    )
    assert (
        states["train_dm05"]["inputs"][0]["uri"] == "{{config.prepared_manifest_uri}}"
    )
    assert (
        "--opendm-python '{{config.opendm_python}}'"
        in states["train_dm05"]["run"]["shell"]
    )
    assert (
        states["serve_rollout"]["inputs"][1]["uri"]
        == "{{config.checkpoint_manifest_uri}}"
    )
    assert (
        states["evaluate_libero"]["inputs"][2]["uri"]
        == "{{config.rollout_manifest_uri}}"
    )
    assert (
        states["emit_reviewable_artifacts"]["inputs"][3]["uri"]
        == "{{config.evaluation_manifest_uri}}"
    )
    assert (
        states["emit_reviewable_artifacts"]["outputs"][1]["uri"] == "{{config.rrd_uri}}"
    )
    adapter_source = (ROOT / "npa/src/npa/workflows/dm05_opendm.py").read_text(
        encoding="utf-8"
    )
    assert "example_dm05_libero.yaml" in adapter_source
    assert '"api_style",\n                    "v1"' in adapter_source

    plan = build_plan(load_spec(SPEC), run_id="dm05-contract")
    assert [step.state for step in plan.steps] == list(states)


def test_workflow_does_not_invent_an_openpi_or_generic_terms_gate() -> None:
    contents = SPEC.read_text(encoding="utf-8") + (
        ROOT / "npa/src/npa/workflows/dm05_opendm.py"
    ).read_text(encoding="utf-8")
    assert "NPA_OPENPI_ACCEPT_GEMMA_TERMS" not in contents
    assert "ACCEPT_" not in contents
    assert "Dexmal/DM05" in contents
    assert "HuggingFaceVLA/libero" in contents
    assert "dexbotic-benchmark" in contents


def test_adjacent_readiness_binds_the_exact_workflow_and_stays_honest() -> None:
    record = json.loads(READINESS.read_text(encoding="utf-8"))
    assert record["schema_version"] == "workflow-readiness/v1"
    assert record["workflow_sha256"] == hashlib.sha256(SPEC.read_bytes()).hexdigest()
    assert record["planning"]["validation"]["status"] == "verified"
    assert record["planning"]["task_fidelity"]["status"] == "verified"
    assert set(record["prerequisites"]) == {
        "output_storage",
        "worker_input",
        "credentials",
        "source_image",
        "target_runtime",
    }
    assert record["prerequisites"]["output_storage"]["status"] == "verified"
    assert record["prerequisites"]["credentials"]["status"] == "verified"
    assert record["prerequisites"]["source_image"]["status"] == "verified"
    assert record["prerequisites"]["source_image"]["evidence"]
    for prerequisite in ("worker_input", "target_runtime"):
        assert record["prerequisites"][prerequisite]["status"] == "unverified"


def test_native_environment_keeps_upstream_launcher_in_its_pinned_venv(
    tmp_path: Path,
) -> None:
    native_python = tmp_path / "opendm-venv" / "bin" / "python"
    native_python.parent.mkdir(parents=True)
    native_python.touch()
    environment = dm05_opendm._native_environment(
        str(native_python), repo_root=tmp_path / "opendm"
    )
    assert environment["PATH"].split(":", 1)[0] == str(native_python.parent)
    assert environment["PYTHONPATH"].split(":", 1)[0] == str(tmp_path / "opendm")
    assert environment["IMAGEIO_FFMPEG_EXE"] == "/usr/bin/ffmpeg"
    parsed = dm05_opendm.build_parser().parse_args(
        [
            "train",
            "--repo-root",
            "/opt/byof",
            "--opendm-python",
            str(native_python),
            "--prepared-uri",
            "s3://bucket/prepared",
            "--checkpoint-uri",
            "s3://bucket/checkpoint",
            "--nproc-per-node",
            "8",
            "--train-steps",
            "100000",
        ]
    )
    assert parsed.opendm_python == str(native_python)


def test_hf_download_uses_the_pinned_native_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native_python = tmp_path / "opendm-venv" / "bin" / "python"
    native_python.parent.mkdir(parents=True)
    native_python.touch()
    captured: dict[str, object] = {}

    def fake_run(
        command: list[str], *, cwd: Path, env: dict[str, str] | None = None
    ) -> None:
        captured.update(command=command, cwd=cwd, env=env)

    monkeypatch.setattr(dm05_opendm, "_run", fake_run)
    dm05_opendm._hf_download(
        repo_id="Dexmal/DM05",
        revision="0123456789abcdef",
        repo_type="model",
        destination=tmp_path / "payload",
        cwd=tmp_path,
        native_python=str(native_python),
    )

    assert captured["command"] == [
        "hf",
        "download",
        "Dexmal/DM05",
        "--repo-type",
        "model",
        "--revision",
        "0123456789abcdef",
        "--local-dir",
        str(tmp_path / "payload"),
    ]
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["PATH"].split(":", 1)[0] == str(native_python.parent)


def test_loopback_inference_refuses_redirects_and_never_uses_a_supplied_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Response:
        status = 302

        @staticmethod
        def read() -> bytes:
            return b""

    class Connection:
        def __init__(self, host: str, port: int, *, timeout: float) -> None:
            captured.update(host=host, port=port, timeout=timeout)

        def request(
            self,
            method: str,
            path: str,
            body: bytes | None = None,
            headers: dict[str, str] | None = None,
        ) -> None:
            captured.update(method=method, path=path, body=body, headers=headers)

        @staticmethod
        def getresponse() -> Response:
            return Response()

        @staticmethod
        def close() -> None:
            return None

    monkeypatch.setattr(dm05_opendm.http.client, "HTTPConnection", Connection)
    payload = {
        "observation": {
            "state": [0.0] * 8,
            "images": {"1": "head", "2": "wrist"},
        }
    }
    with pytest.raises(dm05_opendm.DM05WorkflowError, match="redirect"):
        dm05_opendm._infer_once(port=7891, payload=payload)

    assert captured == {
        "host": "127.0.0.1",
        "port": 7891,
        "timeout": 600,
        "method": "POST",
        "path": "/v1/infer",
        "body": json.dumps(payload).encode("utf-8"),
        "headers": {"Content-Type": "application/json"},
    }
    source = (ROOT / "npa/src/npa/workflows/dm05_opendm.py").read_text(encoding="utf-8")
    assert "urllib.request.urlopen" not in source


@pytest.mark.parametrize("port", [0, -1, 65536, True])
def test_loopback_inference_rejects_invalid_ports(port: int) -> None:
    with pytest.raises(dm05_opendm.DM05WorkflowError, match="invalid loopback"):
        dm05_opendm._loopback_port(port)


def test_revision_check_scopes_git_trust_to_the_pinned_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured.update(command=command, kwargs=kwargs)
        return SimpleNamespace(stdout="0123456789abcdef\n")

    monkeypatch.setattr(dm05_opendm.subprocess, "run", fake_run)
    assert dm05_opendm._git_revision(tmp_path) == "0123456789abcdef"
    assert captured["command"] == [
        "git",
        "-c",
        f"safe.directory={tmp_path}",
        "rev-parse",
        "HEAD",
    ]
