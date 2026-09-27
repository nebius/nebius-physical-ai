from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from npa.clients.storage import StoragePreconditionFailed
from npa.workflows.behavior_challenge import (
    campaign_runner,
    native_comet_policy,
    native_comet_server,
    train_experience,
    train_experience_evaluator,
    trained_comet_policy,
    train_prompt,
)
from npa.workflows.behavior_challenge.campaign import canonical_digest
from npa.workflows.behavior_challenge import policy as campaign_policy
from npa.workflows.behavior_challenge.campaign_runner import (
    _record_originals,
    _record_train_experience,
    _train_experience_evaluator_argv,
    recover_case,
)
from npa.workflows.behavior_challenge.case_store import CaseAlreadyStarted, CaseStore
from npa.workflows.behavior_challenge.autonomous_training_data import (
    AutonomousCometDataset,
    OfficialGoalProgressDataset,
)
from npa.workflows.behavior_challenge.native_comet_policy import _stage_adapters
from npa.workflows.behavior_challenge.train_experience import (
    EvaluatorExperienceRecorder,
    GOAL_PROGRESS_SOURCE,
    PolicyExperienceRecorder,
    RecordingPolicy,
    allowed_observation,
    array_identity,
    file_identity,
    finalize_experience,
    validate_experience_config,
    write_experience_config,
)
from npa.workflows.behavior_challenge.train_experience_evaluator import (
    _apply_hook,
    _official_goal_source,
    _official_step,
    _recording_policy,
    _run_hook,
)

DIGEST = "a" * 64


def _prompt_binding(prompt: str = "pick up the trash") -> dict:
    payload = {
        "schema": train_prompt.SCHEMA,
        "task_name": "picking_up_trash",
        "task_id": 1,
        "effective_prompt": prompt,
        "source_kind": "task_mapping_default",
        "source_files": {
            name: {"bytes": 1, "sha256": DIGEST}
            for name in (
                "task_mapping",
                "wrapper",
                "tokenizer",
                "transforms",
                "training_config",
            )
        },
        "tokenizer_contract": {
            "implementation": "PaligemmaTokenizer",
            "model_uri": "gs://big_vision/paligemma_tokenizer.model",
            "model_bytes_status": "not_observed_bind_at_training_projection",
        },
    }
    return {**payload, "binding_sha256": canonical_digest(payload)}


COMMIT = "b" * 40


class _Random:
    """Track key identity and advancement without the model's JAX runtime."""

    @staticmethod
    def key(seed):
        return np.array([0, seed], dtype=np.uint32)

    @staticmethod
    def split(value):
        return value + _Random.key(1), value + _Random.key(2)

    key_data = staticmethod(np.asarray)


@pytest.fixture(autouse=True)
def _rng_double(monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(random=_Random))


def test_module_collection_does_not_require_jax():
    code = "import runpy,sys; sys.modules['jax']=None; runpy.run_path(sys.argv[1])"
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


_POLICY_CHILD = """
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from npa.workflows.behavior_challenge.native_comet_server import _serve
from npa.workflows.behavior_challenge.train_experience import PolicyExperienceRecorder

root = Path(sys.argv[1])
recorder = PolicyExperienceRecorder(root)
actions = np.zeros((32, 23), dtype=np.float64)
recorder.record_emission(actions, "1" * 64, "2" * 64, 1, 2)
recorder.record_action(actions[0], 0)
try:
    asyncio.run(_serve(None, None, SimpleNamespace(port=int(sys.argv[2])), None, None, recorder))
finally:
    recorder.close()
"""


class _Packer:
    def pack(self, value):
        return value


class _Msgpack:
    Packer = _Packer

    @staticmethod
    def unpackb(value):
        return value


class _Wire:
    def __init__(self, _revision):
        pass

    @staticmethod
    def is_reset(_value):
        return False

    @staticmethod
    def observation_for_policy(value):
        return value

    @staticmethod
    def action_for_evaluator(value):
        return value


class _Socket:
    def __init__(self, observation):
        self.observation = observation
        self.sent = []

    async def send(self, value):
        self.sent.append(value)

    def __aiter__(self):
        async def values():
            yield self.observation
            yield self.observation

        return values()


def _case() -> dict[str, object]:
    return {
        "case_id": "train-case-0",
        "task": "picking_up_trash",
        "instance_id": 0,
        "rollout_id": 0,
        "split": "train",
    }


def _config(*, include_depth: bool = False) -> dict[str, object]:
    return {
        "schema": "npa.behavior.train-experience-config.v2",
        "status": "train_only_recording_enabled",
        "split": "train",
        "cadence": "model_decision_observation_with_all_applied_actions",
        "action_horizon": 32,
        "include_depth": include_depth,
        "case": _case(),
        "panel_sha256": DIGEST,
        "policy_identity_sha256": DIGEST,
        "checkpoint_sha256": DIGEST,
        "rng_contract_sha256": DIGEST,
        "source_commit": COMMIT,
        "prompt_binding": _prompt_binding(),
    }


def test_recording_config_uses_the_declared_panel_id(tmp_path: Path) -> None:
    payload = {
        "schema": "npa.behavior.nonreporting-train-panel.fixture.v1",
        "policy_binding_sha256": DIGEST,
    }
    panel = {**payload, "panel_id": canonical_digest(payload)}
    assert canonical_digest(panel) != panel["panel_id"]
    binding = {"artifacts": {"rng_contract": {"identity": {"sha256": DIGEST}}}}
    verified = {"checkpoint": {"content_sha256": DIGEST}}
    args = SimpleNamespace(train_experience_depth=False)

    native_comet_policy._write_experience_config(
        args,
        panel=panel,
        case=_case(),
        output=tmp_path,
        binding=binding,
        verified=verified,
        prompt_binding=_prompt_binding(),
    )

    config = json.loads((tmp_path / "train-experience/config.json").read_text())
    assert config["panel_sha256"] == panel["panel_id"]


def _observation(state_value: float) -> dict[str, np.ndarray]:
    return {
        "robot_r1::proprio": np.full(61, state_value, dtype=np.float32),
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.zeros(
            (720, 720, 3), dtype=np.uint8
        ),
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": np.zeros(
            (480, 480, 3), dtype=np.uint8
        ),
        "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": np.zeros(
            (480, 480, 3), dtype=np.uint8
        ),
        "robot_r1::robot_r1:zed_link:Camera:0::depth_linear": np.ones(
            (720, 720), dtype=np.float32
        ),
        "robot_r1::robot_r1:left_realsense_link:Camera:0::depth_linear": np.ones(
            (480, 480), dtype=np.float32
        ),
        "robot_r1::robot_r1:right_realsense_link:Camera:0::depth_linear": np.ones(
            (480, 480), dtype=np.float32
        ),
        "privileged.object_pose": np.ones(7, dtype=np.float32),
    }


class _Policy:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = actions
        self._rng = _Random.key(7)
        self.calls = 0

    def infer(self, inputs: dict) -> dict[str, np.ndarray]:
        assert inputs == {"prompt": "pick up the trash"}
        self.calls += 1
        self._rng, _ = _Random.split(self._rng)
        return {"actions": self.actions}


class _ChunkWrapper:
    def __init__(self, policy) -> None:
        self.policy = policy
        self.chunk = None
        self.index = 0

    def reset(self) -> None:
        self.chunk = None
        self.index = 0

    def act(self, inputs):
        if self.chunk is None:
            self.chunk = self.policy.infer(inputs)["actions"]
        action = self.chunk[self.index : self.index + 1]
        self.index += 1
        return action


def _metrics(output: Path, steps: int) -> None:
    directory = output / "json"
    directory.mkdir()
    value = {
        "task": "picking_up_trash",
        "instance_id": 0,
        "rollout_id": 0,
        "steps": steps,
        "success": False,
        "q_score": {"final": 0.0},
    }
    (directory / "picking_up_trash_0_0.json").write_text(json.dumps(value))


def _official_transport(action: np.ndarray) -> np.ndarray:
    torch = pytest.importorskip("torch")
    action_dict = {"action": action}
    converted = torch.from_numpy(deepcopy(action_dict["action"])).to(torch.float32)
    return converted.numpy()


def _native_args(root: Path) -> SimpleNamespace:
    initial = native_comet_server._key_identity(_Random.key(7))
    args = SimpleNamespace(
        upstream_commit="fixture",
        case_id="train-case-0",
        case_seed=7,
        task_name="picking_up_trash",
        instance_id=0,
        rollout_id=0,
        checkpoint_sha256=DIGEST,
        rng_contract_sha256=DIGEST,
        trace_configuration_sha256=DIGEST,
        initial_rng_sha256=initial,
        process_receipt=root / "native-process.json",
        action_trace=None,
    )
    args.process_identity_sha256 = native_comet_server._process_identity(args, initial)
    return args


def _native_sequence(root: Path, monkeypatch, *, record: bool) -> tuple:
    monkeypatch.setitem(
        sys.modules, "openpi_client", SimpleNamespace(msgpack_numpy=_Msgpack)
    )
    monkeypatch.setattr(native_comet_server, "EvaluatorWire", _Wire)
    monkeypatch.setattr(
        native_comet_server,
        "policy_observation",
        lambda _value: {"prompt": "pick up the trash"},
    )
    args = _native_args(root)
    policy = _Policy(np.linspace(0, 1, 32 * 23).reshape(32, 23))
    wrapper = _ChunkWrapper(policy)
    experience = None
    if record:
        write_experience_config(root / "experience", _config())
        experience = PolicyExperienceRecorder(root / "experience")
        wrapper.policy = RecordingPolicy(policy, experience)
    trace = native_comet_server.ActionTrace(None, args)
    progress = native_comet_server.ProcessProgress(root / "progress.jsonl", args)
    socket = _Socket({"robot_r1::proprio": np.zeros(61, dtype=np.float32)})
    asyncio.run(
        native_comet_server._connection(
            socket, wrapper, policy, args, trace, progress, experience
        )
    )
    progress.close()
    terminal = experience.close() if experience is not None else None
    return socket.sent[1:], policy, terminal


def _closed_two_step_experience(root: Path, output: Path) -> None:
    write_experience_config(root, _config())
    raw = np.arange(32 * 23, dtype=np.float64).reshape(32, 23)
    policy_recorder = PolicyExperienceRecorder(root)
    policy = RecordingPolicy(_Policy(raw), policy_recorder)
    policy.infer({"prompt": "pick up the trash"})
    evaluator = EvaluatorExperienceRecorder(root)
    evaluator.reset()
    for frame in range(2):
        policy_recorder.record_action(raw[frame], 0)
        evaluator.record_policy(
            _observation(frame),
            raw[frame].astype(np.float32),
            4 * frame + 1,
            4 * frame + 2,
        )
        evaluator.record_applied(
            _observation(frame + 1),
            raw[frame].astype(np.float32),
            4 * frame + 3,
            4 * frame + 4,
        )
    evaluator.close()
    policy_recorder.close()
    _metrics(output, 2)


def _closed_progress_experience(root: Path, output: Path) -> None:
    write_experience_config(root, _config())
    raw = np.arange(32 * 23, dtype=np.float64).reshape(32, 23)
    policy_recorder = PolicyExperienceRecorder(root)
    policy = RecordingPolicy(_Policy(raw), policy_recorder)
    policy.infer({"prompt": "pick up the trash"})
    evaluator = EvaluatorExperienceRecorder(root, progress_source=GOAL_PROGRESS_SOURCE)
    evaluator.reset()
    statuses = (([], [0, 1]), ([0], [1]))
    for frame, (satisfied, unsatisfied) in enumerate(statuses):
        policy_recorder.record_action(raw[frame], 0)
        action = raw[frame].astype(np.float32)
        evaluator.record_policy(
            _observation(frame), action, 4 * frame + 1, 4 * frame + 2
        )
        evaluator.record_applied(
            _observation(frame + 1),
            action,
            4 * frame + 3,
            4 * frame + 4,
            official_step={
                "goal_status": {
                    "satisfied": list(satisfied),
                    "unsatisfied": list(unsatisfied),
                },
                "terminated": False,
                "truncated": frame == 1,
            },
        )
    evaluator.close()
    policy_recorder.close()
    _metrics(output, 2)


def _experience_version(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        record={
            "case": _case(),
            "train_experience_config": file_identity(root / "config.json"),
        }
    )


def _rewrite_shard(root: Path, branch: str, name: str, mutate) -> None:
    directory = root / branch / name
    path = directory / "part-000000.npz"
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: np.array(source[key], copy=True) for key in source.files}
    mutate(arrays)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    row = train_experience._shard_row(path, arrays, len(next(iter(arrays.values()))))
    (directory / "index.json").write_text(json.dumps({"shards": [row]}))
    terminal_path = root / branch / "terminal.json"
    terminal = json.loads(terminal_path.read_text())
    terminal[{"actions": "action_shards", "transitions": "transition_shards"}[name]] = [
        row
    ]
    terminal_path.write_text(json.dumps(terminal))


def test_train_scope_and_depth_are_explicit() -> None:
    assert validate_experience_config(_config())["include_depth"] is False
    invalid = _config()
    invalid["split"] = "development"
    with pytest.raises(ValueError, match="scope"):
        validate_experience_config(invalid)

    selected = allowed_observation(_observation(0), include_depth=False)
    assert len(selected) == 4
    assert not any("depth" in name for name in selected)
    assert "privileged.object_pose" not in selected

    with_depth = allowed_observation(_observation(0), include_depth=True)
    assert len(with_depth) == 7
    assert sum("depth" in name for name in with_depth) == 3


def test_train_evaluator_entrypoint_replaces_only_module_transport(
    tmp_path: Path,
) -> None:
    command = ["/runtime/python", "-m", "omnigibson.eval.eval", "--mode", "train"]
    assert _train_experience_evaluator_argv(command, tmp_path) == [
        "/runtime/python",
        str(tmp_path / "train_experience_evaluator.py"),
        "--mode",
        "train",
    ]
    with pytest.raises(ValueError, match="command shape"):
        _train_experience_evaluator_argv(["python", "unexpected.py"], tmp_path)


def test_disabled_recording_stages_no_recorder_modules(tmp_path: Path) -> None:
    disabled = tmp_path / "disabled"
    enabled = tmp_path / "enabled"
    disabled.mkdir()
    enabled.mkdir()

    disabled_rows = _stage_adapters(disabled)
    enabled_rows = _stage_adapters(enabled, train_experience=True)

    assert "train_experience.py" not in disabled_rows
    assert "train_prompt.py" not in disabled_rows
    assert not (disabled / "semantic_monitor").exists()
    assert "train_experience.py" in enabled_rows
    assert "train_experience_evaluator.py" in enabled_rows
    assert "train_prompt.py" in enabled_rows
    assert "semantic_monitor/interface.py" in enabled_rows


@pytest.mark.parametrize("adapter", [native_comet_policy, trained_comet_policy])
def test_staged_recorders_import_without_repository(tmp_path: Path, adapter) -> None:
    adapter._stage_adapters(tmp_path, train_experience=True)
    code = """
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
sys.modules['jax'] = None
sys.modules['npa'] = None
import train_experience
import train_experience_evaluator
import train_prompt
import native_comet_server
import semantic_monitor.collector
import semantic_monitor.schema

for name, module in tuple(sys.modules.items()):
    if name.startswith(('native_comet_server', 'train_experience', 'train_prompt', 'semantic_monitor')):
        assert Path(module.__file__).resolve().is_relative_to(root), name
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_policy_recorder_preserves_one_inference_and_chunk_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experience"
    write_experience_config(root, _config())
    recorder = PolicyExperienceRecorder(root)
    actions = np.arange(32 * 23, dtype=np.float32).reshape(32, 23)
    delegate = _Policy(actions)
    policy = RecordingPolicy(delegate, recorder)

    result = policy.infer({"prompt": "pick up the trash"})
    assert result["actions"] is actions
    assert delegate.calls == 1
    recorder.record_action(actions[0], 0)
    with pytest.raises(ValueError, match="source chunk"):
        recorder.record_action(actions[2], 0)
    terminal = recorder.close()

    assert terminal["emission_count"] == 1
    assert terminal["action_count"] == 1


def test_recording_on_off_preserves_actions_rng_and_inference_sequence(
    tmp_path: Path,
) -> None:
    actions = np.arange(32 * 23, dtype=np.float64).reshape(32, 23)
    plain = _Policy(actions)
    recorded = _Policy(actions.copy())
    root = tmp_path / "experience"
    write_experience_config(root, _config())
    wrapped = RecordingPolicy(recorded, PolicyExperienceRecorder(root))

    plain_result = plain.infer({"prompt": "pick up the trash"})
    wrapped_result = wrapped.infer({"prompt": "pick up the trash"})

    assert plain.calls == recorded.calls == 1
    assert array_identity(plain_result["actions"]) == array_identity(
        wrapped_result["actions"]
    )
    assert np.array_equal(_Random.key_data(plain._rng), _Random.key_data(recorded._rng))


def test_native_server_recording_preserves_wire_actions_and_rng(
    tmp_path: Path, monkeypatch
) -> None:
    plain_root = tmp_path / "plain"
    recorded_root = tmp_path / "recorded"
    plain_root.mkdir()
    recorded_root.mkdir()

    plain_actions, plain, _ = _native_sequence(plain_root, monkeypatch, record=False)
    recorded_actions, recorded, terminal = _native_sequence(
        recorded_root, monkeypatch, record=True
    )

    assert [array_identity(value["action"]) for value in plain_actions] == [
        array_identity(value["action"]) for value in recorded_actions
    ]
    assert plain.calls == recorded.calls == terminal["emission_count"] == 1
    assert terminal["action_count"] == 2
    assert np.array_equal(_Random.key_data(plain._rng), _Random.key_data(recorded._rng))


def test_production_supervisor_stop_finalizes_experience(tmp_path: Path) -> None:
    root = tmp_path / "train-experience"
    write_experience_config(root, _config())
    _one_step_evaluator_terminal(root)
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, "-c", _POLICY_CHILD, str(root), str(port)],
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_healthy(process, port)
        campaign_policy._stop_policy(process)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    output = process.stdout.read() if process.stdout is not None else ""
    assert process.returncode == 0, output
    assert (root / "policy/terminal.json").is_file()
    assert (root / "evaluator/terminal.json").is_file()
    _metrics(tmp_path, 1)
    manifest = finalize_experience(root, tmp_path)
    assert manifest["status"] == "complete_exact_train_experience"


def _one_step_evaluator_terminal(root: Path) -> None:
    recorder = EvaluatorExperienceRecorder(root)
    recorder.reset()
    action = np.zeros(23, dtype=np.float32)
    recorder.record_policy(_observation(0), action, 1, 2)
    recorder.record_applied(_observation(1), action, 3, 4)
    recorder.close()


def _free_port() -> int:
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        return int(available.getsockname()[1])


def _wait_healthy(process: subprocess.Popen, port: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and process.poll() is None:
        if campaign_policy._healthy(port):
            return
        time.sleep(0.05)
    raise RuntimeError("recording policy subprocess did not become healthy")


def test_complete_experience_projects_exact_comet_sample(tmp_path: Path) -> None:
    root = tmp_path / "experience"
    write_experience_config(root, _config())
    actions = np.arange(32 * 23, dtype=np.float64).reshape(32, 23)

    policy_recorder = PolicyExperienceRecorder(root)
    policy = RecordingPolicy(_Policy(actions), policy_recorder)
    policy.infer({"prompt": "pick up the trash"})
    policy_recorder.record_action(actions[0], 0)
    policy_recorder.record_action(actions[1], 0)

    evaluator = EvaluatorExperienceRecorder(root)
    observations = [_observation(value) for value in (0.0, 1.0, 2.0)]
    evaluator.reset()
    official = _official_transport(actions)
    evaluator.record_policy(observations[0], official[0], 1, 2)
    evaluator.record_applied(observations[1], official[0], 3, 4)
    evaluator.record_policy(observations[1], official[1], 5, 6)
    evaluator.record_applied(observations[2], official[1], 7, 8)
    evaluator.close()
    policy_recorder.close()

    _metrics(tmp_path, 2)
    manifest = finalize_experience(root, tmp_path)
    dataset = AutonomousCometDataset(root)
    sample = dataset[0]

    assert manifest["frame_count"] == 2
    assert manifest["emission_count"] == 1
    assert len(dataset) == 1
    assert sample["observation.state"].dtype == np.float32
    assert sample["action"].dtype == np.float32
    assert np.array_equal(sample["action"][0], actions[0].astype(np.float32))
    assert np.array_equal(sample["action"][1], actions[1].astype(np.float32))
    terminal = np.repeat(actions[1][None], 30, axis=0).astype(np.float32)
    assert np.array_equal(sample["action"][2:], terminal)
    assert sample["action_valid_mask"].tolist() == [True, True] + [False] * 30
    assert sample["action_is_pad"].tolist() == [False, False] + [True] * 30
    assert sample["prompt"] == "pick up the trash"
    assert all(value.shape[-1] == 3 for key, value in sample.items() if "images" in key)


def test_legacy_dataset_rejects_consistently_rehashed_prompt_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experience"
    _closed_two_step_experience(root, tmp_path)
    finalize_experience(root, tmp_path)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    binding = deepcopy(config.pop("prompt_binding"))
    config["schema"] = "npa.behavior.train-experience-config.v1"
    config["policy_prompt_override"] = binding["effective_prompt"]
    binding["source_kind"] = "literal_override"
    binding["binding_sha256"] = canonical_digest(
        {name: value for name, value in binding.items() if name != "binding_sha256"}
    )
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
    manifest_path = root / "experience-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"] = config
    for row in manifest["members"]:
        if row["path"] == "config.json":
            row.update(file_identity(config_path))
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    receipt = _legacy_prompt_receipt(config_path, binding)
    admitted = train_prompt.legacy_prompt_derivation_sha256(receipt)

    assert (
        AutonomousCometDataset(
            root,
            legacy_prompt_derivation=receipt,
            legacy_prompt_derivation_sha256=admitted,
        )[0]["prompt"]
        == binding["effective_prompt"]
    )
    rewritten = deepcopy(receipt)
    rewritten["prompt_binding"]["effective_prompt"] = "forged prompt"
    rewritten["prompt_binding"]["binding_sha256"] = canonical_digest(
        {
            name: value
            for name, value in rewritten["prompt_binding"].items()
            if name != "binding_sha256"
        }
    )
    rewritten["derivation_sha256"] = canonical_digest(
        {
            name: value
            for name, value in rewritten.items()
            if name != "derivation_sha256"
        }
    )
    with pytest.raises(ValueError, match="identity differs"):
        AutonomousCometDataset(
            root,
            legacy_prompt_derivation=rewritten,
            legacy_prompt_derivation_sha256=admitted,
        )


def _legacy_prompt_receipt(config_path: Path, binding: dict) -> dict:
    payload = {
        "schema": "npa.behavior.legacy-train-prompt-derivation.v1",
        "status": "legacy_recording_prompt_derived_without_mutation",
        "config": file_identity(config_path),
        "qualification": {"bytes": 1, "sha256": DIGEST},
        "source_commit": COMMIT,
        "observed_prompt_artifact": None,
        "prompt_binding": binding,
    }
    return {**payload, "derivation_sha256": canonical_digest(payload)}


def test_finalized_members_publish_before_success_manifest(tmp_path: Path) -> None:
    root = tmp_path / "train-experience"
    _closed_two_step_experience(root, tmp_path)
    finalize_experience(root, tmp_path)

    class Storage:
        def __init__(self):
            self.objects = {}
            self.calls = []

        def put_bytes_conditional(self, payload, uri, *, if_none_match):
            assert if_none_match is True
            self.objects[uri] = (payload, "etag")
            self.calls.append(uri)

        def read_bytes_with_etag(self, uri):
            return self.objects.get(uri)

    storage = Storage()
    version = _experience_version(root)
    store = SimpleNamespace(
        storage=storage,
        panel_id=DIGEST,
        artifact_prefix=lambda _version: "s3://evidence/v1",
    )
    _record_train_experience(store, version, tmp_path)

    manifest = json.loads((root / "experience-manifest.json").read_text())
    assert len(storage.calls) == len(manifest["members"]) + 2
    assert storage.calls[0].endswith("/publication-requirement.json")
    assert storage.calls[-1].endswith("/experience-manifest.json")
    assert set(storage.objects) == {
        f"s3://evidence/v1/train-experience/{row['path']}"
        for row in manifest["members"]
    } | {
        "s3://evidence/v1/train-experience/experience-manifest.json",
        "s3://evidence/v1/train-experience/publication-requirement.json",
    }


def test_partial_experience_cannot_publish(tmp_path: Path) -> None:
    (tmp_path / "train-experience").mkdir()
    store = SimpleNamespace(storage=object(), panel_id=DIGEST)
    version = SimpleNamespace(record={"case": _case()})
    with pytest.raises(ValueError, match="lacks its success manifest"):
        _record_train_experience(store, version, tmp_path)


def test_recorded_experience_requires_local_root(tmp_path: Path) -> None:
    class Storage:
        objects = {}
        calls = []

        def put_bytes_conditional(self, payload, uri, **_kwargs):
            self.calls.append(uri)
            self.objects[uri] = payload

        def read_bytes_with_etag(self, uri):
            value = self.objects.get(uri)
            return None if value is None else (value, "etag")

    completed = []
    storage = Storage()
    store = SimpleNamespace(
        storage=storage,
        panel_id=DIGEST,
        artifact_prefix=lambda _version: "s3://campaign/records/case/v1",
        complete=lambda *_args: completed.append(True),
    )
    version = SimpleNamespace(
        record={
            "case": _case(),
            "train_experience_config": {"bytes": 1, "sha256": "a" * 64},
        }
    )

    with pytest.raises(ValueError, match="experience root is absent"):
        _record_originals(store, version, tmp_path, {**_case(), "files": {}})
    assert storage.calls == []
    assert storage.objects == {}
    assert completed == []


def test_legacy_publication_allows_absent_experience_root(tmp_path: Path) -> None:
    store = SimpleNamespace(storage=object(), panel_id=DIGEST)
    version = SimpleNamespace(record={"case": _case()})

    _record_train_experience(store, version, tmp_path)


class _MemoryStorage:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.revision = 0
        self.fail_suffix: str | None = None
        self.calls: list[str] = []

    def put_bytes_conditional(
        self, payload, uri, *, if_match="", if_none_match=False, **_kwargs
    ):
        if self.fail_suffix and uri.endswith(self.fail_suffix):
            raise ConnectionError("injected experience publication failure")
        current = self.objects.get(uri)
        if (if_none_match and current) or (
            if_match and (current is None or current[1] != if_match)
        ):
            raise StoragePreconditionFailed("conditional conflict")
        self.revision += 1
        etag = str(self.revision)
        self.objects[uri] = payload, etag
        self.calls.append(uri)
        return etag

    def read_bytes_with_etag(self, uri):
        return self.objects.get(uri)

    def download_file(self, uri, destination) -> None:
        Path(destination).write_bytes(self.objects[uri][0])


def test_fresh_recovery_requires_and_restores_published_experience(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "train-experience"
    _closed_two_step_experience(root, tmp_path)
    finalize_experience(root, tmp_path)
    storage = _MemoryStorage()
    store = CaseStore(storage, "s3://campaign/records", DIGEST)
    version = store.start(
        store.claim(_case(), "worker-0"),
        train_experience_config=file_identity(root / "config.json"),
    )
    record = {**_case(), "files": {}}
    storage.fail_suffix = "/train-experience/config.json"

    with pytest.raises(ConnectionError, match="injected experience"):
        _record_originals(store, version, tmp_path, record)
    prefix = store.artifact_prefix(version)
    assert f"{prefix}/validation.json" in storage.objects
    assert f"{prefix}/train-experience/publication-requirement.json" in storage.objects
    assert f"{prefix}/train-experience/experience-manifest.json" not in storage.objects
    assert storage.calls.index(
        f"{prefix}/train-experience/publication-requirement.json"
    ) < storage.calls.index(f"{prefix}/validation.json")

    monkeypatch.setattr(campaign_runner, "_bind_execution_rollout", lambda *_: None)
    monkeypatch.setattr(campaign_runner, "inspect_rollout", lambda *_: record)
    with pytest.raises(CaseAlreadyStarted, match="required TRAIN experience"):
        recover_case(store, version, tmp_path / "fresh-incomplete", {})
    assert store.read(_case()).record["state"] == "started"

    storage.fail_suffix = None
    _record_train_experience(store, version, tmp_path)
    recovered_root = tmp_path / "fresh-complete"
    assert recover_case(store, version, recovered_root, {}) == record
    assert store.read(_case()).record["state"] == "complete"
    assert (
        recovered_root / "train-experience/experience-manifest.json"
    ).read_bytes() == (root / "experience-manifest.json").read_bytes()


@pytest.mark.parametrize("state", ["started", "complete"])
def test_recovery_requires_recorded_experience_requirement(
    tmp_path: Path, state: str
) -> None:
    storage = _MemoryStorage()
    version = SimpleNamespace(
        record={
            "state": state,
            "train_experience_config": {"bytes": 1, "sha256": "a" * 64},
        }
    )
    store = SimpleNamespace(
        storage=storage,
        artifact_prefix=lambda _version: "s3://campaign/records/case/v1",
    )

    with pytest.raises(CaseAlreadyStarted, match="required TRAIN experience"):
        campaign_runner._restore_train_experience(store, version, tmp_path / "fresh")


def test_legacy_recovery_allows_absent_experience_requirement(tmp_path: Path) -> None:
    store = SimpleNamespace(
        storage=_MemoryStorage(),
        artifact_prefix=lambda _version: "s3://campaign/records/case/v1",
    )
    version = SimpleNamespace(record={"state": "complete"})

    campaign_runner._restore_train_experience(store, version, tmp_path / "fresh")


def test_actual_evaluator_hooks_preserve_float32_apply_bytes(tmp_path: Path) -> None:
    root = tmp_path / "experience"
    write_experience_config(root, _config())
    recorder = EvaluatorExperienceRecorder(root)
    returned = np.arange(23, dtype=np.float32)[None]

    class Delegate:
        def reset(self) -> None:
            pass

        def forward(self, *, obs):
            assert obs["robot_r1::proprio"].shape == (1, 61)
            return returned

    policy = _recording_policy(Delegate(), recorder)
    policy.reset()
    before = {name: value[None] for name, value in _observation(0).items()}
    before.pop("privileged.object_pose")
    action = policy.forward(obs=before)

    evaluator = _fake_evaluator(recorder)

    applied = []

    def original(_evaluator, actions, active_env_indices):
        assert active_env_indices == [0]
        applied.append(array_identity(actions))
        return np.asarray([False]), np.asarray([False]), [{}]

    result = _apply_hook(original)(evaluator, action, [0])
    assert result[0].tolist() == [False]
    assert applied == [array_identity(returned)]
    terminal = recorder.close()
    assert terminal["frame_count"] == 1


def test_official_goal_progress_is_separate_and_action_aligned(tmp_path: Path) -> None:
    root = tmp_path / "experience"
    _closed_progress_experience(root, tmp_path)

    manifest = finalize_experience(root, tmp_path)
    actions = AutonomousCometDataset(root)
    progress = OfficialGoalProgressDataset(root)

    assert len(actions) == 1
    assert len(progress) == manifest["frame_count"] == 2
    assert "goal_status" not in actions[0]
    assert progress[0]["delta_satisfied"] is None
    assert progress[0]["potential"] == np.float32(0.0)
    assert progress[1]["delta_satisfied"] == 1
    assert progress[1]["potential"] == np.float32(0.5)
    annotation = manifest["training_annotations"]["official_goal_status"]
    assert annotation["default_comet_projection_includes_annotation"] is False
    assert annotation["source"] == GOAL_PROGRESS_SOURCE


def test_actual_apply_hook_records_official_done_goal_status(tmp_path: Path) -> None:
    root = tmp_path / "experience"
    write_experience_config(root, _config())
    recorder = EvaluatorExperienceRecorder(root, progress_source=GOAL_PROGRESS_SOURCE)
    recorder.reset()
    action = np.zeros((1, 23), dtype=np.float32)
    recorder.record_policy(_observation(0), action, 1, 2)
    evaluator = _fake_evaluator(recorder)

    def original(_evaluator, actions, active_env_indices):
        assert active_env_indices == [0]
        assert actions is action
        return (
            np.asarray([False]),
            np.asarray([True]),
            [{"done": {"goal_status": {"satisfied": [0], "unsatisfied": [1]}}}],
        )

    result = _apply_hook(original)(evaluator, action, [0])
    terminal = recorder.close()
    rows = [
        json.loads(line)
        for line in (root / "evaluator/goal-progress.jsonl").read_text().splitlines()
    ]
    assert result[2][0]["done"]["goal_status"]["satisfied"] == [0]
    assert rows[0]["frame_index"] == 0
    assert rows[0]["truncated"] is True
    assert terminal["progress_source"] == GOAL_PROGRESS_SOURCE


def test_official_step_rejects_non_v393_info_shape() -> None:
    with pytest.raises(ValueError, match="done info"):
        _official_step((np.asarray([False]), np.asarray([False]), [{}]))
    with pytest.raises(ValueError, match="termination array"):
        _official_step((np.asarray([0]), np.asarray([False]), [{"done": {}}]))


def test_goal_progress_source_is_checked_before_install(tmp_path, monkeypatch) -> None:
    classes = [
        type(name, (), {})
        for name in (
            "Evaluator",
            "BehaviorTask",
            "BaseTask",
            "Environment",
            "Predicate",
            "Evaluate",
        )
    ]
    modules = (
        ("omnigibson.eval.evaluator", "BatchedEvaluator"),
        ("omnigibson.tasks.behavior_task", "BehaviorTask"),
        ("omnigibson.tasks.task_base", "BaseTask"),
        ("omnigibson.envs.env_base", "Environment"),
        ("omnigibson.termination_conditions.predicate_goal", "PredicateGoal"),
        ("bddl.condition_evaluation", "evaluate_state"),
    )
    for (module, attribute), cls in zip(modules, classes, strict=True):
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(**{attribute: cls}))
    paths = {}
    for cls, suffix in zip(classes, GOAL_PROGRESS_SOURCE["files"], strict=True):
        path = tmp_path / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
        paths[cls] = path
    monkeypatch.setattr(train_experience_evaluator.inspect, "getsourcefile", paths.get)
    monkeypatch.setattr(
        train_experience_evaluator,
        "file_identity",
        lambda path: GOAL_PROGRESS_SOURCE["files"][
            path.relative_to(tmp_path).as_posix()
        ],
    )
    assert _official_goal_source() == GOAL_PROGRESS_SOURCE

    for changed in (
        "OmniGibson/omnigibson/tasks/task_base.py",
        "OmniGibson/omnigibson/envs/env_base.py",
    ):
        monkeypatch.setattr(
            train_experience_evaluator,
            "file_identity",
            lambda path, changed=changed: (
                {"bytes": 1, "sha256": "0" * 64}
                if path.relative_to(tmp_path).as_posix() == changed
                else GOAL_PROGRESS_SOURCE["files"][
                    path.relative_to(tmp_path).as_posix()
                ]
            ),
        )
        with pytest.raises(ValueError, match="source bytes"):
            _official_goal_source()

    monkeypatch.setattr(
        train_experience_evaluator,
        "file_identity",
        lambda _path: {"bytes": 1, "sha256": "0" * 64},
    )
    with pytest.raises(ValueError, match="source bytes"):
        _official_goal_source()


@pytest.mark.parametrize("mutation", ["partition", "missing", "source", "outcome"])
def test_goal_progress_consistent_rehash_mutants_reject(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / "experience"
    _closed_progress_experience(root, tmp_path)
    terminal_path = root / "evaluator/terminal.json"
    terminal = json.loads(terminal_path.read_text())
    if mutation == "source":
        terminal["progress_source"]["upstream_commit"] = "a" * 40
        terminal_path.write_text(json.dumps(terminal))
    elif mutation == "outcome":
        metrics_path = tmp_path / "json/picking_up_trash_0_0.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["success"] = True
        metrics_path.write_text(json.dumps(metrics))
    else:
        progress_path = root / "evaluator/goal-progress.jsonl"
        rows = [json.loads(line) for line in progress_path.read_text().splitlines()]
        if mutation == "missing":
            rows.pop()
        else:
            rows[1]["goal_status"] = {"satisfied": [0], "unsatisfied": [0]}
        progress_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        terminal["progress_records"] = file_identity(progress_path)
        terminal_path.write_text(json.dumps(terminal))

    with pytest.raises(ValueError, match="goal progress|goal status"):
        finalize_experience(root, tmp_path)


def test_legacy_experience_has_no_progress_annotation(tmp_path: Path) -> None:
    root = tmp_path / "experience"
    _closed_two_step_experience(root, tmp_path)
    manifest = finalize_experience(root, tmp_path)

    assert "training_annotations" not in manifest
    assert len(AutonomousCometDataset(root)) == 1
    with pytest.raises(ValueError, match="no official goal progress"):
        OfficialGoalProgressDataset(root)


def _fake_evaluator(recorder):
    class Evaluator:
        _npa_train_experience = recorder

        def _batch_obs(self):
            value = _observation(1)
            value.pop("privileged.object_pose")
            return {name: array[None] for name, array in value.items()}

    return Evaluator()


def test_incomplete_official_run_has_no_success_terminal(tmp_path: Path) -> None:
    root = tmp_path / "experience"
    write_experience_config(root, _config())
    recorder = EvaluatorExperienceRecorder(root)
    recorder.reset()

    class Evaluator:
        _npa_train_experience = recorder

    def fails(_evaluator, _instances, **_kwargs):
        raise RuntimeError("official evaluator failed")

    with pytest.raises(RuntimeError, match="official evaluator failed"):
        _run_hook(fails)(Evaluator(), [0], rollout_id=0)
    assert not (root / "evaluator/terminal.json").exists()
    _metrics(tmp_path, 1)
    with pytest.raises(ValueError, match="terminal"):
        finalize_experience(root, tmp_path)


def test_final_partial_chunk_and_decision_join_are_complete(tmp_path: Path) -> None:
    root = tmp_path / "experience"
    write_experience_config(root, _config())
    raw = np.linspace(0.0, 1.0, 32 * 23, dtype=np.float64).reshape(32, 23)
    official = raw.astype(np.float32)
    policy_recorder = PolicyExperienceRecorder(root)
    policy = RecordingPolicy(_Policy(raw), policy_recorder)
    evaluator = EvaluatorExperienceRecorder(root)
    evaluator.reset()
    clock = 1
    for frame in range(33):
        if frame % 32 == 0:
            policy.infer({"prompt": "pick up the trash"})
        ordinal, offset = divmod(frame, 32)
        policy_recorder.record_action(raw[offset], ordinal)
        evaluator.record_policy(
            _observation(float(frame)), official[offset], clock, clock + 1
        )
        evaluator.record_applied(
            _observation(float(frame + 1)), official[offset], clock + 2, clock + 3
        )
        clock += 4
    with pytest.raises(ValueError, match="reset"):
        evaluator.reset()
    evaluator.close()
    policy_recorder.close()
    _metrics(tmp_path, 33)

    manifest = finalize_experience(root, tmp_path)
    dataset = AutonomousCometDataset(root)

    assert manifest["episode_count"] == 1
    assert manifest["emission_count"] == 2
    assert len(dataset) == 2
    assert dataset[1]["frame_index"] == 32
    assert dataset[1]["action_valid_mask"].tolist() == [True] + [False] * 31
    terminal = dataset.terminal_observation()
    assert int(terminal["frame_index"]) == 33
    assert np.all(terminal["robot_r1::proprio"] == 33)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (("action_index", 0, "indices"), ("chunk_offset", -1, "offsets")),
)
def test_rehashed_action_mapping_mutants_reject(
    tmp_path: Path, field: str, replacement: int, message: str
) -> None:
    root = tmp_path / "experience"
    _closed_two_step_experience(root, tmp_path)

    def mutate(arrays):
        arrays[field][1] = replacement

    _rewrite_shard(root, "policy", "actions", mutate)
    with pytest.raises(ValueError, match=message):
        finalize_experience(root, tmp_path)


def test_rehashed_applied_action_mutant_rejects(tmp_path: Path) -> None:
    root = tmp_path / "experience"
    _closed_two_step_experience(root, tmp_path)

    def mutate(arrays):
        arrays["applied_action"][0, 0] += 1

    _rewrite_shard(root, "evaluator", "transitions", mutate)
    with pytest.raises(ValueError, match="applied action shard"):
        finalize_experience(root, tmp_path)


def test_unknown_event_and_invalid_official_score_reject(tmp_path: Path) -> None:
    root = tmp_path / "experience"
    _closed_two_step_experience(root, tmp_path)
    events = root / "evaluator/events.jsonl"
    with events.open("a") as stream:
        stream.write(json.dumps({"schema": "attacker.unknown.v1"}) + "\n")
    terminal_path = root / "evaluator/terminal.json"
    terminal = json.loads(terminal_path.read_text())
    terminal["events"] = file_identity(events)
    terminal_path.write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match="journal chronology"):
        finalize_experience(root, tmp_path)

    events.write_text("\n".join(events.read_text().splitlines()[:-1]) + "\n")
    terminal["events"] = file_identity(events)
    terminal_path.write_text(json.dumps(terminal))
    metrics_path = tmp_path / "json/picking_up_trash_0_0.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["q_score"]["final"] = float("nan")
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="official score"):
        finalize_experience(root, tmp_path)


def test_post_observation_mutation_rejects_finalization(tmp_path: Path) -> None:
    root = tmp_path / "experience"
    write_experience_config(root, _config())
    actions = np.zeros((32, 23), dtype=np.float32)
    policy_recorder = PolicyExperienceRecorder(root)
    policy = RecordingPolicy(_Policy(actions), policy_recorder)
    policy.infer({"prompt": "pick up the trash"})
    policy_recorder.record_action(actions[0], 0)
    policy_recorder.record_action(actions[1], 0)
    policy_recorder.close()

    evaluator = EvaluatorExperienceRecorder(root)
    evaluator.reset()
    evaluator.record_policy(_observation(0), actions[0], 1, 2)
    evaluator.record_applied(_observation(1), actions[0], 3, 4)
    evaluator.record_policy(_observation(2), actions[1], 5, 6)
    evaluator.record_applied(_observation(3), actions[1], 7, 8)
    evaluator.close()
    _metrics(tmp_path, 2)

    with pytest.raises(ValueError, match="next policy input"):
        finalize_experience(root, tmp_path)


def test_official_apply_requires_the_returned_action(tmp_path: Path) -> None:
    root = tmp_path / "experience"
    write_experience_config(root, _config())
    recorder = EvaluatorExperienceRecorder(root)
    recorder.reset()
    action = np.zeros(23, dtype=np.float32)
    recorder.record_policy(_observation(0), action, 1, 2)
    changed = action.copy()
    changed[0] = 1
    with pytest.raises(ValueError, match="officially applied"):
        recorder.record_applied(_observation(1), changed, 3, 4)


def test_array_identity_distinguishes_dtype() -> None:
    left = array_identity(np.zeros(23, dtype=np.float32))
    right = array_identity(np.zeros(23, dtype=np.float64))
    assert left["shape"] == right["shape"]
    assert left["dtype"] != right["dtype"]
    assert left["sha256"] != right["sha256"]


def test_autonomous_dataset_rejects_rehashed_escaping_shard(tmp_path: Path) -> None:
    root = tmp_path / "experience"
    _closed_two_step_experience(root, tmp_path)
    finalize_experience(root, tmp_path)
    index_path = root / "evaluator/decisions/index.json"
    index = json.loads(index_path.read_text())
    index["shards"][0]["path"] = "../../outside.npz"
    index_path.write_text(json.dumps(index))
    manifest_path = root / "experience-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["members"]:
        if row["path"] == "evaluator/decisions/index.json":
            row.update(file_identity(index_path))
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="terminal shard index"):
        AutonomousCometDataset(root)


def test_experience_root_symlink_rejected_by_publisher_and_dataset(
    tmp_path: Path,
) -> None:
    root = tmp_path / "train-experience"
    _closed_two_step_experience(root, tmp_path)
    finalize_experience(root, tmp_path)
    outside = tmp_path / "outside-experience"
    root.rename(outside)
    root.symlink_to(outside, target_is_directory=True)
    store = SimpleNamespace(storage=object(), panel_id=DIGEST)
    version = SimpleNamespace(record={"case": _case()})

    with pytest.raises(ValueError, match="root must be a real directory"):
        campaign_runner._train_experience_bundle(store, version, tmp_path)
    with pytest.raises(ValueError, match="root must be a real directory"):
        AutonomousCometDataset(root)


@pytest.mark.parametrize("mutation", ["config", "frame_count", "outcome"])
def test_self_manifest_cannot_rewrite_inventoried_authority(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / "train-experience"
    _closed_two_step_experience(root, tmp_path)
    finalize_experience(root, tmp_path)
    path = root / "experience-manifest.json"
    manifest = json.loads(path.read_text())
    if mutation == "config":
        manifest["config"] = deepcopy(manifest["config"])
        manifest["config"]["case"]["task"] = "forged_task"
    elif mutation == "frame_count":
        manifest["frame_count"] = 1
    else:
        manifest["success"] = True
        manifest["q_score"] = 1.0
    path.write_text(json.dumps(manifest))
    store = SimpleNamespace(storage=object(), panel_id=DIGEST)
    version = SimpleNamespace(record={"case": _case()})

    with pytest.raises(ValueError, match="manifest summary"):
        campaign_runner._train_experience_bundle(store, version, tmp_path)
    with pytest.raises(ValueError, match="manifest summary"):
        AutonomousCometDataset(root)


def test_publication_joins_rehashed_metrics_to_official_output(tmp_path: Path) -> None:
    root = tmp_path / "train-experience"
    _closed_two_step_experience(root, tmp_path)
    finalize_experience(root, tmp_path)
    metrics_path = root / "official-metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["q_score"]["final"] = 1.0
    metrics["success"] = True
    metrics_path.write_text(json.dumps(metrics))
    manifest_path = root / "experience-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["q_score"] = 1.0
    manifest["success"] = True
    manifest["official_metrics"] = file_identity(metrics_path)
    for row in manifest["members"]:
        if row["path"] == "official-metrics.json":
            row.update(file_identity(metrics_path))
    manifest_path.write_text(json.dumps(manifest))
    store = SimpleNamespace(storage=object(), panel_id=DIGEST)
    version = SimpleNamespace(record={"case": _case()})

    with pytest.raises(ValueError, match="official metrics source"):
        campaign_runner._train_experience_bundle(store, version, tmp_path)


def test_started_case_rejects_consistently_rehashed_policy_config(
    tmp_path: Path,
) -> None:
    root = tmp_path / "train-experience"
    _closed_two_step_experience(root, tmp_path)
    finalize_experience(root, tmp_path)
    storage = _MemoryStorage()
    store = CaseStore(storage, "s3://campaign/records", DIGEST)
    started = store.start(
        store.claim(_case(), "worker-0"),
        train_experience_config=file_identity(root / "config.json"),
    )
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    config.update(
        policy_identity_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
        rng_contract_sha256="d" * 64,
        source_commit="e" * 40,
    )
    config_path.write_text(json.dumps(config))
    manifest_path = root / "experience-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"] = config
    for row in manifest["members"]:
        if row["path"] == "config.json":
            row.update(file_identity(config_path))
    manifest_path.write_text(json.dumps(manifest))
    resumed = store.read(_case())
    assert resumed == started

    with pytest.raises(ValueError, match="durable case start"):
        campaign_runner._train_experience_bundle(store, resumed, tmp_path)
