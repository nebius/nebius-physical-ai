"""Verify Native RLC TRAIN tracing stays distinct, durable, and action-neutral."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import numpy as np

from npa.workflows.behavior_challenge import campaign_runner, campaign_workflow
from npa.workflows.behavior_challenge.case_store import CaseStore
from npa.workflows.behavior_challenge.native_semantic_trace import (
    CANS,
    NativeStageObserver,
    StageTrace,
    allowed_observation_sha256,
    array_sha256,
    file_identity,
)
from npa.workflows.behavior_challenge.native_train_arrays import (
    NativeEvaluatorArrays,
    NativePolicyArrays,
)
from npa.workflows.behavior_challenge import native_train_admission
from npa.workflows.behavior_challenge.native_train_trace import (
    finalize_trace,
    validate_finalized_trace,
    write_config,
)
from npa.workflows.behavior_challenge.policy import _stop_policy
from npa.workflows.behavior_challenge.train_official_q import OFFICIAL_Q_SOURCE


class MemoryStorage:
    def __init__(self):
        self.objects = {}
        self.revision = 0

    def read_bytes_with_etag(self, uri):
        return self.objects.get(uri)

    def put_bytes_conditional(
        self, payload, uri, *, if_match=None, if_none_match=False, **_kwargs
    ):
        current = self.objects.get(uri)
        if if_none_match and current is not None:
            raise RuntimeError("exists")
        if if_match and (not current or current[1] != if_match):
            raise RuntimeError("etag")
        self.revision += 1
        etag = str(self.revision)
        self.objects[uri] = (payload, etag)
        return etag


def test_native_stage_observer_accepts_exact_masked_first_inference(tmp_path):
    """The released model masks invalid task stages with negative infinity."""

    wrapper_source = tmp_path / "eval_b1k_wrapper.py"
    wrapper_source.write_text("# exact wrapper identity fixture\n")
    source = {"runtime_wrapper": file_identity(wrapper_source)}

    class Wrapper:
        def __init__(self):
            self.action = np.arange(23, dtype=np.float32)
            self.current_stage = 0
            self.prediction_count = 0
            self.prediction_history = []

        def update_current_stage(self, logits):
            predicted_stage = int(np.argmax(logits))
            self.prediction_history.append(predicted_stage)
            self.current_stage = predicted_stage

        def act(self, observation):
            assert observation == {"allowed": True}
            logits = np.asarray(
                [0.1, 0.2, 0.9, 0.3, 0.0, 0.4] + [float("-inf")] * 9,
                dtype=np.float32,
            )
            assert np.isfinite(logits).sum() == 6
            self.update_current_stage(logits)
            self.prediction_count += 1
            return self.action

    wrapper = Wrapper()
    trace = StageTrace(source)
    returned = NativeStageObserver(wrapper, trace, wrapper_path=wrapper_source).act(
        {"allowed": True}
    )

    assert returned is wrapper.action
    assert wrapper.prediction_history == [2]
    assert trace.rows[0]["raw_predicted_stage"] == int(
        np.argmax([0.1, 0.2, 0.9, 0.3, 0.0, 0.4] + [float("-inf")] * 9)
    )
    assert trace.rows[0]["stage_after"] == 2


@pytest.mark.parametrize(
    "logits",
    [
        np.asarray([float("-inf"), float("-inf")], dtype=np.float32),
        np.asarray([0.0, float("inf")], dtype=np.float32),
        np.asarray([0.0, float("nan")], dtype=np.float32),
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([[0.0, 1.0]], dtype=np.float32),
    ],
)
def test_native_stage_trace_rejects_non_model_logit_contract(logits):
    trace = StageTrace({"runtime_wrapper": {}})
    with pytest.raises(ValueError, match="raw stage proposal"):
        trace.record_inference(
            {"subtask_state": np.asarray(0, dtype=np.int32)},
            {"subtask_logits": logits},
        )


def _config(case):
    exact = native_train_admission.EXACT_FILES
    return {
        "schema": "npa.behavior.native-rlc-train-trace-config.v1",
        "status": "native_train_audit_trace_enabled",
        "case": case,
        "panel_id": "a" * 64,
        "policy_identity_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
        "source_commits": {"rlc": "d" * 40},
        "semantic_component": {
            "module": {
                key: exact["semantic_module"][key] for key in ("bytes", "sha256")
            },
            "official_q": {
                key: exact["official_q"][key] for key in ("bytes", "sha256")
            },
            "lossless_arrays": {
                key: exact["lossless_arrays"][key] for key in ("bytes", "sha256")
            },
            "source": {"exact": True},
        },
        "admission_sha256": "e" * 64,
        "development_or_report_used": False,
        "policy_changed": False,
        "training_ready": False,
    }


def _official_row():
    return {
        "schema": "npa.behavior.train-official-q-row.v1",
        "frame_index": 0,
        "initial_satisfied_options": [[False, False, False]],
        "current_satisfied_options": [[True, False, False]],
        "success": False,
        "q_score": 1 / 3,
        "source": deepcopy(OFFICIAL_Q_SOURCE),
    }


def _semantic_row(source, action, observation):
    return {
        "schema": "npa.private.native-train-semantic-trace-row.v1",
        "frame_index": 0,
        "applied_action_sha256": action,
        "allowed_observation_sha256": allowed_observation_sha256(observation),
        "can_predicates": [
            {
                "slot": index,
                "object_scope_name": name,
                "inside_target": index == 0,
                "grasped_by_arm": {"left": False, "right": False},
            }
            for index, name in enumerate(CANS)
        ],
        "goal_option_masks": [[True, False, False]],
        "official_q": _official_row(),
        "undefined_fields": [
            "target_can_slot",
            "near",
            "grasp_attempted",
            "release_attempted",
            "phase",
            "promotion_label",
        ],
        "source": source,
    }


def _rows(source, observation, action):
    digest = array_sha256(action)
    semantic = _semantic_row(source, digest, observation)
    stage = {
        "schema": "npa.private.native-train-stage-trace-row.v1",
        "action_index": 0,
        "inference_ordinal": 0,
        "new_inference": True,
        "stage_before": 1,
        "raw_predicted_stage": 2,
        "stage_after": 1,
        "action_sha256": digest,
        "source": source,
    }
    return semantic, stage


def _finalized_trace(output, case):
    identity = write_config(output / "native-train-trace", _config(case))
    observation = _observation(2)
    action = np.zeros(23, dtype=np.float32)
    semantic, stage = _rows({"exact": True}, observation, action)
    policy_arrays, evaluator_arrays = _array_fragments(
        output, observation, action, stage
    )
    _jsonl(
        output / "native-trace-evaluator/semantic.jsonl",
        semantic,
        "npa.behavior.native-train-semantic-fragment.v1",
        "official_post_apply_rows_recorded",
        evaluator_arrays,
    )
    _jsonl(
        output / "native-trace-policy/native-stage.jsonl",
        stage,
        "npa.private.native-train-stage-fragment.v1",
        "native_actions_and_stage_votes_recorded",
        policy_arrays,
    )
    finalize_trace(output)
    return identity


def _jsonl(path, value, schema, status, arrays):
    path.parent.mkdir(exist_ok=True)
    payload = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    (path.parent / "terminal.json").write_text(
        json.dumps(
            {
                "schema": schema,
                "status": status,
                "action_count": 1,
                "rows": {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
                "arrays": arrays,
            }
        )
    )


def _observation(fill):
    return {
        "robot_r1::proprio": np.full((1, 61), fill, dtype=np.float32),
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.full(
            (1, 720, 720, 3), fill, dtype=np.uint8
        ),
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": np.full(
            (1, 480, 480, 3), fill, dtype=np.uint8
        ),
        "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": np.full(
            (1, 480, 480, 3), fill, dtype=np.uint8
        ),
    }


def _array_fragments(output, observation, action, stage):
    class Policy:
        def infer(self, _observation, **_kwargs):
            return {"actions": np.full((30, 23), 0.25, dtype=np.float64)}

    class Wrapper:
        def __init__(self):
            self.policy = Policy()
            self.last_actions = np.zeros((20, 23), dtype=np.float64)

    wrapper = Wrapper()
    policy = NativePolicyArrays(output / "native-trace-policy/arrays", wrapper)
    wrapper.policy.infer(observation)
    policy.record(observation, action, stage)
    evaluator = NativeEvaluatorArrays(output / "native-trace-evaluator/arrays")
    evaluator.record(observation, action, 0)
    return policy.close(), evaluator.close()


def test_finalize_and_provider_recovery_keep_native_trace_mandatory(tmp_path):
    case = {
        "case_id": "f" * 64,
        "task": "picking_up_trash",
        "instance_id": 200,
        "rollout_id": 0,
        "split": "train",
    }
    output = tmp_path / "case"
    output.mkdir()
    identity = _finalized_trace(output, case)
    assert validate_finalized_trace(output / "native-train-trace")["action_count"] == 1

    storage = MemoryStorage()
    store = CaseStore(storage, "s3://bucket/native", "a" * 64)
    started = store.start(
        store.claim(case, "worker"), native_train_trace_config=identity
    )
    bundle = campaign_runner._native_trace_bundle(store, started, output)
    campaign_runner._record_native_trace_requirement(store, started, bundle)
    campaign_runner._publish_native_trace(store, started, bundle)

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    campaign_runner._restore_native_trace(store, started, fresh)
    assert validate_finalized_trace(fresh / "native-train-trace")["action_count"] == 1
    shard_uri = next(uri for uri in storage.objects if uri.endswith(".npz"))
    shard_object = storage.objects.pop(shard_uri)
    with pytest.raises(Exception):
        campaign_runner._restore_native_trace(store, started, tmp_path / "partial")
    storage.objects[shard_uri] = shard_object
    requirement = (
        store.artifact_prefix(started)
        + "/native-train-trace/publication-requirement.json"
    )
    storage.objects.pop(requirement)
    with pytest.raises(Exception, match="required Native TRAIN trace"):
        campaign_runner._restore_native_trace(store, started, tmp_path / "missing")


def test_finalized_trace_revalidates_official_q_after_consistent_rehash(tmp_path):
    case = {
        "case_id": "f" * 64,
        "task": "picking_up_trash",
        "instance_id": 200,
        "rollout_id": 0,
        "split": "train",
    }
    output = tmp_path / "case"
    output.mkdir()
    _finalized_trace(output, case)
    root = output / "native-train-trace"
    semantic = json.loads((root / "semantic.jsonl").read_text())
    semantic["official_q"]["q_score"] = 1.0
    payload = (
        json.dumps(semantic, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    (root / "semantic.jsonl").write_bytes(payload)
    manifest_path = root / "trace-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["members"]["semantic.jsonl"] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="[Oo]fficial Q"):
        validate_finalized_trace(root)


def test_lossless_arrays_reject_consistently_rehashed_applied_action(tmp_path):
    case = {
        "case_id": "f" * 64,
        "task": "picking_up_trash",
        "instance_id": 200,
        "rollout_id": 0,
        "split": "train",
    }
    output = tmp_path / "case"
    output.mkdir()
    _finalized_trace(output, case)
    root = output / "native-train-trace"
    shard = next((root / "arrays/evaluator").glob("applied-*.npz"))
    with np.load(shard, allow_pickle=False) as value:
        rows = {name: np.array(value[name]) for name in value.files}
    rows["applied_action"][0, 0] = 1.0
    with shard.open("wb") as stream:
        np.savez_compressed(stream, **rows)
    _rehash_terminal_and_manifest(root, "arrays/evaluator", shard.name)
    with pytest.raises(ValueError, match="returned/applied action join"):
        validate_finalized_trace(root)


def test_lossless_arrays_reject_consistently_rehashed_wrapper_chunk(tmp_path):
    case = {
        "case_id": "f" * 64,
        "task": "picking_up_trash",
        "instance_id": 200,
        "rollout_id": 0,
        "split": "train",
    }
    output = tmp_path / "case"
    output.mkdir()
    _finalized_trace(output, case)
    root = output / "native-train-trace"
    shard = next((root / "arrays/policy").glob("decision-*.npz"))
    with np.load(shard, allow_pickle=False) as value:
        rows = {name: np.array(value[name]) for name in value.files}
    rows["raw_wrapper_actions"][0, 0, 0] = 1.0
    with shard.open("wb") as stream:
        np.savez_compressed(stream, **rows)
    _rehash_terminal_and_manifest(root, "arrays/policy", shard.name)
    with pytest.raises(ValueError, match="wrapper/returned action join"):
        validate_finalized_trace(root)


def test_lossless_arrays_reject_consistently_rehashed_final_observation(tmp_path):
    case = {
        "case_id": "f" * 64,
        "task": "picking_up_trash",
        "instance_id": 200,
        "rollout_id": 0,
        "split": "train",
    }
    output = tmp_path / "case"
    output.mkdir()
    _finalized_trace(output, case)
    root = output / "native-train-trace"
    path = root / "arrays/evaluator/final-observation.npz"
    with np.load(path, allow_pickle=False) as value:
        rows = {name: np.array(value[name]) for name in value.files}
    rows["robot_r1::proprio"][0] += 1
    with path.open("wb") as stream:
        np.savez_compressed(stream, **rows)
    _rehash_final_and_manifest(root, path)
    with pytest.raises(ValueError, match="final observation join"):
        validate_finalized_trace(root)


@pytest.mark.parametrize("directory_link", [False, True])
def test_finalized_trace_rejects_undeclared_symlink(tmp_path, directory_link):
    case = {
        "case_id": "f" * 64,
        "task": "picking_up_trash",
        "instance_id": 200,
        "rollout_id": 0,
        "split": "train",
    }
    output = tmp_path / "case"
    output.mkdir()
    _finalized_trace(output, case)
    root = output / "native-train-trace"
    target = tmp_path / ("outside-directory" if directory_link else "outside-file")
    target.mkdir() if directory_link else target.write_text("outside")
    (root / "undeclared").symlink_to(target, target_is_directory=directory_link)
    with pytest.raises(ValueError, match="contains a link"):
        validate_finalized_trace(root)


def test_finalized_trace_rejects_rehashed_terminal_undeclared_shard(tmp_path):
    root = _one_action_root(tmp_path)
    source = next((root / "arrays/policy").glob("decision-*.npz"))
    extra = source.with_name("decision-999999-1000000.npz")
    extra.write_bytes(source.read_bytes())
    manifest_path = root / "trace-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["members"][f"arrays/policy/{extra.name}"] = _file_identity(extra)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="trace inventory differs"):
        validate_finalized_trace(root)


def test_lossless_arrays_reject_oversized_consistently_rehashed_shard(tmp_path):
    root = _one_action_root(tmp_path)
    shard = next((root / "arrays/policy").glob("decision-*.npz"))
    with np.load(shard, allow_pickle=False) as value:
        rows = {name: np.repeat(value[name], 9, axis=0) for name in value.files}
    replacement = shard.with_name("decision-000000-000009.npz")
    with replacement.open("xb") as stream:
        np.savez_compressed(stream, **rows)
    shard.unlink()
    _replace_shard(root, "arrays/policy", shard.name, replacement.name, 9)
    with pytest.raises(ValueError, match="shard row"):
        validate_finalized_trace(root)


def test_lossless_arrays_reject_invalid_first_observation_dtype(tmp_path):
    root = _one_action_root(tmp_path)
    shard = next((root / "arrays/policy").glob("decision-*.npz"))
    with np.load(shard, allow_pickle=False) as value:
        rows = {name: np.array(value[name]) for name in value.files}
    rows["robot_r1::proprio"] = rows["robot_r1::proprio"].astype(np.float64)
    with shard.open("wb") as stream:
        np.savez_compressed(stream, **rows)
    _rehash_terminal_and_manifest(root, "arrays/policy", shard.name)
    with pytest.raises(ValueError, match="proprioception differs"):
        validate_finalized_trace(root)


def test_lossless_arrays_reject_nonfinite_raw_tail(tmp_path):
    root = _one_action_root(tmp_path)
    shard = next((root / "arrays/policy").glob("decision-*.npz"))
    with np.load(shard, allow_pickle=False) as value:
        rows = {name: np.array(value[name]) for name in value.files}
    rows["raw_wrapper_actions"][0, -1, -1] = np.nan
    with shard.open("wb") as stream:
        np.savez_compressed(stream, **rows)
    _rehash_terminal_and_manifest(root, "arrays/policy", shard.name)
    with pytest.raises(ValueError, match="decision tensor differs"):
        validate_finalized_trace(root)


def _one_action_root(tmp_path):
    output = tmp_path / "case"
    output.mkdir()
    _finalized_trace(
        output,
        {
            "case_id": "f" * 64,
            "task": "picking_up_trash",
            "instance_id": 200,
            "rollout_id": 0,
            "split": "train",
        },
    )
    return output / "native-train-trace"


def _replace_shard(root, directory, old_name, new_name, end):
    terminal_path = root / directory / "terminal.json"
    terminal = json.loads(terminal_path.read_bytes())
    rows = next(value for key, value in terminal.items() if key.endswith("_shards"))
    row = next(value for value in rows if value["path"] == old_name)
    path = root / directory / new_name
    row.update(path=new_name, end=end, **_file_identity(path))
    terminal_path.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n")
    manifest_path = root / "trace-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["members"].pop(f"{directory}/{old_name}")
    manifest["members"][f"{directory}/{new_name}"] = _file_identity(path)
    manifest["members"][f"{directory}/terminal.json"] = _file_identity(terminal_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _file_identity(path):
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _rehash_terminal_and_manifest(root, directory, filename):
    path = root / directory / filename
    identity = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    terminal_path = root / directory / "terminal.json"
    terminal = json.loads(terminal_path.read_bytes())
    keys = [name for name in terminal if name.endswith("_shards")]
    row = next(row for key in keys for row in terminal[key] if row["path"] == filename)
    row.update(identity)
    terminal_path.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n")
    manifest_path = root / "trace-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["members"][f"{directory}/{filename}"] = identity
    payload = terminal_path.read_bytes()
    manifest["members"][f"{directory}/terminal.json"] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _rehash_final_and_manifest(root, path):
    identity = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    terminal_path = path.parent / "terminal.json"
    terminal = json.loads(terminal_path.read_bytes())
    terminal["final_observation"].update(identity)
    terminal_path.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n")
    manifest_path = root / "trace-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["members"]["arrays/evaluator/final-observation.npz"] = identity
    payload = terminal_path.read_bytes()
    manifest["members"]["arrays/evaluator/terminal.json"] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def test_policy_arrays_preserve_partial_second_executed_chunk(tmp_path):
    class Policy:
        def infer(self, _observation):
            return {"actions": np.zeros((30, 23), dtype=np.float32)}

    class Wrapper:
        def __init__(self):
            self.policy = Policy()
            self.last_actions = np.zeros((20, 23), dtype=np.float32)

    wrapper = Wrapper()
    recorder = NativePolicyArrays(tmp_path / "arrays", wrapper)
    observation = _observation(0)
    for index in range(21):
        decision = index in {0, 20}
        if decision:
            wrapper.policy.infer(observation)
        recorder.record(
            observation,
            np.zeros(23, dtype=np.float32),
            {
                "action_index": index,
                "new_inference": decision,
                "inference_ordinal": int(index == 20),
            },
        )
    terminal = recorder.close()
    assert terminal["action_count"] == 21
    assert terminal["inference_count"] == 2
    assert terminal["executed_prefix"] == 20


def test_policy_arrays_preserve_infer_kwargs_and_raw_wrapper_dtype(tmp_path):
    class Policy:
        def infer(self, _observation, *, initial_actions):
            assert initial_actions == "rolling"
            return {"actions": np.full((1, 30, 23), 0.25, dtype=np.float64)}

    class Wrapper:
        def __init__(self):
            self.policy = Policy()
            self.last_actions = np.full((20, 23), 0.5, dtype=np.float64)

    wrapper = Wrapper()
    recorder = NativePolicyArrays(tmp_path / "arrays", wrapper)
    wrapper.policy.infer(_observation(0), initial_actions="rolling")
    recorder.record(
        _observation(0),
        np.full(23, 0.5, dtype=np.float32),
        {"action_index": 0, "new_inference": True, "inference_ordinal": 0},
    )
    terminal = recorder.close()
    shard = tmp_path / "arrays" / terminal["decision_shards"][0]["path"]
    with np.load(shard, allow_pickle=False) as value:
        assert value["raw_model_actions"].dtype == np.float64
        assert value["raw_wrapper_actions"].dtype == np.float64


def test_one_action_fragments_finalize_through_production_writers(
    tmp_path, monkeypatch
):
    class Policy:
        def infer(self, _observation):
            return {"actions": np.zeros((30, 23), dtype=np.float32)}

    class Wrapper:
        def __init__(self):
            self.policy = Policy()
            self.last_actions = np.zeros((20, 23), dtype=np.float32)

    observation = _observation(0)
    action = np.zeros(23, dtype=np.float32)
    stage = {"action_index": 0, "new_inference": True, "inference_ordinal": 0}
    wrapper = Wrapper()
    policy_root = tmp_path / "policy"
    policy_arrays = NativePolicyArrays(policy_root / "arrays", wrapper)
    wrapper.policy.infer(observation)
    policy_arrays.record(observation, action, stage)
    monkeypatch.setitem(sys.modules, "rlc_server", SimpleNamespace())
    policy_fragment = importlib.import_module(
        "npa.workflows.behavior_challenge.native_train_policy_server"
    )
    monkeypatch.setattr(
        policy_fragment,
        "_trace_types",
        lambda: SimpleNamespace(file_identity=_file_identity),
    )
    policy_fragment._write_fragment(
        policy_root,
        SimpleNamespace(pending=None, rows=[stage], source={"exact": True}),
        policy_arrays,
    )
    assert (policy_root / "terminal.json").is_file()

    evaluator_root = tmp_path / "evaluator"
    evaluator_arrays = NativeEvaluatorArrays(evaluator_root / "arrays")
    evaluator_arrays.record(observation, action, 0)
    evaluator_fragment = importlib.import_module(
        "npa.workflows.behavior_challenge.native_train_trace_evaluator"
    )
    evaluator_fragment._write_fragment(
        evaluator_root, [{"frame_index": 0}], evaluator_arrays
    )
    assert (evaluator_root / "terminal.json").is_file()


def test_policy_server_rejects_rewritten_staged_array_identity(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "rlc_server", SimpleNamespace())
    server = importlib.import_module(
        "npa.workflows.behavior_challenge.native_train_policy_server"
    )
    root = Path(server.__file__).parent
    source = {
        "behavior_commit": "a",
        "native_wrapper": {},
        "official": {},
        "rlc_commit": "b",
        "runtime_wrapper": {},
    }
    component = {
        "module": _file_identity(root / "native_semantic_trace.py"),
        "official_q": _file_identity(root / "train_official_q.py"),
        "lossless_arrays": _file_identity(root / "native_train_arrays.py"),
        "source": source,
    }
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema": "npa.behavior.native-rlc-train-trace-config.v1",
                "semantic_component": component,
            }
        )
    )
    assert server._source(config) == source
    component["lossless_arrays"]["sha256"] = "f" * 64
    config.write_text(
        json.dumps(
            {
                "schema": "npa.behavior.native-rlc-train-trace-config.v1",
                "semantic_component": component,
            }
        )
    )
    with pytest.raises(ValueError, match="staged trace source"):
        server._source(config)


def test_native_trace_scope_rejects_dev_comet_and_variant():
    panel = {"schema": "npa.behavior.nonreporting-train-panel.v1"}
    base = dict(
        train_experience=False,
        train_experience_depth=False,
        native_train_trace=True,
        policy_kind="rlc",
        policy_execution_variant="native",
        policy_prompt_override=None,
    )
    campaign_runner._validate_train_experience_scope(SimpleNamespace(**base), panel)
    for changed in (
        {"policy_kind": "comet-native"},
        {"policy_execution_variant": "final-stage-backtrack"},
        {"train_experience": True},
    ):
        with pytest.raises(ValueError):
            campaign_runner._validate_train_experience_scope(
                SimpleNamespace(**{**base, **changed}), panel
            )


def test_native_admission_binds_lossless_recorder_source():
    semantic = {
        "archive": {
            "bytes": 12190,
            "sha256": "a29f700446b57d9e36775ac0419fe25b8f83fc55dba598592febba9e13baa650",
        },
        "independent_go": {"bytes": 2886, "sha256": "0" * 64},
        "lossless_arrays": deepcopy(
            native_train_admission.EXACT_FILES["lossless_arrays"]
        ),
        "manifest": {
            "bytes": 1539,
            "sha256": "bb2c1fa8ad87a952a4709fa22ec2f9bdf4de25cf07d8301c637729d5a74547f8",
        },
        "module": {"bytes": 20594, "sha256": "0" * 64},
        "scope": "audit_only",
    }
    semantic["independent_go"]["sha256"] = native_train_admission.EXACT_FILES[
        "semantic_go"
    ]["sha256"]
    semantic["module"]["sha256"] = native_train_admission.EXACT_FILES[
        "semantic_module"
    ]["sha256"]
    native_train_admission._validate_semantic_authority({"semantic_trace": semantic})
    semantic["lossless_arrays"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="semantic authority"):
        native_train_admission._validate_semantic_authority(
            {"semantic_trace": semantic}
        )


def test_staged_lossless_recorder_identity_is_in_durable_config(tmp_path, monkeypatch):
    from npa.workflows.behavior_challenge import native_semantic_trace, rlc_policy

    monkeypatch.setattr(native_semantic_trace, "source_authority", lambda *_: {"x": 1})
    monkeypatch.setattr(native_semantic_trace, "RUNTIME_WRAPPERS", {(14046, "f" * 64)})
    output = tmp_path / "stage"
    output.mkdir()
    component = rlc_policy._stage_native_trace_sources(
        SimpleNamespace(upstream_root=tmp_path, policy_root=tmp_path), output
    )
    expected = native_train_admission.EXACT_FILES["lossless_arrays"]
    assert component["lossless_arrays"] == {
        key: expected[key] for key in ("bytes", "sha256")
    }
    config = _config({"split": "train"})
    config["semantic_component"] = component
    from npa.workflows.behavior_challenge.native_train_trace import validate_config

    assert validate_config(config) == config


def test_workflow_emits_native_trace_admission_and_flag():
    runtime = campaign_workflow._runtime_config(_native_workflow_runtime())
    argv = campaign_workflow._worker_argv(
        _native_worker_slot(),
        "s3://bucket/train/worker.json",
        runtime,
    )
    assert argv.count("--native-train-trace") == 1
    index = argv.index("--native-train-admission")
    assert argv[index + 1] == "{{config.native_train_admission}}"


def _native_workflow_runtime():
    return {
        "upstream_root": "/opt/BEHAVIOR-1K",
        "evaluator_python": "/opt/behavior/bin/python",
        "data_root": "/data/behavior",
        "host": "127.0.0.1",
        "port": 8000,
        "policy_kind": "rlc",
        "policy_root": "/opt/rlc",
        "policy_python": "/opt/rlc/bin/python",
        "policy_checkpoint": "/models/checkpoint_2",
        "policy_archive": "/models/checkpoint_2.zip",
        "policy_execution_variant": "native",
        "native_train_trace": True,
        "native_train_admission": "/inputs/admission.json",
    }


def _native_worker_slot():
    return {
        "worker_index": 0,
        "resource": {"cloud": "kubernetes", "cpus": 16, "memory": "64Gi"},
        "workspace": "/campaign/worker-0",
    }


def _policy_server_fixture(tmp_path):
    source = Path(__file__).parents[2] / "src/npa/workflows/behavior_challenge"
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "native_train_policy_server.py").write_bytes(
        (source / "native_train_policy_server.py").read_bytes()
    )
    (stage / "rlc_server.py").write_text(
        """import argparse,asyncio\nNATIVE_EXECUTION='native'\ndef parser():\n p=argparse.ArgumentParser();p.add_argument('--execution-variant',default='native');p.add_argument('--source-root');p.add_argument('--checkpoint');p.add_argument('--port',type=int);p.add_argument('--upstream-commit');return p\ndef _policy_source(*a): pass\nclass P:\n def reset(self): pass\ndef _load_stock_policy(a): return P(),None\nasync def _serve(*a):\n while True: await asyncio.sleep(1)\n"""
    )
    (stage / "train_official_q.py").write_text("OFFICIAL_Q_SOURCE={}\n")
    (stage / "native_train_arrays.py").write_text(
        """class NativePolicyArrays:
 def __init__(self,root,policy): root.mkdir(parents=True)
 def record(self,*args): pass
 def close(self): return {'fixture':True}
"""
    )
    (stage / "native_semantic_trace.py").write_text(
        """import hashlib\ndef file_identity(p):\n b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}\nclass StageTrace:\n def __init__(self,s): self.source=s;self.pending=None;self.rows=[]\nclass NativeStageObserver:\n def __init__(self,p,t): t.rows.append({'action_index':0})\n"""
    )
    config = stage / "config.json"
    component = {
        "module": _file_identity(stage / "native_semantic_trace.py"),
        "official_q": _file_identity(stage / "train_official_q.py"),
        "lossless_arrays": _file_identity(stage / "native_train_arrays.py"),
        "source": {
            "behavior_commit": "a",
            "native_wrapper": {},
            "official": {},
            "rlc_commit": "b",
            "runtime_wrapper": {},
        },
    }
    config.write_text(
        json.dumps(
            {
                "schema": "npa.behavior.native-rlc-train-trace-config.v1",
                "semantic_component": component,
            }
        )
    )
    return stage, config


def test_policy_server_sigterm_writes_terminal_in_real_subprocess(tmp_path):
    stage, config = _policy_server_fixture(tmp_path)
    root = tmp_path / "fragment"
    process = subprocess.Popen(
        [
            sys.executable,
            str(stage / "native_train_policy_server.py"),
            "--source-root",
            str(tmp_path),
            "--checkpoint",
            str(tmp_path),
            "--port",
            "9876",
            "--native-trace-root",
            str(root),
            "--native-trace-source",
            str(config),
        ],
        cwd=stage,
    )
    time.sleep(0.2)
    _stop_policy(process)
    assert json.loads((root / "terminal.json").read_text())["action_count"] == 1
