"""Run the official TRAIN evaluator with audit-only Native semantic hooks."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

_ROOT_ENV = "NPA_NATIVE_TRAIN_TRACE_ROOT"


def _trace_modules():
    """Load the exact staged trace and Q modules in the simulator interpreter."""
    try:
        import native_semantic_trace
        import train_official_q
    except ModuleNotFoundError as error:
        if error.name != "npa":
            raise
        import train_official_q

        npa = ModuleType("npa")
        workflows = ModuleType("npa.workflows")
        behavior = ModuleType("npa.workflows.behavior_challenge")
        npa.__path__ = workflows.__path__ = behavior.__path__ = []
        behavior.train_official_q = train_official_q
        sys.modules.update(
            {
                "npa": npa,
                "npa.workflows": workflows,
                "npa.workflows.behavior_challenge": behavior,
                "npa.workflows.behavior_challenge.train_official_q": train_official_q,
            }
        )
        import native_semantic_trace
    import native_train_arrays

    return native_semantic_trace, train_official_q, native_train_arrays


def _config() -> dict[str, Any]:
    root = Path(os.environ[_ROOT_ENV])
    value = json.loads((root / "config.json").read_bytes())
    if value.get("schema") != "npa.behavior.native-rlc-train-trace-config.v1":
        raise ValueError("Native TRAIN trace config differs")
    return value


def install() -> None:
    """Install post-apply observation hooks on the exact official evaluator.

    Args:
        None.
    Returns:
        None.
    Raises:
        ImportError: The exact evaluator or trace modules are unavailable.
    """
    from omnigibson.eval.evaluator import BatchedEvaluator

    trace, official_q, arrays = _trace_modules()
    BatchedEvaluator.__init__ = _initialize_hook(
        BatchedEvaluator.__init__, trace, official_q, arrays
    )
    BatchedEvaluator._apply_actions = _apply_hook(BatchedEvaluator._apply_actions)
    BatchedEvaluator.run = _run_hook(BatchedEvaluator.run)


def _initialize_hook(original, trace, official_q, arrays):
    def initialize(evaluator, config):
        value = _config()
        task = _get(config, "task")
        model = _get(config, "model")
        observed = (
            _get(config, "mode"),
            int(_get(config, "num_envs")),
            _get(task, "name"),
            bool(_get(config, "write_video")),
            _get(config, "policy_name"),
            int(_get(model, "action_chunk_size")),
        )
        expected = ("train", 1, value["case"]["task"], True, "websocket", 0)
        if observed != expected:
            raise ValueError("Native TRAIN evaluator scope differs")
        original(evaluator, config)
        q_observer = official_q.bind_official_q_observer(evaluator)
        source = value["semantic_component"]["source"]
        evaluator._npa_native_semantic = trace.EvaluatorSemanticObserver(
            evaluator, q_observer, source
        )
        evaluator._npa_native_q = q_observer
        evaluator._npa_native_rows = []
        root = Path(os.environ[_ROOT_ENV]).parent / "native-trace-evaluator"
        evaluator._npa_native_arrays = arrays.NativeEvaluatorArrays(root / "arrays")

    return initialize


def _get(value: Any, name: str) -> Any:
    return value.get(name) if hasattr(value, "get") else getattr(value, name)


def _apply_hook(original):
    def apply(evaluator, actions, active_indices):
        if list(active_indices) != [0]:
            raise ValueError("Native TRAIN trace requires environment zero")
        result = original(evaluator, actions, active_indices)
        rows = evaluator._npa_native_rows
        observation = evaluator._batch_obs()
        frame = len(rows)
        rows.append(
            evaluator._npa_native_semantic.observe(frame, actions[0], observation)
        )
        evaluator._npa_native_arrays.record(observation, actions[0], frame)
        return result

    return apply


def _write_fragment(root: Path, rows: list[dict[str, Any]], arrays: Any) -> None:
    if root.is_symlink() or not rows:
        raise ValueError("Native evaluator trace ended incomplete")
    if not root.is_dir() or {path.name for path in root.iterdir()} != {"arrays"}:
        raise ValueError("Native evaluator trace inventory differs")
    payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    rows_path = root / "semantic.jsonl"
    with rows_path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    terminal = {
        "schema": "npa.behavior.native-train-semantic-fragment.v1",
        "status": "official_post_apply_rows_recorded",
        "action_count": len(rows),
        "rows": {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "arrays": arrays.close(),
    }
    payload = (json.dumps(terminal, sort_keys=True, indent=2) + "\n").encode()
    with (root / "terminal.json").open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _run_hook(original):
    def run(evaluator, instances, **kwargs):
        expected = _config()["case"]
        if (
            list(instances) != [expected["instance_id"]]
            or kwargs.get("rollout_id") != 0
        ):
            raise ValueError("Native TRAIN trace case differs")
        result = original(evaluator, instances, **kwargs)
        _, official_q, _ = _trace_modules()
        official_q.validate_official_q_terminal(
            [row["official_q"] for row in evaluator._npa_native_rows],
            result,
            expected["instance_id"],
        )
        _write_fragment(
            Path(os.environ[_ROOT_ENV]).parent / "native-trace-evaluator",
            evaluator._npa_native_rows,
            evaluator._npa_native_arrays,
        )
        return result

    return run


def main() -> None:
    """Install hooks and invoke the unchanged official evaluator CLI.

    Args:
        None.
    Returns:
        None.
    Raises:
        SystemExit: The official evaluator CLI exits.
    """
    install()
    from omnigibson.eval.eval import main as official_main

    official_main()


if __name__ == "__main__":
    main()
