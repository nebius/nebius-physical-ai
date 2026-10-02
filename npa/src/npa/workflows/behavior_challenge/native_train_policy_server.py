"""Serve unchanged Native RLC while recording allowed tensors and audit labels."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import ModuleType
from typing import Any

import rlc_server


def _trace_types():
    """Load the exact staged trace module in the isolated policy interpreter."""
    try:
        import native_semantic_trace
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
    return native_semantic_trace


def _array_types():
    """Load the staged lossless array recorder in the policy interpreter."""
    import native_train_arrays

    return native_train_arrays


class _ObservedPolicy:
    """Preserve native reset and actions while routing actions through the observer."""

    def __init__(self, delegate: Any, observer: Any, arrays: Any) -> None:
        self.delegate = delegate
        self.observer = observer
        self.arrays = arrays

    def reset(self) -> None:
        """Reset the unchanged native policy."""
        self.delegate.reset()

    def act(self, observation: Any) -> Any:
        """Return the unchanged native action after audit-only observation."""
        action = self.observer.act(observation)
        self.arrays.record(observation, action, self.observer.trace.rows[-1])
        return action


def _write_fragment(root: Path, trace: Any, arrays: Any) -> dict[str, Any]:
    """Write terminal-last policy rows without asserting evaluator completion."""
    if root.is_symlink() or trace.pending is not None or not trace.rows:
        raise ValueError("native policy trace ended incomplete")
    if not root.is_dir() or {path.name for path in root.iterdir()} != {"arrays"}:
        raise ValueError("native policy trace inventory differs")
    payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in trace.rows
    )
    rows = root / "native-stage.jsonl"
    with rows.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    terminal = {
        "schema": "npa.private.native-train-stage-fragment.v1",
        "status": "native_actions_and_stage_votes_recorded",
        "action_count": len(trace.rows),
        "rows": _trace_types().file_identity(rows),
        "source": trace.source,
        "policy_changed": False,
        "arrays": arrays.close(),
    }
    raw = json.dumps(terminal, indent=2, sort_keys=True).encode() + b"\n"
    with (root / "terminal.json").open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return terminal


def _source(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if value.get("schema") == "npa.behavior.native-rlc-train-trace-config.v1":
        component = value.get("semantic_component")
        _validate_staged_component(component)
        value = component.get("source")
    required = {
        "behavior_commit",
        "native_wrapper",
        "official",
        "rlc_commit",
        "runtime_wrapper",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("native semantic source authority differs")
    return value


def _validate_staged_component(value: Any) -> None:
    expected = {
        "module": "native_semantic_trace.py",
        "official_q": "train_official_q.py",
        "lossless_arrays": "native_train_arrays.py",
    }
    if not isinstance(value, dict) or set(value) != {*expected, "source"}:
        raise ValueError("native semantic component differs")
    for field, filename in expected.items():
        if value[field] != _file_identity(Path(__file__).with_name(filename)):
            raise ValueError("native staged trace source differs")


def _file_identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _termination_handler() -> dict[signal.Signals, Any]:
    original = {}

    def terminate(signum, frame):
        del frame
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        original[signum] = signal.signal(signum, terminate)
    return original


def parser() -> argparse.ArgumentParser:
    """Extend the exact stock server parser with trace-only paths.

    Args:
        None.
    Returns:
        Native RLC server parser with required trace destinations.
    Raises:
        None.
    """
    value = rlc_server.parser()
    value.description = __doc__
    value.add_argument("--native-trace-root", type=Path, required=True)
    value.add_argument("--native-trace-source", type=Path, required=True)
    return value


def serve(args: argparse.Namespace) -> dict[str, Any]:
    """Load the exact stock policy, record rows, and finalize on shutdown.

    Args:
        args: Validated stock policy and trace arguments.
    Returns:
        Final policy-fragment terminal.
    Raises:
        ValueError: Execution or trace identities differ.
        BaseException: The original server failure after durable finalization.
    """
    if args.execution_variant != rlc_server.NATIVE_EXECUTION:
        raise ValueError("native TRAIN trace requires unchanged native execution")
    source = _source(args.native_trace_source)
    trace_module = _trace_types()
    trace = trace_module.StageTrace(source)
    handlers = _termination_handler()
    try:
        arrays, failure = _run_observed_policy(args, trace_module, trace)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    if arrays is None:
        raise ValueError("native policy arrays were not initialized")
    terminal = _write_fragment(args.native_trace_root, trace, arrays)
    if failure is not None:
        raise failure
    return terminal


def _run_observed_policy(args, trace_module, trace):
    arrays = None
    failure = None
    try:
        with tempfile.TemporaryDirectory(prefix="npa-rlc-source-") as temporary:
            rlc_server._policy_source(args.source_root, Path(temporary))
            policy, _ = rlc_server._load_stock_policy(args)
            arrays = _array_types().NativePolicyArrays(
                args.native_trace_root / "arrays", policy
            )
            observer = trace_module.NativeStageObserver(policy, trace)
            asyncio.run(
                rlc_server._serve(
                    _ObservedPolicy(policy, observer, arrays),
                    args.port,
                    args.upstream_commit,
                )
            )
    except BaseException as error:
        failure = error
    return arrays, failure


def main() -> None:
    """Parse the exact stock/native trace arguments and serve.

    Args:
        None.
    Returns:
        None.
    Raises:
        SystemExit: Argument parsing or managed shutdown exits the process.
    """
    args = parser().parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    serve(args)


if __name__ == "__main__":
    main()
