"""Serve the pinned RLC model with 2026 proprioception and the official wire protocol."""

import argparse
import asyncio
import http
import logging
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile

from rlc_observations import policy_observation

NATIVE_EXECUTION = "native"


def _policy_source(root: Path, overlay: Path) -> None:
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(
        root / "src/b1k",
        overlay / "b1k",
        ignore=ignored,
    )
    shutil.copytree(root / "openpi/src/openpi", overlay / "openpi", ignore=ignored)
    old = "from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES"
    new = "from rlc_observations import PROPRIOCEPTION_INDICES"
    for name in (
        "b1k/policies/b1k_policy.py",
        "b1k/shared/eval_b1k_wrapper.py",
        "openpi/policies/b1k_policy.py",
    ):
        path = overlay / name
        source = path.read_text()
        if source.count(old) != 1:
            raise ValueError("Unexpected pinned RLC proprioception import")
        path.write_text(source.replace(old, new))
    sys.path[:0] = [
        str(overlay),
        str(root / "openpi/packages/openpi-client/src"),
    ]


def _load_policy(args):
    from b1k.policies.policy_config import create_trained_policy
    from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper, B1KWrapperConfig
    from b1k.training.config import get_config

    policy = create_trained_policy(
        get_config("pi_behavior_b1k_fast"),
        args.checkpoint,
        sample_kwargs={"num_steps": 20},
    )
    return B1KPolicyWrapper(policy, task_id=args.task_id, config=B1KWrapperConfig())


def _health(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def _connection(websocket, policy):
    import numpy as np
    from openpi_client import msgpack_numpy

    packer = msgpack_numpy.Packer()
    policy.reset()
    await websocket.send(packer.pack({"policy": "rlc-2026-transfer"}))
    try:
        async for payload in websocket:
            observation = msgpack_numpy.unpackb(payload)
            if set(observation) == {"reset"} and observation["reset"] is True:
                policy.reset()
                continue
            action = policy.act(policy_observation(observation)).cpu().numpy()
            if action.shape != (23,) or not np.isfinite(action).all():
                raise ValueError("Policy returned an invalid R1Pro action")
            await websocket.send(packer.pack({"action": action}))
    finally:
        policy.reset()


async def _serve(policy, port):
    from websockets.asyncio.server import serve

    lock = asyncio.Lock()

    async def handle(websocket):
        if lock.locked():
            await websocket.close(code=1013, reason="Evaluator already connected")
            return
        async with lock:
            await _connection(websocket, policy)

    async with serve(
        handle,
        "127.0.0.1",
        port,
        compression=None,
        max_size=16 * 1024 * 1024,
        process_request=_health,
    ) as server:
        await server.serve_forever()


def parser() -> argparse.ArgumentParser:
    """Build the stock-RLC server parser.

    Args:
        None.
    Returns:
        Configured argument parser with native execution as its default.
    Raises:
        None.
    """
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-root", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--correlation-asset", type=Path)
    value.add_argument("--correlation-sha256")
    value.add_argument("--specialist-state-contract", action="store_true")
    value.add_argument("--task-id", type=int, choices=range(50), required=True)
    value.add_argument("--port", type=int, required=True)
    value.add_argument(
        "--execution-variant",
        choices=(
            NATIVE_EXECUTION,
            "final-stage-backtrack",
            "adaptive-short-chunk",
            "adaptive-short-chunk-transition-refresh",
            "native-stage-transition-refresh",
        ),
        default=NATIVE_EXECUTION,
    )
    return value


def _verify_specialist_options(args, asset: Path | None) -> bool:
    specialist = getattr(args, "specialist_state_contract", False)
    if not specialist:
        return False
    from rlc_specialist import SUPPORTED_TASK_IDS

    if asset is not None:
        raise ValueError("Specialist state contract rejects correlation overrides")
    if args.execution_variant != NATIVE_EXECUTION:
        raise ValueError("Specialist state contract requires native execution")
    if args.task_id not in SUPPORTED_TASK_IDS:
        raise ValueError("Specialist state contract rejects unsupported task")
    return True


def _verify_specialist_wrapper(policy) -> None:
    from rlc_specialist import verify_loaded_state

    native = getattr(policy, "base_policy", None)
    if native is None or native is not getattr(policy, "policy", None):
        raise ValueError("Specialist native wrapper identity differs")
    verify_loaded_state(native)


def _load_stock_policy(args):
    asset = args.correlation_asset
    expected_sha256 = args.correlation_sha256
    if (asset is None) != (expected_sha256 is None):
        raise ValueError("stock correlation requires both artifact and SHA-256")
    specialist = _verify_specialist_options(args, asset)
    if asset is None:
        policy = _load_policy(args)
        if specialist:
            _verify_specialist_wrapper(policy)
        return policy, None
    from b1k.models.pi_behavior import PiBehavior
    from rlc_correlation import (
        load_fp32_correlation,
        pre_policy_fp32_correlation,
        verify_captured_correlation,
    )

    correlation = load_fp32_correlation(asset, expected_sha256)
    with pre_policy_fp32_correlation(
        PiBehavior, correlation, expected_sha256
    ) as installation:
        policy = _load_policy(args)
    captured = verify_captured_correlation(policy, expected_sha256)
    return policy, {"installation": installation[0], "captured": captured}


def _configure_execution(policy, variant: str):
    if variant == NATIVE_EXECUTION:
        return policy
    from rlc_execution import configure_execution

    return configure_execution(policy, variant)


def _install_termination_handlers(policy):
    original_handlers = {}

    def terminate(signum, frame):
        del frame
        policy.finalize_telemetry()
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        original_handlers[signum] = signal.signal(signum, terminate)
    return original_handlers


def _serve_policy(policy, args) -> None:
    if args.execution_variant == NATIVE_EXECUTION:
        asyncio.run(_serve(policy, args.port))
        return
    original_handlers = _install_termination_handlers(policy)
    try:
        asyncio.run(_serve(policy, args.port))
    finally:
        policy.finalize_telemetry()
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)


def _main():
    args = parser().parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory(prefix="npa-rlc-source-") as temporary:
        _policy_source(args.source_root, Path(temporary))
        policy, _ = _load_stock_policy(args)
        policy = _configure_execution(policy, args.execution_variant)
        _serve_policy(policy, args)


if __name__ == "__main__":
    _main()
