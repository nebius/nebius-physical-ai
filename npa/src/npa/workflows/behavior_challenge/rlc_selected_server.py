"""Serve a validated selected RLC export through the existing loopback adapter."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
from pathlib import Path
import signal
import sys
import tempfile

from rlc_selected import load_selected_policy, load_validated_correlation

NATIVE_EXECUTION = "native"
TRANSITION_REFRESH = "transition-refresh"


def parser() -> argparse.ArgumentParser:
    """Build the selected-RLC server argument parser.

    Returns:
        Configured argument parser.
    """

    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-root", type=Path, required=True)
    value.add_argument("--adapter-root", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--selected-export-receipt", type=Path, required=True)
    value.add_argument("--correlation-manifest", type=Path, required=True)
    value.add_argument("--validation-receipt", type=Path, required=True)
    value.add_argument("--task-id", type=int, choices=range(50), required=True)
    value.add_argument("--port", type=int, required=True)
    value.add_argument(
        "--execution-variant",
        choices=(NATIVE_EXECUTION, TRANSITION_REFRESH),
        default=NATIVE_EXECUTION,
    )
    return value


def _configure_execution(policy, variant: str):
    if variant == NATIVE_EXECUTION:
        return policy
    from rlc_transition import configure_selected_execution

    return configure_selected_execution(policy, variant)


def _install_termination_handlers(policy):
    original_handlers = {}

    def terminate(signum, frame):
        del frame
        policy.finalize_telemetry()
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        original_handlers[signum] = signal.signal(signum, terminate)
    return original_handlers


def _serve_policy(adapter, policy, args) -> None:
    if args.execution_variant == NATIVE_EXECUTION:
        asyncio.run(adapter._serve(policy, args.port))
        return
    original_handlers = _install_termination_handlers(policy)
    try:
        asyncio.run(adapter._serve(policy, args.port))
    finally:
        policy.finalize_telemetry()
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)


def main() -> None:
    """Load validated selected state and run the existing RLC server.

    Raises:
        OSError: Required sources or artifacts cannot be read.
        ValueError: Artifact, source, or selected-checkpoint bindings differ.
    """

    args = parser().parse_args()
    correlation, manifest = load_validated_correlation(
        args.correlation_manifest,
        args.validation_receipt,
        args.selected_export_receipt,
        args.adapter_root,
    )
    sys.path.insert(0, str(args.adapter_root))
    adapter = importlib.import_module("rlc_server")
    if (
        Path(adapter.__file__).resolve()
        != (args.adapter_root / "rlc_server.py").resolve()
    ):
        raise ValueError("resolved an unexpected RLC serving adapter")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory(prefix="npa-rlc-selected-") as temporary:
        adapter._policy_source(args.source_root, Path(temporary))
        policy = load_selected_policy(adapter, args, correlation, manifest)
        policy = _configure_execution(policy, args.execution_variant)
        _serve_policy(adapter, policy, args)


if __name__ == "__main__":
    main()
