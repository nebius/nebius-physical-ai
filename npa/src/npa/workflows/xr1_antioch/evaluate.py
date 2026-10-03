"""Run every sealed test seed against one exact XR1 checkpoint in an Antioch session."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .assets import prepare_processor_cache
from .storage import _sha256


def _server(args, socket_path: Path, log):
    prepare_processor_cache(args.assets_path)
    environment = dict(
        os.environ,
        HF_HOME=str(args.assets_path / "hf-cache"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        WANDB_MODE="disabled",
    )
    process = subprocess.Popen(
        [
            str(args.policy_python),
            "-u",
            "-m",
            "npa.workflows.xr1_antioch.policy_server",
            "--checkpoint",
            str(args.checkpoint),
            "--sha256",
            args.checkpoint_sha256,
            "--statistics",
            str(args.statistics),
            "--processor",
            str(args.assets_path / "processor"),
            "--socket",
            str(socket_path),
        ],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    while not socket_path.exists():
        if process.poll() is not None:
            raise RuntimeError(
                f"XR1 policy failed before readiness with exit {process.returncode}"
            )
        time.sleep(1)
    return process


def _episodes(args, socket_path: Path) -> dict:
    manifest = json.loads(args.split_manifest.read_text())
    seeds = [row["seed"] for row in manifest["test"]]
    train_seeds = {
        row["seed"] for split in ("train", "validation") for row in manifest[split]
    }
    if not seeds or len(set(seeds)) != len(seeds) or train_seeds.intersection(seeds):
        raise ValueError("Closed-loop evaluation requires unique, untouched test seeds")
    results = []
    simulation = hashlib.sha256()
    for name in ("scene.py", "cameras.py", "collection.py", "rollout.py"):
        simulation.update(name.encode())
        simulation.update(Path(__file__).with_name(name).read_bytes())
    contract = {
        "statistics_sha256": _sha256(args.statistics),
        "simulation_sha256": simulation.hexdigest(),
    }
    for seed in seeds:
        output = args.output_path / f"test-{seed}"
        with (args.output_path / f"test-{seed}.log").open("w") as log:
            subprocess.run(
                [
                    "python",
                    "-u",
                    "-m",
                    "npa.workflows.xr1_antioch.rollout",
                    "--output-path",
                    str(output),
                    "--policy-socket",
                    str(socket_path),
                    "--checkpoint-sha256",
                    args.checkpoint_sha256,
                    "--seed",
                    str(seed),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        result = json.loads((output / "rollout.json").read_text())
        result.update(contract)
        (output / "rollout.json").write_text(json.dumps(result, allow_nan=False))
        results.append({key: value for key, value in result.items() if key != "trace"})
        print(json.dumps(results[-1]), flush=True)
    report = {
        "checkpoint_sha256": args.checkpoint_sha256,
        "rollouts": results,
        "complete": len(results) == len(seeds),
    }
    (args.output_path / "evaluation.json").write_text(json.dumps(report, indent=2))
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "output-path",
        "policy-python",
        "assets-path",
        "checkpoint",
        "statistics",
        "split-manifest",
    ):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    args = parser.parse_args()
    args.output_path.mkdir(parents=True, exist_ok=False)
    socket_path = args.output_path / "policy.sock"
    with (args.output_path / "policy-server.log").open("w") as log:
        process = _server(args, socket_path, log)
        try:
            _episodes(args, socket_path)
        finally:
            process.terminate()
            process.wait()
            socket_path.unlink(missing_ok=True)


if __name__ == "__main__":
    _main()
