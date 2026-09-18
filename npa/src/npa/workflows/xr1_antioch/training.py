"""Run an isolated, pinned XR1 fine-tune from verified Nebius S3 robot artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .storage import _Store, _relative


def _runtime(uri: str, root: Path) -> Path:
    from .runtime import _cuda_proof, _install

    store = _Store(uri)
    manifest = store.read_json("manifest.json")
    for filename, identity in manifest.items():
        store.download(filename, root / _relative(filename), identity["sha256"])
    proof = json.loads((root / "runtime-proof.json").read_text())
    if not proof["flash_attention_forward_backward"] or not proof["fused_adam_update"]:
        raise ValueError("The requested GPU runtime has not passed native CUDA checks")
    _install(root, root / "wheels")
    print(json.dumps(_cuda_proof()), flush=True)
    return root / "source/xr1"


def _prepare(args) -> Path:
    from mmengine import Config
    from .training_data import _assets, _configuration, _dataset, _normalization

    root = args.work_path
    root.mkdir(parents=True, exist_ok=True)
    source = root / "runtime/source/xr1"
    _assets(args.assets_path, root / "assets")
    report = _dataset(args.input_path, root)
    statistics = _normalization(source, root)
    configuration = _configuration(source, root, statistics)
    output = root / "output"
    output.mkdir()
    configuration["trainer"]["default_root_dir"] = str(output / "checkpoints")
    Config(configuration).dump(str(output / "config.py"))
    shutil.copyfile(root / "normalization.json", output / "normalization.json")
    shutil.copyfile(root / "dataset-report.json", output / "dataset-report.json")
    print("XR1_DATASET " + json.dumps({split: sum(row["success"] for row in report[split])
                                      for split in ("train", "validation")}), flush=True)
    return output


def _train(args) -> None:
    output = _prepare(args)
    _Store(args.output_path.rstrip("/") + "/preparation").publish(output)
    source = args.work_path / "runtime/source/xr1"
    env = dict(os.environ, WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false")
    result = subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
        "-m", "npa.workflows.xr1_antioch.training", "fit", "--configuration", str(output / "config.py"),
    ], cwd=source, env=env)
    if result.returncode:
        raise RuntimeError(f"Native XR1 training failed with exit {result.returncode}; no checkpoint promotion")
    if not (output / "candidate/training.json").is_file():
        raise ValueError("The training process did not export a validated candidate checkpoint")
    _Store(args.output_path).publish(output)
    print("XR1_TRAINING_PUBLISHED_AND_READBACK_VERIFIED", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the workflow adapter's parser without importing the GPU training stack.

    Args:
        None.
    Returns:
        Parser for isolated preparation, training, and distributed rank entrypoints.
    Raises:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "bootstrap", "train"):
        command = commands.add_parser(name)
        for option in ("input-path", "assets-path", "runtime-path", "output-path"):
            command.add_argument(f"--{option}", required=True)
        command.add_argument("--work-path", required=True, type=Path)
    fit = commands.add_parser("fit")
    fit.add_argument("--configuration", required=True, type=Path)
    return parser


def _isolate(args) -> None:
    args.work_path.mkdir(parents=True, exist_ok=False, mode=0o700)
    environment = args.work_path / "vendor-venv"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(environment)], check=True)
    interpreter = environment / "bin/python"
    subprocess.run([str(interpreter), "-m", "npa.workflows.xr1_antioch.training",
                    "bootstrap", *sys.argv[2:]], check=True)


def _main() -> None:
    args = build_parser().parse_args()
    if args.command == "fit":
        from .learning import _fit

        _fit(args.configuration)
    elif args.command == "train":
        _train(args)
    elif args.command == "run":
        _isolate(args)
    else:
        _runtime(args.runtime_path, args.work_path / "runtime")
        # Start fresh after pip installation so imported NumPy/Torch versions
        # cannot survive an upstream dependency replacement in this process.
        subprocess.run([sys.executable, "-m", "npa.workflows.xr1_antioch.training",
                        "train", *sys.argv[2:]], check=True)


if __name__ == "__main__":
    _main()
