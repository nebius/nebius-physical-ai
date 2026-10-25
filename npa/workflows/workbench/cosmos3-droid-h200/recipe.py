"""Plan or launch the single-node native DROID H200 execution check."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

_ROOT = Path(__file__).resolve().parent
_FRAMEWORK = "cf5d68c00d97ccd2480a2320ed652b92dec63102"


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["plan", "dryrun", "run"])
    for name in [
        "framework",
        "dataset-root",
        "manifest",
        "base-dcp",
        "vae",
        "tokenizer",
        "avae-cache",
        "output-dir",
        "filter",
    ]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--endpoint-url", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--aws-profile")
    parser.add_argument("--updates", required=True, type=int)
    return parser


def _settings(args):
    if args.updates < 1:
        raise ValueError("updates must be a positive integer")
    paths = [
        "framework",
        "dataset_root",
        "manifest",
        "base_dcp",
        "vae",
        "tokenizer",
        "avae_cache",
    ]
    settings = {name: str(getattr(args, name).resolve(strict=True)) for name in paths}
    settings.update(
        bucket=args.bucket,
        endpoint_url=args.endpoint_url,
        region=args.region,
        aws_profile=args.aws_profile,
        updates=args.updates,
    )
    if not (Path(settings["base_dcp"]) / "model/.metadata").is_file():
        raise ValueError("base DCP has no model metadata")
    if not (Path(settings["dataset_root"]) / "success/meta/info.json").is_file():
        raise ValueError("dataset root must contain success/meta/info.json")
    _verify_prepared_layout(settings)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=settings["framework"], text=True
    ).strip()
    if revision != _FRAMEWORK:
        raise ValueError("framework revision differs from the measured source")
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD"], cwd=settings["framework"], check=True
    )
    import native_video

    native_video.install(settings)
    native_video.uninstall()
    return settings


def _verify_prepared_layout(settings):
    required = [
        Path(settings["vae"]),
        Path(settings["tokenizer"]) / "tokenizer.json",
        Path(settings["avae_cache"]) / "config.json",
        Path(settings["avae_cache"]) / "diffusion_pytorch_model.safetensors",
        Path(settings["avae_cache"]) / "avae_48k_noncausal_25hz_64ch.ckpt",
        Path(settings["avae_cache"]) / "avae_48k_noncausal_25hz_64ch.json",
    ]
    if not all(path.is_file() for path in required):
        raise ValueError(
            "prepared tokenizer, VAE or materialized AVAE cache is incomplete"
        )


def _toml(settings):
    original = (
        Path(settings["framework"])
        / "examples/toml/sft_config/action_policy_droid_nano.toml"
    )
    text = original.read_text()
    replacements = {
        'wandb_mode   = "online"': 'wandb_mode   = "disabled"',
        "data_parallel_replicate_degree = 32": "data_parallel_replicate_degree = 1",
        "max_iter        = 10000": "max_iter        = " + str(settings["updates"]),
        "logging_iter    = 50": "logging_iter    = 1",
        "grad_accum_iter = 1": "grad_accum_iter = 32",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError("upstream DROID TOML differs from the measured recipe")
        text = text.replace(old, new)
    return text


def _environment(settings, output):
    environment = dict(os.environ)
    environment.update(
        NPA_DROID_SETTINGS=str(output / "settings.json"),
        DROID_ROOT=settings["dataset_root"],
        DATASET_PATH=settings["dataset_root"],
        BASE_CHECKPOINT_PATH=settings["base_dcp"],
        WAN_VAE_PATH=settings["vae"],
        IMAGINAIRE_OUTPUT_ROOT=str(output / "training"),
        TORCH_COMPILE_DISABLE="1",
        COSMOS_TRAINING="1",
        WANDB_MODE="disabled",
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join([str(_ROOT), settings["framework"]]),
    )
    return environment


def _options(settings, output):
    return [
        "--sft-toml",
        str(output / "train.toml"),
        "--",
        "trainer.seed=42",
        "dataloader_train.dataloader.num_workers=4",
        "dataloader_train.dataloader.prefetch_factor=2",
        "dataloader_train.dataloader.batch_size=16",
        "dataloader_train.dataloader.datasets.droid.dataset.use_filter_dict=True",
        "dataloader_train.dataloader.datasets.droid.dataset.filter_dict_path="
        + json.dumps(settings["filter"]),
        "model.config.vlm_config.tokenizer.pretrained_model_name="
        + json.dumps(settings["tokenizer"]),
        "upload_reproducible_setup=False",
        "trainer.callbacks.compile_tokenizer.enabled=False",
        "model.config.compile.enabled=False",
        "trainer.run_validation=False",
        "trainer.run_validation_on_start=False",
    ]


def _argv(settings, output, operation):
    python = str(Path(settings["framework"]) / ".venv/bin/python")
    if operation == "run":
        prefix = [
            python,
            "-B",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            "--max_restarts=0",
            str(_ROOT / "entry.py"),
        ]
    else:
        prefix = [python, "-B", str(_ROOT / "entry.py"), "--dryrun"]
    return prefix + _options(settings, output)


def _execute(args):
    os.umask(0o077)
    settings = _settings(args)
    settings["filter"] = str(args.filter.resolve(strict=True))
    text = _toml(settings)
    output = args.output_dir.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (output / "train.toml").write_text(text)
    command = _argv(settings, output, args.operation)
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    if args.operation != "plan":
        subprocess.run(
            command,
            cwd=settings["framework"],
            env=_environment(settings, output),
            check=True,
        )
    if args.operation == "run":
        _verify_completed_updates(settings, output)
    return {
        "operation": args.operation,
        "global_batch": 8192,
        "output_dir": str(output),
    }


def _verify_completed_updates(settings, output):
    paths = sorted((output / "verification").glob("rank-*.json"))
    ranks = [json.loads(path.read_text()) for path in paths]
    if sorted(row["rank"] for row in ranks) != list(range(8)):
        raise ValueError("missing native verification ranks")
    updates = settings["updates"]
    for row in ranks:
        if (
            row["terminal"] is not True
            or row["finite_losses"] is not True
            or row["microsteps"] != 32 * updates
            or row["samples"] != 1024 * updates
            or row["updates"] != updates
            or row["optimizer_operations"] != updates
        ):
            raise ValueError(
                "native execution did not realize the requested global updates"
            )


if __name__ == "__main__":
    parser = _parser()
    arguments = parser.parse_args()
    try:
        print(json.dumps(_execute(arguments), indent=2))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        print(
            "DROID verification failed; inspect the operator-owned inputs and logs.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
