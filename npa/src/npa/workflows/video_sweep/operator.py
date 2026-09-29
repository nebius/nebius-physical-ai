"""Configure, preflight, submit, resume, and export the canonical video sweep."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

from npa.workflows.video_sweep import artifacts, planning

_REQUIRED_SECRETS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "HF_TOKEN",
    "NEBIUS_TOKEN_FACTORY_KEY",
    "NPA_LINEAGE_POSTGRES_DSN",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_ID",
)
_OPTIONAL_SECRETS = ("MLFLOW_TRACKING_TOKEN", "AWS_SESSION_TOKEN")
_FIELDS = {
    "project",
    "infra",
    "bucket",
    "prefix",
    "run_id",
    "accelerators",
    "sources",
    "variants",
    "reasoner_model",
    "merge_model",
    "samples",
    "threshold",
}


def main(argv: list[str] | None = None) -> int:
    """Run the private operator kit without exposing configuration in logs.

    Args:
        argv: Optional argument list.
    Returns:
        Zero on success, one on a failed action.
    Raises:
        SystemExit: Arguments are invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "check", "run", "resume", "export"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        _operate(args)
    except Exception as error:  # noqa: BLE001 - provider errors can contain credentials
        print(
            f"Sweep {args.action} failed ({type(error).__name__}). Inspect the private operator log and configuration.",
            file=sys.stderr,
        )
        return 1
    return 0


def _operate(args):
    if args.action == "init":
        _initialize(args.config)
        print(
            "Private configuration created. Fill in routing and source URIs; supply service secrets through the environment."
        )
        return
    config = _load(args.config)
    print("Loading configured project storage.", flush=True)
    _credentials(config)
    output = args.output_dir or args.config.parent / (config["run_id"] + "-demo")
    if args.action != "check" and output.exists():
        raise FileExistsError("Choose a new demo output directory")
    if args.action != "export":
        with _private_log(args.config) as log:
            print(
                "Checking credentials, exact model access, and the workflow.",
                flush=True,
            )
            _preflight(config, log)
            if args.action == "check":
                print(
                    "Operator preflight passed. Worker-pod service connectivity is verified during execution."
                )
                return
            print(
                "Staging inputs and submitting the canonical workflow. Progress is recorded in the private operator log.",
                flush=True,
            )
            _stage_inputs(config)
            _invoke(_submit_command(config, resume=args.action == "resume"), log)
    from npa.workflows.video_sweep.demo import export_demo

    stage = SimpleNamespace(root_uri=_root(config), run_id=config["run_id"], workers=2)
    print("Verifying published artifacts and rendering the offline demo.", flush=True)
    summary = export_demo(stage, output)
    print(
        f"Demo exported: {len(summary['candidates'])} variants, {summary['accepted']} accepted. Open index.html or demo.mp4 in the output directory."
    )


def _initialize(path):
    config = {
        "project": "<project-alias>",
        "infra": "k8s/<context>",
        "bucket": "<bucket>",
        "prefix": "video-variant-sweep",
        "run_id": "video-sweep-" + uuid.uuid4().hex[:12],
        "accelerators": "H200:1",
        "sources": ["s3://<bucket>/inputs/source.mp4"],
        "reasoner_model": "nvidia/Cosmos3-Super-Reasoner",
        "merge_model": "nvidia/Nemotron-3_5-Lightning",
        "samples": 8,
        "threshold": 0.8,
        "variants": [
            {
                "hint": "Warm evening light",
                "seed": 7,
                "control": "edge",
                "control_weight": 1.0,
                "guidance": 3.0,
            },
            {
                "hint": "Cool indoor light",
                "seed": 11,
                "control": "edge",
                "control_weight": 1.0,
                "guidance": 3.0,
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")


def _load(path):
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or set(config) != _FIELDS:
        raise ValueError("Configuration fields differ from the generated template")
    for key in (
        "project",
        "infra",
        "bucket",
        "reasoner_model",
        "merge_model",
        "accelerators",
    ):
        value = config[key]
        if not isinstance(value, str) or not value or any(c in value for c in "<>\n\r"):
            raise ValueError("Replace configuration placeholders")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9/_-]*", config["prefix"]):
        raise ValueError("Use a relative S3 prefix without dots or whitespace")
    if not config["infra"].startswith("k8s/"):
        raise ValueError("The reference uses a Kubernetes target")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", config["run_id"]):
        raise ValueError("Use a lowercase alphanumeric run ID with hyphens")
    if type(config["samples"]) is not int or config["samples"] < 2:
        raise ValueError("At least two frame samples are required")
    if (
        type(config["threshold"]) not in (float, int)
        or not 0 <= config["threshold"] <= 1
    ):
        raise ValueError("Threshold must be finite and between zero and one")
    planning._validate_inputs(_sources(config), _variants(config))
    for uri in config["sources"]:
        artifacts._object(uri)
    artifacts._object(_root(config) + "/plan.json")
    return config


def _credentials(config):
    from npa.clients.config import resolve_project_storage
    from npa.clients.credentials import load_credentials, shared_credential_env

    for key, value in shared_credential_env(load_credentials()).items():
        os.environ.setdefault(key, value)
    storage = resolve_project_storage(
        config["project"], include_shared_credentials=False, include_environment=False
    )
    for key, value in (
        ("AWS_ACCESS_KEY_ID", storage.aws_access_key_id),
        ("AWS_SECRET_ACCESS_KEY", storage.aws_secret_access_key),
        ("AWS_ENDPOINT_URL", storage.endpoint_url),
    ):
        if not value:
            raise ValueError("Configure exact-project object storage first")
        os.environ[key] = value


def _private_log(path):
    target = path.with_suffix(".operator.log")
    descriptor = os.open(
        target, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600
    )
    os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, "a")


def _npa(*args):
    return [sys.executable, "-m", "npa", *args]


def _spec():
    from npa.orchestration.npa_workflow import load_spec
    from npa.orchestration.npa_workflow.blueprints import resolve_npa_workflow_spec

    path = resolve_npa_workflow_spec("video-variant-sweep.yaml")
    if path is None:
        raise FileNotFoundError("The canonical workflow catalog is missing")
    load_spec(path)
    return str(path)


def _invoke(command, log):
    subprocess.run(command, check=True, stdout=log, stderr=log)


def _preflight(config, log):
    from npa.clients.token_factory import TokenFactoryClient

    missing = [name for name in _REQUIRED_SECRETS if not os.environ.get(name)]
    if missing:
        raise ValueError("Supply all required service secrets")
    if not {config["reasoner_model"], config["merge_model"]} <= set(
        TokenFactoryClient().list_models()
    ):
        raise ValueError("Select explicitly available hosted models")
    for command in (
        _npa(
            "workbench",
            "health",
            "preflight",
            "--project",
            config["project"],
            "--checks",
            "s3,token_factory,hf",
        ),
        _npa("workbench", "health", "access", "--capability", "cosmos2", "--json"),
        _npa("workbench", "workflow", "validate-spec", _spec(), "--json"),
        _npa(
            "workbench",
            "workflow",
            "plan-spec",
            _spec(),
            "--run-id",
            "preview",
            "--waves",
            "--json",
        ),
        _npa("workbench", "workflow", "preflight-images", _spec(), "--json"),
    ):
        _invoke(command, log)


def _root(config):
    return f"s3://{config['bucket']}/{config['prefix'].rstrip('/')}/{config['run_id']}"


def _sources(config):
    return {"schema": "npa.video_sweep.sources.v1", "clips": config["sources"]}


def _variants(config):
    return {"schema": "npa.video_sweep.variants.v1", "variants": config["variants"]}


def _stage_inputs(config):
    artifacts.write_json(_root(config) + "/inputs/sources.json", _sources(config))
    artifacts.write_json(_root(config) + "/inputs/variants.json", _variants(config))


def _submit_command(config, *, resume=False):
    command = _npa(
        "workbench",
        "workflow",
        "submit",
        _spec(),
        "--project",
        config["project"],
        "--infra",
        config["infra"],
        "--resume-run" if resume else "--run-id",
        config["run_id"],
        "--runtime",
        "--max-wait-seconds",
        "0",
        "--accelerators",
        config["accelerators"],
    )
    variables = {
        key: config[key]
        for key in ("bucket", "reasoner_model", "merge_model", "samples", "threshold")
    }
    variables.update(
        prefix=config["prefix"].rstrip("/") + "/" + config["run_id"],
        sources_uri=_root(config) + "/inputs/sources.json",
        variants_uri=_root(config) + "/inputs/variants.json",
    )
    for key, value in variables.items():
        command.extend(("--var", f"{key}={value}"))
    for name in (*_REQUIRED_SECRETS, *_OPTIONAL_SECRETS):
        if os.environ.get(name):
            command.extend(("--secret-env", name))
    return command


if __name__ == "__main__":
    raise SystemExit(main())
