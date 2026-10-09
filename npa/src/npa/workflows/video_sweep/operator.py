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

from npa.workflows.video_sweep import artifacts, matrix, planning

_REQUIRED_SECRETS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "HF_TOKEN",
    "NEBIUS_TOKEN_FACTORY_KEY",
    "NPA_LINEAGE_POSTGRES_DSN",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_ID",
)
_OPTIONAL_SECRETS = (
    "MLFLOW_TRACKING_TOKEN",
    "MLFLOW_TRACKING_CA_PEM",
    "AWS_SESSION_TOKEN",
)
_FIELDS = {
    "project",
    "infra",
    "bucket",
    "prefix",
    "run_id",
    "accelerators",
    "sources",
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
    parser.add_argument(
        "action", choices=("init", "plan", "check", "run", "resume", "export")
    )
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
    config = _load(args.config, offline=args.action == "plan")
    if args.action == "plan":
        _preview(args, config)
        return
    print("Loading configured project storage.", flush=True)
    _credentials(config)
    output = args.output_dir or args.config.parent / (config["run_id"] + "-demo")
    if args.action != "check" and output.exists():
        raise FileExistsError("Choose a new demo output directory")
    try:
        if args.action != "export" and not _run_workflow(args, config):
            return
    except subprocess.CalledProcessError:
        if args.action in {"run", "resume"}:
            _export_rejection(config, output)
        raise
    from npa.workflows.video_sweep.demo import export_demo

    stage = SimpleNamespace(root_uri=_root(config), run_id=config["run_id"], workers=2)
    print("Verifying run artifacts and rendering the offline demo.", flush=True)
    summary = export_demo(stage, output)
    print(
        f"Demo exported: {len(summary['candidates'])} variants, {summary['accepted']} accepted. Open index.html or demo.mp4 in the output directory."
    )


def _export_rejection(config, output):
    from npa.workflows.video_sweep.demo import export_demo

    root = _root(config)
    if not all(
        artifacts.exists(root + "/" + name) for name in ("review.json", "lineage.json")
    ):
        return
    report = artifacts.read_json(root + "/review.json")
    if not report.get("items") or any(row["accepted"] for row in report["items"]):
        return
    stage = SimpleNamespace(root_uri=root, run_id=config["run_id"], workers=2)
    export_demo(stage, output)
    print(
        "All variants held out. Review demo exported; no dataset published. The workflow failure is retained.",
        flush=True,
    )


def _run_workflow(args, config):
    with _private_log(args.config) as log:
        print("Checking credentials, exact model access, and the workflow.", flush=True)
        _preflight(config, log)
        if args.action == "check":
            print(
                "Operator preflight passed. Worker-pod service connectivity is verified during execution."
            )
            return False
        print(
            "Staging inputs and submitting the canonical workflow. Progress is recorded in the private operator log.",
            flush=True,
        )
        _stage_inputs(config)
        state_dir = args.config.parent / (config["run_id"] + "-runtime")
        state_dir.mkdir(mode=0o700, exist_ok=True)
        command = _submit_command(
            config, resume=args.action == "resume", state_dir=state_dir
        )
        _invoke(command, log)
    return True


def _initialize(path):
    config = {
        "project": "<project-alias>",
        "infra": "k8s/<context>",
        "bucket": "<bucket>",
        "prefix": "video-variant-sweep",
        "run_id": "video-sweep-" + uuid.uuid4().hex[:12],
        "accelerators": "B200:1",
        "generator": "cosmos3-nano",
        "sources": ["s3://<bucket>/inputs/source.mp4"],
        "reasoner_model": "nvidia/Cosmos3-Super-Reasoner",
        "merge_model": "nvidia/Nemotron-3_5-Lightning",
        "samples": 8,
        "threshold": 0.8,
        "sweep": {
            "base": {
                "hint": "Warm warehouse lighting; preserve the vehicle, load and motion",
                "edge_threshold": "medium",
                "num_steps": 35,
                "cfg_normalization": "enabled",
            },
            "axes": {
                "control_guidance": [1.0, 1.5],
                "guidance": [3.0, 5.0],
                "seed": [23, 41],
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")


def _load(path, *, offline=False):
    config = json.loads(path.read_text())
    _validate_fields(config)
    planning._validate_inputs(_sources(config), _variants(config))
    if offline:
        return config
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
    _validate_location(config)
    if type(config["samples"]) is not int or config["samples"] < 2:
        raise ValueError("At least two frame samples are required")
    if (
        type(config["threshold"]) not in (float, int)
        or not 0 <= config["threshold"] <= 1
    ):
        raise ValueError("Threshold must be finite and between zero and one")
    for uri in config["sources"]:
        artifacts._object(uri)
    artifacts._object(_root(config) + "/plan.json")
    return config


def _validate_fields(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be an object")
    selections = set(config) & {"variants", "sweep"}
    optional = {"generator", "image_overrides"}
    if len(selections) != 1 or set(config) - selections - optional != _FIELDS:
        raise ValueError("Configuration fields differ from the generated template")
    _validate_image_overrides(config.get("image_overrides", {}))
    if config.get("generator", "cosmos-transfer2.5") not in (
        "cosmos-transfer2.5",
        "cosmos3-nano",
    ):
        raise ValueError("Unknown generation backend")
    if "sweep" in config and config.get("generator") != "cosmos3-nano":
        raise ValueError("Parameter axes require the native Cosmos3 generator")


def _validate_image_overrides(overrides):
    if not isinstance(overrides, dict):
        raise ValueError("Image overrides must map exact tool references to digests")
    for tool, image in overrides.items():
        if not isinstance(tool, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", tool):
            raise ValueError("An image override requires an exact tool reference")
        if not isinstance(image, str) or not re.fullmatch(
            r"[^\s@=]+@sha256:[0-9a-f]{64}", image
        ):
            raise ValueError("Image overrides must use immutable sha256 digests")


def _image_arguments(config):
    arguments = []
    for tool, image in sorted(config.get("image_overrides", {}).items()):
        arguments.extend(("--image-override", f"{tool}={image}"))
    return arguments


def _image_preflight_command(config):
    command = _npa(
        "workbench",
        "workflow",
        "preflight-images",
        _spec(config),
        "--project",
        config["project"],
        "--infra",
        config["infra"],
        "--json",
    )
    command.extend(_image_arguments(config))
    for key, value in _variables(config).items():
        command.extend(("--var", f"{key}={value}"))
    return command


def _preview(args, config):
    from npa.workflows.video_sweep.matrix_view import write_preview

    if "sweep" not in config:
        raise ValueError("Matrix preview requires a sweep with base and axes")
    summary = matrix.describe(config["sweep"], len(config["sources"]), 2)
    output = args.output_dir or args.config.parent / "matrix-preview"
    write_preview(output, summary)
    print(
        f"Planned {len(summary['jobs'])} candidates: {summary['sources']} sources × {summary['combinations']} parameter combinations across {summary['workers']} workers. No compute submitted."
    )


def _validate_location(config):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9/_-]*", config["prefix"]):
        raise ValueError("Use a relative S3 prefix without dots or whitespace")
    if not config["infra"].startswith("k8s/"):
        raise ValueError("The reference uses a Kubernetes target")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", config["run_id"]):
        raise ValueError("Use a lowercase alphanumeric run ID with hyphens")


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


def _spec(config=None):
    from npa.orchestration.npa_workflow import load_spec
    from npa.orchestration.npa_workflow.blueprints import resolve_npa_workflow_spec

    native = config and config.get("generator") == "cosmos3-nano"
    name = "video-variant-sweep-cosmos3.yaml" if native else "video-variant-sweep.yaml"
    path = resolve_npa_workflow_spec(name)
    if path is None:
        raise FileNotFoundError("The canonical workflow catalog is missing")
    load_spec(path)
    return str(path)


def _invoke(command, log):
    subprocess.run(command, check=True, stdout=log, stderr=log)


def _service_preflight(config):
    from npa.clients.token_factory import TokenFactoryClient

    missing = [name for name in _REQUIRED_SECRETS if not os.environ.get(name)]
    if missing:
        raise ValueError("Supply all required service secrets")
    if not {config["reasoner_model"], config["merge_model"]} <= set(
        TokenFactoryClient().list_models()
    ):
        raise ValueError("Select explicitly available hosted models")


def _preflight(config, log):
    _service_preflight(config)
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
        _npa(
            "workbench",
            "health",
            "access",
            "--capability",
            "cosmos3" if config.get("generator") == "cosmos3-nano" else "cosmos2",
            "--json",
        ),
        _npa("workbench", "workflow", "validate-spec", _spec(config), "--json"),
        _npa(
            "workbench",
            "workflow",
            "plan-spec",
            _spec(config),
            "--run-id",
            "preview",
            "--waves",
            "--json",
        ),
        _image_preflight_command(config),
    ):
        _invoke(command, log)


def _root(config):
    return f"s3://{config['bucket']}/{config['prefix'].rstrip('/')}/{config['run_id']}"


def _sources(config):
    return {"schema": "npa.video_sweep.sources.v1", "clips": config["sources"]}


def _variants(config):
    if "sweep" in config:
        return {
            "schema": "npa.video_sweep.variants.v3",
            "generator": "cosmos3-nano",
            "sweep": config["sweep"],
        }
    if config.get("generator") == "cosmos3-nano":
        return {
            "schema": "npa.video_sweep.variants.v2",
            "generator": "cosmos3-nano",
            "variants": config["variants"],
        }
    return {"schema": "npa.video_sweep.variants.v1", "variants": config["variants"]}


def _stage_inputs(config):
    inputs = {"sources": _sources(config), "variants": _variants(config)}
    execution_uri = _root(config) + "/inputs/execution.json"
    recorded_images = artifacts.exists(execution_uri)
    if config.get("image_overrides"):
        if not recorded_images and artifacts.exists(
            _root(config) + "/inputs/sources.json"
        ):
            raise ValueError("Run image overrides changed; use a new run ID")
        inputs = {
            "execution": {"image_overrides": config["image_overrides"]},
            **inputs,
        }
    elif recorded_images:
        raise ValueError("Run image overrides changed; use a new run ID")
    for name, content in inputs.items():
        uri = _root(config) + f"/inputs/{name}.json"
        if artifacts.exists(uri) and artifacts.digest(
            artifacts.read_json(uri)
        ) != artifacts.digest(content):
            raise ValueError(
                "Run inputs changed; use a new run ID for a different sweep"
            )
    for name, content in inputs.items():
        artifacts.write_json(_root(config) + f"/inputs/{name}.json", content)


def _submit_command(config, *, resume=False, state_dir=None):
    command = _npa(
        "workbench",
        "workflow",
        "submit",
        _spec(config),
        "--project",
        config["project"],
        "--infra",
        config["infra"],
        "--resume-run" if resume else "--run-id",
        config["run_id"],
        "--runtime",
        "--stage-src",
        "--max-wait-seconds",
        "0",
    )
    if state_dir is not None:
        command.extend(("--isolated-config-dir", str(state_dir)))
    command.extend(_image_arguments(config))
    for key, value in _variables(config).items():
        command.extend(("--var", f"{key}={value}"))
    for name in (*_REQUIRED_SECRETS, *_OPTIONAL_SECRETS):
        if os.environ.get(name):
            command.extend(("--secret-env", name))
    return command


def _variables(config):
    variables = {
        key: config[key]
        for key in (
            "bucket",
            "reasoner_model",
            "merge_model",
            "samples",
            "threshold",
            "accelerators",
        )
    }
    variables.update(
        prefix=config["prefix"].rstrip("/") + "/" + config["run_id"],
        sources_uri=_root(config) + "/inputs/sources.json",
        variants_uri=_root(config) + "/inputs/variants.json",
    )
    return variables


if __name__ == "__main__":
    raise SystemExit(main())
