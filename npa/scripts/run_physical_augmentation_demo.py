#!/usr/bin/env python3
"""Launch the canonical RTX workflow and fetch its ready-to-open demo artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import tempfile

from npa._sdk import call_cli_callback
from npa.clients.config import resolve_project_storage
from npa.clients.project_credentials import storage_client_for_project
from npa.workflows.lerobot_transfer_data import file_sha256


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--cluster", default="physical-augmentation")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--fetch-run",
        default="",
        help="Fetch an existing completed run without allocating a GPU.",
    )
    parser.add_argument(
        "--provision",
        action="store_true",
        help="Ensure an RTX-ready cluster before submitting.",
    )
    parser.add_argument(
        "--capacity-block-group",
        default="",
        help="Optional private reservation selector for provisioning.",
    )
    parser.add_argument("--isolated-config-dir", type=Path)
    parser.add_argument("--kubeconfig", type=Path)
    parser.add_argument("--sky-bin", default="")
    parser.add_argument("--no-accept-eula", action="store_true")
    return parser


def _ready(args):
    from npa.cli.workbench.health import preflight_command

    if args.no_accept_eula:
        raise ValueError(
            "Isaac execution was opted out; omit --no-accept-eula to enable its documented runtime-fetch policy."
        )
    call_cli_callback(preflight_command, project=args.project, checks="nebius,s3")
    if not args.provision:
        return
    from npa.cli.provision import provision_if_absent_cmd

    call_cli_callback(
        provision_if_absent_cmd,
        project=args.project,
        cluster_name=args.cluster,
        context_name=args.cluster,
        kubeconfig=args.kubeconfig,
        gpu_nodes=1,
        cpu_nodes=1,
        cpu_platform="cpu-d3",
        cpu_preset="16vcpu-64gb",
        gpu_workload_profile="rtx-rendering",
        capacity_block_group=args.capacity_block_group,
        accelerator="RTXPRO6000:1",
        sky_bin=args.sky_bin,
        sky_smoke=True,
    )


def _submit(args, run_id):
    from npa.cli.workbench.workflow import submit_cmd

    _ready(args)
    spec = (
        Path(__file__).resolve().parents[2]
        / "workflows/testing/physical-augmentation.yaml"
    )
    settings = resolve_project_storage(
        args.project, include_shared_credentials=False, include_environment=False
    )
    bucket = settings.checkpoint_bucket.removeprefix("s3://").split("/")[0]
    call_cli_callback(
        submit_cmd,
        yaml_path=spec,
        project=args.project,
        infra=f"k8s/{args.cluster}",
        run_id=run_id,
        runtime=True,
        stage_src=True,
        var=[f"bucket={bucket}"],
        secret_env=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        isolated_config_dir=args.isolated_config_dir,
        sky_bin=args.sky_bin,
        max_wait_seconds=0,
        image_bootstrap_timeout_seconds=0,
    )


def _fetch(args, run_id):

    settings = resolve_project_storage(
        args.project, include_shared_credentials=False, include_environment=False
    )
    bucket = settings.checkpoint_bucket.removeprefix("s3://").split("/")[0]
    prefix = f"s3://{bucket}/physical-augmentation/{run_id}/reports/"
    storage = storage_client_for_project(args.project)
    destination = args.output_path.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".download-", dir=destination) as temporary:
        staging = Path(temporary)
        _download_artifacts(storage, prefix, staging)
        for path in staging.iterdir():
            path.replace(destination / path.name)
    print(f"Open in your browser: {destination / 'demo.html'}")
    print(f"Shareable video: {destination / 'demo.mp4'}")
    print(f"Interactive recording: {destination / 'demonstrations.rrd'}")


def _download_artifacts(storage, prefix, destination):
    import json

    receipt = destination / "checksums.json"
    storage.download_file(prefix + "checksums.json", str(receipt))
    checksums = json.loads(receipt.read_text())
    for name in (
        "demo.html",
        "demo.mp4",
        "demo-validation.json",
        "report.json",
        "demonstrations.rrd",
    ):
        if name not in checksums:
            raise ValueError(
                f"Completed run does not contain required demo artifact: {name}"
            )
        target = destination / name
        storage.download_file(prefix + name, str(target))
        if file_sha256(target) != checksums[name]:
            raise ValueError(f"Downloaded artifact failed its checksum: {name}")


def main(argv=None) -> int:
    """Submit or fetch a factual demo through existing Workbench interfaces.

    Args:
        argv: Optional command-line arguments.
    Returns:
        Zero when all local demo files pass checksum verification.
    Raises:
        ValueError: Configuration, acceptance or artifact validation fails.
        RuntimeError: Existing provisioning or workflow submission fails.
        OSError: Required files cannot be read or written.
    """
    args = _parser().parse_args(argv)
    run_id = (
        args.fetch_run
        or args.run_id
        or datetime.now(timezone.utc).strftime("physical-augmentation-%Y%m%d-%H%M%S")
    )
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", run_id):
        raise ValueError(
            "Run ID must contain lowercase letters, digits, hyphens or underscores"
        )
    if args.kubeconfig:
        os.environ["KUBECONFIG"] = str(args.kubeconfig.expanduser().resolve())
    if not args.fetch_run:
        _submit(args, run_id)
    _fetch(args, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
