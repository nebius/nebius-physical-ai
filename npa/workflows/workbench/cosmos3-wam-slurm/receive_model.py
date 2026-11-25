"""Pull the qualified update-2 model to a fresh run directory on the evaluation host."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from evaluate import _checkpoint_identity, _source_model


def _pull(host, sources, directory):
    prefix = f"{host}:" if host else ""

    subprocess.run(
        [
            "rsync",
            "-rlt",
            "-s",
            "--",
            *(prefix + str(p) for p in sources),
            str(directory),
        ],
        check=True,
    )


def _source_receipts(destination, trusted_report_sha256):
    if any(
        (destination / name).is_symlink()
        for name in ("native-training.json", "run.json", "qualification.json")
    ):
        raise ValueError("training evidence must use regular files")
    report_bytes = (destination / "native-training.json").read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != trusted_report_sha256:
        raise ValueError("source training report differs from qualification receipt")
    report = json.loads(report_bytes)
    qualified = json.loads((destination / "qualification.json").read_text())
    if (
        qualified["status"] != "passed"
        or qualified["source_training_report_sha256"] != trusted_report_sha256
        or qualified["run_sha256"] != report["run_sha256"]
        or qualified["checkpoint_sha256"] != report["checkpoint_sha256"]
    ):
        raise ValueError(
            "source qualification did not verify this training report and reload"
        )
    run_bytes = (destination / "run.json").read_bytes()
    settings = json.loads(run_bytes)
    if hashlib.sha256(run_bytes).hexdigest() != report["run_sha256"]:
        raise ValueError("transferred run.json differs from qualified run")
    if (
        settings.get("launcher") != "torchrun"
        or settings["steps"] != 2
        or not re.fullmatch(r"[A-Za-z0-9_-]+", settings["name"])
    ):
        raise ValueError("expected the qualified update-2 run")
    expected = _source_model(report, run_bytes, 2)
    relative = (
        Path("output/cosmos3_wam/libero_10")
        / settings["name"]
        / "checkpoints/iter_000000002"
    )
    return report, expected, relative


def _copy_model(host, source_run, destination, relative, expected):
    incoming = destination / relative / "incoming"
    incoming.mkdir(parents=True)
    required = sum(entry["bytes"] for entry in expected.values())
    if shutil.disk_usage(incoming).free < required:
        raise ValueError("destination has insufficient space for the verified model")
    transfer_start = time.monotonic()
    _pull(host, [source_run / relative / "model"], incoming)
    after_copy = time.monotonic()
    model_digest, observed = _checkpoint_identity(incoming)
    if observed != expected:
        raise ValueError("model transfer differs from the source checkpoint")
    verify_seconds = time.monotonic() - after_copy
    # Until verification succeeds the scoring path contains no model.
    (incoming / "model").rename(incoming.parent / "model")
    return {
        "model_manifest_sha256": model_digest,
        "model_bytes": sum(v["bytes"] for v in observed.values()),
        "model_files": len(observed),
        "model_transfer_seconds": after_copy - transfer_start,
        "destination_hash_seconds": verify_seconds,
    }


def _receive(source_run, host, destination, trusted_report_sha256):
    if not re.fullmatch(r"[0-9a-f]{64}", trusted_report_sha256):
        raise ValueError(
            "supply the report SHA-256 from the passing source qualification"
        )
    if not source_run.is_absolute() or (
        host and not re.fullmatch(r"(?:[A-Za-z0-9_][\w.-]*@)?[A-Za-z0-9][\w.-]*", host)
    ):
        raise ValueError("use an absolute source path and an SSH host or user@host")
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    started = time.monotonic()
    _pull(
        host,
        [
            source_run / "run.json",
            source_run / "qualification/native-training.json",
            source_run / "qualification/qualification.json",
        ],
        destination,
    )
    report, expected, relative = _source_receipts(destination, trusted_report_sha256)
    result = _copy_model(host, source_run, destination, relative, expected)
    result.update(
        run_sha256=report["run_sha256"],
        source_training_report_sha256=trusted_report_sha256,
        total_receive_seconds=time.monotonic() - started,
    )
    with (destination / "transfer.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument(
        "--source-host",
        help="SSH host for the training machine; omit for local storage",
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source-report-sha256", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    print(
        json.dumps(
            _receive(
                args.source_run,
                args.source_host,
                args.run_dir,
                args.source_report_sha256,
            )
        )
    )
