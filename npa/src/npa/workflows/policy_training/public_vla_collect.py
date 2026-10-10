"""Retrieve completed public VLA deliverables from the run-owned Kubernetes volume."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time
import uuid

from .public_vla_launch import IMAGE
from .diagnostics import _cleanup


def main() -> None:
    """Wait for completion or failure, then download the report and portable trained policy.

    Args:
        None; --input-path identifies the launch receipt and --output-path a new directory.
    Returns:
        None.
    Raises:
        ValueError: Receipt identity or download containment is invalid.
        RuntimeError: The training job failed.
        subprocess.CalledProcessError: Artifact retrieval failed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the owned namespace and volume after verifying collected artifacts",
    )
    args = parser.parse_args()
    os.umask(0o077)
    receipt = json.loads(args.input_path.read_text())
    kubectl = _kubectl(receipt)
    _wait_for_job(kubectl)
    args.output_path.mkdir(parents=True, exist_ok=False, mode=0o700)
    _download(kubectl, receipt, args.output_path)
    if args.cleanup:
        from .public_vla_cleanup import _cleanup_run

        _cleanup_run(receipt, args.output_path)


def _kubectl(receipt):
    if receipt.get("backend") != "kubernetes" or receipt.get("image") != IMAGE:
        raise ValueError("expected a public VLA Kubernetes launch receipt")
    if not re.fullmatch(r"npa-vla-[a-f0-9]{12}", receipt.get("namespace", "")):
        raise ValueError("receipt does not name a run-owned namespace")
    command = receipt.get("kubectl", [])
    if not command or command[0] != "kubectl" or len(command) % 2 != 1:
        raise ValueError("invalid kubectl configuration in receipt")
    if any(flag not in {"--context", "--kubeconfig"} for flag in command[1::2]):
        raise ValueError("receipt contains unsupported kubectl flags")
    return command + ["-n", receipt["namespace"]]


def _wait_for_job(kubectl):
    print("Waiting for the public VLA job; failures stop collection.", flush=True)
    while True:
        result = subprocess.run(
            kubectl + ["get", "job", "pipeline", "-o", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
        status = json.loads(result.stdout).get("status", {})
        conditions = {
            item["type"]: item["status"] for item in status.get("conditions", [])
        }
        if conditions.get("Failed") == "True":
            raise RuntimeError(
                "training failed; inspect the run-owned job logs and retained volume"
            )
        if conditions.get("Complete") == "True":
            return
        time.sleep(5)


def _download(kubectl, receipt, output):
    name = "collect-" + uuid.uuid4().hex[:12]
    manifest = _reader(name, receipt["namespace"])
    subprocess.run(
        kubectl + ["create", "-f", "-"],
        input=json.dumps(manifest),
        text=True,
        check=True,
    )
    try:
        subprocess.run(
            kubectl + ["wait", "--for=condition=Ready", "pod/" + name, "--timeout=-1s"],
            check=True,
        )
        _extract_deliverables(kubectl, name, output)
        print(
            "Collected HTML, rollout MP4s and the trained policy: "
            + str(output.resolve())
        )
    finally:
        _cleanup(
            lambda: subprocess.run(
                kubectl + ["delete", "pod", name, "--wait=true"],
                capture_output=True,
                check=True,
            ),
            str(output / "collection.json"),
        )


def _extract_deliverables(kubectl, name, output):
    archive = output / "deliverables.tar.gz"
    command = kubectl + [
        "exec",
        name,
        "--",
        "tar",
        "-czf",
        "-",
        "-C",
        "/work/run",
        "report",
        "exported-policy",
        "completed.json",
        "selection.json",
    ]
    with archive.open("xb") as stream:
        subprocess.run(command, stdout=stream, check=True)
    with tarfile.open(archive) as stream:
        stream.extractall(output, filter="data")


def _reader(name, namespace):
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "restartPolicy": "Never",
            "automountServiceAccountToken": False,
            "securityContext": {"runAsUser": 1000, "runAsGroup": 1000, "fsGroup": 1000},
            "containers": [
                {
                    "name": "reader",
                    "image": IMAGE,
                    "command": ["sleep", "infinity"],
                    "resources": {"requests": {"cpu": "100m", "memory": "256Mi"}},
                    "volumeMounts": [
                        {"name": "artifacts", "mountPath": "/work", "readOnly": True}
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "artifacts",
                    "persistentVolumeClaim": {
                        "claimName": "pipeline",
                        "readOnly": True,
                    },
                }
            ],
        },
    }


if __name__ == "__main__":
    main()
