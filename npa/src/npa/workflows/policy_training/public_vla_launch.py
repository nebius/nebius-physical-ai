"""Launch the same public VLA runner on an existing Kubernetes cluster or Slurm allocation."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import uuid

import yaml

IMAGE = "ghcr.io/nebius/nebius-physical-ai/npa-lerobot@sha256:8d513f8558253fc484808a1e53a63a5da5a0c280ff973c4e590dd3e04b228643"
MODULE = "npa.workflows.policy_training.public_vla"
PYTHON = "/opt/lerobot/venv/bin/python"


def main() -> None:
    """Submit a complete reference run and save private recovery instructions.

    Args:
        None; process arguments select the backend and artifact paths.
    Returns:
        None.
    Raises:
        ValueError: Required runtime settings are missing or unsafe.
        subprocess.CalledProcessError: Submission fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("kubernetes", "slurm"), required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--kube-context")
    parser.add_argument("--kubeconfig")
    parser.add_argument("--storage-class")
    parser.add_argument("--shared-path", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume:
        _resume(args)
        return
    from .public_vla import _recipe

    _recipe(args.input_path)
    os.umask(0o077)
    args.output_path.mkdir(parents=True, exist_ok=False)
    if args.backend == "kubernetes":
        _submit_kubernetes(args)
    else:
        _submit_slurm(args)


def _resume(args):
    from .public_vla_collect import _kubectl

    receipt = json.loads((args.output_path / "receipt.json").read_text())
    if args.backend != "kubernetes":
        raise ValueError(
            "for Slurm, resubmit the saved pipeline.sbatch in the shared directory"
        )
    command = _kubectl(receipt)
    job = _saved_job(args.output_path, receipt)
    result = subprocess.check_output(
        command + ["get", "job", "pipeline", "--ignore-not-found", "-o", "json"],
        text=True,
    )
    if result.strip():
        status = json.loads(result).get("status", {})
        failed = any(
            condition.get("type") == "Failed" and condition.get("status") == "True"
            for condition in status.get("conditions", [])
        )
        if not failed or status.get("active") or status.get("succeeded"):
            raise ValueError(
                "resume requires an absent or failed job; retain the active/completed job"
            )
        subprocess.run(
            command + ["delete", "job", "pipeline", "--wait=true"], check=True
        )
    subprocess.run(
        command + ["apply", "-f", "-"], input=json.dumps(job), text=True, check=True
    )
    print(
        "Resumed the saved run using its retained volume, source, recipe and native checkpoint."
    )


def _saved_job(output, receipt):
    objects = json.loads((output / "resources.json").read_text())["items"]
    job = next(item for item in objects if item["kind"] == "Job")
    if job["metadata"] != {"name": "pipeline", "namespace": receipt["namespace"]}:
        raise ValueError("saved Job does not match the run receipt")
    if job["spec"]["template"]["spec"]["containers"][0]["image"] != IMAGE:
        raise ValueError("saved Job changed its pinned image")
    return job


def source_archive() -> bytes:
    """Package only this runner and its import dependencies, excluding workspace data.

    Args:
        None.
    Returns:
        A compressed source archive suitable for a Kubernetes ConfigMap.
    Raises:
        OSError: Installed source cannot be read.
    """
    root = Path(__file__).resolve().parents[2]
    buffer = io.BytesIO()
    files = [root / "__init__.py", root / "workflows" / "__init__.py"]
    names = (
        "__init__",
        "cli",
        "contracts",
        "checkpoints",
        "public_vla",
        "public_vla_data",
        "public_vla_train",
        "public_vla_eval",
        "public_vla_export",
        "public_vla_verify",
        "public_vla_report",
    )
    files += [Path(__file__).parent / (name + ".py") for name in names]
    files += [
        Path(__file__).with_name(name)
        for name in ("public_vla_report.html", "apache-2.0.txt")
    ]
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in files:
            if path.is_file():
                archive.add(
                    path,
                    arcname="npa/" + str(path.relative_to(root)),
                    filter=_anonymous_header,
                )
    return buffer.getvalue()


def _anonymous_header(header):
    header.uid = header.gid = header.mtime = 0
    header.uname = header.gname = ""
    header.pax_headers = {}
    return header


def kubernetes_objects(
    namespace: str, archive: bytes, recipe: dict, storage_class=None
) -> list:
    """Render one GPU job with durable artifacts and no injected credentials.

    Args:
        namespace: New run-owned Kubernetes namespace.
        archive: Runner source archive.
        recipe: Explicit training recipe overrides.
        storage_class: Optional existing persistent-volume storage class.
    Returns:
        Namespace, source ConfigMap, persistent volume claim and training Job.
    Raises:
        ValueError: Namespace is invalid.
    """
    import re

    if not re.fullmatch(r"npa-vla-[a-f0-9]{12}", namespace):
        raise ValueError("expected a generated, run-owned namespace")
    metadata = {"name": "pipeline", "namespace": namespace}
    claim = {
        "accessModes": ["ReadWriteOnce"],
        "resources": {"requests": {"storage": "100Gi"}},
    }
    if storage_class:
        claim["storageClassName"] = storage_class
    return [
        {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": namespace,
                "labels": {"npa.nebius.ai/public-vla-run": namespace},
            },
        },
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": metadata,
            "binaryData": {"source.tgz": base64.b64encode(archive).decode()},
            "data": {"recipe.json": json.dumps(recipe)},
        },
        {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": metadata,
            "spec": claim,
        },
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": metadata,
            "spec": {"backoffLimit": 0, "template": {"spec": _pod_spec()}},
        },
    ]


def _pod_spec():
    command = "\n".join(
        [
            "set -euo pipefail",
            "mkdir -p /work/source",
            "tar -xzf /pipeline/source.tgz -C /work/source",
            "export PYTHONPATH=/work/source",
            f"exec {PYTHON} -u -m {MODULE} --input-path /pipeline/recipe.json --output-path /work/run",
        ]
    )
    template = Path(__file__).with_name("public-vla-pod.yaml")
    spec = yaml.safe_load(template.read_text())["spec"]
    spec["containers"][0].update(image=IMAGE, command=["bash", "-c", command])
    return spec


def _submit_kubernetes(args):
    namespace = "npa-vla-" + uuid.uuid4().hex[:12]
    recipe = json.loads(args.input_path.read_text()) if args.input_path else {}
    objects = kubernetes_objects(
        namespace, source_archive(), recipe, args.storage_class
    )
    manifest = args.output_path / "resources.json"
    manifest.write_text(
        json.dumps({"apiVersion": "v1", "kind": "List", "items": objects})
    )
    kubectl = ["kubectl"]
    for flag, value in (
        ("--context", args.kube_context),
        ("--kubeconfig", args.kubeconfig),
    ):
        if value:
            kubectl += [flag, value]
    subprocess.run(kubectl + ["cluster-info"], stdout=subprocess.DEVNULL, check=True)
    subprocess.run(kubectl + ["apply", "-f", str(manifest)], check=True)
    receipt = {
        "backend": "kubernetes",
        "namespace": namespace,
        "kubectl": kubectl,
        "job": "pipeline",
        "artifact_path": "/work/run",
        "image": IMAGE,
    }
    (args.output_path / "receipt.json").write_text(json.dumps(receipt, indent=2))
    _record_namespace_and_report(args.output_path, receipt)


def _record_namespace_and_report(output, receipt):
    kubectl, namespace = receipt["kubectl"], receipt["namespace"]
    namespace_info = json.loads(
        subprocess.check_output(
            kubectl + ["get", "namespace", namespace, "-o", "json"], text=True
        )
    )
    receipt["namespace_uid"] = namespace_info["metadata"]["uid"]
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    follow = kubectl + ["-n", namespace, "logs", "-f", "job/pipeline"]
    print("Submitted. Follow progress with: " + shlex.join(follow))
    print(
        "Checkpoints and evidence persist on the run's volume after the GPU job exits."
    )


def slurm_script(shared: Path, recipe: Path) -> str:
    """Render a one-GPU Soperator/Pyxis job that executes the complete reference.

    Args:
        shared: Absolute shared run directory visible to login and worker nodes.
        recipe: Recipe file inside that shared directory.
    Returns:
        A shell-safe Slurm batch script without a wall-time or retry limit.
    Raises:
        ValueError: The shared mount is ambiguous or recipe escapes it.
    """
    from .distributed import enroot_image

    if (
        not shared.is_absolute()
        or str(shared) == "/"
        or any(c in str(shared) for c in ",:\n")
    ):
        raise ValueError("shared-path must be an absolute unambiguous mount directory")
    if not recipe.is_relative_to(shared) or ".." in recipe.parts:
        raise ValueError("recipe must be inside the shared run directory")
    command = _slurm_runner_command(shared, recipe)
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --gpus=1",
            "#SBATCH --cpus-per-task=16",
            "#SBATCH --mem=96G",
            "set -euo pipefail",
            "export PYTHONPATH=" + shlex.quote(str(shared / "source")),
            "export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics",
            "srun --kill-on-bad-exit=1 --container-image="
            + shlex.quote(enroot_image(IMAGE))
            + " --container-env=PYTHONPATH,NVIDIA_DRIVER_CAPABILITIES --no-container-mount-home"
            + " --container-mounts="
            + shlex.quote(f"{shared}:{shared}")
            + " --container-workdir="
            + shlex.quote(str(shared))
            + " "
            + shlex.join(command),
            "",
        ]
    )


def _slurm_runner_command(shared, recipe):
    return [
        PYTHON,
        "-u",
        "-m",
        MODULE,
        "--input-path",
        str(recipe),
        "--output-path",
        str(shared / "run"),
    ]


def _submit_slurm(args):
    if args.shared_path is None:
        raise ValueError(
            "Slurm requires --shared-path on the login/worker shared filesystem"
        )
    shared = args.shared_path.resolve()
    shared.mkdir(parents=True, exist_ok=False)
    recipe = shared / "recipe.json"
    recipe.write_text(args.input_path.read_text() if args.input_path else "{}")
    with tarfile.open(fileobj=io.BytesIO(source_archive()), mode="r:gz") as archive:
        archive.extractall(shared / "source", filter="data")
    script = shared / "pipeline.sbatch"
    script.write_text(slurm_script(shared, recipe))
    result = subprocess.run(
        ["sbatch", "--parsable", "--output", str(shared / "slurm.log"), str(script)],
        capture_output=True,
        text=True,
        check=True,
    )
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise ValueError("sbatch did not return a numeric job identifier")
    receipt = {
        "backend": "slurm",
        "job_id": job_id,
        "shared_path": str(shared),
        "image": IMAGE,
    }
    (args.output_path / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print("Submitted the complete public VLA pipeline to Slurm; receipt saved locally.")


if __name__ == "__main__":
    main()
