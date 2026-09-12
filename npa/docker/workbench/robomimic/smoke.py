#!/usr/bin/env python3
"""Deferred live gate: real upstream robomimic BC on official Lift PH data."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
import urllib.parse

import h5py
import numpy as np
import torch
from robomimic.config import config_factory
from robomimic.utils.file_utils import policy_from_checkpoint
from verify_image import verified_source_identity


SOURCE_REVISION = "d309eaecc18acf4152a830a895a6984b8ac71b05"
DATASET_REVISION = "74fa018461f479cd9fd15b924a16103012096203"
DATASET_PATH = "v1.5/lift/ph/low_dim_v15.hdf5"
DATASET_SHA256 = "2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540"
DATASET_BYTES = 21_084_088
BAKED_LOCK_SHA256 = "65efcf0065ad4662b348e54e3f2d86996d934a518fcad0e89ecf012399ce1504"
BAKED_DISTRIBUTION_COUNT = 40
TRAIN_STEPS = 4
VALIDATION_STEPS = 2
CAPABILITIES = [
    "lift_ph_lowdim_bc_train",
    "lift_ph_lowdim_heldout_validate",
    "lift_ph_lowdim_checkpoint_reload_action",
]

_HTTPS_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


def _open_allowed_https(
    url: str,
    *,
    headers: dict[str, str],
    allowed_hosts: tuple[str, ...],
    allow_hf_redirects: bool = False,
    context: ssl.SSLContext | None = None,
) -> tuple[http.client.HTTPSConnection, http.client.HTTPResponse]:
    """Open a tightly scoped HTTPS GET without urllib's multi-scheme opener."""
    current_url = url
    for _ in range(6):
        parsed = urllib.parse.urlsplit(current_url)
        hostname = (parsed.hostname or "").lower()
        allowed = hostname in allowed_hosts or (
            allow_hf_redirects
            and (
                re.fullmatch(r"cdn-lfs(?:-[a-z0-9-]+)?\.hf\.co", hostname) is not None
                or hostname == "cas-bridge.xethub.hf.co"
                or hostname == "cdn.hf.co"
                or hostname.endswith(".cdn.hf.co")
            )
        )
        if (
            parsed.scheme != "https"
            or not allowed
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in {None, 443}
            or parsed.fragment
        ):
            raise RuntimeError("refusing URL outside the approved HTTPS origins")
        target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        connection = http.client.HTTPSConnection(
            hostname,
            port=parsed.port,
            context=context,
            timeout=120,
        )
        try:
            connection.request("GET", target, headers=headers)
            response = connection.getresponse()
        except (OSError, http.client.HTTPException):
            connection.close()
            raise
        if response.status in _HTTPS_REDIRECT_STATUSES:
            location = response.getheader("Location")
            response.close()
            connection.close()
            if not location:
                raise RuntimeError("HTTPS redirect omitted its destination")
            current_url = urllib.parse.urljoin(current_url, location)
            continue
        if response.status != 200:
            status = response.status
            response.close()
            connection.close()
            raise RuntimeError(f"approved HTTPS origin returned status {status}")
        return connection, response
    raise RuntimeError("too many HTTPS redirects")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pod_identity(
    runtime_image: str, expected_digest: str, runtime_root: Path
) -> dict[str, object]:
    service_account_root = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    pod_name = os.environ.get("HOSTNAME", "").strip()
    namespace = (service_account_root / "namespace").read_text(encoding="utf-8").strip()
    expected_namespace = os.environ.get("NPA_ROBOMIMIC_EXPECTED_NAMESPACE", "").strip()
    expected_service_account = os.environ.get(
        "NPA_ROBOMIMIC_EXPECTED_SERVICE_ACCOUNT", ""
    ).strip()
    if namespace != expected_namespace or not expected_service_account:
        raise RuntimeError(
            "workload identity selector mismatch: "
            f"namespace_match={namespace == expected_namespace} "
            f"service_account_set={bool(expected_service_account)}"
        )
    token = (service_account_root / "token").read_text(encoding="utf-8").strip()
    context = ssl.create_default_context(cafile=str(service_account_root / "ca.crt"))
    connection, response = _open_allowed_https(
        f"https://kubernetes.default.svc/api/v1/namespaces/{namespace}/pods/{pod_name}",
        headers={"Authorization": f"Bearer {token}"},
        allowed_hosts=("kubernetes.default.svc",),
        context=context,
    )
    try:
        pod_status = json.load(response)
    finally:
        response.close()
        connection.close()
    observed_service_account = str(
        pod_status.get("spec", {}).get("serviceAccountName", "")
    )
    if (
        pod_status.get("metadata", {}).get("name") != pod_name
        or pod_status.get("metadata", {}).get("namespace") != expected_namespace
        or observed_service_account != expected_service_account
    ):
        raise RuntimeError(
            "Kubernetes Pod identity does not match the run-owned observer"
        )
    matching_statuses = [
        status
        for status in pod_status.get("status", {}).get("containerStatuses", [])
        if status.get("name") == "ray-node"
        and expected_digest in str(status.get("imageID", ""))
    ]
    if len(matching_statuses) != 1:
        raise RuntimeError(
            "Pod status did not prove exactly one executing immutable image"
        )
    status = matching_statuses[0]
    containers = [
        container
        for container in pod_status.get("spec", {}).get("containers", [])
        if container.get("name") == "ray-node"
    ]
    if len(containers) != 1 or containers[0].get("image") != runtime_image:
        raise RuntimeError(
            "executing ray-node container does not use the expected immutable image"
        )
    runtime_mounts = [
        mount
        for mount in containers[0].get("volumeMounts", [])
        if mount.get("name") == "robomimic-runtime"
        and mount.get("mountPath") == str(runtime_root)
    ]
    runtime_volumes = [
        volume
        for volume in pod_status.get("spec", {}).get("volumes", [])
        if volume.get("name") == "robomimic-runtime"
    ]
    mount_read_only = (
        len(runtime_mounts) == 1
        and runtime_mounts[0].get("readOnly") is True
        and len(runtime_volumes) == 1
        and runtime_volumes[0].get("persistentVolumeClaim", {}).get("readOnly") is True
        and bool(os.statvfs(runtime_root).f_flag & os.ST_RDONLY)
    )
    if not mount_read_only:
        raise RuntimeError("runtime volume is not observed as read-only")
    return {
        "pod_image": {
            "runtime_ref": runtime_image,
            "digest": expected_digest,
            "image_id": str(status["imageID"]),
            "container_name": status.get("name", ""),
            "observation_source": "Kubernetes Pod status.containerStatuses[].imageID",
        },
        "runtime_mount": {
            "name": "robomimic-runtime",
            "path": str(runtime_root),
            "read_only": True,
            "observation_source": (
                "Kubernetes Pod spec volumeMount plus statvfs ST_RDONLY"
            ),
        },
        "workload_identity": {
            "namespace": namespace,
            "service_account": observed_service_account,
            "observation_source": (
                "mounted service-account namespace and Kubernetes Pod spec.serviceAccountName"
            ),
        },
    }


def _hardware() -> dict[str, object]:
    gpu_count = torch.cuda.device_count()
    if gpu_count != 1:
        raise RuntimeError(f"expected exactly one visible GPU, got {gpu_count}")
    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = list(torch.cuda.get_device_capability(0))
    architecture = f"sm_{compute_capability[0]}{compute_capability[1]}"
    if "B200" not in gpu_name.upper() or architecture != "sm_100":
        raise RuntimeError(
            f"expected one B200 sm_100, got {gpu_name!r} {architecture!r}"
        )
    smi_rows = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()
    )
    if len(smi_rows) != 1 or "B200" not in smi_rows[0].upper():
        raise RuntimeError(f"nvidia-smi did not prove exactly one B200: {smi_rows!r}")
    return {
        "accelerator_count": gpu_count,
        "model": gpu_name,
        "architecture": architecture,
        "architecture_family": "NVIDIA Blackwell",
        "compute_capability": compute_capability,
        "nvidia_smi_rows": smi_rows,
        "strict_reserved_capacity_attested": True,
    }


def _download_dataset(
    input_dir: Path,
) -> tuple[Path, list[str], dict[str, int], float, float]:
    input_dir.mkdir(parents=True, exist_ok=False)
    partial = input_dir / "lift_ph_lowdim_v15.download"
    dataset = input_dir / "lift_ph_lowdim_v15.hdf5"
    url = (
        "https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/"
        f"{DATASET_REVISION}/{DATASET_PATH}?download=true"
    )
    digest = hashlib.sha256()
    byte_count = 0
    try:
        connection, response = _open_allowed_https(
            url,
            headers={"User-Agent": "npa-robomimic-smoke/1"},
            allowed_hosts=("huggingface.co",),
            allow_hf_redirects=True,
        )
        try:
            with response, partial.open("xb") as handle:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    next_byte_count = byte_count + len(chunk)
                    if next_byte_count > DATASET_BYTES:
                        raise RuntimeError("dataset exceeds its locked byte count")
                    handle.write(chunk)
                    digest.update(chunk)
                    byte_count = next_byte_count
        finally:
            connection.close()
        if digest.hexdigest() != DATASET_SHA256 or byte_count != DATASET_BYTES:
            raise RuntimeError(
                f"dataset identity mismatch: sha256={digest.hexdigest()} bytes={byte_count}"
            )
        with h5py.File(partial, "r") as handle:
            demo_keys = sorted(handle["data"].keys())
            sample_counts = {
                key: int(handle[f"data/{key}"].attrs["num_samples"])
                for key in demo_keys
            }
            action_min = min(
                float(np.min(handle[f"data/{key}/actions"][:])) for key in demo_keys
            )
            action_max = max(
                float(np.max(handle[f"data/{key}/actions"][:])) for key in demo_keys
            )
        if len(demo_keys) != 200 or sum(sample_counts.values()) <= 0:
            raise RuntimeError("official Lift PH trajectory/sample inventory mismatch")
        partial.replace(dataset)
        return dataset, demo_keys, sample_counts, action_min, action_max
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    output_dir = Path(os.environ["NPA_SMOKE_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("NPA_ROBOMIMIC_STRICT_B200_ATTESTED") != "1":
        raise RuntimeError("operator must attest STRICT reserved B200 capacity")
    runtime_image = os.environ.get("BYOF_IMAGE", "")
    match = re.fullmatch(r".+@(sha256:[0-9a-f]{64})", runtime_image)
    if match is None:
        raise RuntimeError("robomimic workload image must be digest-pinned")

    source_root = Path("/opt/robomimic")
    component_root = Path("/opt/npa/robomimic")
    source_identity = verified_source_identity(
        Path("/opt/byof/npa_source_metadata.json")
    )
    if source_identity["revision"] != SOURCE_REVISION:
        raise RuntimeError("unexpected immutable robomimic source identity")
    baked_lock = component_root / "baked-requirements.lock"
    baked_lock_bytes = baked_lock.read_bytes()
    if (
        _sha256(baked_lock) != BAKED_LOCK_SHA256
        or sum(
            bool(re.match(rb"^[A-Za-z0-9_.-]+==", line))
            for line in baked_lock_bytes.splitlines()
        )
        != BAKED_DISTRIBUTION_COUNT
    ):
        raise RuntimeError("neutral baked dependency lock mismatch")
    runtime_lock = component_root / "runtime-requirements.lock"
    runtime_mount_root = Path(os.environ["NPA_ROBOMIMIC_RUNTIME_ROOT"])
    runtime_root = Path(os.environ["NPA_ROBOMIMIC_ACTIVE_RUNTIME_ROOT"])
    snapshot_parent = runtime_root.parent
    if (
        runtime_root == runtime_mount_root
        or snapshot_parent.stat().st_uid != os.getuid()
        or snapshot_parent.stat().st_mode & 0o077
        or runtime_root.stat().st_mode & 0o222
    ):
        raise RuntimeError(
            "runtime execution snapshot is not private or has write bits"
        )
    runtime_inventory = json.loads(
        (runtime_root / "inventory.json").read_text(encoding="utf-8")
    )

    expected_runtime_inventory = os.environ.get(
        "NPA_ROBOMIMIC_RUNTIME_INVENTORY_SHA256", ""
    ).strip()
    if (
        re.fullmatch(r"[0-9a-f]{64}", expected_runtime_inventory) is None
        or _sha256(runtime_root / "inventory.json") != expected_runtime_inventory
    ):
        raise RuntimeError(
            "runtime inventory does not match the manager-approved digest"
        )

    pod = _pod_identity(runtime_image, match.group(1), runtime_mount_root)
    runtime_mount_proof = pod.get("runtime_mount")
    if not isinstance(runtime_mount_proof, dict):
        raise RuntimeError("runtime mount observation is absent")
    hardware = _hardware()
    dataset, demo_keys, sample_counts, action_min, action_max = _download_dataset(
        Path("/workspace/byof-inputs") / output_dir.name
    )

    split_script = source_root / "robomimic" / "scripts" / "split_train_val.py"
    split_program = "\n".join(
        [
            "import numpy as np, runpy, sys",
            "script, dataset = sys.argv[1:3]",
            "np.random.seed(0)",
            "sys.argv = [script, '--dataset', dataset, '--ratio', '0.1']",
            "runpy.run_path(script, run_name='__main__')",
        ]
    )
    split = subprocess.run(
        [sys.executable, "-c", split_program, str(split_script), str(dataset)],
        check=True,
        capture_output=True,
        text=True,
    )
    (output_dir / "split_stdout.log").write_text(
        split.stdout + split.stderr, encoding="utf-8"
    )
    with h5py.File(dataset, "r") as handle:
        train_keys = [value.decode("utf-8") for value in handle["mask/train"][:]]
        valid_keys = [value.decode("utf-8") for value in handle["mask/valid"][:]]
    overlap = sorted(set(train_keys).intersection(valid_keys))
    if (
        overlap
        or not train_keys
        or not valid_keys
        or len(train_keys) + len(valid_keys) != len(demo_keys)
        or set(train_keys + valid_keys) != set(demo_keys)
    ):
        raise RuntimeError("invalid genuine held-out split")

    train_root = output_dir / "training"
    config_path = output_dir / "bc_config.json"
    config = config_factory("bc")
    with config.values_unlocked():
        config.experiment.name = "lift_ph_lowdim_bc_smoke"
        config.experiment.validate = True
        config.experiment.logging.terminal_output_to_txt = False
        config.experiment.logging.log_tb = False
        config.experiment.logging.log_wandb = False
        config.experiment.save.enabled = True
        config.experiment.save.every_n_epochs = 1
        config.experiment.save.on_best_validation = False
        config.experiment.render = False
        config.experiment.render_video = False
        config.experiment.rollout.enabled = False
        config.experiment.epoch_every_n_steps = TRAIN_STEPS
        config.experiment.validation_epoch_every_n_steps = VALIDATION_STEPS
        config.train.data = [{"path": str(dataset), "eval": False}]
        config.train.output_dir = str(train_root)
        config.train.num_data_workers = 0
        config.train.hdf5_cache_mode = "all"
        config.train.hdf5_filter_key = "train"
        config.train.hdf5_validation_filter_key = "valid"
        config.train.batch_size = 32
        config.train.num_epochs = 1
        config.train.cuda = True
    config_path.write_text(config.dump() + "\n", encoding="utf-8")

    training = subprocess.run(
        [
            sys.executable,
            str(source_root / "robomimic" / "scripts" / "train.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    training_text = training.stdout + "\n" + training.stderr
    (output_dir / "training.log").write_text(training_text, encoding="utf-8")
    if (
        "finished run successfully!" not in training_text
        or "run failed with error:" in training_text
    ):
        raise RuntimeError("upstream robomimic training did not finish successfully")

    decoder = json.JSONDecoder()

    def metrics_after(marker: str) -> dict[str, object]:
        marker_pos = training_text.index(marker)
        json_pos = training_text.index("{", marker_pos)
        metrics, _ = decoder.raw_decode(training_text, json_pos)
        return metrics

    train_loss = float(metrics_after("Train Epoch 1")["Loss"])
    validation_loss = float(metrics_after("Validation Epoch 1")["Loss"])
    if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
        raise RuntimeError("training produced non-finite loss")
    checkpoints = list(
        train_root.glob("lift_ph_lowdim_bc_smoke/*/models/model_epoch_1.pth")
    )
    if len(checkpoints) != 1:
        raise RuntimeError("expected exactly one upstream epoch checkpoint")
    checkpoint_path = checkpoints[0]
    checkpoint_hash = _sha256(checkpoint_path)
    policy, checkpoint = policy_from_checkpoint(
        device=torch.device("cuda:0"), ckpt_path=str(checkpoint_path), verbose=False
    )
    optimizer_state = checkpoint["model"]["optimizers"]["policy"]["state"]
    serialized_steps = []
    for state in optimizer_state.values():
        value = state.get("step")
        if value is not None:
            serialized_steps.append(
                int(value.item() if hasattr(value, "item") else value)
            )
    optimizer_step_count = max(serialized_steps, default=0)
    if optimizer_step_count != TRAIN_STEPS:
        raise RuntimeError("serialized optimizer state does not prove four steps")

    heldout_demo = valid_keys[0]
    with h5py.File(dataset, "r") as handle:
        heldout_observation = {
            key: np.asarray(handle[f"data/{heldout_demo}/obs/{key}"][0])
            for key in policy.policy.global_config.all_obs_keys
        }
    policy.start_episode()
    action = np.asarray(policy(heldout_observation))
    finite = bool(np.isfinite(action).all())
    within_range = bool((action >= -1.0).all() and (action <= 1.0).all())
    if action.shape != (7,) or not finite or not within_range:
        raise RuntimeError("invalid held-out action from reloaded checkpoint")

    split_proof = {
        "algorithm": "robomimic.scripts.split_train_val seed=0 ratio=0.1",
        "train_trajectory_count": len(train_keys),
        "validation_trajectory_count": len(valid_keys),
        "train_sample_count": sum(sample_counts[key] for key in train_keys),
        "validation_sample_count": sum(sample_counts[key] for key in valid_keys),
        "overlap_count": len(overlap),
        "train_keys_sha256": hashlib.sha256(
            "\n".join(sorted(train_keys)).encode()
        ).hexdigest(),
        "validation_keys_sha256": hashlib.sha256(
            "\n".join(sorted(valid_keys)).encode()
        ).hexdigest(),
        "heldout_demo": heldout_demo,
    }
    result = {
        "schema": "npa.workbench.robomimic.smoke.v1",
        "solution": "robomimic",
        "capability": "lift_ph_lowdim_checkpoint_reload_action",
        "capabilities_exercised": CAPABILITIES,
        "source": {
            "repository": source_identity["repository"],
            "revision": SOURCE_REVISION,
            "observed_head": source_identity["observed_head"],
            "git_tree_sha1": source_identity["git_tree_sha1"],
            "tree_archive_sha256": source_identity["tree_archive_sha256"],
            "version": "0.5.0",
        },
        "boundaries": {
            "baked_runtime": "neutral-no-cuda",
            "pretrained_weights": False,
            "data_assets": "immutable-runtime-fetch",
            "runtime_cache": "external-read-only-prepopulated",
            "outputs": "run-scoped",
        },
        "dependency_lock": {
            "sha256": BAKED_LOCK_SHA256,
            "distribution_count": BAKED_DISTRIBUTION_COUNT,
            "accepted_sha256_count": len(
                re.findall(rb"--hash=sha256:[0-9a-f]{64}", baked_lock_bytes)
            ),
            "install_contract": "only-binary no-deps require-hashes",
        },
        "external_runtime": {
            "lock_sha256": _sha256(runtime_lock),
            "inventory_sha256": _sha256(runtime_root / "inventory.json"),
            "runtime_id": runtime_inventory["runtime_id"],
            "prepopulated": True,
            "manager_inventory_digest_matched": True,
            "read_only": runtime_mount_proof.get("read_only") is True,
            "atomic_private_snapshot_published": True,
            "snapshot_write_bits_absent": runtime_root.stat().st_mode & 0o222 == 0,
        },
        "dataset": {
            "repository": "robomimic/robomimic_datasets",
            "revision": DATASET_REVISION,
            "path": DATASET_PATH,
            "sha256": DATASET_SHA256,
            "size_bytes": DATASET_BYTES,
            "trajectory_count": len(demo_keys),
            "sample_count": sum(sample_counts.values()),
            "action_observed_range": [action_min, action_max],
        },
        "split": split_proof,
        "training": {
            "entrypoint": "robomimic/scripts/train.py",
            "algorithm": "bc",
            "optimizer": "adam",
            "optimizer_step_count": optimizer_step_count,
            "configured_optimizer_steps": TRAIN_STEPS,
            "configured_validation_forward_steps": VALIDATION_STEPS,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        },
        "checkpoint": {"sha256": checkpoint_hash, "reloaded": True},
        "heldout_action": {
            "demo": heldout_demo,
            "shape": list(action.shape),
            "dtype": str(action.dtype),
            "finite": finite,
            "allowed_range": [-1.0, 1.0],
            "within_allowed_range": within_range,
            "observed_min": float(action.min()),
            "observed_max": float(action.max()),
        },
        "hardware": hardware,
        **pod,
        "exit_status": 0,
        "deferred": [
            "public_image_acceptance",
            "image_policy_sweeps",
            "simulator_rollouts",
            "full_algorithm_matrix",
        ],
    }
    (output_dir / "robomimic-smoke.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
