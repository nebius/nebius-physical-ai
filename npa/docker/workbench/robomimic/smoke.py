#!/usr/bin/env python3
"""Deferred live gate: real upstream robomimic BC on official Lift PH data."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
import re
import ssl
import stat
import subprocess
import sys
import urllib.parse

import h5py
import numpy as np
import torch
from robomimic.config import config_factory
from robomimic.utils.file_utils import policy_from_checkpoint
from verify_image import verify_customer_runtime_entitlement, verified_source_identity


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


def _parsed_https_url(value: str) -> tuple[urllib.parse.SplitResult, str, int | None]:
    """Parse one HTTPS URL behind a value-free diagnostic boundary."""

    parsing_failed = False
    try:
        parsed = urllib.parse.urlsplit(value)
        hostname = (parsed.hostname or "").lower()
        port = parsed.port
    except (UnicodeError, ValueError):
        parsing_failed = True
    if parsing_failed:
        raise RuntimeError("refusing malformed approved HTTPS URL") from None
    return parsed, hostname, port


def _resolved_redirect_url(current_url: str, location: str) -> str:
    """Resolve one redirect without retaining a rejected authority."""

    resolution_failed = False
    try:
        resolved = urllib.parse.urljoin(current_url, location)
    except (UnicodeError, ValueError):
        resolution_failed = True
    if resolution_failed:
        raise RuntimeError("refusing malformed approved HTTPS redirect") from None
    return resolved


def _approved_request_target(parsed: urllib.parse.SplitResult) -> str:
    """Build an ASCII HTTP request target without retaining rejected values."""

    target_failed = False
    try:
        target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        target.encode("ascii")
    except (UnicodeError, ValueError):
        target_failed = True
    if target_failed:
        raise RuntimeError("approved HTTPS transport failed") from None
    return target


def _https_authority_allowed(
    parsed: urllib.parse.SplitResult,
    hostname: str,
    port: int | None,
    allowed_hosts: tuple[str, ...],
    allow_hf_redirects: bool,
) -> None:
    allowed = hostname in allowed_hosts or (
        allow_hf_redirects
        and (
            re.fullmatch(r"cdn-lfs(?:-[a-z0-9-]+)?\.hf\.co", hostname) is not None
            or hostname in {"cas-bridge.xethub.hf.co", "cdn.hf.co"}
            or hostname.endswith(".cdn.hf.co")
        )
    )
    if (
        parsed.scheme != "https"
        or not allowed
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.fragment
    ):
        raise RuntimeError("refusing URL outside the approved HTTPS origins")


def _https_get(
    hostname: str,
    port: int | None,
    target: str,
    headers: dict[str, str],
    context: ssl.SSLContext | None,
) -> tuple[http.client.HTTPSConnection, http.client.HTTPResponse]:
    connection = http.client.HTTPSConnection(
        hostname, port=port, context=context, timeout=120
    )
    try:
        connection.request("GET", target, headers=headers)
        return connection, connection.getresponse()
    except (OSError, http.client.HTTPException, UnicodeError, ValueError):
        connection.close()
        raise RuntimeError("approved HTTPS transport failed") from None


def _open_allowed_https(
    url: str,
    *,
    headers: dict[str, str],
    allowed_hosts: tuple[str, ...],
    allow_hf_redirects: bool = False,
    context: ssl.SSLContext | None = None,
    before_request: Callable[[], object] | None = None,
) -> tuple[http.client.HTTPSConnection, http.client.HTTPResponse]:
    """Open a tightly scoped HTTPS GET without urllib's multi-scheme opener."""
    current_url = url
    for _ in range(6):
        parsed, hostname, port = _parsed_https_url(current_url)
        _https_authority_allowed(
            parsed, hostname, port, allowed_hosts, allow_hf_redirects
        )
        if before_request is not None:
            before_request()
        connection, response = _https_get(
            hostname, port, _approved_request_target(parsed), headers, context
        )
        if response.status in _HTTPS_REDIRECT_STATUSES:
            location = response.getheader("Location")
            response.close()
            connection.close()
            if not location:
                raise RuntimeError("HTTPS redirect omitted its destination")
            current_url = _resolved_redirect_url(current_url, location)
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


def _value_free_customer_entitlement(**arguments: object) -> dict[str, object]:
    """Verify the customer record without retaining rejected private values."""

    refused = False
    try:
        proof = verify_customer_runtime_entitlement(**arguments)
    except Exception:
        refused = True
    if refused:
        raise RuntimeError("customer runtime entitlement refused")
    if not isinstance(proof, dict):
        raise RuntimeError("customer runtime entitlement proof is invalid")
    return proof


def _read_pod_status(
    service_account_root: Path, namespace: str, pod_name: str, token: str
) -> dict[str, object]:
    context = ssl.create_default_context(cafile=str(service_account_root / "ca.crt"))
    connection, response = _open_allowed_https(
        f"https://kubernetes.default.svc/api/v1/namespaces/{namespace}/pods/{pod_name}",
        headers={"Authorization": f"Bearer {token}"},
        allowed_hosts=("kubernetes.default.svc",),
        context=context,
    )
    try:
        return json.load(response)
    finally:
        response.close()
        connection.close()


def _validate_pod_identity(
    pod_status: dict[str, object],
    pod_name: str,
    expected_namespace: str,
    expected_service_account: str,
) -> str:
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
    return observed_service_account


def _validate_pod_container(
    pod_status: dict[str, object], expected_digest: str, runtime_image: str
) -> tuple[dict[str, object], dict[str, object]]:
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
    containers = [
        container
        for container in pod_status.get("spec", {}).get("containers", [])
        if container.get("name") == "ray-node"
    ]
    if len(containers) != 1 or containers[0].get("image") != runtime_image:
        raise RuntimeError(
            "executing ray-node container does not use the expected immutable image"
        )
    return matching_statuses[0], containers[0]


def _validate_pod_status(
    pod_status: dict[str, object],
    pod_name: str,
    expected_namespace: str,
    expected_service_account: str,
    expected_digest: str,
    runtime_image: str,
) -> tuple[dict[str, object], dict[str, object], str]:
    observed_service_account = _validate_pod_identity(
        pod_status, pod_name, expected_namespace, expected_service_account
    )
    status, container = _validate_pod_container(
        pod_status, expected_digest, runtime_image
    )
    return status, container, observed_service_account


def _pod_mount_proof(
    pod_status: dict[str, object], container: dict[str, object], runtime_root: Path
) -> None:
    runtime_mounts = [
        mount
        for mount in container.get("volumeMounts", [])
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


def _pod_selector() -> tuple[Path, str, str, str, str]:
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
            f"namespace_match={namespace == expected_namespace} service_account_set={bool(expected_service_account)}"
        )
    token = (service_account_root / "token").read_text(encoding="utf-8").strip()
    return service_account_root, pod_name, namespace, expected_namespace, token


def _pod_identity_result(
    runtime_image: str,
    expected_digest: str,
    runtime_root: Path,
    status: dict[str, object],
    observed_service_account: str,
    namespace: str,
) -> dict[str, object]:
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


def _pod_identity(
    runtime_image: str, expected_digest: str, runtime_root: Path
) -> dict[str, object]:
    service_account_root, pod_name, namespace, expected_namespace, token = (
        _pod_selector()
    )
    expected_service_account = os.environ.get(
        "NPA_ROBOMIMIC_EXPECTED_SERVICE_ACCOUNT", ""
    ).strip()
    pod_status = _read_pod_status(service_account_root, namespace, pod_name, token)
    status, container, observed_service_account = _validate_pod_status(
        pod_status,
        pod_name,
        expected_namespace,
        expected_service_account,
        expected_digest,
        runtime_image,
    )
    _pod_mount_proof(pod_status, container, runtime_root)
    return _pod_identity_result(
        runtime_image,
        expected_digest,
        runtime_root,
        status,
        observed_service_account,
        namespace,
    )


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
    }


def _require_strict_capacity_observation() -> None:
    """Refuse qualification until an authenticated allocation observer is wired.

    Pod GET access and local GPU observations cannot establish the provider's
    allocation policy. Neither an environment assertion nor a caller-authored
    receipt is an independent observation. A future observer must bind the run,
    Pod UID, scheduled node and actual provider allocation to STRICT capacity;
    this neutral candidate has no such observer and cannot emit a success proof.
    """
    raise RuntimeError(
        "STRICT capacity qualification deferred: authenticated run/Pod-bound "
        "provider allocation observation is unavailable"
    )


def _prepare_dataset_input_dir(
    input_dir: Path, partial: Path, authorize: Callable[[], object]
) -> None:
    authorize()
    try:
        input_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        details = input_dir.lstat()
        if (
            not stat.S_ISDIR(details.st_mode)
            or input_dir.is_symlink()
            or details.st_uid != os.geteuid()
            or stat.S_IMODE(details.st_mode) & 0o022
        ):
            raise RuntimeError("refusing an unsafe existing dataset input directory")
        unexpected = [path for path in input_dir.iterdir() if path != partial]
        if unexpected:
            raise RuntimeError("refusing to reuse a nonempty dataset input directory")
        try:
            partial_details = partial.lstat()
        except FileNotFoundError:
            pass
        else:
            if (
                not stat.S_ISREG(partial_details.st_mode)
                or partial.is_symlink()
                or partial_details.st_uid != os.geteuid()
                or partial_details.st_nlink != 1
            ):
                raise RuntimeError("refusing an unsafe stale dataset partial")
            authorize()
            partial.unlink()


def _fetch_dataset_bytes(
    partial: Path, authorize: Callable[[], object]
) -> tuple[str, int]:
    url = (
        "https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/"
        f"{DATASET_REVISION}/{DATASET_PATH}?download=true"
    )
    digest = hashlib.sha256()
    byte_count = 0
    authorize()
    connection, response = _open_allowed_https(
        url,
        headers={"User-Agent": "npa-robomimic-smoke/1"},
        allowed_hosts=("huggingface.co",),
        allow_hf_redirects=True,
        before_request=authorize,
    )
    try:
        authorize()
        with response, partial.open("xb") as handle:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                next_byte_count = byte_count + len(chunk)
                if next_byte_count > DATASET_BYTES:
                    raise RuntimeError("dataset exceeds its locked byte count")
                authorize()
                handle.write(chunk)
                digest.update(chunk)
                byte_count = next_byte_count
    finally:
        response.close()
        connection.close()
    return digest.hexdigest(), byte_count


def _inspect_dataset(
    partial: Path,
) -> tuple[list[str], dict[str, int], float, float]:
    with h5py.File(partial, "r") as handle:
        demo_keys = sorted(handle["data"].keys())
        sample_counts = {
            key: int(handle[f"data/{key}"].attrs["num_samples"]) for key in demo_keys
        }
        action_min = min(
            float(np.min(handle[f"data/{key}/actions"][:])) for key in demo_keys
        )
        action_max = max(
            float(np.max(handle[f"data/{key}/actions"][:])) for key in demo_keys
        )
    if len(demo_keys) != 200 or sum(sample_counts.values()) <= 0:
        raise RuntimeError("official Lift PH trajectory/sample inventory mismatch")
    return demo_keys, sample_counts, action_min, action_max


def _download_dataset(
    input_dir: Path,
    *,
    entitlement_arguments: dict[str, object],
) -> tuple[Path, list[str], dict[str, int], float, float]:
    # Freeze the exact expected hashes/run once; reread and verify the bound
    # record at each side-effect boundary, including redirected requests.
    bound_arguments = dict(entitlement_arguments)

    def authorize() -> dict[str, object]:
        return _value_free_customer_entitlement(**bound_arguments)

    partial = input_dir / "lift_ph_lowdim_v15.download"
    dataset = input_dir / "lift_ph_lowdim_v15.hdf5"
    _prepare_dataset_input_dir(input_dir, partial, authorize)
    try:
        digest, byte_count = _fetch_dataset_bytes(partial, authorize)
        if digest != DATASET_SHA256 or byte_count != DATASET_BYTES:
            raise RuntimeError(
                f"dataset identity mismatch: sha256={digest} bytes={byte_count}"
            )
        demo_keys, sample_counts, action_min, action_max = _inspect_dataset(partial)
        authorize()
        partial.replace(dataset)
        return dataset, demo_keys, sample_counts, action_min, action_max
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _load_image_context() -> dict[str, object]:
    runtime_image = os.environ.get("BYOF_IMAGE", "")
    match = re.fullmatch(r".+@(sha256:[0-9a-f]{64})", runtime_image)
    if match is None:
        raise RuntimeError("robomimic workload image must be digest-pinned")
    source_identity = verified_source_identity(
        Path("/opt/byof/npa_source_metadata.json")
    )
    if source_identity["revision"] != SOURCE_REVISION:
        raise RuntimeError("unexpected immutable robomimic source identity")
    component_root = Path("/opt/npa/robomimic")
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
    return {
        "runtime_image": runtime_image,
        "image_digest": match.group(1),
        "source_root": Path("/opt/robomimic"),
        "component_root": component_root,
        "source_identity": source_identity,
        "baked_lock_bytes": baked_lock_bytes,
    }


def _load_runtime_context(component_root: Path) -> dict[str, object]:
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
    inventory_path = runtime_root / "inventory.json"
    expected_inventory = os.environ.get(
        "NPA_ROBOMIMIC_RUNTIME_INVENTORY_SHA256", ""
    ).strip()
    if re.fullmatch(r"[0-9a-f]{64}", expected_inventory) is None:
        raise RuntimeError(
            "runtime inventory does not match the operator-selected digest"
        )
    if _sha256(inventory_path) != expected_inventory:
        raise RuntimeError(
            "runtime inventory does not match the operator-selected digest"
        )
    return {
        "runtime_lock": runtime_lock,
        "runtime_mount_root": runtime_mount_root,
        "runtime_root": runtime_root,
        "runtime_inventory": json.loads(inventory_path.read_text(encoding="utf-8")),
        "expected_inventory": expected_inventory,
    }


def _entitlement_arguments(
    runtime_lock: Path, expected_inventory: str
) -> dict[str, object]:
    return dict(
        entitlement_path=Path(os.environ["NPA_ROBOMIMIC_RUNTIME_ENTITLEMENT_FILE"]),
        runtime_lock_path=runtime_lock,
        expected_entitlement_sha256=os.environ.get(
            "NPA_ROBOMIMIC_RUNTIME_ENTITLEMENT_SHA256", ""
        ).strip(),
        expected_customer_binding_sha256=os.environ.get(
            "NPA_ROBOMIMIC_CUSTOMER_BINDING_SHA256", ""
        ).strip(),
        expected_run_id=os.environ.get("NPA_BYOF_RUN_ID", "").strip(),
        expected_inventory_sha256=expected_inventory,
    )


def _load_smoke_context(output_dir: Path) -> dict[str, object]:
    image = _load_image_context()
    runtime = _load_runtime_context(image["component_root"])
    arguments = _entitlement_arguments(
        runtime["runtime_lock"], runtime["expected_inventory"]
    )
    entitlement = _value_free_customer_entitlement(**arguments)
    pod = _pod_identity(
        image["runtime_image"], image["image_digest"], runtime["runtime_mount_root"]
    )
    mount_proof = pod.get("runtime_mount")
    if not isinstance(mount_proof, dict):
        raise RuntimeError("runtime mount observation is absent")
    return {
        **image,
        **runtime,
        "entitlement_arguments": arguments,
        "entitlement": entitlement,
        "pod": pod,
        "runtime_mount_proof": mount_proof,
        "hardware": _hardware(),
        "output_dir": output_dir,
    }


def _split_dataset(
    source_root: Path, dataset: Path, output_dir: Path, demo_keys: list[str]
) -> tuple[list[str], list[str], list[str]]:
    split_script = source_root / "robomimic" / "scripts" / "split_train_val.py"
    program = "\n".join(
        [
            "import numpy as np, runpy, sys",
            "script, dataset = sys.argv[1:3]",
            "np.random.seed(0)",
            "sys.argv = [script, '--dataset', dataset, '--ratio', '0.1']",
            "runpy.run_path(script, run_name='__main__')",
        ]
    )
    split = subprocess.run(
        [sys.executable, "-c", program, str(split_script), str(dataset)],
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
    return train_keys, valid_keys, overlap


def _write_training_config(dataset: Path, output_dir: Path) -> tuple[Path, Path]:
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
    return train_root, config_path


def _run_training(source_root: Path, output_dir: Path, config_path: Path) -> str:
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
    text = training.stdout + "\n" + training.stderr
    (output_dir / "training.log").write_text(text, encoding="utf-8")
    if "finished run successfully!" not in text or "run failed with error:" in text:
        raise RuntimeError("upstream robomimic training did not finish successfully")
    return text


def _training_losses(training_text: str) -> tuple[float, float]:
    decoder = json.JSONDecoder()

    def metrics_after(marker: str) -> dict[str, object]:
        marker_pos = training_text.index(marker)
        metrics, _ = decoder.raw_decode(
            training_text, training_text.index("{", marker_pos)
        )
        return metrics

    train_loss = float(metrics_after("Train Epoch 1")["Loss"])
    validation_loss = float(metrics_after("Validation Epoch 1")["Loss"])
    if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
        raise RuntimeError("training produced non-finite loss")
    return train_loss, validation_loss


def _load_checkpoint_state(
    train_root: Path,
) -> tuple[str, object, int]:
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
    serialized_steps = [
        int(value.item() if hasattr(value, "item") else value)
        for state in checkpoint["model"]["optimizers"]["policy"]["state"].values()
        if (value := state.get("step")) is not None
    ]
    optimizer_steps = max(serialized_steps, default=0)
    if optimizer_steps != TRAIN_STEPS:
        raise RuntimeError("serialized optimizer state does not prove four steps")

    return checkpoint_hash, policy, optimizer_steps


def _heldout_action(
    policy: object, dataset: Path, valid_keys: list[str]
) -> tuple[str, object, bool, bool]:
    heldout_demo = valid_keys[0]
    with h5py.File(dataset, "r") as handle:
        observation = {
            key: np.asarray(handle[f"data/{heldout_demo}/obs/{key}"][0])
            for key in policy.policy.global_config.all_obs_keys
        }
    policy.start_episode()
    action = np.asarray(policy(observation))
    finite = bool(np.isfinite(action).all())
    within_range = bool((action >= -1.0).all() and (action <= 1.0).all())
    if action.shape != (7,) or not finite or not within_range:
        raise RuntimeError("invalid held-out action from reloaded checkpoint")
    return heldout_demo, action, finite, within_range


def _reload_checkpoint(
    train_root: Path, dataset: Path, valid_keys: list[str], training_text: str
) -> dict[str, object]:
    train_loss, validation_loss = _training_losses(training_text)
    checkpoint_hash, policy, optimizer_steps = _load_checkpoint_state(train_root)
    heldout_demo, action, finite, within_range = _heldout_action(
        policy, dataset, valid_keys
    )
    return {
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "checkpoint_hash": checkpoint_hash,
        "optimizer_steps": optimizer_steps,
        "heldout_demo": heldout_demo,
        "action": action,
        "finite": finite,
        "within_range": within_range,
    }


def _split_proof(
    train_keys: list[str],
    valid_keys: list[str],
    overlap: list[str],
    sample_counts: dict[str, int],
    heldout_demo: str,
) -> dict[str, object]:
    return {
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


def _result_source(context: dict[str, object]) -> dict[str, object]:
    source_identity = context["source_identity"]
    return {
        "repository": source_identity["repository"],
        "revision": SOURCE_REVISION,
        "observed_head": source_identity["observed_head"],
        "git_tree_sha1": source_identity["git_tree_sha1"],
        "tree_archive_sha256": source_identity["tree_archive_sha256"],
        "version": "0.5.0",
    }


def _result_runtime(context: dict[str, object]) -> dict[str, object]:
    runtime_inventory = context["runtime_inventory"]
    return {
        "external_runtime": {
            "lock_sha256": _sha256(context["runtime_lock"]),
            "inventory_sha256": _sha256(context["runtime_root"] / "inventory.json"),
            "runtime_id": runtime_inventory["runtime_id"],
            "prepopulated": True,
            "runtime_manifest_digest_matched": True,
            "read_only": context["runtime_mount_proof"].get("read_only") is True,
            "atomic_private_snapshot_published": True,
            "snapshot_write_bits_absent": context["runtime_root"].stat().st_mode & 0o222
            == 0,
        },
    }


def _result_dataset(
    demo_keys: list[str],
    sample_counts: dict[str, int],
    action_range: tuple[float, float],
) -> dict[str, object]:
    return {
        "dataset": {
            "repository": "robomimic/robomimic_datasets",
            "revision": DATASET_REVISION,
            "path": DATASET_PATH,
            "sha256": DATASET_SHA256,
            "size_bytes": DATASET_BYTES,
            "trajectory_count": len(demo_keys),
            "sample_count": sum(sample_counts.values()),
            "action_observed_range": list(action_range),
        },
    }


def _result_training(training: dict[str, object]) -> dict[str, object]:
    return {
        "training": {
            "entrypoint": "robomimic/scripts/train.py",
            "algorithm": "bc",
            "optimizer": "adam",
            "optimizer_step_count": training["optimizer_steps"],
            "configured_optimizer_steps": TRAIN_STEPS,
            "configured_validation_forward_steps": VALIDATION_STEPS,
            "train_loss": training["train_loss"],
            "validation_loss": training["validation_loss"],
        },
        "checkpoint": {"sha256": training["checkpoint_hash"], "reloaded": True},
    }


def _result_action(training: dict[str, object]) -> dict[str, object]:
    action = training["action"]
    return {
        "heldout_action": {
            "demo": training["heldout_demo"],
            "shape": list(action.shape),
            "dtype": str(action.dtype),
            "finite": training["finite"],
            "allowed_range": [-1.0, 1.0],
            "within_allowed_range": training["within_range"],
            "observed_min": float(action.min()),
            "observed_max": float(action.max()),
        },
    }


def _result_header(context: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "npa.workbench.robomimic.smoke.v1",
        "solution": "robomimic",
        "capability": "lift_ph_lowdim_checkpoint_reload_action",
        "capabilities_exercised": CAPABILITIES,
        "source": _result_source(context),
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
                re.findall(rb"--hash=sha256:[0-9a-f]{64}", context["baked_lock_bytes"])
            ),
            "install_contract": "only-binary no-deps require-hashes",
        },
    }


def _build_result(
    context: dict[str, object],
    demo_keys: list[str],
    sample_counts: dict[str, int],
    action_range: tuple[float, float],
    split_proof: dict[str, object],
    training: dict[str, object],
) -> dict[str, object]:
    return {
        **_result_header(context),
        **_result_runtime(context),
        "customer_runtime_entitlement": context["entitlement"],
        **_result_dataset(demo_keys, sample_counts, action_range),
        "split": split_proof,
        **_result_training(training),
        **_result_action(training),
        "hardware": context["hardware"],
        **context["pod"],
        "exit_status": 0,
        "deferred": [
            "public_image_acceptance",
            "image_policy_sweeps",
            "simulator_rollouts",
            "full_algorithm_matrix",
        ],
    }


def main() -> None:
    _require_strict_capacity_observation()
    output_dir = Path(os.environ["NPA_SMOKE_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    context = _load_smoke_context(output_dir)
    dataset, demo_keys, sample_counts, action_min, action_max = _download_dataset(
        Path("/workspace/byof-inputs") / output_dir.name,
        entitlement_arguments=context["entitlement_arguments"],
    )
    train_keys, valid_keys, overlap = _split_dataset(
        context["source_root"], dataset, output_dir, demo_keys
    )
    train_root, config_path = _write_training_config(dataset, output_dir)
    training_text = _run_training(context["source_root"], output_dir, config_path)
    training = _reload_checkpoint(train_root, dataset, valid_keys, training_text)
    result = _build_result(
        context,
        dataset,
        demo_keys,
        sample_counts,
        (action_min, action_max),
        _split_proof(
            train_keys, valid_keys, overlap, sample_counts, training["heldout_demo"]
        ),
        training,
    )
    (output_dir / "robomimic-smoke.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
