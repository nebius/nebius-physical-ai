#!/usr/bin/env python3
"""Materialize the pinned LIBERO runtime into an operator-owned cache.

The public image contains this fetcher and immutable manifests, not LIBERO,
PyTorch/CUDA wheels, demonstrations, task assets, models, or populated caches.
Download success is never treated as permission: ``ensure`` refuses before the
first network or cache effect unless a customer/run authorization is present,
hash-bound, and signed directly by a customer-controlled key. The NPA control
plane authenticates the caller and transports and validates the evidence; it
does not accept or sign the customer's terms assertion.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import ctypes
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import errno
import fcntl
import grp
import hashlib
import hmac
import http.client
import json
import math
import os
import pwd
import re
import shutil
import signal
import shlex
import ssl
import stat
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
from uuid import uuid4
from pathlib import Path
from typing import Any
import zipfile

_Popen = subprocess.Popen

PAYLOAD_RELEASE_ANNOTATION = "npa.nebius.com/libero-release"

SCHEMA = "npa.libero.runtime-manifest.v1"
CUSTOMER_AUTHORIZATION_SCHEMA = "npa.libero.customer-runtime-authorization.v2"
AUTHENTICATED_CALLER_SCHEMA = "npa.libero.authenticated-caller.v1"
OUTPUT_STORAGE_AUTHORIZATION_SCHEMA = "npa.libero.output-storage-authorization.v3"
COMPLETE_SCHEMA = "npa.libero.runtime-cache.v1"
INVENTORY_SCHEMA = "npa.libero.runtime-cache-inventory.v1"
EXPECTED_MANIFEST_KEYS = frozenset(
    {
        "boundaries",
        "demonstration",
        "governing_terms",
        "language_model",
        "runtime_artifact_count",
        "runtime_artifacts",
        "runtime_python",
        "customer_runtime_authorization",
        "schema",
        "solution",
        "source",
        "task",
    }
)
EXPECTED_RUNTIME_PYTHON = {
    "version": "3.10",
    "abi": "cp310",
    "platform": "linux_x86_64",
}
EXPECTED_CUSTOMER_AUTHORIZATION_METADATA = {
    "schema": CUSTOMER_AUTHORIZATION_SCHEMA,
    "required": True,
    "credentials_establish_acceptance": False,
}
EXPECTED_BOUNDARIES = {
    "cache": "operator-owned non-root atomic runtime cache",
    "outputs": "NPA_SMOKE_OUTPUT_DIR only",
    "rendering": False,
}
EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "d999dd97e8f9b324b68ec2a7f19b6360f5599868cd873cd752779106b8ea4f02"
)
EXPECTED_RUNTIME_REQUIREMENTS_SHA256 = (
    "8504f236dcad67ad0e2f5959b916c93aa7ccbd02567c6e323e480366d0f23b99"
)
DEFAULT_MANIFEST = Path("/opt/npa/libero/runtime-manifest.json")
DEFAULT_REQUIREMENTS = Path("/opt/npa/libero/runtime-requirements.txt")
DEFAULT_CACHE = Path("/workspace/.cache/npa/libero")
NVIDIA_TERMS_NORMALIZATION = "nvidia-navigation-uuid-v1"
NVIDIA_SOFTWARE_TERMS_URL = (
    "https://www.nvidia.com/en-us/agreements/enterprise-software/"
    "nvidia-software-license-agreement/"
)
OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY = Path(
    "/opt/npa/libero/output-storage-authorization-public-key.b64"
)
OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY_OWNER_UID = 0
AUTHENTICATED_CALLER_TRUST_ROOT = Path(
    "/run/npa/libero/authenticated-caller-public-key.b64"
)
AUTHENTICATED_CALLER_TRUST_ROOT_OWNER_UID = 0
# Customer signer roots are provisioned outside the control-plane assertion
# path.  A missing or mutable registration is a hard refusal before cache or
# network effects; the caller assertion cannot select or create this root.
CUSTOMER_SIGNER_REGISTRY_ROOT = Path("/run/npa/libero/customer-signer-roots")
CUSTOMER_SIGNER_REGISTRY_OWNER_UID = 0
CUSTOMER_RUN_PUBLIC_KEY = Path("/run/npa/libero/customer-run-public-key.b64")
CUSTOMER_RUN_MODE = "customer-run-v1"
CUSTOMER_RUN_PHASE_ROOT = Path("/workspace/byof-runs")


def _customer_run() -> bool:
    """Select only the explicit customer-operated transport, never acceptance."""
    mode = os.environ.get("NPA_LIBERO_RUNTIME_DELIVERY", "")
    if mode not in {"", CUSTOMER_RUN_MODE}:
        raise BootstrapRefusal("unsupported LIBERO runtime delivery")
    return mode == CUSTOMER_RUN_MODE


def _customer_run_key() -> bytes:
    return _trusted_public_key(
        CUSTOMER_RUN_PUBLIC_KEY, owner_uid=0, label="customer-run signer"
    )

CUSTOMER_AUTHORIZATION_NAMESPACE = b"npa.libero.customer-authorization"
AUTHENTICATED_CALLER_NAMESPACE = b"npa.libero.authenticated-caller"
OUTPUT_STORAGE_AUTHORIZATION_NAMESPACE = b"npa.libero.output-storage-authorization"
ALLOWED_DOWNLOAD_HOSTS = frozenset(
    {
        "files.pythonhosted.org",
        "download-r2.pytorch.org",
        "huggingface.co",
    }
)
ALLOWED_DOWNLOAD_REDIRECT_HOSTS = frozenset(
    {
        "cdn-lfs.huggingface.co",
        "cdn-lfs-us-1.hf.co",
        "cdn-lfs-eu-1.hf.co",
        "cas-bridge.xethub.hf.co",
        "us.aws.cdn.hf.co",
        "us.gcp.cdn.hf.co",
    }
)
SEALED_DIRECTORY_MODE = stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP
SEALED_EXECUTABLE_MODE = SEALED_DIRECTORY_MODE
SEALED_REGULAR_MODE = stat.S_IRUSR | stat.S_IRGRP
CACHE_ROOT_MODE = SEALED_DIRECTORY_MODE | stat.S_IWUSR
INHERITED_CACHE_DESCRIPTOR = 200
INHERITED_AUTHORIZATION_DESCRIPTOR = 201
INHERITED_EXECUTION_LOCK_DESCRIPTOR = 202
INHERITED_BOOTSTRAP_LOCK_DESCRIPTOR = 203
INHERITED_CALLER_DESCRIPTOR = 204
INHERITED_CUSTOMER_TRUST_DESCRIPTOR = 205
PROTECTED_AUTHORIZATION_NAME = ".npa-customer-authorization.json"
ALLOWED_TERMS_HOSTS = frozenset(
    {
        "raw.githubusercontent.com",
        "creativecommons.org",
        "www.apache.org",
        "docs.nvidia.com",
        "www.nvidia.com",
    }
)
EXPECTED_GOVERNING_TERMS = frozenset(
    {
        "libero-mit",
        "dataset-cc-by-4.0",
        "bert-apache-2.0",
        "pytorch-bsd",
        "cuda-eula",
        "nvidia-software-license",
        "cudnn-eula",
    }
)
MAX_RUNTIME_CACHE_DOWNLOAD_BYTES = 32 * 1024 * 1024 * 1024
REVIEWED_SOURCE_BUILD_NAMES = frozenset(
    {
        "antlr4-python3-runtime",
        "bddl",
        "easydict",
        "egl-probe",
        "future",
        "glfw",
        "gym",
        "pathtools",
        "promise",
        "robomimic",
    }
)
SOURCE_BUILD_UNSHARE = Path("/usr/bin/unshare")
SOURCE_BUILD_CHROOT = Path("/usr/sbin/chroot")
RUNTIME_EXECUTION_GROUP = "npa-libero-exec"
RUNTIME_EXECUTION_USER = "npa-libero-exec"
RUNTIME_SUPERVISOR_USER = "ubuntu"
EXECUTABLE_PROFILE_ROOT = Path("/workspace/byof-runs")
EXECUTABLE_PROFILE_NAME = ".npa-executable-profile.json"
STORAGE_SECRET_ENV_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_B64",
        "NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64",
    }
)
RUNTIME_MATERIALIZATION_PASSTHROUGH_ENV_NAMES = frozenset(
    {
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "SOURCE_DATE_EPOCH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TMPDIR",
        "TZ",
    }
)
RUNTIME_EXECUTION_PASSTHROUGH_ENV_NAMES = frozenset(
    {
        "BYOF_CAPABILITY_NAME",
        "BYOF_IMAGE",
        "BYOF_REPO_ROOT",
        "BYOF_SMOKE_ARTIFACT_NAME",
        "BYOF_SMOKE_COMMAND",
        "BYOF_SOLUTION_NAME",
        "CUDA_VISIBLE_DEVICES",
        "HOSTNAME",
        "KUBERNETES_SERVICE_HOST",
        "KUBERNETES_SERVICE_PORT",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "NPA_BYOF_RUN_ID",
        "NPA_LIBERO_RUNTIME_DELIVERY",
        "NPA_LIBERO_EXPECTED_CUSTOMER_AUTHORIZATION_EXPIRES_AT",
        "NPA_LIBERO_EXPECTED_CUSTOMER_SIGNER_PUBLIC_KEY_SHA256",
        "NPA_LIBERO_EXPECTED_EXECUTABLE_PROFILE_SHA256",
        "NPA_LIBERO_EXPECTED_ALLOWED_NODE_SHA256",
        "NPA_LIBERO_EXPECTED_CANONICAL_BUILD_METADATA_SHA256",
        "NPA_LIBERO_EXPECTED_CLUSTER_IDENTITY_SHA256",
        "NPA_LIBERO_EXPECTED_EXECUTION_KUBECONFIG_SHA256",
        "NPA_LIBERO_EXPECTED_EXTERNAL_RBAC_INVENTORY_SHA256",
        "NPA_LIBERO_EXPECTED_INFRASTRUCTURE_BUNDLE_SHA256",
        "NPA_LIBERO_EXPECTED_NAMESPACE_INVENTORY_SHA256",
        "NPA_LIBERO_EXPECTED_NAMESPACE_SHA256",
        "NPA_LIBERO_EXPECTED_NAMESPACE_UID_SHA256",
        "NPA_LIBERO_EXPECTED_OUTPUT_LEASE_ETAG",
        "NPA_LIBERO_EXPECTED_OUTPUT_LEASE_KEY",
        "NPA_LIBERO_AUTHENTICATED_CALLER_SHA256",
        "NPA_LIBERO_EXPECTED_OUTPUT_LEASE_VERSION_ID",
        "NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_AUTHORIZATION_SHA256",
        "NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_POLICY_SHA256",
        "NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_PREFIX_SHA256",
        "NPA_LIBERO_EXPECTED_PAYLOAD_KUBECONFIG_SHA256",
        "NPA_LIBERO_EXPECTED_PUBLICATION_BUNDLE_SHA256",
        "NPA_LIBERO_EXPECTED_RBAC_SPEC_SHA256",
        "NPA_LIBERO_EXPECTED_ROLE_BINDING_UID_SHA256",
        "NPA_LIBERO_EXPECTED_ROLE_UID_SHA256",
        "NPA_LIBERO_EXPECTED_SERVICE_ACCOUNT_UID_SHA256",
        "NPA_LIBERO_EXPECTED_SKYPILOT_CONFIG_SHA256",
        "NPA_LIBERO_BOOTSTRAP_RECEIPT",
        "NPA_LIBERO_CUSTOMER_AUTHORIZATION_SHA256",
        "NPA_LIBERO_CUSTOMER_IDENTITY_SHA256",
        "NPA_SMOKE_OUTPUT_DIR",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "PATH",
        "TZ",
    }
)
OUTPUT_ARTIFACT_SIZE_LIMITS = {
    "libero-bc-rnn-smoke.pth": 256 * 1024 * 1024,
    "libero-smoke.json": 8 * 1024 * 1024,
    "npa_byof_summary.json": 1024 * 1024,
    "npa_runtime_bootstrap.json": 8 * 1024 * 1024,
    "npa_runtime_metadata.json": 8 * 1024 * 1024,
    "nvidia_smi.txt": 1024 * 1024,
    "nvidia_smi_list.txt": 1024 * 1024,
    "solution_smoke_stderr.log": 16 * 1024 * 1024,
    "solution_smoke_stdout.log": 16 * 1024 * 1024,
}
# Only canonical, schema-validated evidence may cross the output boundary.
# Checkpoints, logs, and GPU probe text remain local to the owner-controlled
# staging directory and are never uploaded as workload output.
OUTPUT_UPLOAD_ARTIFACT_NAMES = frozenset(
    {
        "libero-smoke.json",
        "npa_byof_summary.json",
        "npa_runtime_bootstrap.json",
        "npa_runtime_metadata.json",
    }
)
OUTPUT_UPLOAD_SIZE_LIMITS = {
    name: OUTPUT_ARTIFACT_SIZE_LIMITS[name] for name in OUTPUT_UPLOAD_ARTIFACT_NAMES
}
OUTPUT_SIZE_LIMITS = OUTPUT_ARTIFACT_SIZE_LIMITS
FAILURE_OUTPUT_REQUIRED_SIZE_LIMITS = {
    name: limit
    for name, limit in OUTPUT_ARTIFACT_SIZE_LIMITS.items()
    if name not in {"libero-bc-rnn-smoke.pth", "libero-smoke.json"}
}
FAILURE_OUTPUT_OPTIONAL_SIZE_LIMITS = {
    name: OUTPUT_ARTIFACT_SIZE_LIMITS[name]
    for name in ("libero-bc-rnn-smoke.pth", "libero-smoke.json")
}
MAX_OUTPUT_BYTES = 320 * 1024 * 1024
OUTPUT_RECEIPT_NAME = "npa_upload_receipt.json"
OUTPUT_RECEIPT_SCHEMA = "npa.libero.s3-upload-readback.v2"

_OUTPUT_JSON_TOP_LEVEL_KEYS = {
    "libero-smoke.json": frozenset(
        {
            "boundaries",
            "build",
            "capabilities_exercised",
            "capability",
            "checkpoint",
            "dataset",
            "deferred",
            "error",
            "exit_status",
            "heldout_metrics",
            "reloaded_action",
            "runtime",
            "sample_count",
            "schema",
            "solution",
            "source",
            "split",
            "status",
            "task_assets",
            "task_language_model",
            "training",
        }
    ),
    "npa_byof_summary.json": frozenset(
        {
            "capability_name",
            "created_unix",
            "image",
            "run_id",
            "runtime_cache_uploaded",
            "rendering_invoked",
            "smoke_artifact_name",
            "smoke_exit_code",
            "solution_name",
            "status",
            "tool",
            "workload",
        }
    ),
    "npa_runtime_bootstrap.json": frozenset(
        {
            "cache_path",
            "cache_uploaded",
            "content_inventory_entry_count",
            "content_inventory_sha256",
            "customer_authorization_sha256",
            "customer_identity_sha256",
            "demonstration_sha256",
            "demonstration_size_bytes",
            "governing_terms_count",
            "governing_terms_fetched_this_invocation",
            "governing_terms_sha256",
            "git_objects_present",
            "language_model_revision",
            "manifest_sha256",
            "render_assets_present",
            "run_id",
            "runtime_artifact_count",
            "runtime_requirements_sha256",
            "schema",
            "solution",
            "source_license_sha256",
            "source_revision",
            "source_tree",
            "status",
            "warm_reuse",
        }
    ),
    "npa_runtime_metadata.json": frozenset(
        {
            "cache_uploaded",
            "content_inventory_entry_count",
            "content_inventory_sha256",
            "customer_authorization_sha256",
            "customer_identity_sha256",
            "demonstration_sha256",
            "demonstration_size_bytes",
            "governing_terms_count",
            "governing_terms_sha256",
            "git_objects_present",
            "language_model_revision",
            "manifest_sha256",
            "render_assets_present",
            "run_id",
            "runtime_artifact_count",
            "runtime_requirements_sha256",
            "schema",
            "solution",
            "source_license_sha256",
            "source_revision",
            "source_tree",
            "task_bddl_sha256",
            "task_initial_states_sha256",
        }
    ),
}
_OUTPUT_JSON_REQUIRED_KEYS = {
    "libero-smoke.json": frozenset(
        {"schema", "status", "exit_status", "solution", "capability", "source", "dataset"}
    ),
    "npa_byof_summary.json": frozenset(
        {"status", "tool", "workload", "run_id", "smoke_exit_code"}
    ),
    "npa_runtime_bootstrap.json": frozenset(
        {"schema", "solution", "status", "run_id", "manifest_sha256"}
    ),
    "npa_runtime_metadata.json": frozenset(
        {"schema", "solution", "manifest_sha256", "run_id"}
    ),
}
_EMBEDDED_OUTPUT_KEYS = frozenset(
    {"payload", "raw", "raw_bytes", "source_bytes", "checkpoint_bytes", "artifact_bytes"}
)


def _output_object(
    fields: dict[str, tuple], required: frozenset[str]
) -> tuple[str, dict[str, tuple], frozenset[str]]:
    return ("object", fields, required)


_STRING = ("string",)
_BOOL = ("bool",)
_INTEGER = ("integer",)
_NUMBER = ("number",)
_HEX64 = ("hex64",)
_SMOKE_SOURCE = _output_object(
    {
        key: _STRING
        for key in (
            "repository",
            "revision",
            "license",
            "observed_revision",
            "source_prune_path",
        )
    }
    | {
        key: _BOOL
        for key in ("source_prune_path_absent", "git_objects_absent", "layer_scan_required_before_live_use")
    },
    frozenset({"repository", "revision", "license"}),
)
_SMOKE_DATASET = _output_object(
    {
        key: _STRING
        for key in (
            "repository",
            "revision",
            "suite",
            "task",
            "task_description",
            "url",
            "license",
            "attribution",
            "dataset_bddl_path",
        )
    }
    | {
        key: _HEX64 for key in ("expected_sha256", "observed_sha256")
    }
    | {
        key: _INTEGER
        for key in ("expected_size_bytes", "observed_size_bytes", "demo_count", "sample_count")
    }
    | {"downloaded_this_run": _BOOL},
    frozenset(
        {
            "repository",
            "revision",
            "suite",
            "task",
            "task_description",
            "url",
            "expected_sha256",
            "expected_size_bytes",
            "license",
            "attribution",
        }
    ),
)
_LANGUAGE_MODEL_FILE = _output_object(
    {"expected_size_bytes": _INTEGER, "expected_sha256": _HEX64, "observed_sha256": _HEX64},
    frozenset({"expected_size_bytes", "expected_sha256", "observed_sha256"}),
)
_SMOKE_LANGUAGE_MODEL = _output_object(
    {
        "repository": _STRING,
        "revision": _STRING,
        "license": _STRING,
        "delivery": _STRING,
        "source_path": _STRING,
        "source_sha256": _HEX64,
        "files": ("map", _LANGUAGE_MODEL_FILE, 16),
        "embedding_method": _STRING,
        "embedding_shape": ("integer_array", 8),
        "embedding_dtype": _STRING,
        "embedding_finite": _BOOL,
        "downloaded_this_run": _BOOL,
        "cache_uploaded": _BOOL,
    },
    frozenset({"repository", "revision", "license", "delivery"}),
)
_SMOKE_TASK_ASSETS = _output_object(
    {"bddl_path": _STRING, "bddl_sha256": _HEX64, "initial_states_path": _STRING, "initial_states_sha256": _HEX64, "source_license": _STRING},
    frozenset({"bddl_path", "bddl_sha256", "initial_states_path", "initial_states_sha256", "source_license"}),
)
_SMOKE_SPLIT = _output_object(
    {
        "strategy": _STRING,
        "seed": _INTEGER,
        "disjoint": _BOOL,
        "train_demo_count": _INTEGER,
        "heldout_demo_count": _INTEGER,
        "train_demo_ids_sha256": _HEX64,
        "heldout_demo_ids_sha256": _HEX64,
        "train_sample_count": _INTEGER,
        "heldout_sample_count": _INTEGER,
    },
    frozenset(
        {
            "strategy", "seed", "disjoint", "train_demo_count", "heldout_demo_count",
            "train_demo_ids_sha256", "heldout_demo_ids_sha256", "train_sample_count", "heldout_sample_count",
        }
    ),
)
_SMOKE_TRAINING = _output_object(
    {
        "algorithm": _STRING,
        "optimizer": _STRING,
        "optimizer_steps": _INTEGER,
        "requested_optimizer_steps": _INTEGER,
        "parameter_max_abs_delta": _NUMBER,
        "first_loss": _NUMBER,
        "final_loss": _NUMBER,
        "all_losses_finite": _BOOL,
        "sequence_length": _INTEGER,
        "task_embedding": _STRING,
    },
    frozenset({"algorithm", "optimizer", "optimizer_steps", "requested_optimizer_steps", "parameter_max_abs_delta", "first_loss", "final_loss", "all_losses_finite", "sequence_length", "task_embedding"}),
)
_SMOKE_HELDOUT = _output_object(
    {"negative_log_likelihood": _NUMBER, "evaluated_sample_count": _INTEGER, "partition": _STRING},
    frozenset({"negative_log_likelihood", "evaluated_sample_count", "partition"}),
)
_SMOKE_CHECKPOINT = _output_object(
    {"file": _STRING, "sha256": _HEX64, "saved_with": _STRING, "reloaded_with": _STRING, "strict_state_dict_load": _BOOL},
    frozenset({"file", "sha256", "saved_with", "reloaded_with", "strict_state_dict_load"}),
)
_SMOKE_RELOADED = _output_object(
    {"shape": ("integer_array", 8), "dtype": _STRING, "finite": _BOOL, "prediction_sha256": _HEX64, "value_min": _NUMBER, "value_max": _NUMBER, "evaluated_sample_count": _INTEGER},
    frozenset({"shape", "dtype", "finite", "prediction_sha256", "value_min", "value_max", "evaluated_sample_count"}),
)
_SMOKE_RUNTIME = _output_object(
    {
        "gpu_model": _STRING,
        "gpu_architecture": _STRING,
        "compute_capability": ("integer_array", 2),
        "gpu_count": _INTEGER,
        "torch_cuda_arch_list": ("string_array", 32),
        "nvidia_smi": ("string_array", 8),
        "pod_observed_image_digest": _STRING,
        "observation_method": _STRING,
        "pod_name_sha256": _HEX64,
        "namespace_sha256": _HEX64,
        "pod_uid_sha256": _HEX64,
        "node_name_sha256": _HEX64,
        "actual_service_account": _STRING,
        "service_account_uid_sha256": _HEX64,
        "controller_service_account_separated": _BOOL,
        "host_architecture": _STRING,
    },
    frozenset({"gpu_model", "gpu_architecture", "compute_capability", "gpu_count", "torch_cuda_arch_list", "nvidia_smi"}),
)
_SMOKE_BUILD = _output_object(
    {
        "runtime_metadata": ("runtime_metadata",),
        "accepted_canonical_build_metadata_sha256": _HEX64,
        "base_image_digest": _STRING,
        "base_rootfs_material_digest": _STRING,
        "base_image_digest_pinned": _BOOL,
        "base_image_provenance": _STRING,
        "dataset_delivery": _STRING,
        "weights_delivery": _STRING,
        "render_assets_present_in_final_filesystem": _BOOL,
        "render_assets_removed_path": _STRING,
        "git_objects_present_in_final_filesystem": _BOOL,
        "independent_oci_layer_scan_required_before_live_use": _BOOL,
    },
    frozenset({"runtime_metadata", "accepted_canonical_build_metadata_sha256", "base_image_digest", "base_rootfs_material_digest", "base_image_digest_pinned", "base_image_provenance", "dataset_delivery", "weights_delivery", "render_assets_present_in_final_filesystem", "render_assets_removed_path", "git_objects_present_in_final_filesystem", "independent_oci_layer_scan_required_before_live_use"}),
)
_SMOKE_BOUNDARIES = _output_object(
    {"cache": _STRING, "output": _STRING, "cache_uploaded": _BOOL, "rendering_invoked": _BOOL},
    frozenset({"cache", "output", "cache_uploaded", "rendering_invoked"}),
)
_RUNTIME_METADATA_FIELDS = {
    "schema": _STRING,
    "solution": _STRING,
    "manifest_sha256": _HEX64,
    "customer_authorization_sha256": _HEX64,
    "customer_identity_sha256": _HEX64,
    "run_id": _STRING,
    "runtime_requirements_sha256": _HEX64,
    "governing_terms_sha256": _HEX64,
    "governing_terms_count": _INTEGER,
    "source_revision": _STRING,
    "source_tree": _HEX64,
    "source_license_sha256": _HEX64,
    "runtime_artifact_count": _INTEGER,
    "demonstration_sha256": _HEX64,
    "demonstration_size_bytes": _INTEGER,
    "task_bddl_sha256": _HEX64,
    "task_initial_states_sha256": _HEX64,
    "language_model_revision": _STRING,
    "render_assets_present": _BOOL,
    "git_objects_present": _BOOL,
    "cache_uploaded": _BOOL,
    "content_inventory_sha256": _HEX64,
    "content_inventory_entry_count": _INTEGER,
}
_RUNTIME_METADATA_SCHEMA = _output_object(_RUNTIME_METADATA_FIELDS, frozenset(_RUNTIME_METADATA_FIELDS))
_RUNTIME_BOOTSTRAP_FIELDS = {
    **_RUNTIME_METADATA_FIELDS,
    "status": _STRING,
    "cache_path": _STRING,
    "governing_terms_fetched_this_invocation": _BOOL,
    "warm_reuse": _BOOL,
}
_RUNTIME_BOOTSTRAP_SCHEMA = _output_object(
    _RUNTIME_BOOTSTRAP_FIELDS,
    frozenset({"schema", "solution", "run_id", "manifest_sha256"}),
)
_OUTPUT_SCHEMAS = {
    "libero-smoke.json": _output_object(
        {
            "schema": _STRING,
            "status": _STRING,
            "exit_status": _INTEGER,
            "solution": _STRING,
            "capability": _STRING,
            "capabilities_exercised": ("string_array", 16),
            "source": _SMOKE_SOURCE,
            "dataset": _SMOKE_DATASET,
            "task_language_model": _SMOKE_LANGUAGE_MODEL,
            "sample_count": _INTEGER,
            "task_assets": _SMOKE_TASK_ASSETS,
            "split": _SMOKE_SPLIT,
            "training": _SMOKE_TRAINING,
            "heldout_metrics": _SMOKE_HELDOUT,
            "checkpoint": _SMOKE_CHECKPOINT,
            "reloaded_action": _SMOKE_RELOADED,
            "runtime": _SMOKE_RUNTIME,
            "build": _SMOKE_BUILD,
            "boundaries": _SMOKE_BOUNDARIES,
            "deferred": ("string_array", 16),
            "error": _output_object({"type": _STRING, "message": _STRING}, frozenset({"type", "message"})),
        },
        frozenset({"schema", "status", "exit_status", "solution", "capability", "capabilities_exercised", "source", "dataset", "task_language_model"}),
    ),
    "npa_byof_summary.json": _output_object(
        {"status": _STRING, "tool": _STRING, "workload": _STRING, "run_id": _STRING, "image": _STRING, "solution_name": _STRING, "capability_name": _STRING, "smoke_artifact_name": _STRING, "smoke_exit_code": _INTEGER, "runtime_cache_uploaded": _BOOL, "rendering_invoked": _BOOL, "created_unix": _NUMBER},
        frozenset({"status", "tool", "workload", "run_id", "image", "solution_name", "capability_name", "smoke_artifact_name", "smoke_exit_code", "runtime_cache_uploaded", "rendering_invoked", "created_unix"}),
    ),
    "npa_runtime_bootstrap.json": _RUNTIME_BOOTSTRAP_SCHEMA,
    "npa_runtime_metadata.json": _RUNTIME_METADATA_SCHEMA,
}


def _validate_output_value(value: Any, spec: tuple, *, path: str) -> None:
    kind = spec[0]
    if kind == "string":
        valid = isinstance(value, str) and len(value) <= 4096
    elif kind == "bool":
        valid = isinstance(value, bool)
    elif kind == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 2**63 - 1
    elif kind == "number":
        valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    elif kind == "hex64":
        valid = isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    elif kind == "object":
        _validate_output_object(value, spec[1], spec[2], path=path)
        return
    elif kind == "runtime_metadata":
        _validate_output_object(value, _RUNTIME_METADATA_FIELDS, frozenset(_RUNTIME_METADATA_FIELDS), path=path)
        return
    elif kind == "map":
        valid = isinstance(value, dict) and len(value) <= spec[2]
        if valid:
            for key, child in value.items():
                if not isinstance(key, str) or re.fullmatch(r"[A-Za-z0-9._-]{1,128}", key) is None:
                    raise BootstrapRefusal(f"output JSON schema is invalid:{path}.{key}")
                _validate_output_value(child, spec[1], path=f"{path}.{key}")
        if not valid:
            raise BootstrapRefusal(f"output JSON schema is invalid:{path}")
        return
    elif kind in {"string_array", "integer_array"}:
        valid = isinstance(value, list) and len(value) <= spec[1]
        if valid:
            child_kind = "string" if kind == "string_array" else "integer"
            for index, child in enumerate(value):
                _validate_output_value(child, (child_kind,), path=f"{path}[{index}]")
        if not valid:
            raise BootstrapRefusal(f"output JSON schema is invalid:{path}")
        return
    else:
        raise BootstrapRefusal(f"output JSON schema has unsupported type:{kind}")
    if not valid:
        raise BootstrapRefusal(f"output JSON schema is invalid:{path}")


def _validate_output_object(value: Any, fields: dict[str, tuple], required: frozenset[str], *, path: str) -> None:
    if not isinstance(value, dict) or set(value) - set(fields) or not required <= set(value):
        raise BootstrapRefusal(f"output JSON schema is not closed:{path}")
    for key, child in value.items():
        _validate_output_value(child, fields[key], path=f"{path}.{key}")


def _reject_embedded_output(value: Any, *, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _EMBEDDED_OUTPUT_KEYS:
                raise BootstrapRefusal(f"embedded output payload is not allowed:{path}.{key}")
            _reject_embedded_output(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        if len(value) > 1024:
            raise BootstrapRefusal(f"embedded output array is not allowed:{path}")
        for index, child in enumerate(value):
            _reject_embedded_output(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and len(value) > 4096:
        raise BootstrapRefusal(f"embedded output string is not allowed:{path}")


def _canonical_output_payload(name: str, payload: bytes) -> bytes:
    """Return only closed-schema, canonical metadata bytes for upload."""

    allowed = _OUTPUT_JSON_TOP_LEVEL_KEYS.get(name)
    required = _OUTPUT_JSON_REQUIRED_KEYS.get(name)
    if allowed is None or required is None:
        raise BootstrapRefusal(f"output is not an uploadable canonical artifact:{name}")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapRefusal(f"output JSON is invalid:{name}") from exc
    if not isinstance(value, dict):
        raise BootstrapRefusal(f"output JSON schema is not closed:{name}")
    _reject_embedded_output(value, path=name)
    if set(value) - allowed or not required <= set(value):
        raise BootstrapRefusal(f"output JSON schema is not closed:{name}")
    schema = _OUTPUT_SCHEMAS[name]
    _validate_output_object(value, schema[1], schema[2], path=name)
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


class BootstrapRefusal(RuntimeError):
    """A fail-closed identity, permission, or boundary refusal."""


class CustomerAcceptanceRequired(BootstrapRefusal):
    """The customer must use the authenticated acceptance surface."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason.replace("_", " "))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapRefusal(f"cannot read required JSON {path}") from exc
    if not isinstance(value, dict):
        raise BootstrapRefusal(f"required JSON is not an object: {path}")
    return value


def _read_private_regular_bytes(path: Path, *, limit: int, input_name: str) -> bytes:
    """Read one stable owner-private file through a no-follow descriptor."""

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError as exc:
        raise CustomerAcceptanceRequired("authorization_missing") from exc
    except OSError as exc:
        raise BootstrapRefusal(f"{input_name} is unavailable or invalid") from exc
    try:
        return _read_private_regular_descriptor(
            descriptor,
            limit=limit,
            owner_uid=os.geteuid(),
            input_name=input_name,
        )
    finally:
        os.close(descriptor)


def _read_private_regular_descriptor(
    descriptor: int, *, limit: int, owner_uid: int, input_name: str,
    allow_unlinked: bool = False,
) -> bytes:
    """Read stable private bytes without changing an inherited descriptor offset."""

    try:
        before = os.fstat(descriptor)
        payload = os.pread(descriptor, limit + 1, 0)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise BootstrapRefusal(f"{input_name} descriptor is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != owner_uid
        or before.st_nlink not in ({0, 1} if allow_unlinked else {1})
        or stat.S_IMODE(before.st_mode) & 0o077
        or len(payload) > limit
        or before.st_size != len(payload)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise BootstrapRefusal(f"{input_name} is not stable owner-private")
    return payload


def _validate_manifest(
    path: Path, *, require_runtime_closure: bool = True
) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise BootstrapRefusal("runtime manifest must be a regular image file")
    manifest_sha256 = _sha256(path)
    if manifest_sha256 != EXPECTED_RUNTIME_MANIFEST_SHA256:
        raise BootstrapRefusal("runtime manifest bytes differ from the image contract")
    manifest = _load_json(path)
    if manifest.get("schema") != SCHEMA or manifest.get("solution") != "libero":
        raise BootstrapRefusal("runtime manifest identity is invalid")
    if set(manifest) != EXPECTED_MANIFEST_KEYS:
        raise BootstrapRefusal("runtime manifest top-level schema is not closed")
    if manifest.get("runtime_python") != EXPECTED_RUNTIME_PYTHON:
        raise BootstrapRefusal("runtime Python identity is invalid")
    if (
        manifest.get("customer_runtime_authorization")
        != EXPECTED_CUSTOMER_AUTHORIZATION_METADATA
    ):
        raise BootstrapRefusal(
            "customer runtime-authorization metadata is stale or invalid"
        )
    if manifest.get("boundaries") != EXPECTED_BOUNDARIES:
        raise BootstrapRefusal("runtime cache/output boundary metadata is invalid")
    governing_terms = manifest.get("governing_terms")
    if not isinstance(governing_terms, list) or len(governing_terms) != len(
        EXPECTED_GOVERNING_TERMS
    ):
        raise BootstrapRefusal("governing terms inventory is incomplete")
    term_ids: set[str] = set()
    term_boundaries: set[str] = set()
    for term in governing_terms:
        if not isinstance(term, dict) or set(term) - {"normalization"} != {
            "id",
            "name",
            "boundary",
            "version",
            "url",
            "size_bytes",
            "sha256",
        }:
            raise BootstrapRefusal("governing terms entry is not closed")
        if "normalization" in term and (
            term["normalization"] != NVIDIA_TERMS_NORMALIZATION
            or term["id"] != "nvidia-software-license"
            or term["url"] != NVIDIA_SOFTWARE_TERMS_URL
        ):
            raise BootstrapRefusal("governing terms normalization is invalid")
        term_id = str(term.get("id") or "")
        boundary = str(term.get("boundary") or "")
        if (
            term_id in term_ids
            or boundary
            not in {
                "source",
                "runtime_packages",
                "demonstration",
                "task_inputs",
                "language_model",
            }
            or not str(term.get("name") or "").strip()
            or term.get("version") != f"sha256:{term.get('sha256')}"
            or not isinstance(term.get("size_bytes"), int)
            or term["size_bytes"] <= 0
            or not _is_hex(term.get("sha256"), 64)
        ):
            raise BootstrapRefusal("governing terms identity is invalid")
        _validate_terms_url(str(term.get("url") or ""))
        term_ids.add(term_id)
        term_boundaries.add(boundary)
    if (
        term_ids != EXPECTED_GOVERNING_TERMS
        or not {
            "source",
            "runtime_packages",
            "demonstration",
            "language_model",
        }
        <= term_boundaries
    ):
        raise BootstrapRefusal("governing terms inventory is incomplete")
    source = manifest.get("source") or {}
    if (
        source.get("repository")
        != "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
        or not _is_hex(source.get("revision"), 40)
        or not _is_hex(source.get("tree"), 40)
        or source.get("license") != "MIT"
        or not _is_hex(source.get("license_sha256"), 64)
    ):
        raise BootstrapRefusal("pinned LIBERO source contract is invalid")
    sparse_paths = source.get("sparse_paths")
    if (
        not isinstance(sparse_paths, list)
        or len(sparse_paths) != len(set(sparse_paths))
        or not all(isinstance(item, str) and item for item in sparse_paths)
        or "libero/libero/assets" in sparse_paths
        or source.get("forbidden_paths") != ["libero/libero/assets", ".git"]
    ):
        raise BootstrapRefusal("LIBERO sparse-source boundary is invalid")
    demonstration = manifest.get("demonstration") or {}
    if (
        demonstration.get("repository") != "yifengzhu-hf/LIBERO-datasets"
        or not _is_hex(demonstration.get("revision"), 40)
        or demonstration.get("license") != "CC-BY-4.0"
        or not str(demonstration.get("attribution") or "").strip()
        or not str(demonstration.get("filename") or "").endswith(".hdf5")
        or not _is_hex(demonstration.get("sha256"), 64)
        or not isinstance(demonstration.get("size_bytes"), int)
        or demonstration["size_bytes"] <= 0
    ):
        raise BootstrapRefusal("official demonstration contract is invalid")
    _validate_download_url(str(demonstration.get("url") or ""))
    task = manifest.get("task") or {}
    if (
        task.get("suite") != "libero_spatial"
        or not str(task.get("name") or "").strip()
        or not str(task.get("description") or "").strip()
    ):
        raise BootstrapRefusal("LIBERO task identity is invalid")
    for key in ("bddl", "initial_states", "embedding_source"):
        record = task.get(key) or {}
        if not str(record.get("path") or "").strip() or not _is_hex(
            record.get("sha256"), 64
        ):
            raise BootstrapRefusal(f"LIBERO task identity is invalid: {key}")
    model = manifest.get("language_model") or {}
    model_files = model.get("files")
    if (
        model.get("repository") != "google-bert/bert-base-cased"
        or not _is_hex(model.get("revision"), 40)
        or model.get("license") != "Apache-2.0"
        or not isinstance(model_files, list)
        or not model_files
    ):
        raise BootstrapRefusal("task language-model contract is invalid")
    model_names: set[str] = set()
    for item in model_files:
        if not isinstance(item, dict):
            raise BootstrapRefusal("task language-model file is not an object")
        filename = str(item.get("filename") or "")
        if (
            not filename
            or filename in model_names
            or "/" in filename
            or not isinstance(item.get("size_bytes"), int)
            or item["size_bytes"] <= 0
            or not _is_hex(item.get("sha256"), 64)
        ):
            raise BootstrapRefusal("task language-model file identity is invalid")
        model_names.add(filename)
    artifacts = manifest.get("runtime_artifacts")
    if (
        manifest.get("runtime_artifact_count") != 135
        or not isinstance(artifacts, list)
        or len(artifacts) != manifest["runtime_artifact_count"]
    ):
        raise BootstrapRefusal(
            "runtime artifact inventory must contain exactly 135 items"
        )
    names: set[str] = set()
    filenames: set[str] = set()
    total_size_bytes = 0
    for item in artifacts:
        if not isinstance(item, dict):
            raise BootstrapRefusal("runtime artifact entry is not an object")
        name = str(item.get("name") or "")
        filename = str(item.get("filename") or "")
        if (
            name in names
            or filename in filenames
            or not name
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,254}", filename) is None
            or Path(filename).name != filename
            or filename in {".", ".."}
        ):
            raise BootstrapRefusal(
                "runtime artifact names and filenames must be unique"
            )
        names.add(name)
        filenames.add(filename)
        if filename.endswith(".tar.gz"):
            if name not in REVIEWED_SOURCE_BUILD_NAMES:
                raise BootstrapRefusal(
                    f"source runtime artifact is not in the reviewed build allowlist for {name}"
                )
        elif not filename.endswith(".whl"):
            raise BootstrapRefusal(
                f"runtime artifact must be a wheel or reviewed source archive for {name}"
            )
        _validate_download_url(str(item.get("url") or ""))
        version = item.get("version")
        if (
            not isinstance(version, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+!-]*", version) is None
        ):
            raise BootstrapRefusal(f"invalid runtime artifact version for {name}")
        if not _is_hex(item.get("sha256"), 64):
            raise BootstrapRefusal(f"invalid runtime artifact hash for {name}")
        size_bytes = item.get("size_bytes")
        license_expression = str(item.get("license_expression") or "").strip()
        if require_runtime_closure and (
            not isinstance(size_bytes, int) or size_bytes <= 0 or not license_expression
        ):
            raise BootstrapRefusal(
                f"runtime artifact size/license review is incomplete for {name}"
            )
        if isinstance(size_bytes, int) and size_bytes > 0:
            total_size_bytes += size_bytes
    if require_runtime_closure and total_size_bytes > MAX_RUNTIME_CACHE_DOWNLOAD_BYTES:
        raise BootstrapRefusal("runtime artifact inventory exceeds the cache budget")
    return manifest, manifest_sha256


def _validate_requirements(
    path: Path, manifest: dict[str, Any]
) -> tuple[list[str], str]:
    if not path.is_file() or path.is_symlink():
        raise BootstrapRefusal("runtime requirements must be a regular image file")
    requirements_sha256 = _sha256(path)
    if requirements_sha256 != EXPECTED_RUNTIME_REQUIREMENTS_SHA256:
        raise BootstrapRefusal(
            "runtime requirements bytes differ from the image contract"
        )
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    artifacts = manifest["runtime_artifacts"]
    if len(lines) != len(artifacts):
        raise BootstrapRefusal("runtime requirements and artifact inventory differ")
    for line, artifact in zip(lines, artifacts, strict=True):
        expected_prefix = (
            f"{artifact['name']}=={artifact['version']} "
            f"--hash=sha256:{artifact['sha256']} # {artifact['url']}"
        )
        if line != expected_prefix:
            raise BootstrapRefusal(
                "runtime requirements do not bind the manifest order"
            )
    return lines, requirements_sha256


def _is_hex(value: object, length: int) -> bool:
    text = str(value or "")
    return len(text) == length and all(
        character in "0123456789abcdef" for character in text
    )


def _validate_download_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS
        or parsed.fragment
    ):
        raise BootstrapRefusal(
            "runtime download URL is outside the immutable allowlist"
        )


def _validate_terms_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname not in ALLOWED_TERMS_HOSTS
        or parsed.fragment
    ):
        raise BootstrapRefusal("governing terms URL is outside its allowlist")


def _validate_redirect_url(
    url: str, *, terms: bool = False
) -> urllib.parse.SplitResult:
    parsed = urllib.parse.urlsplit(url)
    hostname = parsed.hostname or ""
    allowed_hosts = ALLOWED_TERMS_HOSTS if terms else ALLOWED_DOWNLOAD_HOSTS
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or not (
            hostname in allowed_hosts
            or (not terms and hostname in ALLOWED_DOWNLOAD_REDIRECT_HOSTS)
        )
        or parsed.fragment
    ):
        raise BootstrapRefusal("runtime redirect left the download allowlist")
    return parsed


def wait_for_release() -> dict[str, str]:
    """Block all payload setup until the manager releases this exact Pod UID."""

    host = os.environ.get("KUBERNETES_SERVICE_HOST", "")
    port = int(os.environ.get("KUBERNETES_SERVICE_PORT", "443"))
    name = os.environ.get("HOSTNAME", "")
    service_account = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    namespace = (service_account / "namespace").read_text(encoding="utf-8").strip()
    token = (service_account / "token").read_text(encoding="utf-8").strip()
    if not host or not name or not namespace or not token:
        raise BootstrapRefusal("payload release identity is incomplete")
    context = ssl.create_default_context(cafile=str(service_account / "ca.crt"))
    path = (
        "/api/v1/namespaces/"
        + urllib.parse.quote(namespace, safe="")
        + "/pods/"
        + urllib.parse.quote(name, safe="")
    )
    while True:
        connection = http.client.HTTPSConnection(
            host, port, timeout=10, context=context
        )
        try:
            connection.request(
                "GET", path, headers={"Authorization": f"Bearer {token}"}
            )
            response = connection.getresponse()
            payload = response.read(1024 * 1024)
            status = response.status
        finally:
            connection.close()
        if status == 403:
            time.sleep(1)
            continue
        if status != 200:
            raise BootstrapRefusal("payload release observation failed")
        try:
            pod = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BootstrapRefusal("payload release response is invalid") from exc
        metadata = pod.get("metadata") or {}
        uid = str(metadata.get("uid") or "")
        expected = hashlib.sha256(
            f"{uid}:{name}:npa-libero-release-v1".encode()
        ).hexdigest()
        observed = str(
            (metadata.get("annotations") or {}).get(PAYLOAD_RELEASE_ANNOTATION, "")
        )
        if (
            metadata.get("name") != name
            or metadata.get("namespace") != namespace
            or not uid
            or not hmac.compare_digest(observed, expected)
        ):
            time.sleep(1)
            continue
        return {"pod_uid_sha256": hashlib.sha256(uid.encode()).hexdigest()}
def _open_https_download(
    url: str, *, terms: bool = False, deadline: datetime | None = None
) -> tuple[http.client.HTTPSConnection, http.client.HTTPResponse]:
    current_url = url
    for _ in range(6):
        _ensure_deadline(deadline)
        parsed = _validate_redirect_url(current_url, terms=terms)
        connection = http.client.HTTPSConnection(
            parsed.hostname, parsed.port or 443, timeout=60
        )
        target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        connection.request(
            "GET", target, headers={"User-Agent": "npa-libero-runtime/1"}
        )
        response = connection.getresponse()
        if response.status not in {301, 302, 303, 307, 308}:
            if response.status != 200:
                status = response.status
                response.close()
                connection.close()
                raise BootstrapRefusal(
                    f"runtime download returned unexpected HTTP status {status}"
                )
            return connection, response
        location = response.getheader("Location")
        response.close()
        connection.close()
        if not location:
            raise BootstrapRefusal("runtime redirect omitted its target")
        current_url = urllib.parse.urljoin(current_url, location)
    raise BootstrapRefusal("runtime download exceeded the redirect limit")


def _validate_cache_root(cache_root: Path, output_dir: Path | None) -> Path:
    raw = os.fspath(cache_root)
    if not os.path.isabs(raw) or os.path.normpath(raw) in {"/", "."}:
        raise BootstrapRefusal("runtime cache root must be a narrow absolute path")
    if any(part in {"", ".", ".."} for part in Path(raw).parts[1:]):
        raise BootstrapRefusal("runtime cache root must be canonical")
    resolved = Path(os.path.normpath(raw))
    if output_dir is not None:
        output = Path(os.path.normpath(os.fspath(output_dir)))
        if (
            output == resolved
            or output in resolved.parents
            or resolved in output.parents
        ):
            raise BootstrapRefusal("runtime cache and output boundaries overlap")
    return resolved


def _create_cache_root_component(
    parent_descriptor: int,
    name: str,
    before_create: Callable[[int, str], None] | None,
) -> int:
    """Create and bind the final root component without trusting a pathname."""

    if before_create is not None:
        before_create(parent_descriptor, name)
    try:
        os.mkdir(name, CACHE_ROOT_MODE, dir_fd=parent_descriptor)
    except FileExistsError:
        raise BootstrapRefusal("runtime cache root appeared during creation") from None
    except OSError as exc:
        raise BootstrapRefusal("runtime cache root could not be created") from exc
    child: int | None = None
    try:
        created = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        child = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(child)
    except OSError as exc:
        if child is not None:
            os.close(child)
        raise BootstrapRefusal("runtime cache root changed during creation") from exc
    if not stat.S_ISDIR(created.st_mode) or (created.st_dev, created.st_ino) != (
        opened.st_dev,
        opened.st_ino,
    ):
        os.close(child)
        raise BootstrapRefusal("runtime cache root changed during creation")
    return child


@contextmanager
def _open_cache_root_descriptor(
    cache_root: Path,
    *,
    create: bool,
    owner_uid: int | None = None,
    before_create: Callable[[int, str], None] | None = None,
) -> Iterator[int]:
    """Walk an absolute cache root without following any path component."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise BootstrapRefusal("runtime cache requires no-follow directory support")
    parts = cache_root.parts
    descriptor = os.open("/", os.O_RDONLY | directory | nofollow | os.O_CLOEXEC)
    try:
        for index, part in enumerate(parts[1:], start=1):
            final = index == len(parts) - 1
            try:
                child = os.open(
                    part,
                    os.O_RDONLY | directory | nofollow | os.O_CLOEXEC,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not (create and final):
                    raise BootstrapRefusal(
                        "runtime cache root is unavailable"
                    ) from None
                child = _create_cache_root_component(descriptor, part, before_create)
            except OSError as exc:
                raise BootstrapRefusal(
                    "runtime cache root contains a link or invalid component"
                ) from exc
            os.close(descriptor)
            descriptor = child
        info = os.fstat(descriptor)
        expected_owner = os.getuid() if owner_uid is None else owner_uid
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_nlink < 1
            or info.st_uid != expected_owner
        ):
            raise BootstrapRefusal("runtime cache root ownership or links are invalid")
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _cache_lock(
    cache_root_descriptor: int,
    name: str,
    *,
    exclusive: bool,
    create: bool,
) -> Iterator[int]:
    """Open and hold one validated cache lock through the retained root descriptor."""

    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
    if create:
        flags |= os.O_CREAT
    try:
        descriptor = os.open(name, flags, 0o640, dir_fd=cache_root_descriptor)
    except OSError as exc:
        raise BootstrapRefusal("runtime cache lock is unavailable") from exc
    try:
        root = os.fstat(cache_root_descriptor)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != root.st_uid
        ):
            raise BootstrapRefusal("runtime cache lock identity is invalid")
        if create:
            try:
                group_id = grp.getgrnam(RUNTIME_EXECUTION_GROUP).gr_gid
            except KeyError as exc:
                raise BootstrapRefusal(
                    "runtime execution group is unavailable"
                ) from exc
            os.fchown(descriptor, -1, group_id)
            os.fchmod(descriptor, 0o640)
        elif stat.S_IMODE(opened.st_mode) != 0o640:
            raise BootstrapRefusal("runtime cache lock mode is invalid")
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield descriptor
    finally:
        os.close(descriptor)


def _require_inherited_lock(
    cache_root_descriptor: int, name: str, inherited_descriptor: int
) -> None:
    """Bind an inherited lock descriptor to the exact retained cache root."""

    try:
        named = os.stat(name, dir_fd=cache_root_descriptor, follow_symlinks=False)
        inherited = os.fstat(inherited_descriptor)
    except OSError as exc:
        raise BootstrapRefusal("inherited runtime lock is unavailable") from exc
    if (
        not stat.S_ISREG(named.st_mode)
        or named.st_nlink != 1
        or stat.S_IMODE(named.st_mode) != 0o640
        or (named.st_dev, named.st_ino) != (inherited.st_dev, inherited.st_ino)
    ):
        raise BootstrapRefusal("inherited runtime lock identity is invalid")


def _cache_entry_identity(path: Path) -> tuple[int, int] | None:
    """Return a real directory's identity without following the final component."""

    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise BootstrapRefusal("cannot inspect the runtime cache entry") from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise BootstrapRefusal("runtime cache entry must be a real directory")
    return info.st_dev, info.st_ino


def _cache_entry_identity_at(
    parent_descriptor: int, name: str
) -> tuple[int, int] | None:
    """Inspect one no-follow cache child through the trusted parent descriptor."""

    try:
        info = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise BootstrapRefusal("cannot inspect the runtime cache entry") from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise BootstrapRefusal("runtime cache entry must be a real directory")
    return info.st_dev, info.st_ino


def _require_cache_entry_identity(path: Path, expected: tuple[int, int]) -> None:
    if _cache_entry_identity(path) != expected:
        raise BootstrapRefusal("runtime cache entry changed during validation")


@contextmanager
def _open_cache_entry(path: Path, *, expected: tuple[int, int]) -> Iterator[Path]:
    """Retain the no-follow cache leaf while validating through its descriptor."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise BootstrapRefusal("runtime cache requires no-follow directory support")
    try:
        descriptor = os.open(path, os.O_RDONLY | directory | nofollow | os.O_CLOEXEC)
    except OSError as exc:
        raise BootstrapRefusal(
            "runtime cache entry could not be opened without following links"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != expected
            or stat.S_IMODE(opened.st_mode) & 0o222
        ):
            raise BootstrapRefusal("runtime cache entry identity or mode is invalid")
        _require_cache_entry_identity(path, expected)
        stable_root = Path("/proc/self/fd") / str(descriptor)
        if not stable_root.is_dir():
            raise BootstrapRefusal("runtime cache descriptor is unavailable")
        yield stable_root
        _require_cache_entry_identity(path, expected)
    finally:
        os.close(descriptor)


def _publish_current_cache_link(
    cache_root: Path, final: Path, expected: tuple[int, int]
) -> None:
    """Publish ``current`` only while the validated leaf keeps its identity."""

    current = cache_root / "current"
    temporary_link = cache_root / f".current-{os.getpid()}"
    temporary_link.unlink(missing_ok=True)
    try:
        _require_cache_entry_identity(final, expected)
        temporary_link.symlink_to(final.name)
        _require_cache_entry_identity(final, expected)
        temporary_link.replace(current)
        _require_cache_entry_identity(final, expected)
    except Exception:
        temporary_link.unlink(missing_ok=True)
        _remove_current_cache_link(cache_root, final)
        raise


def _remove_current_cache_link(cache_root: Path, final: Path) -> None:
    current = cache_root / "current"
    if current.is_symlink() and os.readlink(current) == final.name:
        current.unlink()


def _remove_cache_tree(path: Path) -> list[str]:
    """Best-effort no-follow removal that attempts every discovered descendant."""

    errors: list[str] = []
    try:
        descendants = sorted(
            path.rglob("*"),
            key=lambda item: len(item.relative_to(path).parts),
            reverse=True,
        )
    except Exception:  # noqa: BLE001 - rollback records every ambiguity
        return ["descendant inventory failed"]
    for item in (path, *reversed(descendants)):
        try:
            info = item.lstat()
            if stat.S_ISDIR(info.st_mode):
                os.chmod(item, 0o700)
        except FileNotFoundError:
            continue
        except Exception:  # noqa: BLE001 - continue with independent descendants
            errors.append(f"chmod failed:{item.name}")
    for item in descendants:
        try:
            info = item.lstat()
            if stat.S_ISLNK(info.st_mode) or stat.S_ISREG(info.st_mode):
                item.unlink()
            elif stat.S_ISDIR(info.st_mode):
                item.rmdir()
            else:
                errors.append(f"unsupported descendant:{item.name}")
        except FileNotFoundError:
            continue
        except Exception:  # noqa: BLE001 - attempt the rest before reporting
            errors.append(f"remove failed:{item.name}")
    try:
        path.rmdir()
    except FileNotFoundError:
        pass
    except Exception:  # noqa: BLE001 - caller retries once after child failures
        errors.append(f"remove failed:{path.name}")
    return errors


def _discard_new_cache_entry(
    cache_root: Path,
    final: Path,
    expected: tuple[int, int],
    *,
    parent_descriptor: int,
) -> list[str]:
    """Isolate and remove only the exact entry created by this transaction."""

    errors: list[str] = []
    quarantine_root: Path | None = None
    isolated: Path | None = None
    try:
        observed = _cache_entry_identity_at(parent_descriptor, final.name)
    except Exception:  # noqa: BLE001 - identity ambiguity is a fail-closed result
        errors.append("final identity acquisition failed")
        observed = None
    if observed is None:
        return errors
    if observed != expected:
        errors.append("final identity drifted")
        return errors
    try:
        quarantine_root = Path(
            tempfile.mkdtemp(
                prefix=f".{final.name}.failed-{uuid4().hex}-", dir=cache_root
            )
        )
        os.chmod(quarantine_root, 0o700)
        isolated = quarantine_root / "entry"
        os.rename(
            final.name,
            f"{quarantine_root.name}/entry",
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
    except Exception:  # noqa: BLE001 - direct exact-inode removal is the fallback
        errors.append("quarantine creation or rename failed")
        if quarantine_root is not None:
            _remove_cache_tree(quarantine_root)
        try:
            if _cache_entry_identity_at(parent_descriptor, final.name) == expected:
                isolated = final
            else:
                return errors
        except Exception:  # noqa: BLE001 - never remove an unverified replacement
            errors.append("fallback identity acquisition failed")
            return errors
    assert isolated is not None
    try:
        if _cache_entry_identity(isolated) != expected:
            errors.append("isolated cache identity drifted")
            return errors
    except Exception:  # noqa: BLE001 - retain isolated evidence on ambiguity
        errors.append("isolated cache identity acquisition failed")
        return errors
    first_pass = _remove_cache_tree(isolated)
    if isolated.exists():
        errors.extend(first_pass)
        errors.extend(_remove_cache_tree(isolated))
    if quarantine_root is not None and quarantine_root.exists():
        errors.extend(_remove_cache_tree(quarantine_root))
    return errors


def _rollback_new_cache_entry(
    cache_root: Path,
    final: Path,
    expected: tuple[int, int],
    *,
    parent_descriptor: int,
) -> tuple[str, ...]:
    """Independently retract publication and the renamed entry, aggregating errors."""

    errors: list[str] = []
    for _attempt in range(2):
        try:
            _remove_current_cache_link(cache_root, final)
        except Exception:  # noqa: BLE001 - entry cleanup must still run
            errors.append("current-link removal failed")
        errors.extend(
            _discard_new_cache_entry(
                cache_root,
                final,
                expected,
                parent_descriptor=parent_descriptor,
            )
        )
    try:
        remaining = _cache_entry_identity_at(parent_descriptor, final.name)
    except Exception:  # noqa: BLE001
        errors.append("final absence check failed")
    else:
        if remaining == expected:
            errors.append("renamed cache entry remains")
        elif remaining is not None:
            errors.append("different cache entry occupies final name")
    current = cache_root / "current"
    try:
        if current.is_symlink() and os.readlink(current) == final.name:
            errors.append("current link remains")
    except Exception:  # noqa: BLE001
        errors.append("current-link absence check failed")
    return tuple(dict.fromkeys(errors))


def _ssh_string(value: bytes) -> bytes:
    return struct.pack(">I", len(value)) + value


def _canonical_unsigned_customer_authorization(payload: dict[str, Any]) -> bytes:
    unsigned = json.loads(json.dumps(payload))
    unsigned.pop("signature", None)
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()


def _customer_authorization_signature_payload(payload: dict[str, Any]) -> bytes:
    canonical = _canonical_unsigned_customer_authorization(payload)
    return _sshsig_signature_payload(CUSTOMER_AUTHORIZATION_NAMESPACE, canonical)


def _sshsig_signature_payload(namespace: bytes, canonical: bytes) -> bytes:
    return b"".join(
        (
            b"SSHSIG",
            _ssh_string(namespace),
            _ssh_string(b""),
            _ssh_string(b"sha512"),
            _ssh_string(hashlib.sha512(canonical).digest()),
        )
    )


def _customer_authorization_sshsig(public_key: bytes, signature: bytes) -> bytes:
    return _sshsig_envelope(CUSTOMER_AUTHORIZATION_NAMESPACE, public_key, signature)


def _sshsig_envelope(namespace: bytes, public_key: bytes, signature: bytes) -> bytes:
    public_key_blob = _ssh_string(b"ssh-ed25519") + _ssh_string(public_key)
    signature_blob = _ssh_string(b"ssh-ed25519") + _ssh_string(signature)
    payload = b"".join(
        (
            b"SSHSIG",
            struct.pack(">I", 1),
            _ssh_string(public_key_blob),
            _ssh_string(namespace),
            _ssh_string(b""),
            _ssh_string(b"sha512"),
            _ssh_string(signature_blob),
        )
    )
    encoded = base64.b64encode(payload).decode("ascii")
    lines = [encoded[index : index + 70] for index in range(0, len(encoded), 70)]
    return (
        "-----BEGIN SSH SIGNATURE-----\n"
        + "\n".join(lines)
        + "\n-----END SSH SIGNATURE-----\n"
    ).encode()


def _canonical_unsigned_authenticated_caller(payload: dict[str, Any]) -> bytes:
    unsigned = json.loads(json.dumps(payload))
    unsigned.pop("signature", None)
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()


def _validate_authenticated_caller_binding(
    caller_bytes: bytes,
    trusted_public_key: bytes,
    *,
    expected_sha256: str,
    expected_run_id: str,
    expected_customer_identity_sha256: str,
) -> str:
    """Validate the caller assertion that independently binds the customer signer."""

    if hashlib.sha256(caller_bytes).hexdigest() != expected_sha256:
        raise BootstrapRefusal("authenticated caller assertion digest differs")
    try:
        caller = json.loads(caller_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapRefusal("authenticated caller assertion is invalid") from exc
    signature_record = caller.get("signature") if isinstance(caller, dict) else None
    expected_keys = {
        "schema",
        "issuer",
        "session_id",
        "customer_identity_sha256",
        "customer_signer_public_key_sha256",
        "run_id",
        "issued_at",
        "expires_at",
        "nonce",
        "signature",
    }
    if (
        not isinstance(caller, dict)
        or set(caller) != expected_keys
        or caller.get("schema") != AUTHENTICATED_CALLER_SCHEMA
        or caller.get("issuer") != "npa-authenticated-caller-control-plane"
        or caller.get("run_id") != expected_run_id
        or caller.get("customer_identity_sha256") != expected_customer_identity_sha256
        or not _is_hex(caller.get("customer_identity_sha256"), 64)
        or not _is_hex(caller.get("customer_signer_public_key_sha256"), 64)
        or not isinstance(signature_record, dict)
        or set(signature_record) != {"algorithm", "public_key_sha256", "signature_b64"}
        or signature_record.get("algorithm") != "ed25519"
        or len(trusted_public_key) != 32
        or hashlib.sha256(trusted_public_key).hexdigest()
        != signature_record.get("public_key_sha256")
    ):
        raise BootstrapRefusal("authenticated caller assertion is invalid")
    try:
        issued_at = datetime.fromisoformat(
            str(caller["issued_at"]).replace("Z", "+00:00")
        )
        expires_at = datetime.fromisoformat(
            str(caller["expires_at"]).replace("Z", "+00:00")
        )
        signature = base64.b64decode(
            str(signature_record.get("signature_b64") or ""), validate=True
        )
    except (TypeError, ValueError, binascii.Error) as exc:
        raise BootstrapRefusal("authenticated caller assertion is invalid") from exc
    now = datetime.now(timezone.utc)
    if (
        issued_at.tzinfo is None
        or expires_at.tzinfo is None
        or issued_at > now + timedelta(minutes=1)
        or issued_at >= expires_at
        or expires_at <= now
        or expires_at - issued_at > timedelta(minutes=15)
        or len(signature) != 64
    ):
        raise BootstrapRefusal("authenticated caller assertion is expired or invalid")
    canonical = _canonical_unsigned_authenticated_caller(caller)
    allowed_signer = (
        "npa-authenticated-caller-control-plane ssh-ed25519 "
        + base64.b64encode(
            _ssh_string(b"ssh-ed25519") + _ssh_string(trusted_public_key)
        ).decode("ascii")
        + "\n"
    )
    with tempfile.TemporaryDirectory(prefix="npa-libero-caller-binding-") as root:
        root_path = Path(root)
        allowed_path = root_path / "allowed-signers"
        signature_path = root_path / "caller.sig"
        allowed_path.write_text(allowed_signer, encoding="ascii")
        signature_path.write_bytes(
            _sshsig_envelope(AUTHENTICATED_CALLER_NAMESPACE, trusted_public_key, signature)
        )
        os.chmod(allowed_path, 0o600)
        os.chmod(signature_path, 0o600)
        completed = subprocess.run(
            [
                "/usr/bin/ssh-keygen",
                "-Y",
                "verify",
                "-f",
                str(allowed_path),
                "-I",
                "npa-authenticated-caller-control-plane",
                "-n",
                AUTHENTICATED_CALLER_NAMESPACE.decode("ascii"),
                "-s",
                str(signature_path),
            ],
            input=canonical,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"},
            check=False,
        )
    if completed.returncode:
        raise BootstrapRefusal("authenticated caller assertion signature is invalid")
    return str(caller["customer_signer_public_key_sha256"])


def _authenticated_caller_binding_from_environment() -> tuple[bytes, bytes, str]:
    """Read the owner-validated caller binding used before any runtime effect."""

    if _customer_run():
        # The actual customer supplies this key through the private handoff.
        # It is mounted by the controller, never selected by a signed payload.
        key = _customer_run_key()
        return b"", key, hashlib.sha256(key).hexdigest()
    try:
        caller_bytes = base64.b64decode(
            os.environ.get("NPA_LIBERO_AUTHENTICATED_CALLER_B64", ""), validate=True
        )
    except (ValueError, binascii.Error) as exc:
        raise BootstrapRefusal("authenticated caller assertion is unavailable") from exc
    expected_sha256 = os.environ.get("NPA_LIBERO_AUTHENTICATED_CALLER_SHA256", "")
    if not _is_hex(expected_sha256, 64):
        raise BootstrapRefusal("authenticated caller binding is unavailable")
    trusted_public_key = _trusted_public_key(
        AUTHENTICATED_CALLER_TRUST_ROOT,
        owner_uid=AUTHENTICATED_CALLER_TRUST_ROOT_OWNER_UID,
        label="authenticated caller",
    )
    try:
        caller = json.loads(caller_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapRefusal("authenticated caller assertion is unavailable") from exc
    if not isinstance(caller, dict):
        raise BootstrapRefusal("authenticated caller assertion is unavailable")
    signer_sha256 = _validate_authenticated_caller_binding(
        caller_bytes,
        trusted_public_key,
        expected_sha256=expected_sha256,
        expected_run_id=os.environ.get("NPA_BYOF_RUN_ID", ""),
        expected_customer_identity_sha256=os.environ.get(
            "NPA_LIBERO_CUSTOMER_IDENTITY_SHA256", ""
        ),
    )
    return caller_bytes, trusted_public_key, signer_sha256


def _executable_profile_path() -> Path:
    run_id = os.environ.get("NPA_BYOF_RUN_ID", "").strip()
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{15,62}", run_id) is None:
        raise BootstrapRefusal("executable profile run identity is invalid")
    return EXECUTABLE_PROFILE_ROOT / run_id / EXECUTABLE_PROFILE_NAME


def _validate_executable_profile_digest(expected_sha256: str) -> None:
    """Hash the stable owner-private profile bytes, never an environment claim."""

    path = _executable_profile_path()
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        before = os.fstat(descriptor)
        expected_group = grp.getgrnam(RUNTIME_EXECUTION_GROUP).gr_gid
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != pwd.getpwnam(RUNTIME_SUPERVISOR_USER).pw_uid
            or before.st_gid != expected_group
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o440
            or before.st_size <= 0
            or before.st_size > 1024 * 1024
        ):
            raise BootstrapRefusal("executable profile file is invalid")
        payload = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise BootstrapRefusal("executable profile file is unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        len(payload) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        or not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), expected_sha256)
    ):
        raise BootstrapRefusal("executable profile bytes differ from authorization")
    if _customer_run():
        try:
            tasks = [item for item in json.loads(payload) if item.get("resources")]
            task = tasks[0]
            pod = task["resources"]["kubernetes"]["pod_config"]["spec"]
            valid = (len(tasks) == 1
                     and task["envs"].get("NPA_LIBERO_RUNTIME_DELIVERY") == CUSTOMER_RUN_MODE
                     and pod.get("automountServiceAccountToken") is False)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            valid = False
        if not valid:
            raise BootstrapRefusal("customer-run mode is absent from the signed profile")


def _trusted_public_key(path: Path, *, owner_uid: int, label: str) -> bytes:
    """Load one immutable root-owned Ed25519 verification key."""

    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as exc:
        raise BootstrapRefusal(f"{label} trust root is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            encoded = stream.read()
        after = os.fstat(descriptor)
    except OSError as exc:
        raise BootstrapRefusal(f"{label} trust root is unavailable") from exc
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != owner_uid
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o444
        or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_mode,
            before.st_uid,
            before.st_nlink,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_mode,
            after.st_uid,
            after.st_nlink,
        )
        or encoded != encoded.strip()
    ):
        raise BootstrapRefusal(f"{label} trust root is mutable or invalid")
    try:
        public_key = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise BootstrapRefusal(f"{label} trust root is invalid") from exc
    if len(public_key) != 32:
        raise BootstrapRefusal(f"{label} trust root is invalid")
    return public_key


def _trusted_output_storage_authorization_public_key() -> bytes:
    storage_key = _trusted_public_key(
        OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY,
        owner_uid=OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY_OWNER_UID,
        label="output-storage-authorization",
    )
    customer_fingerprint = os.environ.get(
        "NPA_LIBERO_EXPECTED_CUSTOMER_SIGNER_PUBLIC_KEY_SHA256", ""
    )
    if not _is_hex(customer_fingerprint, 64) or hmac.compare_digest(
        customer_fingerprint, hashlib.sha256(storage_key).hexdigest()
    ):
        raise BootstrapRefusal("output-storage trust root is not independent")
    return storage_key


def _trusted_customer_signer_public_key(customer_identity_sha256: str) -> bytes:
    if not _is_hex(customer_identity_sha256, 64):
        raise BootstrapRefusal("customer signer trust root is invalid")
    if _customer_run():
        return _customer_run_key()
    return _trusted_public_key(
        CUSTOMER_SIGNER_REGISTRY_ROOT / f"{customer_identity_sha256}.b64",
        owner_uid=CUSTOMER_SIGNER_REGISTRY_OWNER_UID,
        label="customer signer trust root",
    )


def _verify_customer_authorization_signature(
    payload: dict[str, Any],
    signature_record: dict[str, Any],
    *,
    authenticated_signer_sha256: str,
) -> None:
    try:
        public_key = base64.b64decode(
            str(payload.get("customer_signer_public_key_b64") or ""), validate=True
        )
    except ValueError as exc:
        raise BootstrapRefusal("customer signer identity is invalid") from exc
    claimed_fingerprint = str(signature_record.get("public_key_sha256") or "")
    if (
        len(public_key) != 32
        or not _is_hex(claimed_fingerprint, 64)
        or hashlib.sha256(public_key).hexdigest() != claimed_fingerprint
        or not hmac.compare_digest(claimed_fingerprint, authenticated_signer_sha256)
    ):
        raise BootstrapRefusal("customer signer identity differs")
    registered_key = _trusted_customer_signer_public_key(
        str(payload.get("customer_identity_sha256") or "")
    )
    if not hmac.compare_digest(public_key, registered_key):
        raise BootstrapRefusal("customer signer trust root differs")
    try:
        signature = base64.b64decode(
            str(signature_record.get("signature_b64") or ""), validate=True
        )
    except ValueError as exc:
        raise BootstrapRefusal("customer authorization signature is invalid") from exc
    if len(signature) != 64:
        raise BootstrapRefusal("customer authorization signature is invalid")
    canonical = _canonical_unsigned_customer_authorization(payload)
    public_key_blob = _ssh_string(b"ssh-ed25519") + _ssh_string(public_key)
    allowed_signer = (
        "customer ssh-ed25519 "
        + base64.b64encode(public_key_blob).decode("ascii")
        + "\n"
    )
    with tempfile.TemporaryDirectory(
        prefix="npa-libero-customer-authorization-signature-"
    ) as root:
        root_path = Path(root)
        allowed_path = root_path / "allowed-signers"
        signature_path = root_path / "authorization.sig"
        allowed_path.write_text(allowed_signer, encoding="ascii")
        signature_path.write_bytes(
            _customer_authorization_sshsig(public_key, signature)
        )
        os.chmod(allowed_path, 0o600)
        os.chmod(signature_path, 0o600)
        completed = subprocess.run(
            [
                "/usr/bin/ssh-keygen",
                "-Y",
                "verify",
                "-f",
                str(allowed_path),
                "-I",
                "customer",
                "-n",
                CUSTOMER_AUTHORIZATION_NAMESPACE.decode("ascii"),
                "-s",
                str(signature_path),
            ],
            input=canonical,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"},
            check=False,
        )
    if completed.returncode:
        raise BootstrapRefusal("customer authorization signature is invalid")


def _verify_output_storage_authorization_signature(
    payload: dict[str, Any], signature_record: dict[str, Any]
) -> None:
    public_key = _trusted_output_storage_authorization_public_key()
    fingerprint = hashlib.sha256(public_key).hexdigest()
    if signature_record.get("public_key_sha256") != fingerprint:
        raise BootstrapRefusal("output storage authorization trust root differs")
    try:
        signature = base64.b64decode(
            str(signature_record.get("signature_b64") or ""), validate=True
        )
    except ValueError as exc:
        raise BootstrapRefusal(
            "output storage authorization signature is invalid"
        ) from exc
    if len(signature) != 64:
        raise BootstrapRefusal("output storage authorization signature is invalid")
    unsigned = json.loads(json.dumps(payload))
    unsigned.pop("signature", None)
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    allowed_signer = (
        "npa-output-storage-control-plane ssh-ed25519 "
        + base64.b64encode(
            _ssh_string(b"ssh-ed25519") + _ssh_string(public_key)
        ).decode("ascii")
        + "\n"
    )
    with tempfile.TemporaryDirectory(prefix="npa-libero-storage-signature-") as root:
        allowed_path = Path(root) / "allowed-signers"
        signature_path = Path(root) / "authorization.sig"
        allowed_path.write_text(allowed_signer, encoding="ascii")
        signature_path.write_bytes(
            _sshsig_envelope(
                OUTPUT_STORAGE_AUTHORIZATION_NAMESPACE, public_key, signature
            )
        )
        os.chmod(allowed_path, 0o600)
        os.chmod(signature_path, 0o600)
        completed = subprocess.run(
            [
                "/usr/bin/ssh-keygen",
                "-Y",
                "verify",
                "-f",
                str(allowed_path),
                "-I",
                "npa-output-storage-control-plane",
                "-n",
                OUTPUT_STORAGE_AUTHORIZATION_NAMESPACE.decode("ascii"),
                "-s",
                str(signature_path),
            ],
            input=canonical,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"},
            check=False,
        )
    if completed.returncode:
        raise BootstrapRefusal("output storage authorization signature is invalid")


def _customer_acceptance_notification(
    manifest: dict[str, Any], *, manifest_sha256: str, reason: str
) -> dict[str, Any]:
    customer_identity = os.environ.get("NPA_LIBERO_CUSTOMER_IDENTITY_SHA256", "")
    return {
        "schema": "npa.libero.customer-acceptance-notification.v1",
        "status": "needs_customer_acceptance",
        "solution": "libero",
        "reason": reason,
        "runtime_manifest_sha256": manifest_sha256,
        "candidate_image": os.environ.get("BYOF_IMAGE") or None,
        "run_id": os.environ.get("NPA_BYOF_RUN_ID") or None,
        "customer_identity": (
            f"sha256:{customer_identity[:12]}…"
            if _is_hex(customer_identity, 64)
            else None
        ),
        "terms": [
            {
                "id": term["id"],
                "name": term["name"],
                "official_url": term["url"],
                "version": term["version"],
            }
            for term in manifest["governing_terms"]
        ],
        "acknowledgement": {
            "required": True,
            "instructions": (
                "Review every listed official term and authorize this exact "
                "customer and run using a customer-controlled signing key. "
                "The authenticated NPA surface transports and validates that "
                "evidence but does not accept or sign the terms assertion."
            ),
            "refusal": (
                "Decline or omit authorization to stop before runtime fetch, "
                "installation, cache mutation, or workload execution."
            ),
        },
        "credentials": {
            "purpose": "upstream_access_only",
            "establish_terms_acceptance": False,
        },
    }


def _validate_customer_authorization(
    path: Path,
    expected_sha256: str,
    manifest: dict[str, Any],
    manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Verify a customer/run authorization before all external effects."""

    if not _is_hex(expected_sha256, 64):
        raise CustomerAcceptanceRequired("authorization_missing")
    try:
        authorization_bytes = _read_private_regular_bytes(
            path,
            limit=1024 * 1024,
            input_name="customer authorization file",
        )
    except CustomerAcceptanceRequired:
        raise
    except BootstrapRefusal as exc:
        raise CustomerAcceptanceRequired("authorization_file_invalid") from exc
    if hashlib.sha256(authorization_bytes).hexdigest() != expected_sha256:
        raise CustomerAcceptanceRequired("authorization_hash_mismatch")
    try:
        _caller_bytes, _trusted_key, signer_sha256 = (
            _authenticated_caller_binding_from_environment()
        )
    except BootstrapRefusal as exc:
        raise CustomerAcceptanceRequired("authorization_signature_invalid") from exc
    return _validate_customer_authorization_bytes(
        authorization_bytes,
        expected_sha256,
        manifest,
        manifest_sha256,
        authenticated_signer_sha256=signer_sha256,
    )


def _validate_customer_authorization_bytes(
    authorization_bytes: bytes,
    expected_sha256: str,
    manifest: dict[str, Any],
    manifest_sha256: str,
    *,
    authenticated_signer_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Validate exact signed bytes, classifying customer-actionable failures."""

    observed_sha256 = hashlib.sha256(authorization_bytes).hexdigest()
    if observed_sha256 != expected_sha256:
        raise CustomerAcceptanceRequired("authorization_hash_mismatch")
    try:
        authorization = json.loads(authorization_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CustomerAcceptanceRequired("authorization_malformed") from exc
    if not isinstance(authorization, dict):
        raise CustomerAcceptanceRequired("authorization_malformed")
    signature_record = authorization.get("signature")
    expected_keys = {
        "schema",
        "solution",
        "status",
        "authorization_id",
        "customer_identity_sha256",
        "run_id",
        "candidate_image",
        "runtime_manifest_sha256",
        "workflow_profile_sha256",
        "upstream_source_revision",
        "terms",
        "issuer",
        "evidence_type",
        "customer_signer_public_key_b64",
        "acknowledged_at",
        "issued_at",
        "expires_at",
        "nonce",
        "signature",
    }
    expected_terms = [
        {"id": term["id"], "version": term["version"]}
        for term in manifest["governing_terms"]
    ]
    if (
        set(authorization) != expected_keys
        or authorization.get("schema") != CUSTOMER_AUTHORIZATION_SCHEMA
        or authorization.get("solution") != "libero"
        or authorization.get("status") not in {"authorized", "denied"}
        or re.fullmatch(
            r"[a-z0-9][a-z0-9-]{15,79}",
            str(authorization.get("authorization_id") or ""),
        )
        is None
        or not _is_hex(authorization.get("customer_identity_sha256"), 64)
        or authorization.get("customer_identity_sha256")
        != os.environ.get("NPA_LIBERO_CUSTOMER_IDENTITY_SHA256")
        or authorization.get("run_id") != os.environ.get("NPA_BYOF_RUN_ID")
        or re.fullmatch(
            r"[a-z0-9][a-z0-9-]{15,62}",
            str(authorization.get("run_id") or ""),
        )
        is None
        or authorization.get("candidate_image") != os.environ.get("BYOF_IMAGE")
        or re.fullmatch(
            r"ghcr\.io/nebius/nebius-physical-ai/npa-libero@sha256:[0-9a-f]{64}",
            str(authorization.get("candidate_image") or ""),
        )
        is None
        or authorization.get("runtime_manifest_sha256") != manifest_sha256
        or re.fullmatch(
            r"[0-9a-f]{64}", str(authorization.get("workflow_profile_sha256") or "")
        ) is None
        or authorization.get("workflow_profile_sha256")
        != os.environ.get("NPA_LIBERO_EXPECTED_EXECUTABLE_PROFILE_SHA256")
        or authorization.get("upstream_source_revision")
        != (manifest.get("source") or {}).get("revision")
        or authorization.get("terms") != expected_terms
        or authorization.get("issuer") != "customer"
        or authorization.get("evidence_type") != "customer-controlled-signature"
        or not isinstance(signature_record, dict)
        or set(signature_record) != {"algorithm", "public_key_sha256", "signature_b64"}
        or signature_record.get("algorithm") != "ed25519"
        or re.fullmatch(r"[A-Za-z0-9_-]{32,128}", str(authorization.get("nonce") or ""))
        is None
    ):
        raise CustomerAcceptanceRequired("authorization_wrong_scope")
    expected_signer = os.environ.get(
        "NPA_LIBERO_EXPECTED_CUSTOMER_SIGNER_PUBLIC_KEY_SHA256", ""
    )
    if not _is_hex(expected_signer, 64) or not hmac.compare_digest(
        expected_signer, authenticated_signer_sha256
    ):
        raise CustomerAcceptanceRequired("authorization_signer_binding_differs")
    try:
        acknowledged_at = datetime.fromisoformat(
            str(authorization["acknowledged_at"]).replace("Z", "+00:00")
        )
        issued_at = datetime.fromisoformat(
            str(authorization["issued_at"]).replace("Z", "+00:00")
        )
        expires_at = datetime.fromisoformat(
            str(authorization["expires_at"]).replace("Z", "+00:00")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CustomerAcceptanceRequired("authorization_timestamps_invalid") from exc
    now = datetime.now(timezone.utc)
    if (
        acknowledged_at.tzinfo is None
        or issued_at.tzinfo is None
        or expires_at.tzinfo is None
        or acknowledged_at > issued_at
        or issued_at - acknowledged_at > timedelta(minutes=5)
        or issued_at > now + timedelta(minutes=5)
        or issued_at >= expires_at
        or expires_at <= now
        or expires_at - issued_at > timedelta(hours=24)
    ):
        raise CustomerAcceptanceRequired("authorization_expired_or_replayable")
    expected_expires_raw = os.environ.get(
        "NPA_LIBERO_EXPECTED_CUSTOMER_AUTHORIZATION_EXPIRES_AT", ""
    )
    try:
        expected_expires_at = datetime.fromisoformat(
            expected_expires_raw.replace("Z", "+00:00")
        )
    except (TypeError, ValueError) as exc:
        raise CustomerAcceptanceRequired("authorization_expected_expiry_invalid") from exc
    if (
        expected_expires_at.tzinfo is None
        or expires_at != expected_expires_at
    ):
        raise CustomerAcceptanceRequired("authorization_expected_expiry_mismatch")
    try:
        _verify_customer_authorization_signature(
            authorization,
            signature_record,
            authenticated_signer_sha256=authenticated_signer_sha256,
        )
    except BootstrapRefusal as exc:
        if str(exc) in {
            "customer-authorization trust root is unavailable",
            "customer-authorization trust root is mutable or invalid",
            "customer-authorization trust root is invalid",
            "customer signer trust root is unavailable",
            "customer signer trust root is mutable or invalid",
            "customer signer trust root is invalid",
        }:
            raise
        raise CustomerAcceptanceRequired("authorization_signature_invalid") from exc
    if authorization["status"] == "denied":
        raise CustomerAcceptanceRequired("authorization_denied")
    return authorization, observed_sha256


def _ensure_deadline(deadline: datetime | None) -> None:
    if deadline is not None and datetime.now(timezone.utc) >= deadline:
        raise CustomerAcceptanceRequired("authorization_expired_or_replayable")


def _canonicalize_nvidia_terms(payload: bytes) -> bytes:
    """Normalize only the two equal site-navigation UUIDs, retaining all other bytes."""
    uuid = rb"[0-9a-f]{8}(?:_[0-9a-f]{4}){3}_[0-9a-f]{12}"
    nav = re.findall(
        rb'^        <nav class="global-nav" id="meganavigation(' + uuid + rb')">$',
        payload, re.M,
    )
    script = re.findall(
        rb'^\t        id : "meganavigation(' + uuid + rb')",\n'
        rb'\t        method : "navigation-megamenu",$', payload, re.M,
    )
    if (
        len(nav) != 1 or script != nav
        or len(re.findall(rb"meganavigation" + uuid, payload)) != 2
    ):
        raise BootstrapRefusal("NVIDIA terms navigation identity is invalid")
    return payload.replace(
        b"meganavigation" + nav[0],
        b"meganavigation00000000_0000_0000_0000_000000000000",
    )


def _download_verified(
    destination: Path,
    *,
    url: str,
    sha256: str,
    size: int,
    terms: bool = False,
    deadline: datetime | None = None,
    terms_normalization: str | None = None,
) -> None:
    _ensure_deadline(deadline)
    if size <= 0 or size > MAX_RUNTIME_CACHE_DOWNLOAD_BYTES:
        raise BootstrapRefusal("runtime download has no valid expected size")
    if terms_normalization is not None and (
        not terms or terms_normalization != NVIDIA_TERMS_NORMALIZATION
        or url != NVIDIA_SOFTWARE_TERMS_URL or size > 1024 * 1024
    ):
        raise BootstrapRefusal("governing terms normalization is invalid")
    (_validate_terms_url if terms else _validate_download_url)(url)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    observed_size = 0
    terms_payload = bytearray() if terms_normalization is not None else None
    raw_sha256: str | None = None
    connection: http.client.HTTPSConnection | None = None
    response: http.client.HTTPResponse | None = None
    try:
        connection, response = _open_https_download(url, terms=terms, deadline=deadline)
        content_length = response.getheader("Content-Length")
        if content_length is not None and (
            not content_length.isdigit() or int(content_length) != size
        ):
            raise BootstrapRefusal(
                "runtime download Content-Length differs from expected size"
            )
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            while True:
                _ensure_deadline(deadline)
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                if observed_size + len(chunk) > size:
                    raise BootstrapRefusal("runtime download exceeded expected size")
                stream.write(chunk)
                digest.update(chunk)
                if terms_payload is not None:
                    terms_payload.extend(chunk)
                observed_size += len(chunk)
            if terms_payload is not None:
                raw_sha256 = digest.hexdigest()
                canonical = _canonicalize_nvidia_terms(bytes(terms_payload))
                digest = hashlib.sha256(canonical)
                stream.seek(0)
                stream.write(canonical)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if response is not None:
            response.close()
        if connection is not None:
            connection.close()
    if digest.hexdigest() != sha256 or observed_size != size:
        temporary.unlink(missing_ok=True)
        raise BootstrapRefusal(
            "runtime download bytes do not match their immutable identity"
        )
    temporary.replace(destination)
    if terms_normalization is not None:
        print(json.dumps({
            "schema": "npa.libero.governing-terms-normalization.v1",
            "url": url, "normalization": terms_normalization,
            "raw_sha256": raw_sha256, "canonical_sha256": digest.hexdigest(),
            "size_bytes": observed_size,
        }, sort_keys=True), file=sys.stderr, flush=True)


def _governing_terms_identity(manifest: dict[str, Any]) -> str:
    identities = [
        {
            "id": term["id"],
            "boundary": term["boundary"],
            "sha256": term["sha256"],
            "size_bytes": term["size_bytes"],
            "url": term["url"],
            **({"normalization": term["normalization"]} if "normalization" in term else {}),
        }
        for term in manifest["governing_terms"]
    ]
    encoded = json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verify_governing_terms(
    manifest: dict[str, Any], *, deadline: datetime | None = None
) -> str:
    """Resolve and hash every governing terms source before cold cache mutation."""

    with tempfile.TemporaryDirectory(prefix="npa-libero-terms-") as temporary:
        root = Path(temporary)
        for index, term in enumerate(manifest["governing_terms"]):
            destination = root / f"term-{index}"
            _download_verified(
                destination,
                url=term["url"],
                sha256=term["sha256"],
                size=int(term["size_bytes"]),
                terms=True,
                deadline=deadline,
                terms_normalization=term.get("normalization"),
            )
    return _governing_terms_identity(manifest)


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
    deadline: datetime | None = None,
) -> None:
    selected_environment = dict(os.environ if environment is None else environment)
    selected_environment["GIT_TERMINAL_PROMPT"] = "0"
    _ensure_deadline(deadline)
    if deadline is None:
        subprocess.run(
            command,
            cwd=cwd,
            check=True,
            env=selected_environment,
            stdout=sys.stderr,
        )
        return
    child = _Popen(
        command,
        cwd=cwd,
        env=selected_environment,
        stdout=sys.stderr,
        start_new_session=True,
    )
    while child.poll() is None:
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=5.0)
            raise CustomerAcceptanceRequired("authorization_expired_or_replayable")
        try:
            child.wait(timeout=min(remaining, 1.0))
        except subprocess.TimeoutExpired:
            continue
    if child.returncode:
        raise subprocess.CalledProcessError(child.returncode, command)
    _ensure_deadline(deadline)


def _run_capture(
    command: list[str],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
    deadline: datetime | None = None,
) -> str:
    """Run one fetch command while bounding captured output by authorization."""

    selected_environment = dict(os.environ if environment is None else environment)
    selected_environment["GIT_TERMINAL_PROMPT"] = "0"
    _ensure_deadline(deadline)
    if deadline is None:
        return subprocess.check_output(
            command,
            cwd=cwd,
            env=selected_environment,
            text=True,
        )
    child = _Popen(
        command,
        cwd=cwd,
        env=selected_environment,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        start_new_session=True,
        text=True,
    )
    while child.poll() is None:
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=5.0)
            raise CustomerAcceptanceRequired("authorization_expired_or_replayable")
        try:
            child.wait(timeout=min(remaining, 1.0))
        except subprocess.TimeoutExpired:
            continue
    stdout, _ = child.communicate()
    _ensure_deadline(deadline)
    if child.returncode:
        raise subprocess.CalledProcessError(child.returncode, command, output=stdout)
    return stdout


def _runtime_materialization_environment(root: Path) -> dict[str, str]:
    """Return the credential-free environment shared by every fetch/build child."""

    environment = {
        name: value
        for name, value in os.environ.items()
        if name in RUNTIME_MATERIALIZATION_PASSTHROUGH_ENV_NAMES
    }
    environment.update(
        {
            "GIT_ASKPASS": "/bin/false",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": str(root),
            "PATH": "/usr/bin:/bin",
        }
    )
    return environment


def _fetch_source(
    root: Path,
    source: dict[str, Any],
    *,
    deadline: datetime | None = None,
) -> None:
    destination = root / "source"
    destination.mkdir(mode=0o700)
    environment = _runtime_materialization_environment(root)
    _run(
        ["/usr/bin/git", "init", "--quiet"],
        cwd=destination,
        environment=environment,
        deadline=deadline,
    )
    _run(
        ["/usr/bin/git", "remote", "add", "origin", source["repository"]],
        cwd=destination,
        environment=environment,
        deadline=deadline,
    )
    _run(
        ["/usr/bin/git", "config", "remote.origin.promisor", "true"],
        cwd=destination,
        environment=environment,
        deadline=deadline,
    )
    _run(
        ["/usr/bin/git", "config", "remote.origin.partialclonefilter", "blob:none"],
        cwd=destination,
        environment=environment,
        deadline=deadline,
    )
    _run(
        [
            "/usr/bin/git",
            "fetch",
            "--quiet",
            "--depth=1",
            "--filter=blob:none",
            "origin",
            source["revision"],
        ],
        cwd=destination,
        environment=environment,
        deadline=deadline,
    )
    fetched = _run_capture(
        ["/usr/bin/git", "rev-parse", "FETCH_HEAD^{commit}"],
        cwd=destination,
        environment=environment,
        deadline=deadline,
    ).strip()
    tree = _run_capture(
        ["/usr/bin/git", "rev-parse", "FETCH_HEAD^{tree}"],
        cwd=destination,
        environment=environment,
        deadline=deadline,
    ).strip()
    if fetched != source["revision"] or tree != source["tree"]:
        raise BootstrapRefusal(
            "fetched LIBERO source identity differs from the manifest"
        )
    _run(
        ["/usr/bin/git", "sparse-checkout", "init", "--no-cone"],
        cwd=destination,
        environment=environment,
        deadline=deadline,
    )
    _run(
        [
            "/usr/bin/git",
            "sparse-checkout",
            "set",
            "--no-cone",
            "--",
            *source["sparse_paths"],
        ],
        cwd=destination,
        environment=environment,
        deadline=deadline,
    )
    _run(
        ["/usr/bin/git", "checkout", "--quiet", "--detach", fetched],
        cwd=destination,
        environment=environment,
        deadline=deadline,
    )
    if _sha256(destination / source["license_file"]) != source["license_sha256"]:
        raise BootstrapRefusal(
            "LIBERO source license does not match the reviewed MIT file"
        )
    if (destination / "libero" / "libero" / "assets").exists():
        raise BootstrapRefusal("forbidden LIBERO render payload survived sparse fetch")
    shutil.rmtree(destination / ".git")


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _built_wheel_identity(path: Path) -> tuple[str, str]:
    """Read a generated wheel's identity without importing build tooling."""

    if path.is_symlink() or not path.is_file():
        raise BootstrapRefusal("generated runtime wheel is not a regular file")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077 or path.stat().st_uid != os.getuid():
        raise BootstrapRefusal("generated runtime wheel is not owner-private")
    try:
        with zipfile.ZipFile(path) as archive:
            metadata = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA") and "/" in name
            ]
            if len(metadata) != 1:
                raise BootstrapRefusal("generated runtime wheel metadata is ambiguous")
            payload = archive.read(metadata[0])
            if len(payload) > 1024 * 1024:
                raise BootstrapRefusal("generated runtime wheel metadata is too large")
            text = payload.decode("utf-8")
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise BootstrapRefusal("generated runtime wheel is unreadable") from exc
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in {"Name", "Version"}:
            values[key] = value.strip()
    if not values.get("Name") or not values.get("Version"):
        raise BootstrapRefusal("generated runtime wheel identity is incomplete")
    return values["Name"], values["Version"]


def _validate_read_only_system_submounts(source: Path, mountinfo: str) -> None:
    """Keep GPU-injected child mounts without introducing writable subtrees."""

    source = source.resolve(strict=True)
    for line in mountinfo.splitlines():
        fields = line.split()
        if len(fields) < 10 or "-" not in fields:
            raise BootstrapRefusal("source build mount inventory is invalid")
        mountpoint = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4]))
        if mountpoint.is_relative_to(source) and "ro" not in fields[5].split(","):
            raise BootstrapRefusal("source build system subtree contains a writable mount")


def _build_source_wheels(
    pip: str,
    source_artifacts: list[dict[str, Any]],
    source_lines: list[str],
    wheelhouse: Path,
    build_wheelhouse: Path,
    environment: dict[str, str],
    deadline: datetime | None = None,
) -> list[str]:
    """Build reviewed source archives into identity-checked local wheels."""

    if not source_artifacts:
        return []
    build_wheelhouse.mkdir(mode=0o700)
    source_lock = build_wheelhouse.parent / ".source-requirements.txt"
    source_lock.write_text("\n".join(source_lines) + "\n", encoding="utf-8")
    os.chmod(source_lock, 0o400)
    for sandbox_tool in (SOURCE_BUILD_UNSHARE, SOURCE_BUILD_CHROOT):
        try:
            sandbox_info = sandbox_tool.stat()
        except OSError as exc:
            raise BootstrapRefusal("source runtime sandbox is unavailable") from exc
        if (
            not stat.S_ISREG(sandbox_info.st_mode)
            or sandbox_info.st_uid != 0
            or stat.S_IMODE(sandbox_info.st_mode) & 0o022
            or not stat.S_IMODE(sandbox_info.st_mode) & 0o111
        ):
            raise BootstrapRefusal("source runtime sandbox is not trusted")
    wheelhouse = wheelhouse.resolve(strict=True)
    build_wheelhouse = build_wheelhouse.resolve(strict=True)
    source_lock = source_lock.resolve(strict=True)
    venv = Path(pip).resolve().parent.parent
    sandbox_root = build_wheelhouse.parent / ".source-build-sandbox"
    sandbox_root.mkdir(mode=0o700)
    for relative in (
        "bin",
        "etc",
        "lib",
        "lib64",
        "npa-build/input",
        "npa-build/output",
        "npa-build/venv",
        "sbin",
        "tmp",
        "usr",
    ):
        (sandbox_root / relative).mkdir(mode=0o700, parents=True)
    # Container /etc contains locked hosts/DNS/SSH submounts that cannot be
    # detached by a non-recursive bind inside an unprivileged user namespace.
    # Builds need only neutral account/loader data, never SSH keys or sudoers.
    for name in ("passwd", "group", "nsswitch.conf", "ld.so.cache"):
        source = Path("/etc") / name
        info = source.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise BootstrapRefusal("source build account/loader file is not trusted")
        shutil.copyfile(source, sandbox_root / "etc" / name)
        os.chmod(sandbox_root / "etc" / name, 0o400)
    (sandbox_root / "npa-build/source-requirements.txt").touch(mode=0o400)
    quote = shlex.quote
    readonly_binds = [
        ("/bin", sandbox_root / "bin"),
        ("/lib", sandbox_root / "lib"),
        ("/lib64", sandbox_root / "lib64"),
        ("/sbin", sandbox_root / "sbin"),
        ("/usr", sandbox_root / "usr"),
        (str(wheelhouse), sandbox_root / "npa-build/input"),
        (str(source_lock), sandbox_root / "npa-build/source-requirements.txt"),
        (str(venv), sandbox_root / "npa-build/venv"),
    ]
    bind_commands = []
    mountinfo = Path("/proc/self/mountinfo").read_text()
    system_roots = {"/bin", "/lib", "/lib64", "/sbin", "/usr"}
    for source, destination in readonly_binds:
        recursive = source in system_roots
        if recursive:
            _validate_read_only_system_submounts(Path(source), mountinfo)
        bind_commands.extend(
            [
                f"mount {'--rbind' if recursive else '--bind'} {quote(source)} {quote(str(destination))}",
                f"mount -o remount,ro,bind {quote(str(destination))}",
            ]
        )
    bind_commands.extend(
        [
            f"mount --bind {quote(str(build_wheelhouse))} {quote(str(sandbox_root / 'npa-build/output'))}",
            f"mount -t tmpfs tmpfs {quote(str(sandbox_root / 'tmp'))}",
        ]
    )
    build_command = " ".join(
        quote(part)
        for part in (
            "/npa-build/venv/bin/python",
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--require-hashes",
            "--no-index",
            "--find-links",
            "/npa-build/input",
            "--wheel-dir",
            "/npa-build/output",
            "-r",
            "/npa-build/source-requirements.txt",
        )
    )
    script = "set -eu; mount --make-rprivate /; " + "; ".join(bind_commands)
    script += (
        "; exec "
        + quote(str(SOURCE_BUILD_CHROOT))
        + " "
        + quote(str(sandbox_root))
        + " /usr/bin/env -i HOME=/tmp TMPDIR=/tmp PATH=/npa-build/venv/bin:/usr/bin:/bin "
        + build_command
    )
    sandbox_command = [
        str(SOURCE_BUILD_UNSHARE),
        "--user",
        "--map-root-user",
        "--mount",
        "--net",
        "--pid",
        "--fork",
        "--",
        "/bin/sh",
        "-ceu",
        script,
    ]
    try:
        _run(sandbox_command, environment=environment, deadline=deadline)
    finally:
        shutil.rmtree(sandbox_root, ignore_errors=False)
    wheels = sorted(path for path in build_wheelhouse.iterdir() if path.is_file())
    if len(wheels) != len(source_artifacts) or any(
        path.suffix != ".whl" for path in wheels
    ):
        raise BootstrapRefusal("source runtime build did not produce one wheel per archive")
    for path in wheels:
        os.chmod(path, 0o400)
    expected = {
        (_canonical_distribution_name(item["name"]), str(item["version"]))
        for item in source_artifacts
    }
    observed = {_built_wheel_identity(path) for path in wheels}
    observed = {(_canonical_distribution_name(name), version) for name, version in observed}
    if observed != expected:
        raise BootstrapRefusal("source runtime build produced unexpected wheel identities")
    by_identity = {
        (_canonical_distribution_name(_built_wheel_identity(path)[0]), _built_wheel_identity(path)[1]): path
        for path in wheels
    }
    return [
        f"{item['name']}=={item['version']} --hash=sha256:{_sha256(by_identity[(_canonical_distribution_name(item['name']), str(item['version']))])}"
        for item in source_artifacts
    ]


def _prepare_published_venv(venv: Path, published_venv: Path) -> None:
    """Remove venv's redundant alias and relocate generated text entrypoints."""

    alias = venv / "lib64"
    if alias.is_symlink():
        if os.readlink(alias) != "lib":
            raise BootstrapRefusal("runtime venv lib64 alias is unexpected")
        alias.unlink()
    old_prefix = str(venv).encode()
    new_prefix = str(published_venv).encode()
    for path in [venv / "pyvenv.cfg", *sorted((venv / "bin").iterdir())]:
        mode = path.lstat().st_mode
        # pip compiles installed .py scripts here. Recursive cache inventory
        # and sealing still reject symlinks or special entries inside it.
        if path.name == "__pycache__" and stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise BootstrapRefusal("runtime venv entrypoint is not a regular file")
        payload = path.read_bytes()
        # Python executables remain byte-identical; only generated UTF-8 scripts
        # and venv configuration embed the temporary installation prefix.
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if old_prefix in payload:
            path.write_bytes(payload.replace(old_prefix, new_prefix))


def _install_runtime(
    root: Path,
    artifacts: list[dict[str, Any]],
    requirement_lines: list[str],
    *,
    published_root: Path,
    deadline: datetime | None = None,
) -> None:
    materialization_environment = _runtime_materialization_environment(root)
    wheelhouse = root / "downloads"
    wheelhouse.mkdir(mode=0o700)
    resolved_wheelhouse = wheelhouse.resolve(strict=True)
    for item in artifacts:
        destination = wheelhouse / item["filename"]
        if destination.resolve(strict=False).parent != resolved_wheelhouse:
            raise BootstrapRefusal("runtime artifact destination leaves the wheelhouse")
        _download_verified(
            destination,
            url=item["url"],
            sha256=item["sha256"],
            size=int(item["size_bytes"]),
            deadline=deadline,
        )
    venv = root / "venv"
    _run(
        [sys.executable, "-m", "venv", "--copies", str(venv)],
        environment=materialization_environment,
        deadline=deadline,
    )
    bootstrap_names = {"pip", "setuptools", "wheel"}
    bootstrap_lock = root / ".bootstrap-requirements.txt"
    runtime_lock = root / ".runtime-requirements.txt"
    bootstrap_lock.write_text(
        "\n".join(
            line
            for line, item in zip(requirement_lines, artifacts, strict=True)
            if item["name"] in bootstrap_names
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_entries = [
        (line, item)
        for line, item in zip(requirement_lines, artifacts, strict=True)
        if item["name"] not in bootstrap_names
    ]
    wheel_entries = [(line, item) for line, item in runtime_entries if item["filename"].endswith(".whl")]
    source_entries = [(line, item) for line, item in runtime_entries if item["filename"].endswith(".tar.gz")]
    build_wheelhouse = root / ".built-wheelhouse"
    pip = str(venv / "bin" / "python")
    os.chmod(bootstrap_lock, 0o400)
    for artifact in wheelhouse.iterdir():
        os.chmod(artifact, 0o400)
    os.chmod(wheelhouse, 0o500)
    common = [
        pip,
        "-m",
        "pip",
        "install",
        "--only-binary=:all:",
        "--require-hashes",
        "--no-index",
        "--no-deps",
        "--no-cache-dir",
        "--find-links",
        str(wheelhouse),
    ]
    _run(
        [*common, "-r", str(bootstrap_lock)],
        environment=materialization_environment,
        deadline=deadline,
    )
    generated_source_lines = _build_source_wheels(
        pip,
        [item for _line, item in source_entries],
        [line for line, _item in source_entries],
        wheelhouse,
        build_wheelhouse,
        materialization_environment,
        deadline,
    )
    runtime_lock.write_text(
        "\n".join([line for line, _item in wheel_entries] + generated_source_lines)
        + "\n",
        encoding="utf-8",
    )
    os.chmod(runtime_lock, 0o400)
    _run(
        [
            *common,
            "--find-links",
            str(build_wheelhouse),
            "-r",
            str(runtime_lock),
        ],
        environment=materialization_environment,
        deadline=deadline,
    )
    _ensure_deadline(deadline)
    site_packages = _run_capture(
        [pip, "-c", "import site; print(site.getsitepackages()[0])"],
        environment=materialization_environment,
        deadline=deadline,
    ).strip()
    Path(site_packages, "npa-libero-source.pth").write_text(
        str(published_root / "source") + "\n", encoding="utf-8"
    )
    _prepare_published_venv(venv, published_root / "venv")
    os.chmod(wheelhouse, 0o700)
    shutil.rmtree(wheelhouse)
    if build_wheelhouse.exists():
        shutil.rmtree(build_wheelhouse)
    source_lock = root / ".source-requirements.txt"
    if source_lock.exists():
        source_lock.unlink()
    bootstrap_lock.unlink()
    runtime_lock.unlink()


def _fetch_inputs(
    root: Path, manifest: dict[str, Any], *, deadline: datetime | None = None
) -> None:
    demonstration = manifest["demonstration"]
    _download_verified(
        root / "data" / demonstration["filename"],
        url=demonstration["url"],
        sha256=demonstration["sha256"],
        size=int(demonstration["size_bytes"]),
        deadline=deadline,
    )
    model = manifest["language_model"]
    model_root = root / "models" / f"bert-base-cased-{model['revision']}"
    for item in model["files"]:
        url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{item['filename']}?download=true"
        _download_verified(
            model_root / item["filename"],
            url=url,
            sha256=item["sha256"],
            size=int(item["size_bytes"]),
            deadline=deadline,
        )
    model_record = {
        "repository": model["repository"],
        "revision": model["revision"],
        "files": {item["filename"]: item["sha256"] for item in model["files"]},
    }
    record_path = model_root / "npa-language-model.json"
    record_path.write_text(
        json.dumps(model_record, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(record_path, 0o600)


def _validate_task_inputs(root: Path, manifest: dict[str, Any]) -> None:
    source = root / "source"
    task = manifest["task"]
    for key in ("bddl", "initial_states", "embedding_source"):
        record = task[key]
        path = source / record["path"]
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise BootstrapRefusal(
                f"genuine upstream task input failed identity: {key}"
            )


def _inventory_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    excluded = {".complete.json", ".content-inventory.json"}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise BootstrapRefusal(
                f"runtime cache may not contain symlinks: {relative}"
            )
        elif stat.S_ISDIR(info.st_mode):
            entries.append({"path": relative, "type": "directory"})
        elif stat.S_ISREG(info.st_mode):
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "size_bytes": info.st_size,
                    "sha256": _sha256(path),
                }
            )
        else:
            raise BootstrapRefusal(
                f"runtime cache contains unsupported filesystem entry: {relative}"
            )
    return entries


def _write_content_inventory(root: Path) -> tuple[str, int]:
    entries = _inventory_entries(root)
    path = root / ".content-inventory.json"
    path.write_text(
        json.dumps({"schema": INVENTORY_SCHEMA, "entries": entries}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)
    return _sha256(path), len(entries)


def _seal_cache_tree(root: Path) -> None:
    try:
        execution_gid = grp.getgrnam(RUNTIME_EXECUTION_GROUP).gr_gid
    except KeyError as exc:
        raise BootstrapRefusal("runtime execution group is unavailable") from exc
    paths = sorted(
        (root, *root.rglob("*")),
        key=lambda item: len(item.relative_to(root).parts),
        reverse=True,
    )
    for path in paths:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise BootstrapRefusal("runtime cache may not contain symlinks")
        os.chown(path, -1, execution_gid)
        if stat.S_ISDIR(info.st_mode):
            os.chmod(path, SEALED_DIRECTORY_MODE)
        elif stat.S_ISREG(info.st_mode):
            os.chmod(
                path,
                SEALED_EXECUTABLE_MODE if info.st_mode & 0o111 else SEALED_REGULAR_MODE,
            )
        else:
            raise BootstrapRefusal("runtime cache contains an unsupported entry")


def _validate_read_only_tree(root: Path) -> None:
    for path in (root, *root.rglob("*")):
        # _open_cache_entry intentionally exposes the stable root through a
        # descriptor symlink under /proc/self/fd. Validate that descriptor's
        # target; no descendant symlink is part of the cache contract.
        info = path.stat() if path == root else path.lstat()
        if path != root and stat.S_ISLNK(info.st_mode):
            raise BootstrapRefusal("runtime cache may not contain symlinks")
        if stat.S_IMODE(info.st_mode) & 0o222:
            raise BootstrapRefusal("existing runtime cache is writable")


def _complete_record(
    root: Path,
    manifest: dict[str, Any],
    manifest_sha256: str,
    authorization_sha256: str,
    customer_identity_sha256: str,
    run_id: str,
    requirements_sha256: str,
    governing_terms_sha256: str,
) -> dict[str, Any]:
    source = manifest["source"]
    demonstration = manifest["demonstration"]
    inventory_sha256, inventory_entry_count = _write_content_inventory(root)
    record = {
        "schema": COMPLETE_SCHEMA,
        "solution": "libero",
        "manifest_sha256": manifest_sha256,
        "customer_authorization_sha256": authorization_sha256,
        "customer_identity_sha256": customer_identity_sha256,
        "run_id": run_id,
        "runtime_requirements_sha256": requirements_sha256,
        "governing_terms_sha256": governing_terms_sha256,
        "governing_terms_count": len(manifest["governing_terms"]),
        "source_revision": source["revision"],
        "source_tree": source["tree"],
        "source_license_sha256": source["license_sha256"],
        "runtime_artifact_count": len(manifest["runtime_artifacts"]),
        "demonstration_sha256": demonstration["sha256"],
        "demonstration_size_bytes": demonstration["size_bytes"],
        "task_bddl_sha256": manifest["task"]["bddl"]["sha256"],
        "task_initial_states_sha256": manifest["task"]["initial_states"]["sha256"],
        "language_model_revision": manifest["language_model"]["revision"],
        "render_assets_present": False,
        "git_objects_present": False,
        "cache_uploaded": False,
        "content_inventory_sha256": inventory_sha256,
        "content_inventory_entry_count": inventory_entry_count,
    }
    complete = root / ".complete.json"
    complete.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(complete, 0o600)
    return record


def _validate_complete(
    root: Path,
    manifest: dict[str, Any],
    manifest_sha256: str,
    authorization_sha256: str,
    customer_identity_sha256: str,
    run_id: str,
    requirements_sha256: str,
    governing_terms_sha256: str,
) -> dict[str, Any]:
    _validate_read_only_tree(root)
    record = _load_json(root / ".complete.json")
    expected = _complete_record_values(
        manifest,
        manifest_sha256,
        authorization_sha256,
        customer_identity_sha256,
        run_id,
        requirements_sha256,
        governing_terms_sha256,
    )
    dynamic_keys = {"content_inventory_sha256", "content_inventory_entry_count"}
    if set(record) != set(expected) | dynamic_keys or any(
        record.get(key) != value for key, value in expected.items()
    ):
        raise BootstrapRefusal(
            "existing runtime cache completion record does not match"
        )
    inventory_path = root / ".content-inventory.json"
    if (
        not inventory_path.is_file()
        or inventory_path.is_symlink()
        or not _is_hex(record["content_inventory_sha256"], 64)
        or _sha256(inventory_path) != record["content_inventory_sha256"]
    ):
        raise BootstrapRefusal("existing runtime cache inventory is invalid")
    inventory = _load_json(inventory_path)
    entries = inventory.get("entries")
    if (
        inventory.get("schema") != INVENTORY_SCHEMA
        or not isinstance(entries, list)
        or record["content_inventory_entry_count"] != len(entries)
        or entries != _inventory_entries(root)
    ):
        raise BootstrapRefusal("existing runtime cache content differs from inventory")
    if (root / "source" / "libero" / "libero" / "assets").exists() or (
        root / "source" / ".git"
    ).exists():
        raise BootstrapRefusal(
            "existing runtime cache contains forbidden source payload"
        )
    if not (root / "venv" / "bin" / "python").is_file():
        raise BootstrapRefusal("existing runtime cache is incomplete")
    demonstration = manifest["demonstration"]
    data = root / "data" / demonstration["filename"]
    if (
        not data.is_file()
        or data.stat().st_size != demonstration["size_bytes"]
        or _sha256(data) != demonstration["sha256"]
    ):
        raise BootstrapRefusal("existing demonstration cache does not match")
    model = manifest["language_model"]
    model_root = root / "models" / f"bert-base-cased-{model['revision']}"
    for item in model["files"]:
        path = model_root / item["filename"]
        if (
            not path.is_file()
            or path.stat().st_size != item["size_bytes"]
            or _sha256(path) != item["sha256"]
        ):
            raise BootstrapRefusal("existing task language-model cache does not match")
    _validate_task_inputs(root, manifest)
    return record


def _validate_and_publish_cache(
    *,
    cache_root: Path,
    final: Path,
    identity: tuple[int, int],
    manifest: dict[str, Any],
    manifest_sha256: str,
    authorization_sha256: str,
    customer_identity_sha256: str,
    run_id: str,
    requirements_sha256: str,
    governing_terms_sha256: str,
) -> dict[str, Any]:
    try:
        with _open_cache_entry(final, expected=identity) as stable_final:
            record = _validate_complete(
                stable_final,
                manifest,
                manifest_sha256,
                authorization_sha256,
                customer_identity_sha256,
                run_id,
                requirements_sha256,
                governing_terms_sha256,
            )
            _publish_current_cache_link(cache_root, final, identity)
    except Exception:
        _remove_current_cache_link(cache_root, final)
        raise
    return record


def _complete_record_values(
    manifest: dict[str, Any],
    manifest_sha256: str,
    authorization_sha256: str,
    customer_identity_sha256: str,
    run_id: str,
    requirements_sha256: str,
    governing_terms_sha256: str,
) -> dict[str, Any]:
    source = manifest["source"]
    demonstration = manifest["demonstration"]
    return {
        "schema": COMPLETE_SCHEMA,
        "solution": "libero",
        "manifest_sha256": manifest_sha256,
        "customer_authorization_sha256": authorization_sha256,
        "customer_identity_sha256": customer_identity_sha256,
        "run_id": run_id,
        "runtime_requirements_sha256": requirements_sha256,
        "governing_terms_sha256": governing_terms_sha256,
        "governing_terms_count": len(manifest["governing_terms"]),
        "source_revision": source["revision"],
        "source_tree": source["tree"],
        "source_license_sha256": source["license_sha256"],
        "runtime_artifact_count": len(manifest["runtime_artifacts"]),
        "demonstration_sha256": demonstration["sha256"],
        "demonstration_size_bytes": demonstration["size_bytes"],
        "task_bddl_sha256": manifest["task"]["bddl"]["sha256"],
        "task_initial_states_sha256": manifest["task"]["initial_states"]["sha256"],
        "language_model_revision": manifest["language_model"]["revision"],
        "render_assets_present": False,
        "git_objects_present": False,
        "cache_uploaded": False,
    }


def _runtime_cache_scope_sha256(
    manifest_sha256: str,
    customer_identity_sha256: str,
    run_id: str,
    authorization_sha256: str,
) -> str:
    if (
        not _is_hex(manifest_sha256, 64)
        or not _is_hex(customer_identity_sha256, 64)
        or not _is_hex(authorization_sha256, 64)
        or re.fullmatch(r"[a-z0-9][a-z0-9-]{15,62}", run_id) is None
    ):
        raise BootstrapRefusal("runtime cache scope is invalid")
    return hashlib.sha256(
        json.dumps(
            {
                "schema": "npa.libero.customer-runtime-cache-scope.v2",
                "customer_authorization_sha256": authorization_sha256,
                "customer_identity_sha256": customer_identity_sha256,
                "run_id": run_id,
                "runtime_manifest_sha256": manifest_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def ensure(args: argparse.Namespace) -> dict[str, Any]:
    manifest, manifest_sha256 = _validate_manifest(Path(args.manifest))
    authorization, authorization_sha256 = _validate_customer_authorization(
        Path(args.authorization), args.authorization_sha256, manifest, manifest_sha256
    )
    _validate_executable_profile_digest(authorization["workflow_profile_sha256"])
    customer_identity_sha256 = authorization["customer_identity_sha256"]
    run_id = authorization["run_id"]
    authorization_expires_at = _parse_utc(
        authorization["expires_at"], "customer authorization expires_at"
    )
    requirement_lines, requirements_sha256 = _validate_requirements(
        Path(args.requirements), manifest
    )
    output = Path(args.output_dir) if args.output_dir else None
    cache_root = _validate_cache_root(Path(args.cache_root), output)
    scope_sha256 = _runtime_cache_scope_sha256(
        manifest_sha256,
        customer_identity_sha256,
        run_id,
        authorization_sha256,
    )
    display_final = cache_root / scope_sha256
    governing_terms_sha256: str | None = None
    cache_root_created = False

    def revalidate_authorization() -> None:
        """Recheck the signed customer/run binding before every fetch boundary."""

        try:
            _caller_bytes, _trusted_key, signer_sha256 = (
                _authenticated_caller_binding_from_environment()
            )
            current_bytes = _read_private_regular_bytes(
                Path(args.authorization),
                limit=1024 * 1024,
                input_name="customer authorization file",
            )
        except CustomerAcceptanceRequired:
            raise
        except BootstrapRefusal as exc:
            raise CustomerAcceptanceRequired("authorization_signature_invalid") from exc
        current, current_sha256 = _validate_customer_authorization_bytes(
            current_bytes,
            authorization_sha256,
            manifest,
            manifest_sha256,
            authenticated_signer_sha256=signer_sha256,
        )
        if (
            current_sha256 != authorization_sha256
            or current["customer_identity_sha256"] != customer_identity_sha256
            or current["run_id"] != run_id
        ):
            raise CustomerAcceptanceRequired("authorization_changed")

    def verify_terms_before_cache_creation(_parent_descriptor: int, _name: str) -> None:
        nonlocal cache_root_created, governing_terms_sha256
        revalidate_authorization()
        governing_terms_sha256 = _verify_governing_terms(
            manifest, deadline=authorization_expires_at
        )
        cache_root_created = True

    with _open_cache_root_descriptor(
        cache_root,
        create=True,
        before_create=verify_terms_before_cache_creation,
    ) as root_fd:
        initial_identity = _cache_entry_identity_at(root_fd, scope_sha256)
        if cache_root_created and initial_identity is not None:
            raise BootstrapRefusal("runtime cache entry appeared during creation")
        if initial_identity is not None:
            governing_terms_sha256 = _governing_terms_identity(manifest)
        elif governing_terms_sha256 is None:
            revalidate_authorization()
            governing_terms_sha256 = _verify_governing_terms(
                manifest, deadline=authorization_expires_at
            )
        try:
            group_id = grp.getgrnam(RUNTIME_EXECUTION_GROUP).gr_gid
        except KeyError as exc:
            raise BootstrapRefusal("runtime execution group is unavailable") from exc
        os.fchown(root_fd, -1, group_id)
        os.fchmod(root_fd, CACHE_ROOT_MODE)
        stable_cache_root = Path("/proc/self/fd") / str(root_fd)
        final = stable_cache_root / scope_sha256
        assert governing_terms_sha256 is not None
        with _cache_lock(root_fd, ".bootstrap.lock", exclusive=True, create=True):
            locked_identity = _cache_entry_identity_at(root_fd, scope_sha256)
            if locked_identity != initial_identity:
                raise BootstrapRefusal("runtime cache entry changed before validation")
            warm_reuse = locked_identity is not None
            if warm_reuse:
                assert locked_identity is not None
                revalidate_authorization()
                record = _validate_and_publish_cache(
                    cache_root=stable_cache_root,
                    final=final,
                    identity=locked_identity,
                    manifest=manifest,
                    manifest_sha256=manifest_sha256,
                    authorization_sha256=authorization_sha256,
                    customer_identity_sha256=customer_identity_sha256,
                    run_id=run_id,
                    requirements_sha256=requirements_sha256,
                    governing_terms_sha256=governing_terms_sha256,
                )
            else:
                partial = Path(
                    tempfile.mkdtemp(
                        prefix=f".{manifest_sha256}.partial-",
                        # Keep the retained directory inode anchored across venv,
                        # ensurepip and pip descendants that close inherited FDs.
                        # The owning materializer holds root_fd until publication.
                        dir=Path("/proc") / str(os.getpid()) / "fd" / str(root_fd),
                    )
                )
                os.chmod(partial, 0o700)
                renamed = False
                committed = False
                renamed_identity: tuple[int, int] | None = None
                try:
                    revalidate_authorization()
                    _fetch_source(
                        partial,
                        manifest["source"],
                        deadline=authorization_expires_at,
                    )
                    _validate_task_inputs(partial, manifest)
                    revalidate_authorization()
                    _install_runtime(
                        partial,
                        manifest["runtime_artifacts"],
                        requirement_lines,
                        published_root=display_final,
                        deadline=authorization_expires_at,
                    )
                    revalidate_authorization()
                    _fetch_inputs(partial, manifest, deadline=authorization_expires_at)
                    record = _complete_record(
                        partial,
                        manifest,
                        manifest_sha256,
                        authorization_sha256,
                        customer_identity_sha256,
                        run_id,
                        requirements_sha256,
                        governing_terms_sha256,
                    )
                    _seal_cache_tree(partial)
                    renamed_identity = _cache_entry_identity(partial)
                    if renamed_identity is None:
                        raise BootstrapRefusal(
                            "materialized runtime cache is unavailable"
                        )
                    os.rename(
                        partial.name,
                        scope_sha256,
                        src_dir_fd=root_fd,
                        dst_dir_fd=root_fd,
                    )
                    renamed = True
                    locked_identity = _cache_entry_identity_at(root_fd, scope_sha256)
                    if locked_identity != renamed_identity:
                        raise BootstrapRefusal(
                            "materialized runtime cache is unavailable"
                        )
                    revalidate_authorization()
                    record = _validate_and_publish_cache(
                        cache_root=stable_cache_root,
                        final=final,
                        identity=locked_identity,
                        manifest=manifest,
                        manifest_sha256=manifest_sha256,
                        authorization_sha256=authorization_sha256,
                        customer_identity_sha256=customer_identity_sha256,
                        run_id=run_id,
                        requirements_sha256=requirements_sha256,
                        governing_terms_sha256=governing_terms_sha256,
                    )
                    committed = True
                except Exception as exc:
                    rollback_errors: tuple[str, ...] = ()
                    if renamed and not committed and renamed_identity is not None:
                        rollback_errors = _rollback_new_cache_entry(
                            stable_cache_root,
                            final,
                            renamed_identity,
                            parent_descriptor=root_fd,
                        )
                    partial_errors = _remove_cache_tree(partial)
                    if partial.exists():
                        partial_errors.append("partial cache remains")
                    cleanup_errors = (*rollback_errors, *partial_errors)
                    if cleanup_errors:
                        cleanup_kind = (
                            "runtime cache rollback"
                            if rollback_errors
                            else "runtime cache partial cleanup"
                        )
                        raise BootstrapRefusal(
                            f"{cleanup_kind} was incomplete: "
                            + ", ".join(dict.fromkeys(cleanup_errors))
                        ) from exc
                    raise
    return {
        **record,
        "cache_path": str(display_final),
        "warm_reuse": warm_reuse,
        "governing_terms_fetched_this_invocation": not warm_reuse,
    }


def status(args: argparse.Namespace) -> dict[str, Any]:
    manifest, manifest_sha256 = _validate_manifest(
        Path(args.manifest), require_runtime_closure=False
    )
    _, requirements_sha256 = _validate_requirements(Path(args.requirements), manifest)
    cache_root = _validate_cache_root(Path(args.cache_root), None)
    materialized = False
    current = cache_root / "current"
    if current.is_symlink():
        target = os.readlink(current)
        if not _is_hex(target, 64):
            raise BootstrapRefusal("runtime current link has an invalid scope")
        final = cache_root / target
        identity = _cache_entry_identity(final)
        if identity is None:
            raise BootstrapRefusal("runtime current link target is unavailable")
        with _open_cache_entry(final, expected=identity) as stable_final:
            record = _load_json(stable_final / ".complete.json")
            authorization_sha256 = str(
                record.get("customer_authorization_sha256") or ""
            )
            customer_identity_sha256 = str(record.get("customer_identity_sha256") or "")
            run_id = str(record.get("run_id") or "")
            if not _is_hex(authorization_sha256, 64):
                raise BootstrapRefusal(
                    "existing runtime cache authorization identity is invalid"
                )
            _validate_complete(
                stable_final,
                manifest,
                manifest_sha256,
                authorization_sha256,
                customer_identity_sha256,
                run_id,
                requirements_sha256,
                _governing_terms_identity(manifest),
            )
            materialized = True
    return {
        "schema": "npa.libero.runtime-status.v1",
        "solution": "libero",
        "manifest_sha256": manifest_sha256,
        "materialized": materialized,
        "source_revision": manifest["source"]["revision"],
    }


def _runtime_execution_environment(stable_root: Path) -> dict[str, str]:
    return {
        **{
            name: value
            for name, value in os.environ.items()
            if name in RUNTIME_EXECUTION_PASSTHROUGH_ENV_NAMES
        },
        "HOME": "/nonexistent",
        "LIBERO_RUNTIME_ROOT": str(stable_root),
        "PATH": "/usr/bin:/bin",
    }


def _execution_authorization_values(
    manifest_sha256: str,
) -> tuple[str, str, str, str]:
    authorization_sha256 = os.environ.get(
        "NPA_LIBERO_CUSTOMER_AUTHORIZATION_SHA256", ""
    ).strip()
    customer_identity_sha256 = os.environ.get(
        "NPA_LIBERO_CUSTOMER_IDENTITY_SHA256", ""
    ).strip()
    run_id = os.environ.get("NPA_BYOF_RUN_ID", "").strip()
    if not _is_hex(authorization_sha256, 64):
        raise BootstrapRefusal("customer authorization SHA-256 is unavailable")
    scope_sha256 = _runtime_cache_scope_sha256(
        manifest_sha256,
        customer_identity_sha256,
        run_id,
        authorization_sha256,
    )
    return authorization_sha256, customer_identity_sha256, run_id, scope_sha256


def _require_current_cache_link(
    cache_root_descriptor: int, expected: str, refusal: str
) -> None:
    """Require the descriptor-relative current link to name the expected cache."""

    try:
        current = os.readlink("current", dir_fd=cache_root_descriptor)
    except OSError:
        raise BootstrapRefusal(refusal) from None
    if current != expected:
        raise BootstrapRefusal(refusal)


def _close_sensitive_execution_descriptor(descriptor: int, name: str) -> None:
    """Remove one customer-evidence descriptor before launching runtime code."""

    try:
        os.close(descriptor)
    except OSError as exc:
        if exc.errno != errno.EBADF:
            raise BootstrapRefusal(f"{name} descriptor could not be closed") from exc


def _run_authorized_smoke(
    *,
    cache_descriptor: int,
    environment: dict[str, str],
    expires_at: datetime,
) -> int:
    """Run the untrusted child only while the signed authorization is valid."""

    child = _Popen(
        ["/opt/npa/libero/smoke.sh"],
        env=environment,
        pass_fds=(cache_descriptor,),
        start_new_session=True,
    )
    while child.poll() is None:
        remaining = (expires_at - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=5.0)
            raise CustomerAcceptanceRequired("authorization_expired_or_replayable")
        try:
            child.wait(timeout=min(remaining, 1.0))
        except subprocess.TimeoutExpired:
            continue
    return int(child.returncode)


def execute(
    cache_descriptor: int = INHERITED_CACHE_DESCRIPTOR,
    authorization_descriptor: int = INHERITED_AUTHORIZATION_DESCRIPTOR,
    execution_lock_descriptor: int = INHERITED_EXECUTION_LOCK_DESCRIPTOR,
    bootstrap_lock_descriptor: int = INHERITED_BOOTSTRAP_LOCK_DESCRIPTOR,
    caller_descriptor: int = INHERITED_CALLER_DESCRIPTOR,
    customer_trust_descriptor: int = INHERITED_CUSTOMER_TRUST_DESCRIPTOR,
) -> int:
    """Run the fixed smoke from one locked, descriptor-stable cache snapshot."""

    manifest, manifest_sha256 = _validate_manifest(DEFAULT_MANIFEST)
    _, requirements_sha256 = _validate_requirements(DEFAULT_REQUIREMENTS, manifest)
    expected_authorization_sha256, expected_customer, expected_run, scope_sha256 = (
        _execution_authorization_values(manifest_sha256)
    )
    try:
        supervisor_uid = pwd.getpwnam(RUNTIME_SUPERVISOR_USER).pw_uid
    except KeyError as exc:
        raise BootstrapRefusal("runtime supervisor account is unavailable") from exc
    caller_bytes = _read_private_regular_descriptor(
        caller_descriptor,
        limit=1024 * 1024,
        owner_uid=supervisor_uid,
        input_name="authenticated caller assertion",
        allow_unlinked=True,
    )
    trusted_public_key = _read_private_regular_descriptor(
        customer_trust_descriptor,
        limit=1024,
        owner_uid=supervisor_uid,
        input_name="authenticated caller trust root",
        allow_unlinked=True,
    )
    try:
        if _customer_run():
            if caller_bytes or not hmac.compare_digest(
                trusted_public_key, _customer_run_key()
            ):
                raise BootstrapRefusal("customer-run signer handoff differs")
            caller_signer_sha256 = hashlib.sha256(trusted_public_key).hexdigest()
        else:
            caller_signer_sha256 = _validate_authenticated_caller_binding(
                caller_bytes,
                trusted_public_key,
                expected_sha256=os.environ.get(
                    "NPA_LIBERO_AUTHENTICATED_CALLER_SHA256", ""
                ),
                expected_run_id=expected_run,
                expected_customer_identity_sha256=expected_customer,
            )
    except BootstrapRefusal as exc:
        reason = (
            "authorization_expired_or_replayable"
            if "expired" in str(exc)
            else "authorization_signature_invalid"
        )
        raise CustomerAcceptanceRequired(reason) from exc
    try:
        authorization_bytes = _read_private_regular_descriptor(
            authorization_descriptor,
            limit=1024 * 1024,
            owner_uid=supervisor_uid,
            input_name="customer authorization file",
        )
    except BootstrapRefusal as exc:
        raise CustomerAcceptanceRequired("authorization_file_invalid") from exc
    authorization, authorization_sha256 = _validate_customer_authorization_bytes(
        authorization_bytes,
        expected_authorization_sha256,
        manifest,
        manifest_sha256,
        authenticated_signer_sha256=caller_signer_sha256,
    )
    _validate_executable_profile_digest(authorization["workflow_profile_sha256"])
    customer_identity_sha256 = authorization["customer_identity_sha256"]
    run_id = authorization["run_id"]
    if customer_identity_sha256 != expected_customer or run_id != expected_run:
        raise CustomerAcceptanceRequired("authorization_wrong_scope")
    expires_at = _parse_utc(
        authorization["expires_at"], "customer authorization expires_at"
    )
    cache_root = _validate_cache_root(DEFAULT_CACHE, None)
    with _open_cache_root_descriptor(
        cache_root, create=False, owner_uid=supervisor_uid
    ) as root_fd:
        identity = _cache_entry_identity_at(root_fd, scope_sha256)
        if identity is None:
            raise BootstrapRefusal("runtime cache is not materialized")
        _require_inherited_lock(root_fd, ".execution.lock", execution_lock_descriptor)
        _require_inherited_lock(root_fd, ".bootstrap.lock", bootstrap_lock_descriptor)
        _require_current_cache_link(
            root_fd,
            scope_sha256,
            "runtime current link differs from the accepted cache",
        )
        governing_terms_sha256 = _governing_terms_identity(manifest)
        try:
            opened = os.fstat(cache_descriptor)
        except OSError as exc:
            raise BootstrapRefusal(
                "inherited runtime cache descriptor is unavailable"
            ) from exc
        if (opened.st_dev, opened.st_ino) != identity or not stat.S_ISDIR(
            opened.st_mode
        ):
            raise BootstrapRefusal("runtime cache descriptor identity changed")
        stable_root = Path("/proc/self/fd") / str(cache_descriptor)
        _validate_complete(
            stable_root,
            manifest,
            manifest_sha256,
            authorization_sha256,
            customer_identity_sha256,
            run_id,
            requirements_sha256,
            governing_terms_sha256,
        )
        _close_sensitive_execution_descriptor(
            authorization_descriptor, "customer authorization"
        )
        _close_sensitive_execution_descriptor(caller_descriptor, "authenticated caller")
        _close_sensitive_execution_descriptor(
            customer_trust_descriptor, "authenticated caller trust root"
        )
        environment = _runtime_execution_environment(stable_root)
        if _customer_run():
            # The trusted supervisor needs sudo for this UID transition; the
            # subsequently executed third-party runtime must never regain it.
            if ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0) != 0:
                raise BootstrapRefusal("training no-new-privileges setup failed")
        smoke_exit_code = _run_authorized_smoke(
            cache_descriptor=cache_descriptor,
            environment=environment,
            expires_at=expires_at,
        )
        _validate_complete(
            stable_root,
            manifest,
            manifest_sha256,
            authorization_sha256,
            customer_identity_sha256,
            run_id,
            requirements_sha256,
            governing_terms_sha256,
        )
        if _cache_entry_identity_at(root_fd, scope_sha256) != identity:
            raise BootstrapRefusal("runtime cache entry changed during execution")
        _require_current_cache_link(
            root_fd,
            scope_sha256,
            "runtime current link changed during execution",
        )
        return smoke_exit_code


def _parse_utc(value: object, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise BootstrapRefusal(f"{label} is not a valid timestamp") from exc
    if parsed.tzinfo is None:
        raise BootstrapRefusal(f"{label} must include a timezone")
    return parsed


def _storage_authorization(
    output_prefix: str, run_id: str, endpoint: str
) -> dict[str, Any]:
    encoded = os.environ.get("NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_B64", "").strip()
    expected = os.environ.get(
        "NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_AUTHORIZATION_SHA256", ""
    ).strip()
    try:
        payload = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise BootstrapRefusal(
            "output storage authorization is not valid Base64"
        ) from exc
    if not _is_hex(expected, 64) or hashlib.sha256(payload).hexdigest() != expected:
        raise BootstrapRefusal(
            "output storage authorization is not control-plane-bound"
        )
    try:
        authorization = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapRefusal(
            "output storage authorization is not valid JSON"
        ) from exc
    keys = {
        "schema",
        "issuer",
        "capability_id",
        "customer_identity_sha256",
        "run_id",
        "candidate_image",
        "runtime_manifest_sha256",
        "output_prefix",
        "endpoint_url",
        "access_key_id_sha256",
        "secret_access_key_sha256",
        "session_token_sha256",
        "policy_sha256",
        "lease_key",
        "lease_etag",
        "lease_version_id",
        "lease_nonce",
        "issued_at",
        "expires_at",
        "nonce",
        "signature",
    }
    if not isinstance(authorization, dict) or set(authorization) != keys:
        raise BootstrapRefusal("output storage authorization schema is not closed")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    session_token = os.environ.get("AWS_SESSION_TOKEN", "")
    prefix_sha256 = hashlib.sha256(output_prefix.encode()).hexdigest()
    _, runtime_manifest_sha256 = _validate_manifest(DEFAULT_MANIFEST)
    issued_at = _parse_utc(authorization.get("issued_at"), "authorization issued_at")
    expires_at = _parse_utc(authorization.get("expires_at"), "authorization expires_at")
    expected_lease_key = os.environ.get("NPA_LIBERO_EXPECTED_OUTPUT_LEASE_KEY", "")
    expected_lease_etag = os.environ.get("NPA_LIBERO_EXPECTED_OUTPUT_LEASE_ETAG", "")
    expected_lease_version_id = os.environ.get(
        "NPA_LIBERO_EXPECTED_OUTPUT_LEASE_VERSION_ID", ""
    )
    customer_authorization_expires_at = _parse_utc(
        os.environ.get("NPA_LIBERO_EXPECTED_CUSTOMER_AUTHORIZATION_EXPIRES_AT"),
        "customer authorization expires_at",
    )
    now = datetime.now(timezone.utc)
    valid = (
        authorization.get("schema") == OUTPUT_STORAGE_AUTHORIZATION_SCHEMA
        and authorization.get("issuer") == "npa-output-storage-control-plane"
        and re.fullmatch(
            r"[a-z0-9][a-z0-9-]{15,79}",
            str(authorization.get("capability_id") or ""),
        )
        is not None
        and authorization.get("customer_identity_sha256")
        == os.environ.get("NPA_LIBERO_CUSTOMER_IDENTITY_SHA256")
        and authorization.get("run_id") == run_id
        and authorization.get("candidate_image") == os.environ.get("BYOF_IMAGE")
        and authorization.get("runtime_manifest_sha256") == runtime_manifest_sha256
        and authorization.get("output_prefix") == output_prefix
        and authorization.get("endpoint_url") == endpoint
        and prefix_sha256
        == os.environ.get("NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_PREFIX_SHA256", "")
        and authorization.get("policy_sha256")
        == os.environ.get("NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_POLICY_SHA256", "")
        and authorization.get("lease_key") == expected_lease_key
        and authorization.get("lease_etag") == expected_lease_etag
        and authorization.get("lease_version_id") == expected_lease_version_id
        and authorization.get("lease_key")
        and re.fullmatch(r'"[^\"]+"', str(authorization.get("lease_etag") or ""))
        is not None
        and authorization.get("lease_version_id")
        and re.fullmatch(
            r"[A-Za-z0-9_-]{32,128}", str(authorization.get("lease_nonce") or "")
        )
        is not None
        and all((access_key, secret_key, session_token))
        and authorization.get("access_key_id_sha256")
        == hashlib.sha256(access_key.encode()).hexdigest()
        and authorization.get("secret_access_key_sha256")
        == hashlib.sha256(secret_key.encode()).hexdigest()
        and authorization.get("session_token_sha256")
        == hashlib.sha256(session_token.encode()).hexdigest()
        and issued_at <= now + timedelta(minutes=5)
        and issued_at < expires_at
        and expires_at > now
        and expires_at - issued_at <= timedelta(hours=24)
        and expires_at <= customer_authorization_expires_at
        and isinstance(authorization.get("signature"), dict)
        and set(authorization["signature"])
        == {"algorithm", "public_key_sha256", "signature_b64"}
        and authorization["signature"].get("algorithm") == "ed25519"
    )
    if not valid:
        raise BootstrapRefusal("output storage authorization is invalid or expired")
    _verify_output_storage_authorization_signature(
        authorization, authorization["signature"]
    )
    return authorization


def _output_version_query(method: str, query: str) -> str:
    """Permit only one canonical immutable-version selector for object reads/deletes."""

    if not query:
        return ""
    name, separator, encoded = query.partition("=")
    if method not in {"GET", "HEAD", "DELETE"} or name != "versionId" or not separator:
        raise BootstrapRefusal("output storage version query is not allowed")
    try:
        version_id = urllib.parse.unquote(encoded, errors="strict")
    except UnicodeError as exc:
        raise BootstrapRefusal("output storage version query is not canonical") from exc
    if (
        not version_id
        or version_id == "null"
        or len(version_id.encode()) > 1024
        or any(ord(character) < 32 or ord(character) == 127 for character in version_id)
    ):
        raise BootstrapRefusal("output storage immutable version identity is invalid")
    canonical = "versionId=" + urllib.parse.quote(version_id, safe="-_.~")
    if query != canonical:
        raise BootstrapRefusal("output storage version query is not canonical")
    return canonical


def _sigv4_request(
    method: str,
    url: str,
    *,
    payload: bytes = b"",
    extra_headers: dict[str, str] | None = None,
    expected_statuses: frozenset[int] = frozenset({200}),
) -> tuple[int, dict[str, str], bytes]:
    access_key = os.environ["AWS_ACCESS_KEY_ID"]
    secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    session_token = os.environ["AWS_SESSION_TOKEN"]
    region = os.environ.get("AWS_DEFAULT_REGION", "us-central1")
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise BootstrapRefusal(
            "output storage endpoint must be a credential-free HTTPS URL"
        )
    canonical_query = _output_version_query(method, parsed.query)
    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    headers = {
        "host": parsed.netloc,
        "x-amz-content-sha256": payload_sha256,
        "x-amz-date": amz_date,
        "x-amz-security-token": session_token,
        **{key.lower(): value for key, value in (extra_headers or {}).items()},
    }
    signed_names = ";".join(sorted(headers))
    canonical_headers = "".join(
        f"{name}:{' '.join(headers[name].split())}\n" for name in sorted(headers)
    )
    canonical_uri = urllib.parse.quote(urllib.parse.unquote(parsed.path), safe="/-_.~")
    if parsed.path != canonical_uri:
        raise BootstrapRefusal("output storage object path is not canonical")
    canonical_request = "\n".join(
        (
            method,
            canonical_uri,
            canonical_query,
            canonical_headers,
            signed_names,
            payload_sha256,
        )
    )
    scope = f"{date_stamp}/{region}/s3/aws4_request"
    string_to_sign = "\n".join(
        (
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        )
    )

    def sign(key: bytes, value: str) -> bytes:
        return hmac.new(key, value.encode(), hashlib.sha256).digest()

    signing_key = sign(
        sign(sign(sign(("AWS4" + secret_key).encode(), date_stamp), region), "s3"),
        "aws4_request",
    )
    signature = hmac.new(
        signing_key, string_to_sign.encode(), hashlib.sha256
    ).hexdigest()
    request_headers = {
        **headers,
        "authorization": (
            f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
            f"SignedHeaders={signed_names}, Signature={signature}"
        ),
    }
    connection = http.client.HTTPSConnection(
        parsed.hostname, parsed.port or 443, timeout=120
    )
    target = urllib.parse.urlunsplit(
        ("", "", canonical_uri or "/", canonical_query, "")
    )
    try:
        connection.request(
            method,
            target,
            body=payload if method == "PUT" else None,
            headers=request_headers,
        )
        response = connection.getresponse()
        try:
            if response.status not in expected_statuses:
                raise BootstrapRefusal(
                    f"output storage {method} refused with HTTP {response.status}"
                )
            status_code = response.status
            body = response.read(MAX_OUTPUT_BYTES + 1)
            response_headers = {
                key.lower(): value for key, value in response.getheaders()
            }
        finally:
            response.close()
    except (http.client.HTTPException, OSError) as exc:
        raise BootstrapRefusal(f"output storage {method} request failed") from exc
    finally:
        connection.close()
    if len(body) > MAX_OUTPUT_BYTES:
        raise BootstrapRefusal(
            "output storage response exceeds the aggregate size budget"
        )
    return status_code, response_headers, body


def _s3_object_url(endpoint: str, bucket: str, object_key: str) -> str:
    """Encode each S3 key segment while preserving separators on the wire."""

    segments = object_key.split("/")
    if (
        not bucket
        or bucket in {".", ".."}
        or not object_key
        or object_key.startswith("/")
        or any(segment in {"", ".", ".."} for segment in segments)
    ):
        raise BootstrapRefusal("output storage object identity is invalid")
    path = "/".join(
        urllib.parse.quote(segment, safe="-_.~") for segment in (bucket, *segments)
    )
    return f"{endpoint}/{path}"


def _immutable_output_bytes(
    directory_fd: int, name: str, limit: int
) -> tuple[bytes, str]:
    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
        raise BootstrapRefusal("output violates its type, link, or size boundary")
    file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    with os.fdopen(file_fd, "rb") as stream:
        opened_before = os.fstat(stream.fileno())
        payload = stream.read(limit + 1)
        opened_after = os.fstat(stream.fileno())
    stable_identity = (
        opened_after.st_dev,
        opened_after.st_ino,
        opened_after.st_size,
        opened_after.st_mtime_ns,
    ) == (
        opened_before.st_dev,
        opened_before.st_ino,
        len(payload),
        opened_before.st_mtime_ns,
    )
    if len(payload) > limit or not stable_identity:
        raise BootstrapRefusal("output changed or exceeded its budget during snapshot")
    return payload, hashlib.sha256(payload).hexdigest()


def _verified_output_put(
    *,
    endpoint: str,
    bucket: str,
    object_key: str,
    name: str,
    payload: bytes,
    digest: str,
    transaction_token: str,
    attempted: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{32}", transaction_token):
        raise BootstrapRefusal("output transaction identity is invalid")
    checksum = base64.b64encode(bytes.fromhex(digest)).decode()
    url = _s3_object_url(endpoint, bucket, object_key)
    attempted[object_key] = {
        "size_bytes": len(payload),
        "checksum": checksum,
        "etag": "",
        "version_id": "",
        "transaction_token_sha256": hashlib.sha256(transaction_token.encode()).hexdigest(),
    }

    def reconcile_put() -> dict[str, str]:
        try:
            status_code, reconciled_headers, _ = _sigv4_request(
                "HEAD",
                url,
                extra_headers={"x-amz-checksum-mode": "ENABLED"},
                expected_statuses=frozenset({200, 404}),
            )
        except BootstrapRefusal as reconcile_exc:
            key_hash = hashlib.sha256(object_key.encode()).hexdigest()
            raise BootstrapRefusal(
                "output storage PUT outcome is ambiguous; ownership "
                f"reconciliation failed for key hash {key_hash}"
            ) from reconcile_exc
        if status_code == 404:
            attempted.pop(object_key, None)
            key_hash = hashlib.sha256(object_key.encode()).hexdigest()
            raise BootstrapRefusal(
                "output storage PUT outcome is ambiguous; object proven absent "
                f"for key hash {key_hash}"
            )
        version_id = reconciled_headers.get("x-amz-version-id", "")
        etag = reconciled_headers.get("etag", "")
        if (
            reconciled_headers.get("x-amz-meta-npa-transaction-token")
            != transaction_token
            or reconciled_headers.get("x-amz-checksum-sha256") != checksum
            or not version_id
            or re.fullmatch(r'"[^\"]+"', etag) is None
        ):
            key_hash = hashlib.sha256(object_key.encode()).hexdigest()
            raise BootstrapRefusal(
                "output storage PUT outcome is ambiguous; immutable ownership "
                f"identity unavailable for key hash {key_hash}"
            )
        attempted[object_key].update({"etag": etag, "version_id": version_id})
        return reconciled_headers

    try:
        _, created_headers, _ = _sigv4_request(
            "PUT",
            url,
            payload=payload,
            extra_headers={
                "if-none-match": "*",
                "x-amz-checksum-sha256": checksum,
                "x-amz-meta-npa-transaction-token": transaction_token,
            },
        )
    except BootstrapRefusal as exc:
        if not str(exc).startswith("output storage PUT "):
            attempted.pop(object_key, None)
            raise
        created_headers = reconcile_put()
    version_id = created_headers.get("x-amz-version-id", "")
    etag = created_headers.get("etag", "")
    if not version_id or re.fullmatch(r'"[^\"]+"', etag) is None:
        created_headers = reconcile_put()
        version_id = created_headers.get("x-amz-version-id", "")
        etag = created_headers.get("etag", "")
    attempted[object_key].update({"etag": etag, "version_id": version_id})
    version_url = url + "?versionId=" + urllib.parse.quote(version_id, safe="")
    _, headers, observed = _sigv4_request(
        "GET", version_url, extra_headers={"x-amz-checksum-mode": "ENABLED"}
    )
    if (
        observed != payload
        or hashlib.sha256(observed).hexdigest() != digest
        or headers.get("x-amz-checksum-sha256") != checksum
        or headers.get("x-amz-version-id") != version_id
        or headers.get("etag") != etag
    ):
        raise BootstrapRefusal("output storage checksum/readback differs")
    return {
        "name": name,
        "object_key": object_key,
        "size_bytes": len(payload),
        "sha256": digest,
        "etag": etag,
        "version_id": version_id,
        "transaction_token_sha256": hashlib.sha256(transaction_token.encode()).hexdigest(),
    }


def _cleanup_object_etag(
    headers: dict[str, str], *, size_bytes: int, checksum: str
) -> str:
    if headers.get("x-amz-checksum-sha256") != checksum:
        return ""
    try:
        observed_size = int(headers.get("content-length", "-1"))
    except ValueError:
        return ""
    etag = headers.get("etag", "")
    if observed_size != size_bytes or re.fullmatch(r'"[^"]+"', etag) is None:
        return ""
    return etag


def _cleanup_output_attempts(
    *, endpoint: str, bucket: str, attempted: dict[str, dict[str, Any]]
) -> None:
    failures: list[str] = []
    for object_key, identity in reversed(attempted.items()):
        url = _s3_object_url(endpoint, bucket, object_key)
        version_id = str(identity["version_id"])
        version_url = url + "?versionId=" + urllib.parse.quote(version_id, safe="")
        key_hash = hashlib.sha256(object_key.encode()).hexdigest()
        try:
            status_code, headers, _ = _sigv4_request(
                "HEAD",
                version_url,
                extra_headers={"x-amz-checksum-mode": "ENABLED"},
                expected_statuses=frozenset({200, 404}),
            )
            if status_code == 404:
                continue
            etag = _cleanup_object_etag(
                headers,
                size_bytes=int(identity["size_bytes"]),
                checksum=str(identity["checksum"]),
            )
            if (
                not etag
                or etag != identity["etag"]
                or headers.get("x-amz-version-id") != version_id
            ):
                failures.append(key_hash)
                continue
            delete_status, _, _ = _sigv4_request(
                "DELETE",
                version_url,
                extra_headers={"if-match": str(identity["etag"])},
                expected_statuses=frozenset({200, 204, 412}),
            )
            if delete_status == 412:
                failures.append(key_hash)
                continue
            status_code, _, _ = _sigv4_request(
                "HEAD", version_url, expected_statuses=frozenset({200, 404})
            )
            if status_code != 404:
                failures.append(key_hash)
        except (BootstrapRefusal, OSError, ValueError):
            failures.append(key_hash)
    if failures:
        raise BootstrapRefusal(
            "output transaction cleanup is incomplete for exact key hashes: "
            + ",".join(sorted(set(failures)))
        )


def _verify_output_lease(
    *, endpoint: str, bucket: str, authorization: dict[str, Any]
) -> tuple[str, str, str, str]:
    """Require the signed, manager-created, version-bound output lease."""

    key = str(authorization.get("lease_key") or "")
    etag = str(authorization.get("lease_etag") or "")
    version_id = str(authorization.get("lease_version_id") or "")
    nonce = str(authorization.get("lease_nonce") or "")
    if (
        not key
        or re.fullmatch(r'"[^\"]+"', etag) is None
        or not version_id
        or re.fullmatch(r"[A-Za-z0-9_-]{32,128}", nonce) is None
    ):
        raise BootstrapRefusal("output transaction lease binding is incomplete")
    url = _s3_object_url(endpoint, bucket, key)
    version_url = url + "?versionId=" + urllib.parse.quote(version_id, safe="")
    status, headers, _ = _sigv4_request(
        "HEAD", version_url, expected_statuses=frozenset({200, 404})
    )
    if (
        status != 200
        or headers.get("etag") != etag
        or headers.get("x-amz-version-id") != version_id
        or headers.get("x-amz-meta-npa-lease-nonce") != nonce
    ):
        raise BootstrapRefusal("output transaction lease changed or disappeared")
    return key, etag, version_id, nonce


def _release_output_lease(
    *,
    endpoint: str,
    bucket: str,
    key: str,
    etag: str,
    version_id: str,
    nonce: str,
) -> None:
    url = _s3_object_url(endpoint, bucket, key)
    version_url = url + "?versionId=" + urllib.parse.quote(version_id, safe="")
    status, headers, _ = _sigv4_request(
        "HEAD", version_url, expected_statuses=frozenset({200, 404})
    )
    if (
        status != 200
        or headers.get("etag") != etag
        or headers.get("x-amz-version-id") != version_id
        or headers.get("x-amz-meta-npa-lease-nonce") != nonce
    ):
        raise BootstrapRefusal("output transaction lease identity changed")
    status, _, _ = _sigv4_request(
        "DELETE",
        version_url,
        extra_headers={"if-match": etag},
        expected_statuses=frozenset({200, 204, 412}),
    )
    if status == 412:
        raise BootstrapRefusal("output transaction lease identity changed")
    status, _, _ = _sigv4_request(
        "HEAD", version_url, expected_statuses=frozenset({200, 404})
    )
    if status != 404:
        raise BootstrapRefusal("output transaction lease cleanup is incomplete")


def _remove_unexpected_output_entries(root_fd: int, names: set[str]) -> list[str]:
    """Remove only unexpected entries from the owner-controlled staging root."""

    errors: list[str] = []
    root = Path("/proc/self/fd") / str(root_fd)
    for name in sorted(names):
        try:
            info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                errors.extend(_remove_cache_tree(root / name))
            elif stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                os.unlink(name, dir_fd=root_fd)
            else:
                errors.append(f"unsupported unexpected output:{name}")
        except FileNotFoundError:
            continue
        except OSError as exc:
            errors.append(f"unexpected output cleanup failed:{name}:{exc}")
    return errors


def _output_limits_for_exit(smoke_exit_code: int, *, root_fd: int) -> dict[str, int]:
    """Apply the exact success/failure inventory before reading any payload."""

    observed_names = set(os.listdir(root_fd))
    if smoke_exit_code == 0:
        required_names = set(OUTPUT_ARTIFACT_SIZE_LIMITS)
        allowed_names = required_names
        output_limits = OUTPUT_ARTIFACT_SIZE_LIMITS
    else:
        required_names = set(FAILURE_OUTPUT_REQUIRED_SIZE_LIMITS)
        optional_names = set(FAILURE_OUTPUT_OPTIONAL_SIZE_LIMITS)
        allowed_names = required_names | optional_names
        output_limits = {
            **FAILURE_OUTPUT_REQUIRED_SIZE_LIMITS,
            **{
                name: FAILURE_OUTPUT_OPTIONAL_SIZE_LIMITS[name]
                for name in sorted(optional_names & observed_names)
            },
        }
    unexpected = observed_names - allowed_names
    if unexpected:
        cleanup_errors = _remove_unexpected_output_entries(root_fd, unexpected)
        detail = ", ".join(sorted(unexpected))
        if cleanup_errors:
            detail += "; cleanup=" + ", ".join(cleanup_errors)
        raise BootstrapRefusal(
            "output differs from the exact artifact allowlist: " + detail
        )
    if not required_names <= observed_names:
        raise BootstrapRefusal("output is missing a required artifact")
    return output_limits


def _write_output_summary(smoke_exit_code: int, *, root_fd: int) -> None:
    summary = {
        "status": "success" if smoke_exit_code == 0 else "failed",
        "tool": "byof",
        "workload": "solution-smoke-libero-b200",
        "run_id": os.environ.get("NPA_BYOF_RUN_ID", ""),
        "image": os.environ.get("BYOF_IMAGE", ""),
        "solution_name": os.environ.get("BYOF_SOLUTION_NAME", ""),
        "capability_name": os.environ.get("BYOF_CAPABILITY_NAME", ""),
        "smoke_artifact_name": os.environ.get("BYOF_SMOKE_ARTIFACT_NAME", ""),
        "smoke_exit_code": smoke_exit_code,
        "runtime_cache_uploaded": False,
        "rendering_invoked": False,
        "created_unix": round(datetime.now(timezone.utc).timestamp(), 3),
    }
    summary_payload = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode()
    summary_fd = os.open(
        "npa_byof_summary.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP,
        dir_fd=root_fd,
    )
    with os.fdopen(summary_fd, "wb") as stream:
        stream.write(summary_payload)



def seal_for_controller(smoke_exit_code: int, *, root_fd: int) -> None:
    """Seal the existing bounded proof files; no object-store authority enters the Pod."""
    if not _customer_run():
        raise BootstrapRefusal("controller retrieval requires the customer-run profile")
    if any(os.environ.get(name) for name in (
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
        "NEBIUS_IAM_TOKEN", "NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_B64",
    )):
        raise BootstrapRefusal("customer-run workload received cloud or storage credentials")
    info = os.fstat(root_fd)
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise BootstrapRefusal("controller output descriptor is not sealed")
    _write_output_summary(smoke_exit_code, root_fd=root_fd)
    limits = _output_limits_for_exit(smoke_exit_code, root_fd=root_fd)
    records = []
    total = 0
    for name, limit in sorted(limits.items()):
        payload, digest = _immutable_output_bytes(root_fd, name, limit)
        total += len(payload)
        if total > MAX_OUTPUT_BYTES:
            raise BootstrapRefusal("controller output exceeds aggregate size budget")
        if name in OUTPUT_UPLOAD_SIZE_LIMITS:
            _canonical_output_payload(name, payload)
        records.append({"name": name, "size_bytes": len(payload), "sha256": digest})
    payload = {
        "schema": "npa.libero.controller-output-inventory.v1",
        "run_id": os.environ["NPA_BYOF_RUN_ID"],
        "image": os.environ["BYOF_IMAGE"],
        "authorization_sha256": os.environ["NPA_LIBERO_CUSTOMER_AUTHORIZATION_SHA256"],
        "smoke_exit_code": smoke_exit_code,
        "files": records,
    }
    path = _customer_phase_path("completed.json")
    _write_phase_file(path, payload)


def _customer_phase_path(name: str) -> Path:
    run_id = os.environ.get("NPA_BYOF_RUN_ID", "")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{15,62}", run_id) is None:
        raise BootstrapRefusal("customer-run phase requires an exact run ID")
    return CUSTOMER_RUN_PHASE_ROOT / run_id / name


def _write_phase_file(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True)


def wait_for_customer_phase(phase: str) -> dict[str, Any]:
    """Wait for the owned controller; fetched code cannot write supervisor phase files."""
    if not _customer_run() or phase not in {"training", "retrieved"}:
        raise BootstrapRefusal("invalid customer-run phase")
    path = _customer_phase_path(phase + ".json")
    while not path.exists():
        expected = os.environ.get("NPA_LIBERO_EXPECTED_CUSTOMER_AUTHORIZATION_EXPIRES_AT", "")
        _ensure_deadline(_parse_utc(expected, "customer authorization expiry"))
        time.sleep(1)
    payload = json.loads(_immutable_supervisor_bytes(path, 64 * 1024))
    if (
        payload.get("schema") != "npa.libero.customer-controller-phase.v1"
        or payload.get("phase") != phase
        or payload.get("run_id") != os.environ.get("NPA_BYOF_RUN_ID")
        or payload.get("authorization_sha256") != os.environ.get("NPA_LIBERO_CUSTOMER_AUTHORIZATION_SHA256")
        or payload.get("image") != os.environ.get("BYOF_IMAGE")
    ):
        raise BootstrapRefusal("customer-run controller phase differs")
    return payload


def upload_outputs(smoke_exit_code: int, *, root_fd: int) -> dict[str, Any]:
    """Upload through image-owned stdlib code after untrusted runtime execution ends."""

    run_id = os.environ.get("NPA_BYOF_RUN_ID", "")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{15,62}", run_id):
        raise BootstrapRefusal("output upload requires the accepted run ID")
    output_prefix = os.environ.get("S3_OUTPUT_PREFIX", "").rstrip("/") + "/"
    parsed = urllib.parse.urlsplit(output_prefix)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise BootstrapRefusal("output prefix must be an S3 URI")
    endpoint = (
        os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("NEBIUS_S3_ENDPOINT") or ""
    ).rstrip("/")
    if not endpoint:
        raise BootstrapRefusal("output storage endpoint is required")
    endpoint_parts = urllib.parse.urlsplit(endpoint)
    if endpoint_parts.scheme != "https" or not endpoint_parts.netloc:
        raise BootstrapRefusal("output storage endpoint must be HTTPS")
    storage_authorization = _storage_authorization(output_prefix, run_id, endpoint)
    root_info = os.fstat(root_fd)
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or root_info.st_uid != os.getuid()
        or stat.S_IMODE(root_info.st_mode) != 0o700
    ):
        raise BootstrapRefusal("output upload descriptor is not sealed")
    _write_output_summary(smoke_exit_code, root_fd=root_fd)

    prefix = parsed.path.lstrip("/")
    transaction_id = uuid4().hex
    transaction_token = uuid4().hex
    transaction_prefix = prefix + ".npa-transactions/" + transaction_id + "/"
    attempted: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    committed = False
    lease_released = False
    lease_key = lease_etag = lease_version_id = lease_nonce = ""
    try:
        lease_key, lease_etag, lease_version_id, lease_nonce = _verify_output_lease(
            endpoint=endpoint,
            bucket=parsed.netloc,
            authorization=storage_authorization,
        )
        output_limits = _output_limits_for_exit(smoke_exit_code, root_fd=root_fd)
        observed_total = sum(
            os.stat(name, dir_fd=root_fd, follow_symlinks=False).st_size
            for name in output_limits
        )
        if observed_total > MAX_OUTPUT_BYTES:
            raise BootstrapRefusal("output exceeds the aggregate size budget")
        snapshots = []
        for name, limit in sorted(OUTPUT_UPLOAD_SIZE_LIMITS.items()):
            payload, digest = _immutable_output_bytes(root_fd, name, limit)
            payload = _canonical_output_payload(name, payload)
            digest = hashlib.sha256(payload).hexdigest()
            snapshots.append((name, payload, digest))
        for name, payload, digest in snapshots:
            current_authorization = _storage_authorization(
                output_prefix, run_id, endpoint
            )
            if (
                current_authorization["capability_id"]
                != storage_authorization["capability_id"]
            ):
                raise BootstrapRefusal(
                    "output storage authorization changed during upload"
                )
            receipts.append(
                _verified_output_put(
                    endpoint=endpoint,
                    bucket=parsed.netloc,
                    object_key=transaction_prefix + name,
                    name=name,
                    payload=payload,
                    digest=digest,
                    transaction_token=transaction_token,
                    attempted=attempted,
                )
            )
        receipt_payload = (
            json.dumps(
                {
                    "schema": OUTPUT_RECEIPT_SCHEMA,
                    "run_id": run_id,
                    "customer_identity_sha256": os.environ.get(
                        "NPA_LIBERO_CUSTOMER_IDENTITY_SHA256", ""
                    ),
                    "candidate_image": os.environ.get("BYOF_IMAGE", ""),
                    "output_storage_capability_id": storage_authorization[
                        "capability_id"
                    ],
                    "output_storage_authorization_sha256": os.environ.get(
                        "NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_AUTHORIZATION_SHA256",
                        "",
                    ),
                    "transaction_id": transaction_id,
                    "transaction_prefix": transaction_prefix,
                    "status": "verified",
                    "commit_marker": OUTPUT_RECEIPT_NAME,
                    "artifacts": receipts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
        receipt_fd = os.open(
            OUTPUT_RECEIPT_NAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            stat.S_IRUSR | stat.S_IWUSR,
            dir_fd=root_fd,
        )
        with os.fdopen(receipt_fd, "wb") as stream:
            stream.write(receipt_payload)
        _immutable_output_bytes(root_fd, OUTPUT_RECEIPT_NAME, 8 * 1024 * 1024)
        current_authorization = _storage_authorization(output_prefix, run_id, endpoint)
        if (
            current_authorization["capability_id"]
            != storage_authorization["capability_id"]
        ):
            raise BootstrapRefusal("output storage authorization changed before commit")
        _verified_output_put(
            endpoint=endpoint,
            bucket=parsed.netloc,
            object_key=prefix + OUTPUT_RECEIPT_NAME,
            name=OUTPUT_RECEIPT_NAME,
            payload=receipt_payload,
            digest=receipt_sha256,
            transaction_token=transaction_token,
            attempted=attempted,
        )
        _release_output_lease(
            endpoint=endpoint,
            bucket=parsed.netloc,
            key=lease_key,
            etag=lease_etag,
            version_id=lease_version_id,
            nonce=lease_nonce,
        )
        lease_released = True
        committed = True
    except Exception as exc:
        cleanup_errors: list[str] = []
        if not committed:
            try:
                _cleanup_output_attempts(
                    endpoint=endpoint, bucket=parsed.netloc, attempted=attempted
                )
            except BootstrapRefusal as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))
        if not lease_released and lease_key and lease_nonce:
            try:
                _release_output_lease(
                    endpoint=endpoint,
                    bucket=parsed.netloc,
                    key=lease_key,
                    etag=lease_etag,
                    version_id=lease_version_id,
                    nonce=lease_nonce,
                )
            except BootstrapRefusal as lease_exc:
                cleanup_errors.append(str(lease_exc))
        if cleanup_errors:
            raise BootstrapRefusal(
                "output transaction cleanup is incomplete: "
                + "; ".join(cleanup_errors)
            ) from exc
        raise
    return {
        "schema": "npa.libero.output-upload.v2",
        "status": "verified",
        "transaction_id": transaction_id,
        "commit_sha256": receipt_sha256,
    }


def _run_output_root(run_id: str) -> Path:
    # Keep the executable profile in the run root, outside the output inventory.
    return Path(f"/workspace/byof-runs/{run_id}/output")


def _bootstrap_receipt_path(cache_root: Path, run_id: str) -> Path:
    return cache_root / "run-receipts" / f"{run_id}.json"


def _immutable_supervisor_bytes(path: Path, limit: int) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise BootstrapRefusal("supervisor evidence is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(limit + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.getuid()
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o027
        or len(payload) > limit
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or before.st_size != len(payload)
    ):
        raise BootstrapRefusal("supervisor evidence is mutable or invalid")
    return payload


def _execution_uid_processes() -> list[int]:
    try:
        try:
            execution_uid = pwd.getpwnam(RUNTIME_EXECUTION_USER).pw_uid
        except KeyError as exc:
            raise BootstrapRefusal("runtime execution account is unavailable") from exc
        discovered = []
        for process in Path("/proc").iterdir():
            if not process.name.isdigit():
                continue
            try:
                status_lines = (
                    (process / "status").read_text(encoding="utf-8").splitlines()
                )
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            # A dead child retained by container PID 1 cannot execute or write;
            # only its parent can reap it, so signals cannot remove that entry.
            if any(line.startswith("State:\tZ") for line in status_lines):
                continue
            uid_line = next(
                (line for line in status_lines if line.startswith("Uid:")), None
            )
            if uid_line is None:
                raise BootstrapRefusal("execution process identity is unavailable")
            fields = uid_line.split()
            if len(fields) != 5:
                raise BootstrapRefusal("execution process identity is invalid")
            if int(fields[2]) == execution_uid and int(process.name) != os.getpid():
                discovered.append(int(process.name))
        return sorted(discovered)
    except OSError as exc:
        raise BootstrapRefusal("execution process inventory is unavailable") from exc


def _execution_process_group_ids(processes: list[int]) -> list[int]:
    """Read process groups for the exact execution-UID processes we own."""

    groups: set[int] = set()
    for pid in processes:
        try:
            stat_text = (Path("/proc") / str(pid) / "stat").read_text(
                encoding="utf-8"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError) as exc:
            raise BootstrapRefusal(
                "execution process group identity is unavailable"
            ) from exc
        closing = stat_text.rfind(")")
        if closing < 0:
            raise BootstrapRefusal("execution process group identity is invalid")
        fields = stat_text[closing + 2 :].split()
        if len(fields) < 3:
            raise BootstrapRefusal("execution process group identity is invalid")
        try:
            groups.add(int(fields[2]))
        except ValueError as exc:
            raise BootstrapRefusal(
                "execution process group identity is invalid"
            ) from exc
    return sorted(groups)


def _terminate_execution_processes(processes: list[int]) -> None:
    """Terminate every owned execution process and its process groups."""

    initial = sorted(set(processes))
    if not initial:
        return
    if _customer_run() and os.getuid() == pwd.getpwnam(RUNTIME_SUPERVISOR_USER).pw_uid:
        subprocess.run(
            ["/usr/bin/sudo", "-n", "-u", RUNTIME_EXECUTION_USER,
             "/opt/npa/libero/runtime-bootstrap.py", "terminate"],
            check=True, start_new_session=True,
        )
        return
    groups = _execution_process_group_ids(initial)
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            raise BootstrapRefusal(
                "runtime execution process-group termination failed"
            ) from exc
    for pid in initial:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            raise BootstrapRefusal(
                "runtime execution termination failed"
            ) from exc

    deadline = time.monotonic() + 5.0
    remaining = _execution_uid_processes()
    while remaining and time.monotonic() < deadline:
        time.sleep(0.05)
        remaining = _execution_uid_processes()
    if remaining:
        for group in _execution_process_group_ids(remaining):
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                raise BootstrapRefusal(
                    "runtime execution process-group kill failed"
                ) from exc
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                raise BootstrapRefusal("runtime execution kill failed") from exc
        deadline = time.monotonic() + 5.0
        while remaining and time.monotonic() < deadline:
            time.sleep(0.05)
            remaining = _execution_uid_processes()
    if remaining:
        raise BootstrapRefusal(
            "runtime execution cleanup is incomplete: "
            + ",".join(str(pid) for pid in remaining)
        )


def _materialize_supervisor_artifact(root_fd: int, name: str, payload: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP,
        dir_fd=root_fd,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


@contextmanager
def _bind_inherited_descriptors(sources: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    """Expose exact retained descriptors at the sudo close-from boundary."""

    targets = tuple(
        range(INHERITED_CACHE_DESCRIPTOR, INHERITED_CACHE_DESCRIPTOR + len(sources))
    )
    preserved = [os.dup(source) for source in sources]
    try:
        for source, target in zip(preserved, targets, strict=True):
            os.dup2(source, target, inheritable=True)
        yield targets
    finally:
        for target in targets:
            try:
                os.close(target)
            except OSError:
                pass
        for descriptor in preserved:
            os.close(descriptor)


def execute_and_upload(*, controller_retrieval: bool = False) -> int:
    """Hold exact authorization and exclusive UID ownership through readback."""

    if controller_retrieval != _customer_run():
        raise BootstrapRefusal("execution output transport differs from customer profile")

    manifest, manifest_sha256 = _validate_manifest(DEFAULT_MANIFEST)
    _, requirements_sha256 = _validate_requirements(DEFAULT_REQUIREMENTS, manifest)
    authorization_sha256, customer_identity_sha256, run_id, scope_sha256 = (
        _execution_authorization_values(manifest_sha256)
    )
    cache_root = _validate_cache_root(DEFAULT_CACHE, None)
    output_root = _run_output_root(run_id)
    caller_bytes, trusted_public_key, caller_signer_sha256 = (
        _authenticated_caller_binding_from_environment()
    )
    caller_file = tempfile.TemporaryFile(mode="w+b")
    trust_file = tempfile.TemporaryFile(mode="w+b")
    caller_file.write(caller_bytes)
    caller_file.flush()
    caller_file.seek(0)
    trust_file.write(trusted_public_key)
    trust_file.flush()
    trust_file.seek(0)
    output_fd = -1
    authorization_fd = -1
    try:
        output_fd = os.open(output_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        output_info = os.fstat(output_fd)
        if (
            output_info.st_uid != os.getuid()
            or stat.S_IMODE(output_info.st_mode) != 0o1770
        ):
            raise BootstrapRefusal("execution output staging directory is invalid")
        try:
            authorization_fd = os.open(
                PROTECTED_AUTHORIZATION_NAME,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=output_fd,
            )
        except OSError as exc:
            raise CustomerAcceptanceRequired(
                "authorization_missing_at_execution"
            ) from exc
        try:
            authorization_bytes = _read_private_regular_descriptor(
                authorization_fd,
                limit=1024 * 1024,
                owner_uid=os.getuid(),
                input_name="customer authorization file",
            )
        except BootstrapRefusal as exc:
            raise CustomerAcceptanceRequired("authorization_file_invalid") from exc
        authorization, observed_authorization_sha256 = (
            _validate_customer_authorization_bytes(
                authorization_bytes,
                authorization_sha256,
                manifest,
                manifest_sha256,
                authenticated_signer_sha256=caller_signer_sha256,
            )
        )
        if (
            observed_authorization_sha256 != authorization_sha256
            or authorization["customer_identity_sha256"] != customer_identity_sha256
            or authorization["run_id"] != run_id
        ):
            raise CustomerAcceptanceRequired("authorization_wrong_scope")
        bootstrap_receipt = _bootstrap_receipt_path(cache_root, run_id)
        bootstrap_payload = _immutable_supervisor_bytes(
            bootstrap_receipt, OUTPUT_SIZE_LIMITS["npa_runtime_bootstrap.json"]
        )
        with _open_cache_root_descriptor(cache_root, create=False) as cache_root_fd:
            identity = _cache_entry_identity_at(cache_root_fd, scope_sha256)
            if identity is None:
                raise BootstrapRefusal("runtime cache is not materialized")
            if os.readlink("current", dir_fd=cache_root_fd) != scope_sha256:
                raise BootstrapRefusal(
                    "runtime current link differs from accepted cache"
                )
            with _cache_lock(
                cache_root_fd, ".execution.lock", exclusive=True, create=True
            ) as execution_lock_fd:
                if _execution_uid_processes():
                    raise BootstrapRefusal("runtime execution UID is already active")
                with _cache_lock(
                    cache_root_fd, ".bootstrap.lock", exclusive=False, create=False
                ) as bootstrap_lock_fd:
                    descriptor = os.open(
                        scope_sha256,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=cache_root_fd,
                    )
                    try:
                        opened = os.fstat(descriptor)
                        if (opened.st_dev, opened.st_ino) != identity:
                            raise BootstrapRefusal(
                                "runtime cache descriptor identity changed"
                            )
                        stable_root = Path("/proc/self/fd") / str(descriptor)
                        governing_terms_sha256 = _governing_terms_identity(manifest)
                        _validate_complete(
                            stable_root,
                            manifest,
                            manifest_sha256,
                            authorization_sha256,
                            customer_identity_sha256,
                            run_id,
                            requirements_sha256,
                            governing_terms_sha256,
                        )
                        runtime_metadata = _immutable_supervisor_bytes(
                            stable_root / ".complete.json",
                            OUTPUT_SIZE_LIMITS["npa_runtime_metadata.json"],
                        )
                        stdout_fd = os.open(
                            "solution_smoke_stdout.log",
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                            0o640,
                            dir_fd=output_fd,
                        )
                        try:
                            stderr_fd = os.open(
                                "solution_smoke_stderr.log",
                                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                0o640,
                                dir_fd=output_fd,
                            )
                        except Exception:
                            os.close(stdout_fd)
                            raise
                        with (
                            os.fdopen(stdout_fd, "wb") as stdout,
                            os.fdopen(stderr_fd, "wb") as stderr,
                            _bind_inherited_descriptors(
                                (
                                    descriptor,
                                    authorization_fd,
                                    execution_lock_fd,
                                    bootstrap_lock_fd,
                                    caller_file.fileno(),
                                    trust_file.fileno(),
                                )
                            ) as inherited,
                        ):
                            _validate_customer_authorization_bytes(
                                authorization_bytes,
                                authorization_sha256,
                                manifest,
                                manifest_sha256,
                                authenticated_signer_sha256=caller_signer_sha256,
                            )
                            environment = _runtime_execution_environment(
                                Path("/proc/self/fd") / str(INHERITED_CACHE_DESCRIPTOR)
                            )
                            environment["NPA_LIBERO_BOOTSTRAP_RECEIPT"] = str(
                                bootstrap_receipt
                            )
                            try:
                                completed = subprocess.run(
                                    [
                                        "/usr/bin/sudo",
                                        "--close-from",
                                        str(INHERITED_CUSTOMER_TRUST_DESCRIPTOR + 1),
                                        "--user=npa-libero-exec",
                                        "/opt/npa/libero/runtime-bootstrap.py",
                                        "execute",
                                    ],
                                    check=False,
                                    env=environment,
                                    stdout=stdout,
                                    stderr=stderr,
                                    pass_fds=inherited,
                                    start_new_session=True,
                                )
                            finally:
                                remaining_processes = _execution_uid_processes()
                                if remaining_processes:
                                    _terminate_execution_processes(remaining_processes)
                                    raise BootstrapRefusal(
                                        "runtime execution left processes behind: "
                                        + ",".join(
                                            str(pid) for pid in remaining_processes
                                        )
                                    )
                        smoke_exit_code = completed.returncode
                        if smoke_exit_code == 3:
                            # The child emits this code only for a customer-actionable
                            # authorization refusal. Revalidate the same retained bytes
                            # so the supervisor returns the structured acceptance state
                            # and never publishes it as a workload failure.
                            try:
                                _validate_customer_authorization_bytes(
                                    authorization_bytes,
                                    authorization_sha256,
                                    manifest,
                                    manifest_sha256,
                                    authenticated_signer_sha256=caller_signer_sha256,
                                )
                            except CustomerAcceptanceRequired:
                                raise
                            raise BootstrapRefusal(
                                "runtime child reported an inconsistent authorization refusal"
                            )
                        current_output = output_root.stat(follow_symlinks=False)
                        if not stat.S_ISDIR(current_output.st_mode) or (
                            current_output.st_dev,
                            current_output.st_ino,
                        ) != (output_info.st_dev, output_info.st_ino):
                            raise BootstrapRefusal(
                                "execution output staging directory changed during execution"
                            )
                        os.fchmod(output_fd, 0o700)
                        authorization_info = os.fstat(authorization_fd)
                        named_authorization = os.stat(
                            PROTECTED_AUTHORIZATION_NAME,
                            dir_fd=output_fd,
                            follow_symlinks=False,
                        )
                        if (authorization_info.st_dev, authorization_info.st_ino) != (
                            named_authorization.st_dev,
                            named_authorization.st_ino,
                        ):
                            raise BootstrapRefusal(
                                "protected authorization identity changed"
                            )
                        os.unlink(PROTECTED_AUTHORIZATION_NAME, dir_fd=output_fd)
                        _materialize_supervisor_artifact(
                            output_fd, "npa_runtime_bootstrap.json", bootstrap_payload
                        )
                        _materialize_supervisor_artifact(
                            output_fd, "npa_runtime_metadata.json", runtime_metadata
                        )
                        artifact_name = os.environ.get("BYOF_SMOKE_ARTIFACT_NAME", "")
                        try:
                            artifact_info = os.stat(
                                artifact_name, dir_fd=output_fd, follow_symlinks=False
                            )
                            artifact_is_regular = (
                                stat.S_ISREG(artifact_info.st_mode)
                                and artifact_info.st_nlink == 1
                            )
                        except OSError:
                            artifact_is_regular = False
                        if (
                            artifact_name not in OUTPUT_ARTIFACT_SIZE_LIMITS
                            or not artifact_is_regular
                        ):
                            error_fd = os.open(
                                "solution_smoke_stderr.log",
                                os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW,
                                dir_fd=output_fd,
                            )
                            with os.fdopen(error_fd, "a", encoding="utf-8") as stderr:
                                stderr.write(
                                    f"missing required smoke artifact: {artifact_name}\n"
                                )
                            smoke_exit_code = 1
                        _validate_customer_authorization_bytes(
                            authorization_bytes,
                            authorization_sha256,
                            manifest,
                            manifest_sha256,
                            authenticated_signer_sha256=caller_signer_sha256,
                        )
                        if controller_retrieval:
                            seal_for_controller(smoke_exit_code, root_fd=output_fd)
                        else:
                            upload_outputs(smoke_exit_code, root_fd=output_fd)
                        _validate_complete(
                            stable_root,
                            manifest,
                            manifest_sha256,
                            authorization_sha256,
                            customer_identity_sha256,
                            run_id,
                            requirements_sha256,
                            governing_terms_sha256,
                        )
                        if (
                            _cache_entry_identity_at(cache_root_fd, scope_sha256)
                            != identity
                        ):
                            raise BootstrapRefusal(
                                "runtime cache changed before readback"
                            )
                        if os.readlink("current", dir_fd=cache_root_fd) != scope_sha256:
                            raise BootstrapRefusal(
                                "runtime current link changed before readback"
                            )
                        return smoke_exit_code
                    finally:
                        os.close(descriptor)
    finally:
        if authorization_fd >= 0:
            os.close(authorization_fd)
        if output_fd >= 0:
            os.close(output_fd)
        caller_file.close()
        trust_file.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "wait-for-release",
            "ensure",
            "status",
            "execute",
            "execute-and-upload",
            "execute-and-seal",
            "customer-ready",
            "wait-for-retrieval",
            "terminate",
        ),
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--requirements", default=str(DEFAULT_REQUIREMENTS))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE))
    parser.add_argument("--authorization", default="")
    parser.add_argument("--authorization-sha256", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--smoke-exit-code", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "terminate":
            if os.getuid() != pwd.getpwnam(RUNTIME_EXECUTION_USER).pw_uid:
                raise BootstrapRefusal("termination requires the execution UID")
            _terminate_execution_processes(_execution_uid_processes())
            return 0
        if args.command == "wait-for-release":
            print(
                json.dumps({**wait_for_release(), "status": "released"}, sort_keys=True)
            )
            return 0
        if args.command == "execute":
            return execute()
        if args.command == "execute-and-upload":
            return execute_and_upload()
        if args.command == "execute-and-seal":
            wait_for_customer_phase("training")
            return execute_and_upload(controller_retrieval=True)
        if args.command == "customer-ready":
            if not _customer_run():
                raise BootstrapRefusal("customer-ready requires the customer-run profile")
            _write_phase_file(_customer_phase_path("materialized.json"), {
                "run_id": os.environ["NPA_BYOF_RUN_ID"],
                "authorization_sha256": os.environ["NPA_LIBERO_CUSTOMER_AUTHORIZATION_SHA256"],
            })
            return 0
        if args.command == "wait-for-retrieval":
            wait_for_customer_phase("retrieved")
            return 0
        else:
            payload = ensure(args) if args.command == "ensure" else status(args)
    except CustomerAcceptanceRequired as exc:
        try:
            manifest, manifest_sha256 = _validate_manifest(Path(args.manifest))
            notification = _customer_acceptance_notification(
                manifest, manifest_sha256=manifest_sha256, reason=exc.reason
            )
        except Exception:
            notification = {
                "schema": "npa.libero.customer-acceptance-notification.v1",
                "status": "needs_customer_acceptance",
                "solution": "libero",
                "reason": exc.reason,
            }
        print(json.dumps(notification, sort_keys=True), file=sys.stderr)
        return 3
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": "npa.libero.runtime-bootstrap-result.v1",
                    "status": "refused",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps({**payload, "status": "ready"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
