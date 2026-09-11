"""Fail-closed authorization transport for the fixed RoboTwin BYOF workflow."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import yaml

from npa.deploy.images import is_public_registry


PUBLIC_CONTEXT_ENV = "NPA_BYOF_ROBOTWIN_RUNTIME_CONTEXT"
TRANSPORT_CONTEXT_ENV = "NPA_INTERNAL_BYOF_ROBOTWIN_CONTEXT_V1"
MATERIALIZED_KUBECONFIG_ENV = "NPA_INTERNAL_BYOF_ROBOTWIN_KUBECONFIG"
MATERIALIZED_SKYPILOT_CONFIG_ENV = "NPA_INTERNAL_BYOF_ROBOTWIN_SKYPILOT_CONFIG"
CONTEXT_ENV_NAMES = (
    PUBLIC_CONTEXT_ENV,
    TRANSPORT_CONTEXT_ENV,
    MATERIALIZED_KUBECONFIG_ENV,
    MATERIALIZED_SKYPILOT_CONFIG_ENV,
)
MAX_CONTEXT_BYTES = 64 * 1024
MAX_CONFIG_BYTES = 24 * 1024
MAX_TRANSPORT_SOURCE_BYTES = 64 * 1024
MAX_TRANSPORT_BYTES = 96 * 1024
TRANSPORT_SCHEMA = "npa.byof.robotwin.runtime-transport.v1"
SOURCE_REVISION = "96c1feab536306b50c26af200044fcdf126e8904"
ASSET_REVISION = "785feb15aa4a4f532395ad2b1d2be5f28cb561ad"
RTX_ACCELERATOR = "RTXPRO-6000-BLACKWELL-SERVER-EDITION"
BUILD_COMMAND_SHA256 = (
    "87cb636a259cbec5d59283e9e03cc076b139182fb01f9b4504c81040ae384931"
)
SMOKE_COMMAND_SHA256 = (
    "bc26a3019d36863326cf4e55bd3536c303355c7b661ede5bc20ed9a4607deb2d"
)
REQUIRED_DECISIONS = (
    "nvidia_cuda_eula",
    "nvidia_cudnn_sla",
    "curobo_noncommercial_research_or_evaluation",
    "robotwin2_aggregate_asset_and_output_terms",
)
INVOCATION = {
    "repo_url": "https://github.com/RoboTwin-Platform/RoboTwin.git",
    "repo_ref": SOURCE_REVISION,
    "base_profile": "ubuntu",
    "base_image": (
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@"
        "sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719"
    ),
    "workload": "solution-smoke",
    "solution_name": "robotwin",
    "capability_name": "beat_block_hammer_successful_seed_replay_collection",
    "smoke_artifact_name": "robotwin-smoke.json",
    "runtime_context_env": PUBLIC_CONTEXT_ENV,
    "yaml": "byof-solution-smoke-robotwin-rtxpro-gpu",
    "task": "beat_block_hammer",
}
_CONTEXT_FIELDS = frozenset(
    {
        "solution",
        "ownership_provenance",
        "reservation",
        "license_acceptance",
        "project",
        "nebius_profile",
        "kubeconfig",
        "kubernetes_context",
        "skypilot_config_path",
        "registry",
        "bucket",
        "output_root",
        "run_id",
    }
)


class RobotwinPreflightError(ValueError):
    """A fixed-category RoboTwin authorization or contract refusal."""

    def __init__(self, category: str, context_sha256: str = "") -> None:
        digest = context_sha256 or "unavailable"
        super().__init__(
            f"RoboTwin authorization refused ({category}; context_sha256={digest})"
        )
        self.category = category
        self.context_sha256 = digest


@dataclass(frozen=True)
class RobotwinAuthorization:
    """Validated private runtime authorization; private fields never enter repr."""

    context_sha256: str
    project: str = field(repr=False)
    profile: str = field(repr=False)
    kubeconfig_source: str = field(repr=False)
    kubeconfig_bytes: bytes = field(repr=False)
    kubernetes_context: str = field(repr=False)
    skypilot_config_source: str = field(repr=False)
    skypilot_config_bytes: bytes = field(repr=False)
    registry: str = field(repr=False)
    bucket: str = field(repr=False)
    output_root: str = field(repr=False)
    run_id: str = field(repr=False)
    summary_uri: str = field(repr=False)
    raw_context: bytes = field(repr=False)
    redactions: tuple[str, ...] = field(repr=False)

    @property
    def kubeconfig(self) -> str:
        """Return the validated kubeconfig path for this process."""

        return self.kubeconfig_source

    @property
    def skypilot_config_path(self) -> str:
        """Return the validated SkyPilot config path for this process."""

        return self.skypilot_config_source


@dataclass(frozen=True)
class RobotwinSubmitContext:
    """Private live-submit binding returned before any external side effect."""

    context_sha256: str
    authorization: RobotwinAuthorization = field(repr=False)
    transport_value: str = field(repr=False)


@dataclass(frozen=True)
class MaterializedRobotwinContext:
    """Owner-only worker paths for one decoded transport envelope."""

    context_path: Path = field(repr=False)
    kubeconfig_path: Path = field(repr=False)
    skypilot_config_path: Path = field(repr=False)
    authorization: RobotwinAuthorization = field(repr=False)


def _refusal(category: str, raw: bytes = b"") -> RobotwinPreflightError:
    digest = hashlib.sha256(raw).hexdigest() if raw else ""
    return RobotwinPreflightError(category, digest)


def _read_owner_file(path_text: str, *, label: str, limit: int) -> bytes:
    path = Path(path_text).expanduser()
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise _refusal(f"{label}-not-regular")
        if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
            raise _refusal(f"{label}-not-owner-only")
        if metadata.st_size > limit:
            raise _refusal(f"{label}-too-large")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            raw = stream.read(limit + 1)
    except RobotwinPreflightError:
        raise
    except OSError as exc:
        raise _refusal(f"{label}-unreadable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(raw) > limit:
        raise _refusal(f"{label}-too-large")
    return raw


def read_owner_context(environ: Mapping[str, str] | None = None) -> bytes:
    """Read the public context file once through an owner-only no-follow fd."""

    source = os.environ if environ is None else environ
    reference = str(source.get(PUBLIC_CONTEXT_ENV) or "").strip()
    if not reference:
        raise _refusal("context-missing")
    if reference.startswith(("{", "[")):
        raise _refusal("context-file-reference-required")
    return _read_owner_file(reference, label="context", limit=MAX_CONTEXT_BYTES)


def _context_text(payload: Mapping[str, Any], field_name: str, raw: bytes) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise _refusal(f"context-{field_name}-invalid", raw)
    return value.strip()


def _load_yaml_mapping(raw: bytes, *, label: str, context: bytes) -> dict[str, Any]:
    try:
        decoded = raw.decode("utf-8")
        value = yaml.safe_load(decoded)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise _refusal(f"{label}-invalid", context) from exc
    if not isinstance(value, dict):
        raise _refusal(f"{label}-invalid", context)
    return value


def _has_external_reference(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in {
                "include",
                "includes",
                "credential_process",
                "token_file",
                "client_certificate",
                "client_key",
                "certificate_authority",
            }:
                return True
            if _has_external_reference(child):
                return True
    elif isinstance(value, list):
        return any(_has_external_reference(item) for item in value)
    elif isinstance(value, str):
        text = value.strip()
        return text.startswith(("/", "~/", "../", "file://"))
    return False


def _validate_kubeconfig(
    raw: bytes, *, context_name: str, context: bytes
) -> None:
    payload = _load_yaml_mapping(raw, label="kubeconfig", context=context)

    def named_entry(section: str, name: str) -> Mapping[str, Any]:
        entries = payload.get(section)
        if not isinstance(entries, list):
            raise _refusal("kubeconfig-structure-invalid", context)
        matches = [
            entry
            for entry in entries
            if isinstance(entry, Mapping) and entry.get("name") == name
        ]
        if len(matches) != 1:
            raise _refusal(f"kubeconfig-{section}-mismatch", context)
        return matches[0]

    selected = named_entry("contexts", context_name).get("context")
    if not isinstance(selected, Mapping):
        raise _refusal("kubeconfig-context-invalid", context)
    cluster_name = selected.get("cluster")
    user_name = selected.get("user")
    if not isinstance(cluster_name, str) or not isinstance(user_name, str):
        raise _refusal("kubeconfig-context-mismatch", context)
    cluster = named_entry("clusters", cluster_name).get("cluster")
    user = named_entry("users", user_name).get("user")
    if not isinstance(cluster, Mapping) or not isinstance(user, Mapping):
        raise _refusal("kubeconfig-structure-invalid", context)
    server = cluster.get("server")
    parsed_server = urlparse(server if isinstance(server, str) else "")
    if (
        parsed_server.scheme != "https"
        or not parsed_server.netloc
        or parsed_server.username
        or parsed_server.password
        or not isinstance(cluster.get("certificate-authority-data"), str)
        or not cluster["certificate-authority-data"].strip()
    ):
        raise _refusal("kubeconfig-cluster-not-portable", context)
    if "exec" in user:
        raise _refusal("kubeconfig-exec-plugin-refused", context)
    inline_token = isinstance(user.get("token"), str) and bool(user["token"].strip())
    inline_cert = all(
        isinstance(user.get(name), str) and bool(user[name].strip())
        for name in ("client-certificate-data", "client-key-data")
    )
    if not (inline_token or inline_cert):
        raise _refusal("kubeconfig-user-not-portable", context)
    if _has_external_reference(payload):
        raise _refusal("kubeconfig-external-reference", context)


def _validate_skypilot_config(
    raw: bytes, *, context_name: str, context: bytes
) -> None:
    payload = _load_yaml_mapping(raw, label="skypilot-config", context=context)
    if _has_external_reference(payload):
        raise _refusal("skypilot-config-external-reference", context)
    kubernetes = payload.get("kubernetes")
    if not isinstance(kubernetes, Mapping):
        raise _refusal("skypilot-config-kubernetes-invalid", context)
    allowed = kubernetes.get("allowed_contexts")
    if allowed is not None and (
        not isinstance(allowed, list)
        or any(not isinstance(item, str) for item in allowed)
        or context_name not in allowed
    ):
        raise _refusal("skypilot-config-context-mismatch", context)


def _portable_config_redactions(raw: bytes) -> tuple[str, ...]:
    """Return complete config text and scalar values for downstream redaction."""

    payload = yaml.safe_load(raw.decode("utf-8"))
    values: list[str] = [raw.decode("utf-8")]

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
        elif isinstance(value, str) and len(value.strip()) >= 4:
            values.append(value.strip())

    collect(payload)
    return tuple(dict.fromkeys(values))


def _parse_context_payload(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _refusal("context-invalid-json", raw) from exc
    if not isinstance(payload, dict):
        raise _refusal("context-not-object", raw)
    if set(payload) != _CONTEXT_FIELDS:
        raise _refusal("context-schema-mismatch", raw)
    return payload


def _validate_authorization_fields(payload: dict[str, Any], raw: bytes) -> dict[str, str]:
    if payload.get("solution") != "robotwin":
        raise _refusal("context-wrong-solution", raw)
    if payload.get("ownership_provenance") != "manager-issued":
        raise _refusal("context-ownership-provenance-invalid", raw)
    reservation = payload.get("reservation")
    if not isinstance(reservation, dict) or set(reservation) != {
        "policy", "accelerator", "count"
    }:
        raise _refusal("reservation-schema-mismatch", raw)
    if reservation.get("policy") != "STRICT":
        raise _refusal("reservation-policy-not-strict", raw)
    if reservation.get("accelerator") != RTX_ACCELERATOR:
        raise _refusal("reservation-accelerator-mismatch", raw)
    if type(reservation.get("count")) is not int or reservation["count"] != 1:
        raise _refusal("reservation-count-not-one", raw)
    decisions = payload.get("license_acceptance")
    if not isinstance(decisions, dict) or set(decisions) != set(REQUIRED_DECISIONS):
        raise _refusal("license-decision-schema-mismatch", raw)
    for decision in REQUIRED_DECISIONS:
        if decisions.get(decision) is not True:
            raise _refusal(f"license-decision-{decision}-missing", raw)
    return {
        name: _context_text(payload, name, raw)
        for name in _CONTEXT_FIELDS
        if name not in {"solution", "reservation", "license_acceptance"}
    }


def _validate_destination(values: Mapping[str, str], raw: bytes) -> str:
    registry = values["registry"]
    if is_public_registry(registry):
        raise _refusal("registry-not-private", raw)
    bucket = values["bucket"]
    parsed = urlparse(values["output_root"])
    if (
        parsed.scheme != "s3"
        or parsed.netloc != bucket
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
        or any(part in {".", ".."} for part in parsed.path.split("/"))
    ):
        raise _refusal("output-root-bucket-mismatch", raw)
    run_id = values["run_id"]
    if re.fullmatch(r"robotwin-[A-Za-z0-9_.-]{1,80}", run_id) is None:
        raise _refusal("run-id-invalid", raw)
    return f"{values['output_root'].rstrip('/')}/{run_id}/npa_byof_summary.json"


def validate_context_bytes(
    raw: bytes,
    *,
    config_bytes: Mapping[str, bytes] | None = None,
) -> RobotwinAuthorization:
    """Validate exact manager context bytes and their portable config payloads."""

    if not raw or len(raw) > MAX_CONTEXT_BYTES:
        raise _refusal("context-size-invalid", raw[:MAX_CONTEXT_BYTES])
    payload = _parse_context_payload(raw)
    values = _validate_authorization_fields(payload, raw)
    summary_uri = _validate_destination(values, raw)
    for field_name in ("kubeconfig", "skypilot_config_path"):
        path = Path(values[field_name]).expanduser()
        if not path.is_absolute():
            raise _refusal(f"context-{field_name}-not-absolute", raw)
        values[field_name] = str(path)
    supplied = dict(config_bytes or {})
    kube_bytes = (
        supplied["kubeconfig"]
        if "kubeconfig" in supplied
        else _read_owner_file(
            values["kubeconfig"], label="kubeconfig", limit=MAX_CONFIG_BYTES
        )
    )
    sky_bytes = (
        supplied["skypilot_config_path"]
        if "skypilot_config_path" in supplied
        else _read_owner_file(
            values["skypilot_config_path"],
            label="skypilot-config",
            limit=MAX_CONFIG_BYTES,
        )
    )
    if len(kube_bytes) > MAX_CONFIG_BYTES or len(sky_bytes) > MAX_CONFIG_BYTES:
        raise _refusal("transport-config-too-large", raw)
    if len(raw) + len(kube_bytes) + len(sky_bytes) > MAX_TRANSPORT_SOURCE_BYTES:
        raise _refusal("transport-source-too-large", raw)
    _validate_kubeconfig(
        kube_bytes, context_name=values["kubernetes_context"], context=raw
    )
    _validate_skypilot_config(
        sky_bytes, context_name=values["kubernetes_context"], context=raw
    )
    redactions = tuple(
        dict.fromkeys(
            (
                *values.values(),
                raw.decode("utf-8", errors="replace"),
                *_portable_config_redactions(kube_bytes),
                *_portable_config_redactions(sky_bytes),
            )
        )
    )
    return RobotwinAuthorization(
        context_sha256=hashlib.sha256(raw).hexdigest(),
        project=values["project"],
        profile=values["nebius_profile"],
        kubeconfig_source=values["kubeconfig"],
        kubeconfig_bytes=kube_bytes,
        kubernetes_context=values["kubernetes_context"],
        skypilot_config_source=values["skypilot_config_path"],
        skypilot_config_bytes=sky_bytes,
        registry=values["registry"],
        bucket=values["bucket"],
        output_root=values["output_root"],
        run_id=values["run_id"],
        summary_uri=summary_uri,
        raw_context=raw,
        redactions=redactions,
    )


def load_runtime_authorization(
    environ: Mapping[str, str] | None = None,
) -> RobotwinAuthorization:
    """Load and validate the fixed public file-reference authorization."""

    source = os.environ if environ is None else environ
    raw = read_owner_context(source)
    kube_path = str(source.get(MATERIALIZED_KUBECONFIG_ENV) or "").strip()
    sky_path = str(source.get(MATERIALIZED_SKYPILOT_CONFIG_ENV) or "").strip()
    if bool(kube_path) != bool(sky_path):
        raise _refusal("materialized-config-set-incomplete", raw)
    if not kube_path:
        return validate_context_bytes(raw)
    config_bytes = {
        "kubeconfig": _read_owner_file(
            kube_path, label="materialized-kubeconfig", limit=MAX_CONFIG_BYTES
        ),
        "skypilot_config_path": _read_owner_file(
            sky_path, label="materialized-skypilot-config", limit=MAX_CONFIG_BYTES
        ),
    }
    authorization = validate_context_bytes(raw, config_bytes=config_bytes)
    return replace(
        authorization,
        kubeconfig_source=kube_path,
        skypilot_config_source=sky_path,
        redactions=tuple(
            dict.fromkeys((*authorization.redactions, kube_path, sky_path))
        ),
    )


def encode_transport(authorization: RobotwinAuthorization) -> str:
    """Encode one validated context and config byte set for secret-value transport."""

    def record(raw: bytes) -> dict[str, str]:
        return {
            "base64": base64.b64encode(raw).decode("ascii"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    payload = {
        "schema_version": TRANSPORT_SCHEMA,
        "context": record(authorization.raw_context),
        "files": {
            "kubeconfig": record(authorization.kubeconfig_bytes),
            "skypilot_config_path": record(authorization.skypilot_config_bytes),
        },
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _decode_transport_record(value: Any, *, label: str, limit: int) -> bytes:
    if not isinstance(value, dict) or set(value) != {"base64", "sha256"}:
        raise _refusal(f"transport-{label}-schema-mismatch")
    encoded = value.get("base64")
    digest = value.get("sha256")
    if not isinstance(encoded, str) or not isinstance(digest, str):
        raise _refusal(f"transport-{label}-schema-mismatch")
    if len(encoded) > 4 * ((limit + 2) // 3):
        raise _refusal(f"transport-{label}-too-large")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise _refusal(f"transport-{label}-invalid") from exc
    if len(raw) > limit:
        raise _refusal(f"transport-{label}-too-large")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise _refusal(f"transport-{label}-digest-mismatch", raw)
    return raw


def decode_transport(value: str) -> RobotwinAuthorization:
    """Strictly decode and revalidate the private worker transport value."""

    if not isinstance(value, str) or len(value.encode("utf-8")) > MAX_TRANSPORT_BYTES:
        raise _refusal("transport-size-invalid")
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise _refusal("transport-invalid-json") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "context", "files"
    }:
        raise _refusal("transport-schema-mismatch")
    if payload.get("schema_version") != TRANSPORT_SCHEMA:
        raise _refusal("transport-version-mismatch")
    files = payload.get("files")
    if not isinstance(files, dict) or set(files) != {
        "kubeconfig", "skypilot_config_path"
    }:
        raise _refusal("transport-files-schema-mismatch")
    raw = _decode_transport_record(
        payload["context"], label="context", limit=MAX_CONTEXT_BYTES
    )
    config_bytes = {
        name: _decode_transport_record(record, label=name, limit=MAX_CONFIG_BYTES)
        for name, record in files.items()
    }
    return validate_context_bytes(raw, config_bytes=config_bytes)


def _contract_marker(spec: Any) -> bool:
    config = getattr(spec, "config", {}) or {}
    return any(
        (
            getattr(spec, "name", "") == "byof-robotwin",
            config.get("solution_name") == "robotwin",
            config.get("runtime_context_env") == PUBLIC_CONTEXT_ENV,
            config.get("repo_ref") == SOURCE_REVISION,
        )
    )


def _expected_config() -> dict[str, Any]:
    return {
        **{key: value for key, value in INVOCATION.items() if key != "yaml"},
        "resource_profile_yaml": INVOCATION["yaml"],
        "repo_auth": "none",
        "repo_token_env": "GH_TOKEN",
        "iterations": 1,
        "num_envs": 1,
        "num_demos": 1,
        "wait_timeout": -1,
        "poll_interval": 60,
        "bucket": "example-bucket",
        "output_root": "s3://{{config.bucket}}/oss-solutions/robotwin",
        "summary_uri": "{{config.output_root}}/{{run.id}}/npa_byof_summary.json",
    }


def recognize_contract(spec: Any) -> bool:
    """Recognize only the complete immutable public RoboTwin workflow contract."""

    if not _contract_marker(spec):
        return False
    config = dict(getattr(spec, "config", {}) or {})
    expected = _expected_config()
    if set(config) != {*expected, "build_command", "smoke_command"}:
        raise _refusal("workflow-config-schema-mismatch")
    for key, value in expected.items():
        if config.get(key) != value:
            raise _refusal(f"workflow-config-{key}-mismatch")
    if hashlib.sha256(str(config.get("build_command") or "").encode()).hexdigest() != BUILD_COMMAND_SHA256:
        raise _refusal("workflow-build-command-mismatch")
    smoke = str(config.get("smoke_command") or "")
    if hashlib.sha256(smoke.encode()).hexdigest() != SMOKE_COMMAND_SHA256:
        raise _refusal("workflow-smoke-command-mismatch")
    if ASSET_REVISION not in smoke:
        raise _refusal("workflow-asset-revision-mismatch")
    states = getattr(spec, "states", {}) or {}
    if set(states) != {"byof-run"} or getattr(spec, "initial", "") != "byof-run":
        raise _refusal("workflow-state-shape-mismatch")
    state = states["byof-run"]
    if state.tool_ref != "workbench.byof.repo" or state.resources != "launcher":
        raise _refusal("workflow-tool-resource-mismatch")
    if not state.terminal or len(state.outputs) != 1:
        raise _refusal("workflow-output-shape-mismatch")
    output = state.outputs[0]
    if output.uri != "{{config.summary_uri}}" or output.schema != "npa.workbench.byof.summary.v1":
        raise _refusal("workflow-output-contract-mismatch")
    if getattr(spec, "resources", {}) != {
        "launcher": {"cloud": "kubernetes", "cpus": "4+", "memory": "8+"}
    }:
        raise _refusal("workflow-outer-resource-mismatch")
    return True


def validate_invocation(args: Any) -> None:
    """Validate the immutable BYOF runner invocation before any work begins."""

    for name, expected in INVOCATION.items():
        if getattr(args, name) != expected:
            raise _refusal(f"invocation-{name}-mismatch")
    hashes = {
        "build_command": BUILD_COMMAND_SHA256,
        "smoke_command": SMOKE_COMMAND_SHA256,
    }
    for name, expected in hashes.items():
        if hashlib.sha256(getattr(args, name).encode()).hexdigest() != expected:
            raise _refusal(f"invocation-{name}-mismatch")
    scalar_match = (
        args.repo_auth == "none"
        and args.iterations == 1
        and args.num_envs == 1
        and args.num_demos == 1
        and args.wait_timeout == -1
        and args.poll_interval == 60
        and args.cleanup
    )
    if not scalar_match or args.skip_build or args.skip_push or args.skip_run:
        raise _refusal("invocation-smoke-contract-mismatch")
    private_empty = all(
        not getattr(args, name).strip()
        for name in ("project", "registry", "image", "config_path")
    )
    if not private_empty or args.output_root.strip() not in {
        "", "s3://example-bucket/oss-solutions/robotwin"
    }:
        raise _refusal("invocation-private-coordinate-override")


def prepare_live_submit(
    spec: Any,
    *,
    requested_secret_envs: Sequence[str],
    environ: Mapping[str, str] | None = None,
) -> RobotwinSubmitContext | None:
    """Validate RoboTwin locally and return its value-only secret transport."""

    if not recognize_contract(spec):
        return None
    if PUBLIC_CONTEXT_ENV not in requested_secret_envs:
        raise _refusal("context-secret-not-requested")
    authorization = load_runtime_authorization(environ)
    return RobotwinSubmitContext(
        authorization.context_sha256,
        authorization,
        encode_transport(authorization),
    )


def bind_submit_coordinates(
    context: RobotwinSubmitContext,
    *,
    project: str,
    infra: str,
    config_path: Path | None,
    s3_bucket: str,
    resume_requested: bool,
    alternate_output_requested: bool,
    outer_override_requested: bool,
    runtime_requested: bool,
) -> tuple[str, str, Path]:
    """Bind live launcher coordinates to the validated manager authorization."""

    authorization = context.authorization
    expected_infra = f"k8s/{authorization.kubernetes_context}"
    if project and project != authorization.project:
        raise _refusal("submit-project-conflict", authorization.raw_context)
    if infra and infra != expected_infra:
        raise _refusal("submit-infra-conflict", authorization.raw_context)
    if config_path is not None and str(config_path) != authorization.skypilot_config_source:
        raise _refusal("submit-skypilot-config-conflict", authorization.raw_context)
    if s3_bucket and s3_bucket != authorization.bucket:
        raise _refusal("submit-bucket-conflict", authorization.raw_context)
    if resume_requested:
        raise _refusal("submit-resume-refused", authorization.raw_context)
    if alternate_output_requested:
        raise _refusal("submit-output-override-refused", authorization.raw_context)
    if outer_override_requested:
        raise _refusal("submit-outer-resource-override-refused", authorization.raw_context)
    if runtime_requested:
        raise _refusal("submit-runtime-mode-refused", authorization.raw_context)
    return (
        authorization.project,
        expected_infra,
        Path(authorization.skypilot_config_source),
    )


def write_owner_file(path: Path, raw: bytes) -> None:
    """Create one owner-only regular file without following or replacing paths."""

    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def materialize_transport(
    value: str, directory: Path
) -> MaterializedRobotwinContext:
    """Decode a transport and write its exact bytes below one 0700 directory."""

    authorization = decode_transport(value)
    directory.chmod(0o700)
    context_path = directory / "runtime-context.json"
    kubeconfig_path = directory / "kubeconfig.yaml"
    skypilot_path = directory / "skypilot-config.yaml"
    write_owner_file(context_path, authorization.raw_context)
    write_owner_file(kubeconfig_path, authorization.kubeconfig_bytes)
    write_owner_file(skypilot_path, authorization.skypilot_config_bytes)
    return MaterializedRobotwinContext(
        context_path,
        kubeconfig_path,
        skypilot_path,
        authorization,
    )
