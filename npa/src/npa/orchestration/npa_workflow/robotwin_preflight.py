"""Fail-closed authorization transport for the fixed RoboTwin BYOF workflow."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import posixpath
import re
import stat
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse, urlsplit

import yaml

PUBLIC_CONTEXT_ENV = "NPA_BYOF_ROBOTWIN_RUNTIME_CONTEXT"
TRANSPORT_CONTEXT_ENV = "NPA_INTERNAL_BYOF_ROBOTWIN_CONTEXT_V1"
MATERIALIZED_KUBECONFIG_ENV = "NPA_INTERNAL_BYOF_ROBOTWIN_KUBECONFIG"
MATERIALIZED_SKYPILOT_CONFIG_ENV = "NPA_INTERNAL_BYOF_ROBOTWIN_SKYPILOT_CONFIG"
CHILD_IMAGE_ENV = "NPA_INTERNAL_BYOF_ROBOTWIN_IMAGE"
CHILD_RUN_ID_ENV = "NPA_INTERNAL_BYOF_ROBOTWIN_RUN_ID"
CHILD_OUTPUT_ROOT_ENV = "NPA_INTERNAL_BYOF_ROBOTWIN_OUTPUT_ROOT"
CHILD_OUTPUT_PREFIX_ENV = "NPA_INTERNAL_BYOF_ROBOTWIN_OUTPUT_PREFIX"
CHILD_BUCKET_ENV = "NPA_INTERNAL_BYOF_ROBOTWIN_BUCKET"
CHILD_RUNTIME_AUTH_ENV = "NPA_INTERNAL_BYOF_ROBOTWIN_RUNTIME_AUTH_V1"
CHILD_CONFIG_PATH_ENV = "NPA_INTERNAL_BYOF_ROBOTWIN_CHILD_SKYPILOT_CONFIG"
CONTEXT_ENV_NAMES = (
    PUBLIC_CONTEXT_ENV,
    TRANSPORT_CONTEXT_ENV,
    MATERIALIZED_KUBECONFIG_ENV,
    MATERIALIZED_SKYPILOT_CONFIG_ENV,
    CHILD_IMAGE_ENV,
    CHILD_RUN_ID_ENV,
    CHILD_OUTPUT_ROOT_ENV,
    CHILD_OUTPUT_PREFIX_ENV,
    CHILD_BUCKET_ENV,
    CHILD_RUNTIME_AUTH_ENV,
    CHILD_CONFIG_PATH_ENV,
)
MAX_CONTEXT_BYTES = 64 * 1024
MAX_CONFIG_BYTES = 24 * 1024
MAX_TRANSPORT_SOURCE_BYTES = 64 * 1024
MAX_TRANSPORT_BYTES = 96 * 1024
TRANSPORT_SCHEMA = "npa.byof.robotwin.runtime-transport.v1"
SOURCE_REVISION = "96c1feab536306b50c26af200044fcdf126e8904"
CUROBO_REVISION = "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"
ASSET_REVISION = "785feb15aa4a4f532395ad2b1d2be5f28cb561ad"
WORKFLOW_SHA256 = "718bb6ae47c8e5e7e761303ebda9e962afa446a6b84030dade7c224cd255ece3"
RUNTIME_LOCK_SHA256 = "c42c4037392f51ad6c2473eb3f07843738a4c5147328ace1686ddb9cf553b4ef"
RUNTIME_LOCK_STATUS = "incomplete"
RUNTIME_AUTH_SCHEMA = "npa.byof.robotwin.runtime-authorization.v1"
RTX_ACCELERATOR = "RTXPRO-6000-BLACKWELL-SERVER-EDITION"
BUILD_COMMAND_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SMOKE_COMMAND_SHA256 = (
    "6ef4b9cf7691a5b33aa2daf10b45909df024fa836ff7c7c5d2bbc2faa3e31f13"
)
OUTER_RUN_SHA256 = (
    "ece33734929e9dfedbd68215e3a530c90e93f849a891ea669114370f95b27beb"
)
INNER_SETUP_SHA256 = (
    "601a7e430674e8172e49c00d0f8e428e18d6e3f605f5680146d0b671bb4ee81d"
)
INNER_RUN_SHA256 = (
    "de11d89c4016819f10d53bbdaaaba3bc3b9e957193d67b0d270263a33aac31d7"
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
    "base_profile": "prebuilt",
    "base_image": "tool://robotwin",
    "workload": "solution-smoke",
    "solution_name": "robotwin",
    "capability_name": "beat_block_hammer_successful_seed_replay_collection",
    "smoke_artifact_name": "robotwin-smoke.json",
    "runtime_context_env": PUBLIC_CONTEXT_ENV,
    "yaml": "byof-solution-smoke-robotwin-rtxpro-gpu",
    "task": "beat_block_hammer",
}


def _is_official_robotwin_repository(value: str) -> bool:
    """Recognize equivalent Git spellings of the pinned official repository."""

    text = value.strip()
    if not text:
        return False
    scp_match = re.fullmatch(
        r"(?:[^@/:\s]+@)?(?P<host>(?:www\.)?github\.com\.?):(?P<path>[^?#]+)",
        text,
        re.I,
    )
    if scp_match:
        host = scp_match.group("host").lower().rstrip(".")
        path = scp_match.group("path")
    else:
        candidate = text if "://" in text else f"//{text}"
        try:
            parsed = urlsplit(candidate)
            port = parsed.port
        except ValueError:
            return False
        host = (parsed.hostname or "").lower().rstrip(".")
        if port not in (None, 443):
            return False
        path = parsed.path
    if host not in {"github.com", "www.github.com"}:
        return False
    normalized = posixpath.normpath("/" + unquote(path).lstrip("/"))
    normalized = normalized.rstrip("/").removesuffix(".git").lower()
    return normalized == "/robotwin-platform/robotwin"


def _is_robotwin_image_reference(value: str) -> bool:
    """Recognize immutable and mutable references to the RoboTwin image repo."""

    reference = value.strip().removeprefix("docker:")
    if not reference or any(character.isspace() for character in reference):
        return False
    repository = reference.split("@", 1)[0]
    last_slash = repository.rfind("/")
    last_colon = repository.rfind(":")
    if last_colon > last_slash:
        repository = repository[:last_colon]
    return repository.rsplit("/", 1)[-1].lower() == "npa-robotwin"


def is_robotwin_request(
    *,
    solution_name: str = "",
    repo_url: str = "",
    base_image: str = "",
    image: str = "",
    smoke_command: str = "",
    capability_name: str = "",
    yaml_path: str = "",
) -> bool:
    """Recognize public inputs that could otherwise relabel the RoboTwin path."""

    return any(
        (
            solution_name.strip().lower() == "robotwin",
            _is_official_robotwin_repository(repo_url),
            base_image.strip().lower() == "tool://robotwin",
            _is_robotwin_image_reference(base_image),
            _is_robotwin_image_reference(image),
            smoke_command.strip() == "/opt/npa/robotwin/robotwin-runtime run",
            capability_name.strip() == INVOCATION["capability_name"],
            Path(yaml_path).name.removesuffix(".yaml") == INVOCATION["yaml"],
        )
    )


_CONTEXT_FIELDS = frozenset(
    {
        "solution",
        "ownership_provenance",
        "workflow_sha256",
        "source_revision",
        "curobo_revision",
        "asset_revision",
        "runtime_lock_sha256",
        "bootstrap_image",
        "reservation",
        "license_acceptance",
        "project",
        "nebius_profile",
        "kubeconfig",
        "kubernetes_context",
        "skypilot_config_path",
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
    bootstrap_image: str = field(repr=False)
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
    layer: str = "outer"
    private_values: tuple[str, ...] = field(default=(), repr=False)
    private_environment: tuple[tuple[str, str], ...] = field(default=(), repr=False)


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
    raw = b""
    unreadable = False
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
    except OSError:
        unreadable = True
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if unreadable:
        raise _refusal(f"{label}-unreadable")
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
    value: Any = None
    invalid = False
    try:
        decoded = raw.decode("utf-8")
        value = yaml.safe_load(decoded)
    except (UnicodeDecodeError, yaml.YAMLError):
        invalid = True
    if invalid:
        raise _refusal(f"{label}-invalid", context)
    if not isinstance(value, dict):
        raise _refusal(f"{label}-invalid", context)
    return value


def _has_external_reference(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"(?<!^)(?=[A-Z])", "_", str(key)).lower().replace(
                "-", "_"
            )
            if normalized in {
                "include",
                "includes",
                "auth_provider",
                "credential_process",
                "token_file",
                "client_certificate",
                "client_key",
                "certificate_authority",
                "proxy_url",
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
    if set(payload) != {
        "apiVersion",
        "kind",
        "clusters",
        "contexts",
        "users",
        "current-context",
    } or payload.get("apiVersion") != "v1" or payload.get("kind") != "Config":
        raise _refusal("kubeconfig-schema-mismatch", context)
    if payload.get("current-context") != context_name:
        raise _refusal("kubeconfig-current-context-mismatch", context)

    def named_entry(section: str, name: str) -> Mapping[str, Any]:
        entries = payload.get(section)
        if not isinstance(entries, list) or len(entries) != 1:
            raise _refusal("kubeconfig-structure-invalid", context)
        matches = [
            entry
            for entry in entries
            if isinstance(entry, Mapping) and entry.get("name") == name
        ]
        if len(matches) != 1:
            raise _refusal(f"kubeconfig-{section}-mismatch", context)
        entry = matches[0]
        singular = {"clusters": "cluster", "contexts": "context", "users": "user"}[
            section
        ]
        if set(entry) != {"name", singular}:
            raise _refusal("kubeconfig-structure-invalid", context)
        return entry

    selected = named_entry("contexts", context_name).get("context")
    if (
        not isinstance(selected, Mapping)
        or not set(selected).issubset({"cluster", "user", "namespace"})
        or not {"cluster", "user"}.issubset(selected)
    ):
        raise _refusal("kubeconfig-context-invalid", context)
    namespace = selected.get("namespace")
    if namespace is not None and (
        not isinstance(namespace, str) or not namespace.strip()
    ):
        raise _refusal("kubeconfig-context-invalid", context)
    cluster_name = selected.get("cluster")
    user_name = selected.get("user")
    if not isinstance(cluster_name, str) or not isinstance(user_name, str):
        raise _refusal("kubeconfig-context-mismatch", context)
    cluster = named_entry("clusters", cluster_name).get("cluster")
    user = named_entry("users", user_name).get("user")
    if not isinstance(cluster, Mapping) or not isinstance(user, Mapping):
        raise _refusal("kubeconfig-structure-invalid", context)
    if set(cluster) != {"server", "certificate-authority-data"}:
        raise _refusal("kubeconfig-cluster-schema-mismatch", context)
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
    certificate_authority_valid = True
    try:
        certificate_authority_valid = bool(
            base64.b64decode(cluster["certificate-authority-data"], validate=True)
        )
    except (TypeError, ValueError):
        certificate_authority_valid = False
    if not certificate_authority_valid:
        raise _refusal("kubeconfig-cluster-not-portable", context)
    if "exec" in user:
        raise _refusal("kubeconfig-exec-plugin-refused", context)
    inline_token = isinstance(user.get("token"), str) and bool(user["token"].strip())
    inline_cert = all(
        isinstance(user.get(name), str) and bool(user[name].strip())
        for name in ("client-certificate-data", "client-key-data")
    )
    allowed_user_keys = (
        {"token"}
        if inline_token and not inline_cert
        else {"client-certificate-data", "client-key-data"}
        if inline_cert and not inline_token
        else set()
    )
    if not allowed_user_keys or set(user) != allowed_user_keys:
        raise _refusal("kubeconfig-user-not-portable", context)
    if inline_cert:
        inline_certificate_valid = True
        try:
            inline_certificate_valid = not any(
                not base64.b64decode(user[name], validate=True)
                for name in ("client-certificate-data", "client-key-data")
            )
        except (TypeError, ValueError):
            inline_certificate_valid = False
        if not inline_certificate_valid:
            raise _refusal("kubeconfig-user-not-portable", context)
    if _has_external_reference(payload):
        raise _refusal("kubeconfig-external-reference", context)


def _validate_skypilot_config(
    raw: bytes, *, context_name: str, context: bytes
) -> None:
    payload = _load_yaml_mapping(raw, label="skypilot-config", context=context)
    if _has_external_reference(payload):
        raise _refusal("skypilot-config-external-reference", context)
    if set(payload) != {"kubernetes"}:
        raise _refusal("skypilot-config-schema-mismatch", context)
    kubernetes = payload.get("kubernetes")
    if not isinstance(kubernetes, Mapping) or not set(kubernetes).issubset(
        {"allowed_contexts", "pod_config"}
    ) or "allowed_contexts" not in kubernetes:
        raise _refusal("skypilot-config-kubernetes-invalid", context)
    allowed = kubernetes.get("allowed_contexts")
    if allowed != [context_name]:
        raise _refusal("skypilot-config-context-mismatch", context)
    pod_config = kubernetes.get("pod_config")
    if pod_config is None:
        return
    if not isinstance(pod_config, Mapping) or set(pod_config) != {"spec"}:
        raise _refusal("skypilot-config-pod-schema-mismatch", context)
    pod_spec = pod_config.get("spec")
    if not isinstance(pod_spec, Mapping) or set(pod_spec) != {"imagePullSecrets"}:
        raise _refusal("skypilot-config-pod-schema-mismatch", context)
    pull_secrets = pod_spec.get("imagePullSecrets")
    if (
        not isinstance(pull_secrets, list)
        or not pull_secrets
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"name"}
            or not isinstance(item.get("name"), str)
            or re.fullmatch(
                r"[a-z0-9](?:[-a-z0-9.]{0,251}[a-z0-9])?",
                item["name"],
            )
            is None
            for item in pull_secrets
        )
    ):
        raise _refusal("skypilot-config-pod-schema-mismatch", context)


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
        elif (
            isinstance(value, str)
            and len(value.strip()) >= 4
            and value.strip() != "Config"
        ):
            values.append(value.strip())

    collect(payload)
    return tuple(dict.fromkeys(values))


def _parse_context_payload(raw: bytes) -> dict[str, Any]:
    payload: Any = None
    invalid = False
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        invalid = True
    if invalid:
        raise _refusal("context-invalid-json", raw)
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
    values = {
        name: _context_text(payload, name, raw)
        for name in _CONTEXT_FIELDS
        if name not in {"solution", "reservation", "license_acceptance"}
    }
    immutable_values = {
        "workflow_sha256": WORKFLOW_SHA256,
        "source_revision": SOURCE_REVISION,
        "curobo_revision": CUROBO_REVISION,
        "asset_revision": ASSET_REVISION,
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
    }
    for name, expected in immutable_values.items():
        if values[name] != expected:
            raise _refusal(f"context-{name}-mismatch", raw)
    return values


def _validate_destination(values: Mapping[str, str], raw: bytes) -> str:
    image = values["bootstrap_image"]
    if (
        "://" in image
        or any(character.isspace() for character in image)
        or image.startswith("/")
        or re.fullmatch(r"[^@]+/npa-robotwin@sha256:[0-9a-f]{64}", image) is None
    ):
        raise _refusal("bootstrap-image-not-immutable", raw)
    authority, separator, namespace = image.partition("/")
    parsed_registry = urlparse(f"//{authority}")
    valid_registry_port = True
    try:
        parsed_registry.port
    except ValueError:
        valid_registry_port = False
    if not valid_registry_port:
        raise _refusal("bootstrap-image-not-immutable", raw)
    host = (parsed_registry.hostname or "").lower()
    explicit_authority = bool(
        separator
        and namespace
        and parsed_registry.netloc == authority
        and parsed_registry.username is None
        and parsed_registry.password is None
        and host not in {"", "localhost"}
        and ("." in host or ":" in authority)
        and all(
            re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", part) is not None
            for part in namespace.split("@", 1)[0].split("/")
        )
    )
    if not explicit_authority:
        raise _refusal("bootstrap-image-not-immutable", raw)
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
                *(
                    values[name]
                    for name in (
                        "ownership_provenance",
                        "bootstrap_image",
                        "project",
                        "nebius_profile",
                        "kubeconfig",
                        "kubernetes_context",
                        "skypilot_config_path",
                        "bucket",
                        "output_root",
                        "run_id",
                    )
                ),
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
        bootstrap_image=values["bootstrap_image"],
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


def require_runtime_lock_complete(
    authorization: RobotwinAuthorization,
) -> RobotwinAuthorization:
    """Stop the Phase A candidate before image, network, scheduler, or GPU work."""

    if RUNTIME_LOCK_STATUS != "complete":
        raise _refusal("runtime-lock-incomplete", authorization.raw_context)
    return authorization


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


def encode_runtime_authorization(authorization: RobotwinAuthorization) -> str:
    """Encode only validated context bytes for the GPU bootstrap secret."""

    return json.dumps(
        {
            "schema_version": RUNTIME_AUTH_SCHEMA,
            "context_base64": base64.b64encode(authorization.raw_context).decode("ascii"),
            "context_sha256": authorization.context_sha256,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_transport_record(value: Any, *, label: str, limit: int) -> bytes:
    if not isinstance(value, dict) or set(value) != {"base64", "sha256"}:
        raise _refusal(f"transport-{label}-schema-mismatch")
    encoded = value.get("base64")
    digest = value.get("sha256")
    if not isinstance(encoded, str) or not isinstance(digest, str):
        raise _refusal(f"transport-{label}-schema-mismatch")
    if len(encoded) > 4 * ((limit + 2) // 3):
        raise _refusal(f"transport-{label}-too-large")
    raw = b""
    invalid = False
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        invalid = True
    if invalid:
        raise _refusal(f"transport-{label}-invalid")
    if len(raw) > limit:
        raise _refusal(f"transport-{label}-too-large")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise _refusal(f"transport-{label}-digest-mismatch", raw)
    return raw


def decode_transport(value: str) -> RobotwinAuthorization:
    """Strictly decode and revalidate the private worker transport value."""

    if not isinstance(value, str) or len(value.encode("utf-8")) > MAX_TRANSPORT_BYTES:
        raise _refusal("transport-size-invalid")
    payload: Any = None
    invalid = False
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        invalid = True
    if invalid:
        raise _refusal("transport-invalid-json")
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
        and args.repo_token_env == "GH_TOKEN"
        and not args.wan_acceptance_candidate_image
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
        for name in ("project", "registry", "image", "config_path", "sky_bin")
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
    authorization = require_runtime_lock_complete(load_runtime_authorization(environ))
    return RobotwinSubmitContext(
        authorization.context_sha256,
        authorization,
        encode_transport(authorization),
    )


def prepare_inner_submit(
    authorization: RobotwinAuthorization,
    environment: Mapping[str, str],
) -> RobotwinSubmitContext:
    """Bind the inner launcher to the same in-process validated authorization."""

    output_prefix = f"{authorization.output_root.rstrip('/')}/{authorization.run_id}/"
    expected = {
        "KUBECONFIG": authorization.kubeconfig_source,
        "KUBECONTEXT": authorization.kubernetes_context,
        "NPA_BYOF_K8S_CONTEXT": authorization.kubernetes_context,
        "NPA_BYOF_PROJECT": authorization.project,
        "NPA_NEBIUS_PROFILE": authorization.profile,
        "NEBIUS_PROFILE": authorization.profile,
        CHILD_BUCKET_ENV: authorization.bucket,
        CHILD_CONFIG_PATH_ENV: authorization.skypilot_config_source,
        CHILD_OUTPUT_PREFIX_ENV: output_prefix,
        CHILD_OUTPUT_ROOT_ENV: authorization.output_root,
        CHILD_RUN_ID_ENV: authorization.run_id,
        "NPA_BYOF_ROBOTWIN_RESERVATION_EVIDENCE_SHA256": (
            authorization.context_sha256
        ),
    }
    if any(str(environment.get(name) or "") != value for name, value in expected.items()):
        raise _refusal("inner-environment-mismatch", authorization.raw_context)
    image = str(environment.get(CHILD_IMAGE_ENV) or "")
    scan_sha256 = str(environment.get("NPA_BYOF_ROBOTWIN_IMAGE_SCAN_SHA256") or "")
    scan_archives = str(
        environment.get("NPA_BYOF_ROBOTWIN_IMAGE_SCAN_ARCHIVES") or ""
    )
    runtime_authorization = encode_runtime_authorization(authorization)
    if image != authorization.bootstrap_image:
        raise _refusal("inner-image-mismatch", authorization.raw_context)
    if environment.get(CHILD_RUNTIME_AUTH_ENV) != runtime_authorization:
        raise _refusal("inner-runtime-authorization-mismatch", authorization.raw_context)
    if re.fullmatch(r"[0-9a-f]{64}", scan_sha256) is None or not (
        scan_archives.isdecimal() and int(scan_archives) >= 2
    ):
        raise _refusal("inner-image-scan-evidence-missing", authorization.raw_context)
    endpoint = str(environment.get("AWS_ENDPOINT_URL") or "")
    if (
        re.fullmatch(r"https://storage\.[a-z0-9-]+\.nebius\.cloud", endpoint)
        is None
        or environment.get("NEBIUS_S3_ENDPOINT") != endpoint
    ):
        raise _refusal("inner-storage-endpoint-mismatch", authorization.raw_context)
    bound_environment = {
        "KUBECONFIG": authorization.kubeconfig_source,
        CHILD_BUCKET_ENV: authorization.bucket,
        CHILD_IMAGE_ENV: image,
        CHILD_OUTPUT_PREFIX_ENV: output_prefix,
        CHILD_RUN_ID_ENV: authorization.run_id,
        CHILD_RUNTIME_AUTH_ENV: runtime_authorization,
        "AWS_ENDPOINT_URL": endpoint,
        "NEBIUS_S3_ENDPOINT": endpoint,
        "NPA_BYOF_ROBOTWIN_RESERVATION_EVIDENCE_SHA256": (
            authorization.context_sha256
        ),
        "NPA_BYOF_ROBOTWIN_IMAGE_SCAN_SHA256": scan_sha256,
        "NPA_BYOF_ROBOTWIN_IMAGE_SCAN_ARCHIVES": scan_archives,
    }
    private_values = tuple(
        dict.fromkeys(
            (
                *authorization.redactions,
                image,
                endpoint,
                scan_sha256,
                authorization.context_sha256,
                runtime_authorization,
            )
        )
    )
    return RobotwinSubmitContext(
        authorization.context_sha256,
        authorization,
        "",
        layer="inner",
        private_values=private_values,
        private_environment=tuple(bound_environment.items()),
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


def _recognize_rendered_contract(documents: Sequence[Mapping[str, Any]]) -> bool:
    """Recognize the one rendered task produced by the immutable public spec."""

    if len(documents) != 2 or documents[0] != {
        "name": "byof-robotwin",
        "execution": "serial",
    }:
        return False
    task = documents[1]
    resources = task.get("resources")
    environment = task.get("envs")
    command = task.get("run")
    if task.get("name") != "byof-run" or resources != {
        "cloud": "kubernetes",
        "cpus": "4+",
        "memory": "8+",
    }:
        return False
    if not isinstance(environment, Mapping) or not isinstance(command, str):
        return False
    required_environment = {
        "NPA_WORKFLOW_NAME": "byof-robotwin",
        "NPA_WORKFLOW_STATE": "byof-run",
        "NPA_EXECUTION_OUTPUTS": "[]",
        "NPA_WORKFLOW_FENCE_SEQUENCE": "1",
        "NPA_WORKFLOW_FENCE_ATTEMPT": "1",
    }
    allowed_environment = {
        *required_environment,
        "NPA_WORKFLOW_RUN_ID",
        "NPA_WORKFLOW_ATTEMPT_ID",
        "NPA_SRC_S3_URI",
        "AWS_ENDPOINT_URL",
    }
    if set(environment) != allowed_environment or any(
        environment.get(name) != value
        for name, value in required_environment.items()
    ):
        return False
    run_id = environment.get("NPA_WORKFLOW_RUN_ID")
    attempt_id = environment.get("NPA_WORKFLOW_ATTEMPT_ID")
    source_uri = urlparse(str(environment.get("NPA_SRC_S3_URI") or ""))
    endpoint = str(environment.get("AWS_ENDPOINT_URL") or "")
    if (
        not isinstance(run_id, str)
        or not run_id
        or command.count(run_id) != 1
        or re.fullmatch(r"[0-9a-f]{64}", str(attempt_id or "")) is None
        or source_uri.scheme != "s3"
        or not source_uri.netloc
        or re.fullmatch(r"https://storage\.[a-z0-9-]+\.nebius\.cloud", endpoint)
        is None
    ):
        return False
    normalized_command = command.replace(run_id, "<run-id>")
    return (
        hashlib.sha256(normalized_command.encode()).hexdigest() == OUTER_RUN_SHA256
        and TRANSPORT_CONTEXT_ENV not in command
    )


def _recognize_inner_rendered_contract(
    documents: Sequence[Mapping[str, Any]],
) -> bool:
    """Recognize the exact one-GPU profile rendered by the validated runner."""

    if len(documents) != 2 or documents[0] != {
        "name": "byof-solution-smoke-robotwin-rtxpro-gpu",
        "execution": "serial",
    }:
        return False
    task = documents[1]
    expected_resources = {
        "cloud": "kubernetes",
        "accelerators": f"{RTX_ACCELERATOR}:1",
        "cpus": "16+",
        "memory": "64+",
        "disk_size": 100,
        "image_id": f"docker:${{{CHILD_IMAGE_ENV}}}",
        "kubernetes": {
            "pod_config": {
                "spec": {
                    "containers": [
                        {
                            "name": "ray-node",
                            "env": [
                                {
                                    "name": "POD_NAME",
                                    "valueFrom": {
                                        "fieldRef": {"fieldPath": "metadata.name"}
                                    },
                                },
                                {
                                    "name": "POD_NAMESPACE",
                                    "valueFrom": {
                                        "fieldRef": {"fieldPath": "metadata.namespace"}
                                    },
                                },
                            ],
                        }
                    ]
                }
            }
        },
    }
    if task.get("name") != "byof-solution-smoke-robotwin-rtxpro":
        return False
    if task.get("resources") != expected_resources:
        return False
    if hashlib.sha256(str(task.get("setup") or "").encode()).hexdigest() != INNER_SETUP_SHA256:
        return False
    if hashlib.sha256(str(task.get("run") or "").encode()).hexdigest() != INNER_RUN_SHA256:
        return False
    environment = task.get("envs")
    if not isinstance(environment, Mapping):
        return False
    required_environment = {
        "NPA_BYOF_RUN_ID": f"${{{CHILD_RUN_ID_ENV}}}",
        "BYOF_REPO_ROOT": "/opt/byof",
        "BYOF_SOLUTION_NAME": "robotwin",
        "BYOF_CAPABILITY_NAME": INVOCATION["capability_name"],
        "BYOF_SMOKE_ARTIFACT_NAME": INVOCATION["smoke_artifact_name"],
        "BYOF_IMAGE": f"${{{CHILD_IMAGE_ENV}}}",
        CHILD_RUNTIME_AUTH_ENV: f"${{{CHILD_RUNTIME_AUTH_ENV}}}",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
        "S3_OUTPUT_PREFIX": f"${{{CHILD_OUTPUT_PREFIX_ENV}}}",
        "NPA_S3_BUCKET": f"${{{CHILD_BUCKET_ENV}}}",
    }
    if any(environment.get(name) != value for name, value in required_environment.items()):
        return False
    if hashlib.sha256(str(environment.get("BYOF_SMOKE_COMMAND") or "").encode()).hexdigest() != SMOKE_COMMAND_SHA256:
        return False
    allowed = {
        *required_environment,
        "BYOF_SMOKE_COMMAND",
        "AWS_ENDPOINT_URL",
        "NEBIUS_S3_ENDPOINT",
    }
    if set(environment) != allowed:
        return False
    return environment.get("AWS_ENDPOINT_URL") == "${AWS_ENDPOINT_URL}" and (
        environment.get("NEBIUS_S3_ENDPOINT") == "${NEBIUS_S3_ENDPOINT}"
    )


def _validate_preverified_target(
    authorization: RobotwinAuthorization,
    target: Any,
    report: Mapping[str, Any] | None,
) -> None:
    """Require the exact owner/output target and its completed verification report."""

    from npa.execution_preflight import ExecutionTarget

    checks = report.get("checks") if isinstance(report, Mapping) else None
    if (
        not isinstance(checks, Mapping)
        or report.get("schema_version") != "npa.execution-preflight.v1"
        or report.get("presence") != "pass"
        or report.get("access") != "pass"
        or report.get("execution_readiness") != "pass"
        or report.get("destination_count") != 1
        or checks.get("scope") != "pass"
        or checks.get("storage_owner") != "pass"
        or checks.get("cluster_owner") != "pass"
        or checks.get("storage_write_read") != "pass"
    ):
        raise _refusal("submit-target-not-preverified", authorization.raw_context)
    if (
        not isinstance(target, ExecutionTarget)
        or getattr(target, "project", None) != authorization.project
        or getattr(target, "context", None) != authorization.kubernetes_context
    ):
        raise _refusal("submit-target-mismatch", authorization.raw_context)
    output_uris = tuple(getattr(target, "output_uris", ()) or ())
    summary_outputs = [
        uri for uri in output_uris if str(uri).endswith("/npa_byof_summary.json")
    ]
    output_kinds = dict(getattr(target, "output_kinds", {}) or {})
    if (
        summary_outputs != [authorization.summary_uri]
        or output_kinds.get(authorization.summary_uri) != "file"
    ):
        raise _refusal("submit-output-binding-mismatch", authorization.raw_context)


def validate_confidential_submit_bridge(
    context: RobotwinSubmitContext,
    *,
    documents: Sequence[Mapping[str, Any]],
    infra: str,
    config_path: Path | None,
    secret_envs: Sequence[str],
    extra_env: Mapping[str, str],
    execution_target: Any,
    execution_report: Mapping[str, Any] | None,
) -> RobotwinAuthorization:
    """Authorize the narrow shared submit bridge without exposing private values."""

    if not isinstance(context, RobotwinSubmitContext):
        raise _refusal("submit-bridge-contract-mismatch")
    authorization = context.authorization
    if context.context_sha256 != authorization.context_sha256:
        raise _refusal("submit-bridge-context-mismatch", authorization.raw_context)
    expected_infra = f"k8s/{authorization.kubernetes_context}"
    expected_config = authorization.skypilot_config_source
    common_valid = (
        infra == expected_infra
        and config_path is not None
        and str(Path(config_path).expanduser()) == expected_config
        and extra_env.get("KUBECONFIG") == authorization.kubeconfig_source
        and PUBLIC_CONTEXT_ENV not in secret_envs
    )
    if context.layer == "outer":
        if not _recognize_rendered_contract(documents):
            raise _refusal("submit-bridge-contract-mismatch", authorization.raw_context)
        transport = extra_env.get(TRANSPORT_CONTEXT_ENV)
        layer_valid = (
            TRANSPORT_CONTEXT_ENV in secret_envs
            and transport == context.transport_value
            and transport == encode_transport(authorization)
        )
    elif context.layer == "inner":
        if not _recognize_inner_rendered_contract(documents):
            raise _refusal("submit-bridge-contract-mismatch", authorization.raw_context)
        required_secrets = {
            CHILD_BUCKET_ENV,
            CHILD_IMAGE_ENV,
            CHILD_OUTPUT_PREFIX_ENV,
            CHILD_RUN_ID_ENV,
            CHILD_RUNTIME_AUTH_ENV,
            "NPA_BYOF_ROBOTWIN_IMAGE_SCAN_ARCHIVES",
            "NPA_BYOF_ROBOTWIN_IMAGE_SCAN_SHA256",
            "NPA_BYOF_ROBOTWIN_RESERVATION_EVIDENCE_SHA256",
        }
        layer_valid = (
            not context.transport_value
            and TRANSPORT_CONTEXT_ENV not in secret_envs
            and TRANSPORT_CONTEXT_ENV not in extra_env
            and execution_target is None
            and execution_report is None
            and required_secrets.issubset(secret_envs)
            and CHILD_CONFIG_PATH_ENV not in secret_envs
            and CHILD_OUTPUT_ROOT_ENV not in secret_envs
            and all(extra_env.get(name) == value for name, value in context.private_environment)
        )
    else:
        layer_valid = False
    if not common_valid or not layer_valid:
        raise _refusal("submit-bridge-input-mismatch", authorization.raw_context)
    if context.layer == "outer":
        _validate_preverified_target(
            authorization, execution_target, execution_report
        )
    return authorization


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
