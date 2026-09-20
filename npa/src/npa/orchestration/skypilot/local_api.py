"""Task-owned SkyPilot API processes for isolated runtime directories.

SkyPilot 0.12.2 has one default localhost API port even with distinct HOME.
Its server module supports --port; the standard client endpoint setting then
uses the remote API protocol (including file uploads). Never stop a shared API.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Mapping
from urllib.error import URLError
from urllib.request import urlopen
import uuid

import yaml


_ENDPOINT = "SKYPILOT_API_SERVER_ENDPOINT"
_MARKER = "NPA_OWNED_SKYPILOT_API_ID"
_METADATA_TOKEN_ROOT = Path("/mnt/cloud-metadata")
_METADATA_CREDENTIAL_SOURCE = "instance_metadata"
_AGENT_RECOVERY_REBIND_ENV = "NPA_AGENT_ISOLATED_RECOVERY_REBIND"
# These resolved values are private runtime configuration, never credentials.
_RUNTIME_SETTINGS = {
    "storage_bucket": "NPA_S3_BUCKET",
    "storage_prefix": "NPA_S3_PREFIX",
    "aws_region": "AWS_REGION",
    "aws_default_region": "AWS_DEFAULT_REGION",
    "provider_profile": "NEBIUS_PROFILE",
    "aws_profile": "AWS_PROFILE",
}


class IsolatedApiError(ValueError):
    """Secret-safe failure; no fallback to a shared daemon is permitted."""


def _require_linux_host() -> None:
    """Reject hosts without the kernel evidence needed for safe API ownership."""

    if sys.platform != "linux" or not Path("/proc/self").is_dir():
        raise IsolatedApiError(
            "isolated SkyPilot execution requires a Linux operator host with /proc "
            "for process and socket ownership verification; run setup, submission, "
            "monitoring, recovery, and cleanup on that same Linux host. "
            "See docs/orchestration/skypilot-setup.md"
        )


def _isolated_api_root(isolated_dir: Path) -> Path:
    """Return one canonical owner directory for equivalent isolated paths.

    macOS exposes ``/tmp`` through ``/private/tmp``.  A controller created
    through one spelling must remain discoverable through the other; otherwise
    a safe owner record is incorrectly treated as foreign on a later status,
    cancellation, or submit operation.
    """

    return Path(isolated_dir).expanduser().resolve(strict=False) / "local-api"


@contextmanager
def _locked(root: Path):
    _require_linux_host()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    with open(
        root / "lock", "a", opener=lambda p, flags: os.open(p, flags, 0o600)
    ) as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _write(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    with open(
        temporary, "x", opener=lambda p, flags: os.open(p, flags, 0o600)
    ) as handle:
        json.dump(record, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read(root: Path) -> dict[str, Any] | None:
    path = root / "daemon.json"
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
        if record["schema_version"] != 1 or record["root"] != str(root.absolute()):
            raise ValueError
        if not isinstance(record["port"], int) or not 1024 <= record["port"] <= 65535:
            raise ValueError
        if not isinstance(record["marker"], str) or not record["marker"]:
            raise ValueError
        return record
    except (ValueError, KeyError, TypeError, OSError):
        raise IsolatedApiError(
            "isolated SkyPilot API ownership record is invalid; refusing shared fallback"
        ) from None


def _available_ports() -> tuple[int, int, int]:
    # Hold all reservations together so the kernel cannot return a port twice.
    sockets = []
    ports = []
    try:
        while len(ports) < 3:
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            sockets.append(listener)
            port = int(listener.getsockname()[1])
            if port not in {46580, 50011, 9090}:
                ports.append(port)
        return tuple(ports)
    finally:
        for listener in sockets:
            listener.close()


def _yaml_document(contents: str | bytes) -> Any:
    try:
        return yaml.safe_load(contents) or {}
    except yaml.YAMLError:
        # Parser diagnostics can include credential-bearing source lines.
        raise IsolatedApiError(
            "executing credential/configuration YAML is invalid"
        ) from None


class _UniqueMappingLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError("duplicate mapping key")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _strict_mapping(contents: bytes) -> dict | None:
    try:
        loader = _UniqueMappingLoader(contents)
        try:
            value = loader.get_single_data()
        finally:
            loader.dispose()
        return value if isinstance(value, dict) else None
    except (yaml.YAMLError, ValueError, TypeError):
        # No parser diagnostic or credential-bearing input may leave this boundary.
        return None


def _nebius_exec_environment(environment, spec):
    if not isinstance(spec, dict):
        return None
    env = dict(environment)
    if not spec:
        return env
    if Path(str(spec.get("command", ""))).name != "nebius":
        return None
    entries = spec.get("env") or []
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"name", "value"}:
            return None
        if not all(isinstance(value, str) for value in entry.values()):
            return None
    names = [entry["name"] for entry in entries]
    if len(set(names)) != len(names) or "HOME" in names:
        return None
    env.update({entry["name"]: entry["value"] for entry in entries})
    return env


def _unsupported_nebius_auth_argument(arg):
    auth_words = (
        "token",
        "auth",
        "endpoint",
        "service-account",
        "private-key",
        "public-key",
        "federat",
        "impersonat",
    )
    if arg == "--" or arg.startswith(("-I", "-p", "-c", "--profile", "--config")):
        return True
    return arg.startswith("-") and any(word in arg for word in auth_words)


def _nebius_exec_selectors(spec):
    args = spec.get("args") or []
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        return None
    selectors = {}
    index = 0
    while index < len(args):
        arg = args[index]
        flag, separator, value = arg.partition("=")
        if flag in {"--profile", "-p", "--config", "-c"}:
            name = "profile" if flag in {"--profile", "-p"} else "config"
            if not separator:
                index += 1
                value = args[index] if index < len(args) else ""
            if name in selectors or not value or value.startswith("-"):
                return None
            selectors[name] = value
        elif _unsupported_nebius_auth_argument(arg):
            return None
        index += 1
    return selectors


def _supported_service_account_profile(selected):
    fields = {
        "auth-type",
        "service-account-id",
        "public-key-id",
        "private-key-file-path",
        "endpoint",
        "parent-id",
        "tenant-id",
    }
    required = ("service-account-id", "public-key-id", "private-key-file-path")
    if not isinstance(selected, dict) or set(selected) - fields:
        return False
    if selected.get("auth-type") != "service account":
        return False
    if not all(isinstance(value, str) and value for value in selected.values()):
        return False
    if not all(selected.get(key) for key in required):
        return False
    for key in ("service-account-id", "public-key-id"):
        if not re.fullmatch(r"[^/\s]+", selected[key]):
            return False
    return Path(selected["private-key-file-path"]).is_absolute()


def _supported_metadata_token_profile(selected):
    """Accept only the mounted-token profile created for an attached identity."""
    minimal_fields = {"endpoint", "parent-id", "token-file"}
    extended_fields = minimal_fields | {"token-endpoint", "virtual"}
    if not isinstance(selected, dict) or (
        set(selected) != minimal_fields and set(selected) != extended_fields
    ):
        return False
    if not all(
        isinstance(selected.get(key), str) and selected[key] for key in minimal_fields
    ):
        return False
    if "token-endpoint" in selected and (
        not isinstance(selected["token-endpoint"], str)
        or not selected["token-endpoint"]
    ):
        return False
    if "virtual" in selected and type(selected["virtual"]) is not bool:
        return False
    try:
        token_file = Path(selected["token-file"])
        return (
            token_file.is_absolute()
            and token_file.is_file()
            and token_file.resolve().is_relative_to(_METADATA_TOKEN_ROOT.resolve())
        )
    except OSError:
        return False


def _nebius_profile_selection(config_path, profile, environment, profile_supported):
    if not config_path.is_absolute():
        return None
    try:
        contents = config_path.read_bytes()
    except FileNotFoundError:
        return None
    data = _strict_mapping(contents)
    if (
        not data
        or set(data) - {"default", "profiles"}
        or not isinstance(data.get("profiles"), dict)
    ):
        return None
    profile = profile or environment.get("NEBIUS_PROFILE") or data.get("default")
    if profile is None and len(data["profiles"]) == 1:
        profile = next(iter(data["profiles"]))
    if not isinstance(profile, str) or not profile:
        return None
    selected = data["profiles"].get(profile)
    if not profile_supported(selected):
        return None
    return str(config_path), hashlib.sha256(contents).hexdigest(), profile, selected


def _nebius_config_dir(environment, home):
    return Path(environment.get("NEBIUS_CONFIG_DIR") or home / ".nebius")


def _service_account_key_binding(selection, provider_dir):
    from cryptography.exceptions import UnsupportedAlgorithm
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

    config_name, config_hash, profile, selected = selection
    account = selected["service-account-id"]
    public_key = selected["public-key-id"]
    key_name = str(Path(selected["private-key-file-path"]))
    try:
        key_bytes = Path(key_name).read_bytes()
        if not isinstance(
            load_pem_private_key(key_bytes, password=None), RSAPrivateKey
        ):
            raise ValueError
    except (OSError, ValueError, TypeError, UnsupportedAlgorithm):
        raise IsolatedApiError(
            "selected Nebius service account private key cannot be verified"
        ) from None
    key_hash = hashlib.sha256(key_bytes).hexdigest()
    binding = json.dumps(
        [config_name, config_hash, profile, account, public_key, key_name, key_hash]
    )
    return {
        Path(config_name): config_hash,
        Path(key_name): key_hash,
        provider_dir / "credentials.yaml": "derived-nebius-sa-cache-v1:"
        + hashlib.sha256(binding.encode()).hexdigest(),
    }


def _selected_nebius_identity(
    environment: Mapping[str, str],
    home: Path,
    execs: list[dict],
    *,
    profile_supported,
    credential_source="",
):
    """Resolve one exact Nebius CLI profile used by every selected kube exec."""
    selections = []
    # Nebius CLI 0.12.254 keeps its token cache in HOME/.nebius, even where a
    # kube exec supplies NEBIUS_CONFIG_DIR. The supported config selector is
    # --config; do not let an ignored environment directory relocate either
    # the selected profile or the cache identity we protect.
    provider_dir = home / ".nebius"
    alternate_auth = (
        "NEBIUS_ENDPOINT",
        "NEBIUS_IAM_TOKEN",
        "NEBIUS_IAM_TOKEN_FILE",
        "NPA_NEBIUS_IAM_TOKEN",
        "NPA_NEBIUS_IAM_TOKEN_FILE",
    )
    for spec in execs or [{}]:
        env = _nebius_exec_environment(environment, spec)
        if env is None or any(env.get(key) for key in alternate_auth):
            return None
        if (
            credential_source
            and env.get("NPA_NEBIUS_CREDENTIAL_SOURCE") != credential_source
        ):
            return None
        selectors = _nebius_exec_selectors(spec)
        if selectors is None:
            return None
        config_path = Path(selectors.get("config", provider_dir / "config.yaml"))
        selected = _nebius_profile_selection(
            config_path,
            selectors.get("profile"),
            env,
            profile_supported,
        )
        if selected is None:
            return None
        selections.append(selected)
    if not selections or any(
        selection != selections[0] for selection in selections[1:]
    ):
        return None
    return provider_dir, selections[0]


def _nebius_service_account_identity(
    environment: Mapping[str, str], home: Path, execs: list[dict]
) -> dict[Path, str]:
    """Bind one supported CLI RSA profile and its durable key; never fetch tokens.

    CLI --config/--profile override exec environment and default selection.
    CLI 0.12.254 ignores NEBIUS_CONFIG_DIR for this selection and keeps the
    cache in HOME/.nebius. Relative paths, extra auth sources, and multiple
    effective selections remain byte-strict.
    """
    resolved = _selected_nebius_identity(
        environment,
        home,
        execs,
        profile_supported=_supported_service_account_profile,
    )
    if resolved is None:
        return {}
    provider_dir, selection = resolved
    return _service_account_key_binding(selection, provider_dir)


def _metadata_token_binding(selection, provider_dir):
    config_name, config_hash, profile, selected = selection
    token_file = Path(selected["token-file"]).resolve()
    binding = json.dumps(
        [
            config_name,
            config_hash,
            profile,
            selected["endpoint"],
            selected["parent-id"],
            selected.get("token-endpoint", ""),
            str(token_file),
            selected.get("virtual", False),
        ]
    )
    digest = hashlib.sha256(binding.encode()).hexdigest()
    return {
        Path(config_name): config_hash,
        token_file: "derived-nebius-metadata-source-v1:" + digest,
        provider_dir / "credentials.yaml": "derived-nebius-metadata-cache-v1:" + digest,
    }


def _nebius_metadata_token_identity(
    environment: Mapping[str, str], home: Path, execs: list[dict]
) -> dict[Path, str]:
    """Bind the mounted token source for an explicitly attached VM identity."""
    resolved = _selected_nebius_identity(
        environment,
        home,
        execs,
        profile_supported=_supported_metadata_token_profile,
        credential_source=_METADATA_CREDENTIAL_SOURCE,
    )
    if resolved is None:
        return {}
    provider_dir, selection = resolved
    return _metadata_token_binding(selection, provider_dir)


def _derived_service_account_cache(contents: bytes | None) -> bool:
    """Recognize derived SA tokens, including creation/pruning of an empty cache.

    This identifies durable auth configuration, not the subject of an arbitrary
    bearer. Provider/principal access preflight remains required. Unknown or
    mixed OAuth/token formats receive no exception to full-byte verification.
    """
    if contents is None:
        return True
    data = _strict_mapping(contents)
    if data is None or set(data) != {"tokens"} or not isinstance(data["tokens"], dict):
        return False
    for name, value in data["tokens"].items():
        if not isinstance(name, str) or not re.fullmatch(
            r"service-account/[^/\s]+/[^/\s]+", name
        ):
            return False
        if not isinstance(value, dict) or set(value) != {"token", "expires_at"}:
            return False
        if (
            not isinstance(value["token"], str)
            or not value["token"]
            or type(value["expires_at"]) is not int
            or value["expires_at"] < 0
        ):
            return False
    return True


def _configured_identity_paths(environment, home, kube_paths, config):
    paths = [*kube_paths, home / ".aws" / "config", home / ".aws" / "credentials"]
    filenames = (
        "config.yaml",
        "credentials.yaml",
        "credentials.json",
        "NEBIUS_IAM_TOKEN.txt",
        "NEBIUS_TENANT_ID.txt",
        "NEBIUS_DOMAIN.txt",
    )
    provider_dirs = (home / ".nebius", _nebius_config_dir(environment, home))
    designated_caches = {directory / "credentials.yaml" for directory in provider_dirs}
    npa_dir = Path(environment.get("NPA_CONFIG_DIR") or home / ".npa")
    protected = [npa_dir / name for name in filenames]
    for directory in (*provider_dirs, npa_dir):
        paths.extend(directory / name for name in filenames)
    for key in (
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "NPA_NEBIUS_IAM_TOKEN_FILE",
        "NEBIUS_IAM_TOKEN_FILE",
    ):
        if environment.get(key):
            paths.append(Path(environment[key]).expanduser())
            protected.append(paths[-1])
    workspace = (config.get("workspaces") or {}).get(
        config.get("active_workspace") or "default"
    ) or {}
    native = workspace.get("nebius") or {}
    if native.get("credentials_file_path"):
        paths.append(
            Path(str(native["credentials_file_path"]).replace("~", str(home), 1))
        )
        protected.append(paths[-1])
    return paths, protected, designated_caches


def _selected_kube_principals(kube, allowed, execs):
    requested = allowed or [kube.get("current-context")]
    selected_contexts = [
        item for item in kube.get("contexts", []) if item.get("name") in requested
    ]
    contexts = [item.get("context", {}) for item in selected_contexts]
    valid_selection = (
        contexts
        and all(isinstance(name, str) and name for name in requested)
        and {item.get("name") for item in selected_contexts} == set(requested)
        and len(selected_contexts) == len(set(requested))
    )
    if not valid_selection:
        execs.append({"command": "unsupported"})
    users = {item.get("user") for item in contexts}
    clusters = {item.get("cluster") for item in contexts}
    selected_users = [
        item for item in kube.get("users", []) if item.get("name") in users
    ]
    if (
        not users
        or None in users
        or {item.get("name") for item in selected_users} != users
        or len(selected_users) != len(users)
    ):
        execs.append({"command": "unsupported"})
    return users, clusters


def _kube_referenced_credentials(kube, kube_path, allowed):
    paths, execs = [], []
    users, clusters = _selected_kube_principals(kube, allowed, execs)
    references = (
        ("users", users, ("client-key", "client-certificate", "tokenFile")),
        ("clusters", clusters, ("certificate-authority",)),
    )
    for key, selected, settings in references:
        for item in kube.get(key, []):
            if item.get("name") not in selected:
                continue
            details = item.get("user" if key == "users" else "cluster") or {}
            if not isinstance(details, dict):
                execs.append({"command": "unsupported"})
                continue
            if key == "users":
                extra_auth = set(details) & {
                    "token",
                    "tokenFile",
                    "client-key",
                    "client-key-data",
                    "auth-provider",
                    "username",
                    "password",
                }
                execs.append(
                    details.get("exec")
                    if not extra_auth and details.get("exec")
                    else {"command": "unsupported"}
                )
            for setting in settings:
                if details.get(setting):
                    referenced = Path(details[setting]).expanduser()
                    paths.append(
                        referenced
                        if referenced.is_absolute()
                        else kube_path.parent / referenced
                    )
    return paths, execs


def _selected_kube_identity_paths(kube_paths, config):
    # Unrelated contexts in a merged kubeconfig must not select private keys.
    allowed = (config.get("kubernetes") or {}).get("allowed_contexts") or []
    paths, execs = [], []
    for kube_path in kube_paths:
        if not kube_path.is_file():
            continue
        kube = _yaml_document(kube_path.read_text())
        if not isinstance(kube, dict):
            continue
        referenced, selected_execs = _kube_referenced_credentials(
            kube, kube_path, allowed
        )
        paths.extend(referenced)
        execs.extend(selected_execs)
    return paths, execs


def _hash_identity_paths(paths, protected, designated_caches, durable, cache):
    result = {}
    cache_binding = durable.pop(cache, None)
    # Explicit bearer/NPA/AWS/key aliases never become derived token caches.
    protected.extend(path for path in paths if path not in designated_caches)
    protected.extend(durable)
    cache_allowed = cache_binding and all(
        path.resolve() != cache.resolve() for path in protected
    )
    for path in paths:
        contents = path.read_bytes() if path.is_file() else None
        value = (
            hashlib.sha256(contents).hexdigest() if contents is not None else "absent"
        )
        if (
            cache_allowed
            and path in designated_caches
            and path.resolve() == cache.resolve()
            and _derived_service_account_cache(contents)
        ):
            value = cache_binding
        result[str(path.absolute())] = value
    result.update({str(path.absolute()): value for path, value in durable.items()})
    return result


def _identity_files(
    environment: Mapping[str, str], *, config: Mapping[str, Any] | None = None
) -> dict[str, str]:
    home = Path(environment.get("HOME") or "").expanduser()
    kube_paths = [
        Path(item).expanduser()
        for item in environment.get("KUBECONFIG", "").split(os.pathsep)
        if item
    ]
    if config is None:
        config_path = Path(environment.get("SKYPILOT_GLOBAL_CONFIG") or "")
        config = (
            _yaml_document(config_path.read_text()) if config_path.is_file() else {}
        )
    config = config or {}
    paths, protected, designated_caches = _configured_identity_paths(
        environment, home, kube_paths, config
    )
    referenced, execs = _selected_kube_identity_paths(kube_paths, config)
    paths.extend(referenced)
    protected.extend(referenced)
    try:
        durable = _nebius_service_account_identity(environment, home, execs)
        if not durable:
            durable = _nebius_metadata_token_identity(environment, home, execs)
        cache = home / ".nebius" / "credentials.yaml"
        return _hash_identity_paths(paths, protected, designated_caches, durable, cache)
    except OSError:
        raise IsolatedApiError(
            "executing credential/configuration file identity cannot be inspected"
        ) from None


def _agent_profile_rebind_allowed(
    record: Mapping[str, Any],
    *,
    binding: Mapping[str, str],
    files: Mapping[str, str],
    environment: Mapping[str, str],
) -> bool:
    """Allow a stopped Agent daemon to refresh only its staged config/cache.

    An Agent bootstrap rewrites its own NPA profile and receives a fresh
    instance-metadata token. These files can change while the selected project,
    Kubernetes context, provider configuration, and storage identity are
    unchanged. This opt-in migration is deliberately unavailable to regular
    callers and never permits a live daemon or any other identity file to be
    rebound.
    """
    if environment.get(_AGENT_RECOVERY_REBIND_ENV) != "v1":
        return False
    recorded_binding = record.get("environment_binding")
    recorded_files = record.get("identity_files")
    if not isinstance(recorded_binding, dict) or not isinstance(recorded_files, dict):
        return False
    if dict(recorded_binding) != dict(binding):
        return False
    project = str(environment.get("NPA_SKYPILOT_PROJECT") or "")
    if not project or record.get("project_alias") != project:
        return False
    configured = str(environment.get("NPA_CONFIG_DIR") or "")
    home = str(environment.get("HOME") or "")
    if not configured or not home:
        return False
    if environment.get("NPA_NEBIUS_CREDENTIAL_SOURCE") != _METADATA_CREDENTIAL_SOURCE:
        return False
    try:
        config_root = Path(configured).expanduser().resolve(strict=False)
        metadata_cache = (
            Path(home).expanduser().absolute() / ".nebius" / "credentials.yaml"
        )
        allowed = {
            str(config_root / "config.yaml"),
            str(config_root / "credentials.yaml"),
            str(metadata_cache),
        }
        changed = {
            path
            for path in set(recorded_files) | set(files)
            if recorded_files.get(path) != files.get(path)
        }
        return bool(changed) and all(
            path in allowed
            or Path(path)
            .expanduser()
            .resolve(strict=False)
            .is_relative_to(_METADATA_TOKEN_ROOT.resolve())
            for path in changed
        )
    except (OSError, ValueError):
        return False


def _session_members(record: Mapping[str, Any]) -> list[int]:
    if not record.get("pid"):
        return []
    if sys.platform == "darwin":
        # macOS does not expose Linux's procfs tree. The owned API server is
        # started in its own session. Its queue child can briefly outlive the
        # leader during graceful shutdown, so retain only listeners proven to
        # belong to the recorded process group.
        process = _process(record)
        group = int(record["pid"])
        members = set(_darwin_owned_listener_pids(record, process_group=group))
        if process:
            members.add(int(process["pid"]))
        return sorted(members)
    snapshots: dict[int, tuple[int, int, bool]] = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = directory.joinpath("stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[3]) != record["pid"] or fields[0] == "Z":
                continue
            if directory.stat().st_uid != os.getuid():
                raise IsolatedApiError(
                    "isolated SkyPilot API session changed process owner"
                )
            try:
                environment = dict(
                    item.split(b"=", 1)
                    for item in directory.joinpath("environ").read_bytes().split(b"\0")
                    if b"=" in item
                )
            except PermissionError:
                # Exiting/setproctitle workers may hide environ. Their exact
                # saved lifetime or verified live lineage is still required.
                environment = {}
            snapshots[int(directory.name)] = (
                int(fields[1]),
                int(fields[19]),
                environment.get(_MARKER.encode(), b"").decode() == record["marker"],
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
    if not snapshots:
        return []
    saved = record.get("session_processes", {})
    trusted = {
        pid
        for pid, (_, ticks, marked) in snapshots.items()
        if marked or saved.get(str(pid)) == ticks
    }
    root_pid = int(record["pid"])
    if root_pid in snapshots:
        if snapshots[root_pid][1] != record.get("start_ticks"):
            raise IsolatedApiError(
                "isolated SkyPilot API session leader lifetime changed"
            )
        # A known lifetime remains ours while the kernel clears argv/env on exit.
        trusted.add(root_pid)
    # Sky 0.12.2 setproctitle executor workers erase their /proc environment.
    # Corroborate them through a live parent chain in this exact owned session;
    # persist PID/starttime proof before signaling, so orphaned workers remain
    # identifiable after their leader exits. A reused PID cannot satisfy it.
    while True:
        descendants = {
            pid for pid, (parent, _, _) in snapshots.items() if parent in trusted
        }
        added = descendants - trusted
        if not added:
            break
        trusted.update(added)
    if trusted != set(snapshots):
        raise IsolatedApiError(
            "isolated SkyPilot API session contains a process with uncertain ownership"
        )
    if isinstance(record, dict):
        record["session_processes"] = {
            str(pid): value[1] for pid, value in snapshots.items()
        }
    return sorted(snapshots)


def _endpoint(record: Mapping[str, Any]) -> str:
    return f"http://127.0.0.1:{record['port']}"


def _process(
    record: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any] | None:
    """Match the private intent marker even across a Popen/PID-save crash."""
    if sys.platform == "darwin":
        return _darwin_process(record)
    matches = []
    candidates = (
        [Path("/proc") / str(record["pid"])]
        if record.get("pid")
        else Path("/proc").iterdir()
    )
    for directory in candidates:
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            command = directory.joinpath("cmdline").read_bytes().split(b"\0")
            if b"sky.server.server" not in command or b"-m" not in command:
                if record.get("pid"):
                    fields = (
                        directory.joinpath("stat").read_text().rsplit(")", 1)[1].split()
                    )
                    if fields[0] != "Z":
                        raise IsolatedApiError(
                            "isolated SkyPilot API process lifetime disagrees with its ownership record"
                        )
                continue
            environment = dict(
                item.split(b"=", 1)
                for item in directory.joinpath("environ").read_bytes().split(b"\0")
                if b"=" in item
            )
            if environment.get(_MARKER.encode(), b"").decode() != record["marker"]:
                if record.get("pid"):
                    raise IsolatedApiError(
                        "isolated SkyPilot API process lifetime disagrees with its ownership record"
                    )
                continue
            # The marker alone is insufficient: bind the exact interpreter,
            # home, user, selected credentials/config and process lifetime.
            if str(Path(command[0].decode()).absolute()) != record.get("interpreter"):
                raise IsolatedApiError(
                    "isolated SkyPilot API process executable disagrees with its ownership record"
                )
            for key, wanted in record.get("environment_binding", {}).items():
                actual = hashlib.sha256(environment.get(key.encode(), b"")).hexdigest()
                if actual != wanted:
                    raise IsolatedApiError(
                        "isolated SkyPilot API process environment disagrees with its ownership record"
                    )
            stat_fields = (
                directory.joinpath("stat").read_text().rsplit(")", 1)[1].split()
            )
            if int(stat_fields[3]) != int(directory.name):
                # Forked uvicorn/executor children can retain the same argv and
                # environment; the independently launched root is session leader.
                continue
            decoded = {
                key.decode(): value.decode() for key, value in environment.items()
            }
            if verify_files and record.get("config_sha256"):
                config_path = Path(decoded.get("SKYPILOT_GLOBAL_CONFIG") or "")
                if (
                    not config_path.is_file()
                    or hashlib.sha256(config_path.read_bytes()).hexdigest()
                    != record["config_sha256"]
                ):
                    raise IsolatedApiError(
                        "isolated SkyPilot API verified configuration changed on disk"
                    )
            if (
                verify_files
                and record.get("identity_files")
                and _identity_files(decoded) != record["identity_files"]
            ):
                raise IsolatedApiError(
                    "isolated SkyPilot API credential configuration changed after verification"
                )
            matches.append(
                {"pid": int(directory.name), "start_ticks": int(stat_fields[19])}
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            raise IsolatedApiError(
                "isolated SkyPilot API process ownership cannot be inspected"
            ) from None
    if len(matches) > 1:
        raise IsolatedApiError(
            "isolated SkyPilot API ownership is ambiguous; refusing duplicate startup"
        )
    if (
        matches
        and record.get("pid")
        and (record["pid"], record.get("start_ticks"))
        != (matches[0]["pid"], matches[0]["start_ticks"])
    ):
        raise IsolatedApiError(
            "isolated SkyPilot API process lifetime disagrees with its ownership record"
        )
    return matches[0] if matches else None


def _darwin_process_fingerprint(line: str, record: Mapping[str, Any]) -> str:
    """Keep CPython's framework launcher transition within one process lifetime."""
    original = hashlib.sha256(line.encode()).hexdigest()
    interpreter = record.get("interpreter")
    if interpreter:
        fields = line.split(None, 6)
        command = fields[-1] if len(fields) == 7 else ""
        executable, separator, arguments = command.partition(" -m sky.server.server")
        resolved = Path(interpreter).resolve()
        equivalents = {str(interpreter), str(resolved)}
        framework = resolved.parent.parent
        if (
            resolved.parent.name == "bin"
            and framework.parent.parent.name == "Python.framework"
        ):
            application = framework / "Resources/Python.app/Contents/MacOS/Python"
            if application.is_file():
                equivalents.add(str(application))
        if not separator or executable not in equivalents:
            raise IsolatedApiError(
                "isolated SkyPilot API process executable disagrees with its ownership record"
            )
        # CPython replaces argv[0] when its macOS launcher execs the framework
        # application. Preserve the PID, start time and all server arguments.
        line = (
            line[: len(line) - len(command)] + str(interpreter) + separator + arguments
        )
    fingerprint = hashlib.sha256(line.encode()).hexdigest()
    previous = str(record.get("darwin_process_fingerprint") or "")
    # Existing receipts may contain the unnormalized framework command hash.
    if previous and previous not in {original, fingerprint}:
        raise IsolatedApiError(
            "isolated SkyPilot API process lifetime disagrees with its ownership record"
        )
    return fingerprint


def _darwin_process(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate the task-owned server on macOS, where ``/proc`` is unavailable.

    Linux can inspect the private environment marker and descendant tree through
    procfs. On macOS we require an exact saved PID, a stable process
    start/command fingerprint, and a loopback-listener check before use. A
    missing or changed process is never adopted.
    """

    raw_pid = record.get("pid")
    if raw_pid in (None, ""):
        return None
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        raise IsolatedApiError(
            "isolated SkyPilot API ownership record has an invalid process ID"
        ) from None
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid=,lstart=,command="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise IsolatedApiError(
            "isolated SkyPilot API process ownership cannot be inspected on macOS"
        ) from exc
    line = (result.stdout or "").strip()
    if result.returncode != 0 or not line:
        return None
    expected_port = str(record.get("port") or "")
    if not expected_port:
        raise IsolatedApiError(
            "isolated SkyPilot API process lifetime disagrees with its ownership record"
        )
    # A former server's PID can be recycled after a graceful shutdown. It is
    # not an owned process merely because the integer matches our receipt; do
    # not block recovery (or later signal that foreign process) when it is no
    # longer a Sky server on the recorded port.
    if "sky.server.server" not in line or f"--port {expected_port}" not in line:
        return None
    fingerprint = _darwin_process_fingerprint(line, record)
    if isinstance(record, dict):
        record["darwin_process_fingerprint"] = fingerprint
    return {"pid": pid, "start_ticks": fingerprint}


def _darwin_owned_listener_pids(
    record: Mapping[str, Any],
    *,
    process_group: int,
    ports: tuple[int, ...] | None = None,
) -> list[int]:
    """Return only listener PIDs corroborated in one owned process group."""

    selected_ports = ports or tuple(
        int(record[key])
        for key in ("port", "queue_port", "metrics_port")
        if record.get(key) not in (None, "")
    )
    owned: set[int] = set()
    for selected_port in selected_ports:
        try:
            listeners = subprocess.run(
                ["lsof", "-nP", "-t", f"-iTCP:{selected_port}", "-sTCP:LISTEN"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise IsolatedApiError(
                "isolated SkyPilot API listener ownership cannot be inspected on macOS"
            ) from exc
        if listeners.returncode != 0:
            continue
        for raw_pid in (listeners.stdout or "").splitlines():
            try:
                listener_pid = int(raw_pid.strip())
            except ValueError:
                continue
            try:
                process = subprocess.run(
                    ["ps", "-p", str(listener_pid), "-o", "pgid="],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if (
                    process.returncode == 0
                    and int((process.stdout or "").strip()) == process_group
                ):
                    owned.add(listener_pid)
            except (OSError, ValueError):
                continue
    return sorted(owned)


def _listener_owned(
    record: Mapping[str, Any], process: Mapping[str, Any], *, port: int | None = None
) -> bool:
    """Corroborate actual loopback LISTEN inode held by the daemon process tree."""
    root_pid = int(process["pid"])
    selected_port = port or record["port"]
    if sys.platform == "darwin":
        return bool(
            _darwin_owned_listener_pids(
                record, process_group=root_pid, ports=(int(selected_port),)
            )
        )
    try:
        namespace = Path(f"/proc/{root_pid}/ns/net").readlink()
    except FileNotFoundError:
        return False
    if namespace != Path("/proc/self/ns/net").readlink():
        raise IsolatedApiError(
            "isolated SkyPilot API belongs to another network namespace"
        )
    inodes = set()
    for row in Path(f"/proc/{root_pid}/net/tcp").read_text().splitlines()[1:]:
        fields = row.split()
        if fields[1] == f"0100007F:{selected_port:04X}" and fields[3] == "0A":
            inodes.add(fields[9])
    if not inodes:
        return False
    # Workers inherit the root's session ID because Popen starts a new session.
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            stat_fields = (
                directory.joinpath("stat").read_text().rsplit(")", 1)[1].split()
            )
            if int(stat_fields[3]) != root_pid:
                continue
            for fd in directory.joinpath("fd").iterdir():
                link = str(fd.readlink())
                if link.startswith("socket:[") and link[8:-1] in inodes:
                    return True
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    raise IsolatedApiError("isolated SkyPilot API port is held by an unowned process")


def isolated_api_environment(
    isolated_dir: Path, environment: Mapping[str, str]
) -> dict[str, str]:
    """Persist endpoint intent before any client could connect to a shared API."""
    root = _isolated_api_root(isolated_dir)
    with _locked(root):
        record = _read(root)
        explicit = str(environment.get(_ENDPOINT) or "")
        if explicit and (record is None or explicit != _endpoint(record)):
            raise IsolatedApiError(
                "an isolated SkyPilot runtime cannot use a different configured API endpoint"
            )
        if record is None:
            http_port, metrics_port, queue_port = _available_ports()
            record = {
                "schema_version": 1,
                "root": str(root),
                "port": http_port,
                "metrics_port": metrics_port,
                "queue_port": queue_port,
                "marker": uuid.uuid4().hex,
                "state": "intent",
            }
            _write(root / "daemon.json", record)
        process = _process(record) if record.get("interpreter") else None
        if process:
            _listener_owned(record, process)
        selected = {
            **environment,
            _ENDPOINT: _endpoint(record),
            "NPA_SKYPILOT_ISOLATED_API_DIR": str(root.parent),
        }
        for setting, name in _RUNTIME_SETTINGS.items():
            value = record.get("runtime_settings", {}).get(setting)
            if value and not selected.get(name):
                selected[name] = value
        recover = bool(record.get("interpreter") and not process)
    if recover:
        # A fresh status/reconcile/cancel client must reconnect to the same
        # persistent API database, never SkyPilot's shared fallback endpoint.
        recovery_env = dict(selected)
        recovery_env["SKYPILOT_GLOBAL_CONFIG"] = str(root / "server-config.yaml")
        alias = record.get("project_alias")
        if alias:
            from npa.orchestration.npa_workflow.submit_credentials import (
                STORAGE_ENDPOINT_ENV_NAMES,
                resolve_submit_credentials,
            )

            expected_config = record["environment_binding"].get("NPA_CONFIG_DIR")
            if (
                expected_config
                and hashlib.sha256(
                    recovery_env.get("NPA_CONFIG_DIR", "").encode()
                ).hexdigest()
                != expected_config
            ):
                raise IsolatedApiError(
                    "isolated SkyPilot API recovery requires the original selected NPA configuration"
                )
            credentials = resolve_submit_credentials(
                project=alias, environ=recovery_env
            )
            values = {
                "AWS_ACCESS_KEY_ID": credentials.access_key_id,
                "AWS_SECRET_ACCESS_KEY": credentials.secret_access_key,
                **dict.fromkeys(STORAGE_ENDPOINT_ENV_NAMES, credentials.endpoint_url),
            }
            for key, value in values.items():
                if not recovery_env.get(key) and value:
                    recovery_env[key] = value
            recovery_env["NPA_SKYPILOT_PROJECT"] = alias
        ensure_isolated_api(
            isolated_dir=isolated_dir,
            sky_executable=str(Path(record["interpreter"]).parent / "sky"),
            environment=recovery_env,
            cwd=str(root.parent),
        )
    return selected


def ensure_isolated_api(
    *,
    isolated_dir: Path,
    sky_executable: str,
    environment: Mapping[str, str],
    cwd: str,
) -> dict[str, Any]:
    """Start/adopt only this scope's exact daemon; never create a cloud job."""
    root = _isolated_api_root(isolated_dir)
    with _locked(root):
        record = _read(root)
        if record is None or environment.get(_ENDPOINT) != _endpoint(record):
            raise IsolatedApiError(
                "isolated SkyPilot API endpoint intent is missing or inconsistent"
            )
        interpreter = str(Path(sky_executable).absolute().parent / "python")
        # A stopped receipt deliberately retains its executable so lifecycle
        # commands can recover the same persistent controller endpoint.  It
        # must not, however, become an approval to replace that controller
        # with an arbitrary SkyPilot installation.
        if (
            record.get("state") == "stopped"
            and record.get("interpreter") != interpreter
        ):
            raise IsolatedApiError(
                "isolated SkyPilot API recovery requires the original recorded interpreter"
            )
        config_source = Path(environment.get("SKYPILOT_GLOBAL_CONFIG") or "")
        if not config_source.is_file():
            raise IsolatedApiError(
                "isolated SkyPilot API requires the verified runtime configuration"
            )
        if environment.get("SKYPILOT_DB_CONNECTION_URI"):
            raise IsolatedApiError(
                "isolated SkyPilot API cannot use a shared external database"
            )
        config_bytes = config_source.read_bytes()
        parsed_config = _yaml_document(config_bytes)
        if parsed_config.get("db"):
            raise IsolatedApiError(
                "isolated SkyPilot API cannot use a shared external database"
            )
        configured_endpoint = (parsed_config.get("api_server") or {}).get("endpoint")
        if configured_endpoint and configured_endpoint != _endpoint(record):
            raise IsolatedApiError(
                "isolated SkyPilot API configuration names a different endpoint"
            )
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        daemon_env = dict(environment)
        config_path = root / "server-config.yaml"
        daemon_env.update(
            {
                _MARKER: record["marker"],
                "IS_SKYPILOT_SERVER": "true",
                "SKYPILOT_GLOBAL_CONFIG": str(config_path),
            }
        )
        plugins_path = root / "server-plugins.yaml"
        inherited_plugins = daemon_env.get("SKYPILOT_SERVER_PLUGINS_CONFIG")
        if inherited_plugins and Path(inherited_plugins) != plugins_path:
            raise IsolatedApiError(
                "isolated SkyPilot API cannot inherit unverified server plugins"
            )
        daemon_env["SKYPILOT_SERVER_PLUGINS_CONFIG"] = str(plugins_path)
        # Retain only hashes of settings that determine executing identity.
        # Artifact destination varies for each workflow, while this daemon is
        # the shared control plane for all of them. Binding it to a run prefix
        # would reject a valid later workflow before it can submit.
        identity_keys = (
            "HOME",
            "SKYPILOT_USER_ID",
            "KUBECONFIG",
            "NEBIUS_CONFIG_DIR",
            "NEBIUS_PROFILE",
            "NPA_NEBIUS_CREDENTIAL_SOURCE",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_ENDPOINT_URL",
            "AWS_ENDPOINT_URL_S3",
            "AWS_REGION",
            "AWS_DEFAULT_REGION",
            "AWS_PROFILE",
            "AWS_CONFIG_FILE",
            "AWS_SHARED_CREDENTIALS_FILE",
            "S3_ENDPOINT_URL",
            "NEBIUS_S3_ENDPOINT",
            "NPA_STORAGE_ENDPOINT",
            "NPA_CONFIG_DIR",
            "NPA_SKYPILOT_PROJECT",
            "NEBIUS_IAM_TOKEN",
            "NEBIUS_IAM_TOKEN_FILE",
            "NPA_NEBIUS_IAM_TOKEN",
            "NPA_NEBIUS_IAM_TOKEN_FILE",
            "SKYPILOT_GLOBAL_CONFIG",
            "SKYPILOT_SERVER_PLUGINS_CONFIG",
            "PYTHONPATH",
            _ENDPOINT,
            _MARKER,
        )
        binding = {
            key: hashlib.sha256(daemon_env.get(key, "").encode()).hexdigest()
            for key in identity_keys
        }
        files = _identity_files(daemon_env, config=parsed_config)
        spawned = None
        process = _process(record) if record.get("interpreter") else None
        if process:
            if (
                record.get("interpreter") != interpreter
                or record.get("config_sha256") != config_hash
            ):
                raise IsolatedApiError(
                    "running isolated SkyPilot API has a different verified configuration; preserve its jobs before restarting"
                )
            if (
                record["environment_binding"] != binding
                or record.get("identity_files") != files
            ):
                raise IsolatedApiError(
                    "running isolated SkyPilot API has a different executing "
                    "identity or its credential configuration changed"
                )
            runtime_settings = {
                setting: daemon_env[name]
                for setting, name in _RUNTIME_SETTINGS.items()
                if daemon_env.get(name)
            }
            if record.get("runtime_settings") != runtime_settings:
                record["runtime_settings"] = runtime_settings
                _write(root / "daemon.json", record)
        else:
            # No process with this marker exists; starting the same persistent
            # API database recovers controller/job identity, never submits again.
            if _session_members(record):
                raise IsolatedApiError(
                    "owned SkyPilot API children survived their leader; finish its process-session cleanup before recovery"
                )
            if record.get("environment_binding") and (
                record["environment_binding"] != binding
                or record.get("identity_files") != files
            ):
                if not _agent_profile_rebind_allowed(
                    record,
                    binding=binding,
                    files=files,
                    environment=daemon_env,
                ):
                    raise IsolatedApiError(
                        "isolated SkyPilot API recovery requires the original executing identity and credential configuration"
                    )
                record.update(environment_binding=binding, identity_files=files)
                _write(root / "daemon.json", record)
            for port in (record["port"], record["queue_port"], record["metrics_port"]):
                with socket.socket() as listener:
                    try:
                        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        listener.bind(("127.0.0.1", port))
                    except OSError:
                        raise IsolatedApiError(
                            "isolated SkyPilot API endpoint is occupied by an unowned listener"
                        ) from None
            with open(
                plugins_path, "w", opener=lambda p, flags: os.open(p, flags, 0o600)
            ) as handle:
                yaml.safe_dump(
                    {
                        "plugins": [
                            {
                                "class": "npa.orchestration.skypilot.local_api_plugin.IsolatedQueuePlugin",
                                "parameters": {"port": record["queue_port"]},
                            }
                        ]
                    },
                    handle,
                )
            with open(
                config_path, "wb", opener=lambda p, flags: os.open(p, flags, 0o600)
            ) as handle:
                handle.write(config_bytes)
            record.update(
                interpreter=interpreter,
                environment_binding=binding,
                config_sha256=config_hash,
                identity_files=files,
                project_alias=daemon_env.get("NPA_SKYPILOT_PROJECT", ""),
                runtime_settings={
                    setting: daemon_env[name]
                    for setting, name in _RUNTIME_SETTINGS.items()
                    if daemon_env.get(name)
                },
                pid=None,
                start_ticks=None,
                state="starting",
            )
            _write(root / "daemon.json", record)
            with open(
                root / "server.log",
                "ab",
                opener=lambda p, flags: os.open(p, flags, 0o600),
            ) as log:
                spawned = subprocess.Popen(
                    [
                        interpreter,
                        "-m",
                        "sky.server.server",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(record["port"]),
                        "--metrics-port",
                        str(record["metrics_port"]),
                    ],
                    env=daemon_env,
                    cwd=cwd,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            if sys.platform == "darwin":
                # Linux discovers the marker through /proc. macOS has no
                # equivalent, so persist only the PID and immediately
                # revalidate its start/command fingerprint before use.
                record.update(pid=spawned.pid, start_ticks=None)
                _write(root / "daemon.json", record)
        while True:
            process = _process(record)
            if process is None:
                # A fresh /proc scan can miss the child during startup. Its
                # absence is not exit evidence while our Popen handle is alive.
                if spawned is not None and spawned.poll() is None:
                    time.sleep(0.2)
                    continue
                raise IsolatedApiError(
                    "owned SkyPilot API exited before readiness; inspect its private server log"
                )
            record.update(process)
            _write(root / "daemon.json", record)
            queue_ready = _listener_owned(record, process, port=record["queue_port"])
            metrics_ready = not daemon_env.get(
                "SKY_API_SERVER_METRICS_ENABLED"
            ) or _listener_owned(record, process, port=record["metrics_port"])
            if _listener_owned(record, process) and queue_ready and metrics_ready:
                try:
                    with urlopen(f"{_endpoint(record)}/api/health") as response:
                        health = json.load(response)
                    if (
                        health.get("version") != "0.12.2"
                        or str(health.get("status") or "").lower() != "healthy"
                    ):
                        raise IsolatedApiError(
                            "owned SkyPilot API readiness/version evidence is inconsistent"
                        )
                    record["state"] = "ready"
                    _write(root / "daemon.json", record)
                    return {
                        "healthy": True,
                        "outcome": "owned_isolated_api",
                        "process_count": 1,
                    }
                except (URLError, ConnectionError, json.JSONDecodeError):
                    pass
            time.sleep(0.2)


def owned_daemon_environment(isolated_dir: Path) -> dict[str, str]:
    """Read an owned daemon's exact environment in memory for a cleanup transaction.

    Credentials must never be written to a transaction record or log. PID,
    lifetime, marker, executable, config and identity-file checks precede and
    follow the read; no foreign or ambiguous process may supply credentials.
    """
    root = _isolated_api_root(isolated_dir)
    with _locked(root):
        record = _read(root)
        if not record or not record.get("interpreter"):
            raise IsolatedApiError(
                "controller transaction requires an owned source API"
            )
        process = _process(record)
        if not process or not _listener_owned(record, process):
            raise IsolatedApiError(
                "controller transaction source API is not verified ready"
            )
        try:
            raw = (Path("/proc") / str(process["pid"]) / "environ").read_bytes()
        except OSError:
            raise IsolatedApiError(
                "controller transaction source identity is unreadable"
            ) from None
        environment = dict(
            entry.decode().split("=", 1) for entry in raw.split(b"\0") if b"=" in entry
        )
        if _process(record) != process:
            raise IsolatedApiError(
                "controller transaction source API changed during inspection"
            )
        return environment


def stop_isolated_api(isolated_dir: Path) -> None:
    """Stop only the owned local process group, after callers finish cloud jobs."""
    root = _isolated_api_root(isolated_dir)
    with _locked(root):
        record = _read(root)
        if record is None or not record.get("interpreter"):
            return
        process = _process(record, verify_files=False)
        if process:
            # Recover a Popen-success/PID-save crash before any signal is sent.
            record.update(process)
        members = _session_members(record)
        if process or members:
            _write(root / "daemon.json", record)
            if process:
                # Sky's graceful shutdown joins executor threads before killing
                # their queue manager. Signaling the whole session first kills
                # that queue prematurely and deadlocks the parent's shutdown.
                try:
                    os.kill(record["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
                while record["pid"] in _session_members(record):
                    time.sleep(0.2)
                _write(root / "daemon.json", record)
            # Sky executors can retain signal handlers after their parent exits.
            # The parent's graceful shutdown has finished; force-remove only
            # the corroborated orphan session, as Sky's own API stop does for
            # its local executor tree. No live leader/request is interrupted.
            remaining = _session_members(record)
            if remaining:
                try:
                    os.killpg(record["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
                while _session_members(record):
                    time.sleep(0.2)
        # A stopped controller has no authority to pin the next controller's
        # credentials or generated configuration. Retain the endpoint, known
        # interpreter, and non-secret runtime settings so a status/reconcile
        # client can restart the same persistent local API without submitting
        # work; clear every identity/configuration binding before that start.
        for key in (
            "environment_binding",
            "config_sha256",
            "identity_files",
            "project_alias",
            "darwin_process_fingerprint",
            "session_processes",
        ):
            record.pop(key, None)
        record.update(state="stopped", pid=None, start_ticks=None)
        _write(root / "daemon.json", record)
