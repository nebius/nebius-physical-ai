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
import time
from typing import Any, Mapping
from urllib.error import URLError
from urllib.request import urlopen
import uuid

import yaml


_ENDPOINT = "SKYPILOT_API_SERVER_ENDPOINT"
_MARKER = "NPA_OWNED_SKYPILOT_API_ID"
# These resolved values are private runtime configuration, never credentials.
_RUNTIME_SETTINGS = {"storage_bucket": "NPA_S3_BUCKET", "storage_prefix": "NPA_S3_PREFIX",
                     "aws_region": "AWS_REGION", "aws_default_region": "AWS_DEFAULT_REGION",
                     "provider_profile": "NEBIUS_PROFILE", "aws_profile": "AWS_PROFILE"}


class IsolatedApiError(ValueError):
    """Secret-safe failure; no fallback to a shared daemon is permitted."""


@contextmanager
def _locked(root: Path):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    with open(root / "lock", "a", opener=lambda p, flags: os.open(p, flags, 0o600)) as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _write(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    with open(temporary, "x", opener=lambda p, flags: os.open(p, flags, 0o600)) as handle:
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
        raise IsolatedApiError("isolated SkyPilot API ownership record is invalid; refusing shared fallback") from None


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
        raise IsolatedApiError("executing credential/configuration YAML is invalid") from None


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
    auth_words = ("token", "auth", "endpoint", "service-account", "private-key",
                  "public-key", "federat", "impersonat")
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
    fields = {"auth-type", "service-account-id", "public-key-id", "private-key-file-path",
              "endpoint", "parent-id", "tenant-id"}
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


def _nebius_profile_selection(config_path, profile, environment):
    if not config_path.is_absolute():
        return None
    try:
        contents = config_path.read_bytes()
    except FileNotFoundError:
        return None
    data = _strict_mapping(contents)
    if not data or set(data) - {"default", "profiles"} or not isinstance(data.get("profiles"), dict):
        return None
    profile = profile or environment.get("NEBIUS_PROFILE") or data.get("default")
    if profile is None and len(data["profiles"]) == 1:
        profile = next(iter(data["profiles"]))
    if not isinstance(profile, str) or not profile:
        return None
    selected = data["profiles"].get(profile)
    if not _supported_service_account_profile(selected):
        return None
    return (str(config_path), hashlib.sha256(contents).hexdigest(), profile,
            selected["service-account-id"], selected["public-key-id"],
            str(Path(selected["private-key-file-path"])))


def _nebius_config_dir(environment, home):
    return Path(environment.get("NEBIUS_CONFIG_DIR") or home / ".nebius")


def _service_account_key_binding(selection, provider_dir):
    from cryptography.exceptions import UnsupportedAlgorithm
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

    config_name, config_hash, profile, account, public_key, key_name = selection
    try:
        key_bytes = Path(key_name).read_bytes()
        if not isinstance(load_pem_private_key(key_bytes, password=None), RSAPrivateKey):
            raise ValueError
    except (OSError, ValueError, TypeError, UnsupportedAlgorithm):
        raise IsolatedApiError("selected Nebius service account private key cannot be verified") from None
    key_hash = hashlib.sha256(key_bytes).hexdigest()
    binding = json.dumps([config_name, config_hash, profile, account, public_key, key_name, key_hash])
    return {Path(config_name): config_hash, Path(key_name): key_hash,
            provider_dir / "credentials.yaml": "derived-nebius-sa-cache-v1:" + hashlib.sha256(binding.encode()).hexdigest()}


def _nebius_service_account_identity(environment: Mapping[str, str], home: Path, execs: list[dict]) -> dict[Path, str]:
    """Bind one supported CLI RSA profile and its durable key; never fetch tokens.

    CLI --config/--profile override exec environment and default selection.
    NEBIUS_CONFIG_DIR selects the default profile directory and token cache.
    Relative paths, extra auth sources, and multiple effective selections remain
    byte-strict, including exec overrides selecting a different cache directory.
    """
    selections = set()
    provider_dir = _nebius_config_dir(environment, home)
    alternate_auth = ("NEBIUS_ENDPOINT", "NEBIUS_IAM_TOKEN", "NEBIUS_IAM_TOKEN_FILE",
                      "NPA_NEBIUS_IAM_TOKEN", "NPA_NEBIUS_IAM_TOKEN_FILE")
    for spec in execs or [{}]:
        env = _nebius_exec_environment(environment, spec)
        if env is None or any(env.get(key) for key in alternate_auth):
            return {}
        if _nebius_config_dir(env, home) != provider_dir:
            return {}
        selectors = _nebius_exec_selectors(spec)
        if selectors is None:
            return {}
        config_path = Path(selectors.get("config", provider_dir / "config.yaml"))
        selected = _nebius_profile_selection(config_path, selectors.get("profile"), env)
        if selected is None:
            return {}
        selections.add(selected)
    if len(selections) != 1:
        return {}
    return _service_account_key_binding(selections.pop(), provider_dir)


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
        if not isinstance(name, str) or not re.fullmatch(r"service-account/[^/\s]+/[^/\s]+", name):
            return False
        if not isinstance(value, dict) or set(value) != {"token", "expires_at"}:
            return False
        if not isinstance(value["token"], str) or not value["token"] or type(value["expires_at"]) is not int or value["expires_at"] < 0:
            return False
    return True


def _configured_identity_paths(environment, home, kube_paths, config):
    paths = [*kube_paths, home / ".aws" / "config", home / ".aws" / "credentials"]
    filenames = ("config.yaml", "credentials.yaml", "credentials.json", "NEBIUS_IAM_TOKEN.txt", "NEBIUS_TENANT_ID.txt", "NEBIUS_DOMAIN.txt")
    provider_dirs = (home / ".nebius", _nebius_config_dir(environment, home))
    designated_caches = {directory / "credentials.yaml" for directory in provider_dirs}
    npa_dir = Path(environment.get("NPA_CONFIG_DIR") or home / ".npa")
    protected = [npa_dir / name for name in filenames]
    for directory in (*provider_dirs, npa_dir):
        paths.extend(directory / name for name in filenames)
    for key in ("AWS_CONFIG_FILE", "AWS_SHARED_CREDENTIALS_FILE", "NPA_NEBIUS_IAM_TOKEN_FILE", "NEBIUS_IAM_TOKEN_FILE"):
        if environment.get(key):
            paths.append(Path(environment[key]).expanduser())
            protected.append(paths[-1])
    workspace = (config.get("workspaces") or {}).get(config.get("active_workspace") or "default") or {}
    native = workspace.get("nebius") or {}
    if native.get("credentials_file_path"):
        paths.append(Path(str(native["credentials_file_path"]).replace("~", str(home), 1)))
        protected.append(paths[-1])
    return paths, protected, designated_caches


def _selected_kube_principals(kube, allowed, execs):
    requested = allowed or [kube.get("current-context")]
    selected_contexts = [item for item in kube.get("contexts", []) if item.get("name") in requested]
    contexts = [item.get("context", {}) for item in selected_contexts]
    valid_selection = (contexts and all(isinstance(name, str) and name for name in requested)
                       and {item.get("name") for item in selected_contexts} == set(requested)
                       and len(selected_contexts) == len(set(requested)))
    if not valid_selection:
        execs.append({"command": "unsupported"})
    users = {item.get("user") for item in contexts}
    clusters = {item.get("cluster") for item in contexts}
    selected_users = [item for item in kube.get("users", []) if item.get("name") in users]
    if not users or None in users or {item.get("name") for item in selected_users} != users or len(selected_users) != len(users):
        execs.append({"command": "unsupported"})
    return users, clusters


def _kube_referenced_credentials(kube, kube_path, allowed):
    paths, execs = [], []
    users, clusters = _selected_kube_principals(kube, allowed, execs)
    references = (("users", users, ("client-key", "client-certificate", "tokenFile")),
                  ("clusters", clusters, ("certificate-authority",)))
    for key, selected, settings in references:
        for item in kube.get(key, []):
            if item.get("name") not in selected:
                continue
            details = item.get("user" if key == "users" else "cluster") or {}
            if not isinstance(details, dict):
                execs.append({"command": "unsupported"})
                continue
            if key == "users":
                extra_auth = set(details) & {"token", "tokenFile", "client-key", "client-key-data", "auth-provider", "username", "password"}
                execs.append(details.get("exec") if not extra_auth and details.get("exec") else {"command": "unsupported"})
            for setting in settings:
                if details.get(setting):
                    referenced = Path(details[setting]).expanduser()
                    paths.append(referenced if referenced.is_absolute() else kube_path.parent / referenced)
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
        referenced, selected_execs = _kube_referenced_credentials(kube, kube_path, allowed)
        paths.extend(referenced)
        execs.extend(selected_execs)
    return paths, execs


def _hash_identity_paths(paths, protected, designated_caches, durable, cache):
    result = {}
    cache_binding = durable.pop(cache, None)
    # Explicit bearer/NPA/AWS/key aliases never become derived token caches.
    protected.extend(path for path in paths if path not in designated_caches)
    protected.extend(durable)
    cache_allowed = cache_binding and all(path.resolve() != cache.resolve() for path in protected)
    for path in paths:
        contents = path.read_bytes() if path.is_file() else None
        value = hashlib.sha256(contents).hexdigest() if contents is not None else "absent"
        if cache_allowed and path in designated_caches and path.resolve() == cache.resolve() and _derived_service_account_cache(contents):
            value = cache_binding
        result[str(path.absolute())] = value
    result.update({str(path.absolute()): value for path, value in durable.items()})
    return result


def _identity_files(environment: Mapping[str, str], *, config: Mapping[str, Any] | None = None) -> dict[str, str]:
    home = Path(environment.get("HOME") or "").expanduser()
    kube_paths = [Path(item).expanduser() for item in environment.get("KUBECONFIG", "").split(os.pathsep) if item]
    if config is None:
        config_path = Path(environment.get("SKYPILOT_GLOBAL_CONFIG") or "")
        config = _yaml_document(config_path.read_text()) if config_path.is_file() else {}
    config = config or {}
    paths, protected, designated_caches = _configured_identity_paths(environment, home, kube_paths, config)
    referenced, execs = _selected_kube_identity_paths(kube_paths, config)
    paths.extend(referenced)
    protected.extend(referenced)
    try:
        durable = _nebius_service_account_identity(environment, home, execs)
        cache = _nebius_config_dir(environment, home) / "credentials.yaml"
        return _hash_identity_paths(paths, protected, designated_caches, durable, cache)
    except OSError:
        raise IsolatedApiError("executing credential/configuration file identity cannot be inspected") from None


def _session_members(record: Mapping[str, Any]) -> list[int]:
    if not record.get("pid"):
        return []
    snapshots: dict[int, tuple[int, int, bool]] = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = directory.joinpath("stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[3]) != record["pid"] or fields[0] == "Z":
                continue
            if directory.stat().st_uid != os.getuid():
                raise IsolatedApiError("isolated SkyPilot API session changed process owner")
            try:
                environment = dict(item.split(b"=", 1) for item in directory.joinpath("environ").read_bytes().split(b"\0") if b"=" in item)
            except PermissionError:
                # Exiting/setproctitle workers may hide environ. Their exact
                # saved lifetime or verified live lineage is still required.
                environment = {}
            snapshots[int(directory.name)] = (
                int(fields[1]), int(fields[19]),
                environment.get(_MARKER.encode(), b"").decode() == record["marker"],
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
    if not snapshots:
        return []
    saved = record.get("session_processes", {})
    trusted = {
        pid for pid, (_, ticks, marked) in snapshots.items()
        if marked or saved.get(str(pid)) == ticks
    }
    root_pid = int(record["pid"])
    if root_pid in snapshots:
        if snapshots[root_pid][1] != record.get("start_ticks"):
            raise IsolatedApiError("isolated SkyPilot API session leader lifetime changed")
        # A known lifetime remains ours while the kernel clears argv/env on exit.
        trusted.add(root_pid)
    # Sky 0.12.2 setproctitle executor workers erase their /proc environment.
    # Corroborate them through a live parent chain in this exact owned session;
    # persist PID/starttime proof before signaling, so orphaned workers remain
    # identifiable after their leader exits. A reused PID cannot satisfy it.
    while True:
        descendants = {pid for pid, (parent, _, _) in snapshots.items() if parent in trusted}
        added = descendants - trusted
        if not added:
            break
        trusted.update(added)
    if trusted != set(snapshots):
        raise IsolatedApiError("isolated SkyPilot API session contains a process with uncertain ownership")
    if isinstance(record, dict):
        record["session_processes"] = {str(pid): value[1] for pid, value in snapshots.items()}
    return sorted(snapshots)


def _endpoint(record: Mapping[str, Any]) -> str:
    return f"http://127.0.0.1:{record['port']}"


def _process(record: Mapping[str, Any], *, verify_files: bool = True) -> dict[str, Any] | None:
    """Match the private intent marker even across a Popen/PID-save crash."""
    matches = []
    candidates = [Path("/proc") / str(record["pid"])] if record.get("pid") else Path("/proc").iterdir()
    for directory in candidates:
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            command = directory.joinpath("cmdline").read_bytes().split(b"\0")
            if b"sky.server.server" not in command or b"-m" not in command:
                if record.get("pid"):
                    fields = directory.joinpath("stat").read_text().rsplit(")", 1)[1].split()
                    if fields[0] != "Z":
                        raise IsolatedApiError("isolated SkyPilot API process lifetime disagrees with its ownership record")
                continue
            environment = dict(item.split(b"=", 1) for item in directory.joinpath("environ").read_bytes().split(b"\0") if b"=" in item)
            if environment.get(_MARKER.encode(), b"").decode() != record["marker"]:
                if record.get("pid"):
                    raise IsolatedApiError("isolated SkyPilot API process lifetime disagrees with its ownership record")
                continue
            # The marker alone is insufficient: bind the exact interpreter,
            # home, user, selected credentials/config and process lifetime.
            if str(Path(command[0].decode()).absolute()) != record.get("interpreter"):
                raise IsolatedApiError("isolated SkyPilot API process executable disagrees with its ownership record")
            for key, wanted in record.get("environment_binding", {}).items():
                actual = hashlib.sha256(environment.get(key.encode(), b"")).hexdigest()
                if actual != wanted:
                    raise IsolatedApiError("isolated SkyPilot API process environment disagrees with its ownership record")
            stat_fields = directory.joinpath("stat").read_text().rsplit(")", 1)[1].split()
            if int(stat_fields[3]) != int(directory.name):
                # Forked uvicorn/executor children can retain the same argv and
                # environment; the independently launched root is session leader.
                continue
            decoded = {key.decode(): value.decode() for key, value in environment.items()}
            if verify_files and record.get("config_sha256"):
                config_path = Path(decoded.get("SKYPILOT_GLOBAL_CONFIG") or "")
                if not config_path.is_file() or hashlib.sha256(config_path.read_bytes()).hexdigest() != record["config_sha256"]:
                    raise IsolatedApiError("isolated SkyPilot API verified configuration changed on disk")
            if verify_files and record.get("identity_files") and _identity_files(decoded) != record["identity_files"]:
                raise IsolatedApiError("isolated SkyPilot API credential configuration changed after verification")
            matches.append({"pid": int(directory.name), "start_ticks": int(stat_fields[19])})
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            raise IsolatedApiError("isolated SkyPilot API process ownership cannot be inspected") from None
    if len(matches) > 1:
        raise IsolatedApiError("isolated SkyPilot API ownership is ambiguous; refusing duplicate startup")
    if matches and record.get("pid") and (record["pid"], record.get("start_ticks")) != (matches[0]["pid"], matches[0]["start_ticks"]):
        raise IsolatedApiError("isolated SkyPilot API process lifetime disagrees with its ownership record")
    return matches[0] if matches else None


def _listener_owned(record: Mapping[str, Any], process: Mapping[str, Any], *, port: int | None = None) -> bool:
    """Corroborate actual loopback LISTEN inode held by the daemon process tree."""
    root_pid = int(process["pid"])
    selected_port = port or record["port"]
    try:
        namespace = Path(f"/proc/{root_pid}/ns/net").readlink()
    except FileNotFoundError:
        return False
    if namespace != Path("/proc/self/ns/net").readlink():
        raise IsolatedApiError("isolated SkyPilot API belongs to another network namespace")
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
            stat_fields = directory.joinpath("stat").read_text().rsplit(")", 1)[1].split()
            if int(stat_fields[3]) != root_pid:
                continue
            for fd in directory.joinpath("fd").iterdir():
                link = str(fd.readlink())
                if link.startswith("socket:[") and link[8:-1] in inodes:
                    return True
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    raise IsolatedApiError("isolated SkyPilot API port is held by an unowned process")


def isolated_api_environment(isolated_dir: Path, environment: Mapping[str, str]) -> dict[str, str]:
    """Persist endpoint intent before any client could connect to a shared API."""
    root = Path(isolated_dir).absolute() / "local-api"
    with _locked(root):
        record = _read(root)
        explicit = str(environment.get(_ENDPOINT) or "")
        if explicit and (record is None or explicit != _endpoint(record)):
            raise IsolatedApiError("an isolated SkyPilot runtime cannot use a different configured API endpoint")
        if record is None:
            http_port, metrics_port, queue_port = _available_ports()
            record = {"schema_version": 1, "root": str(root), "port": http_port,
                      "metrics_port": metrics_port, "queue_port": queue_port, "marker": uuid.uuid4().hex, "state": "intent"}
            _write(root / "daemon.json", record)
        process = _process(record) if record.get("interpreter") else None
        if process:
            _listener_owned(record, process)
        selected = {**environment, _ENDPOINT: _endpoint(record),
                    "NPA_SKYPILOT_ISOLATED_API_DIR": str(Path(isolated_dir).absolute())}
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
            from npa.orchestration.npa_workflow.submit_credentials import STORAGE_ENDPOINT_ENV_NAMES, resolve_submit_credentials

            expected_config = record["environment_binding"].get("NPA_CONFIG_DIR")
            if expected_config and hashlib.sha256(recovery_env.get("NPA_CONFIG_DIR", "").encode()).hexdigest() != expected_config:
                raise IsolatedApiError("isolated SkyPilot API recovery requires the original selected NPA configuration")
            credentials = resolve_submit_credentials(project=alias, environ=recovery_env)
            values = {"AWS_ACCESS_KEY_ID": credentials.access_key_id,
                      "AWS_SECRET_ACCESS_KEY": credentials.secret_access_key,
                      **dict.fromkeys(STORAGE_ENDPOINT_ENV_NAMES, credentials.endpoint_url)}
            for key, value in values.items():
                if not recovery_env.get(key) and value:
                    recovery_env[key] = value
            recovery_env["NPA_SKYPILOT_PROJECT"] = alias
        ensure_isolated_api(isolated_dir=isolated_dir,
                            sky_executable=str(Path(record["interpreter"]).parent / "sky"),
                            environment=recovery_env, cwd=str(Path(isolated_dir).absolute()))
    return selected


def ensure_isolated_api(
    *, isolated_dir: Path, sky_executable: str, environment: Mapping[str, str], cwd: str,
) -> dict[str, Any]:
    """Start/adopt only this scope's exact daemon; never create a cloud job."""
    root = Path(isolated_dir).absolute() / "local-api"
    with _locked(root):
        record = _read(root)
        if record is None or environment.get(_ENDPOINT) != _endpoint(record):
            raise IsolatedApiError("isolated SkyPilot API endpoint intent is missing or inconsistent")
        interpreter = str(Path(sky_executable).absolute().parent / "python")
        config_source = Path(environment.get("SKYPILOT_GLOBAL_CONFIG") or "")
        if not config_source.is_file():
            raise IsolatedApiError("isolated SkyPilot API requires the verified runtime configuration")
        if environment.get("SKYPILOT_DB_CONNECTION_URI"):
            raise IsolatedApiError("isolated SkyPilot API cannot use a shared external database")
        config_bytes = config_source.read_bytes()
        parsed_config = _yaml_document(config_bytes)
        if parsed_config.get("db"):
            raise IsolatedApiError("isolated SkyPilot API cannot use a shared external database")
        configured_endpoint = (parsed_config.get("api_server") or {}).get("endpoint")
        if configured_endpoint and configured_endpoint != _endpoint(record):
            raise IsolatedApiError("isolated SkyPilot API configuration names a different endpoint")
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        daemon_env = dict(environment)
        config_path = root / "server-config.yaml"
        daemon_env.update({_MARKER: record["marker"], "IS_SKYPILOT_SERVER": "true",
                           "SKYPILOT_GLOBAL_CONFIG": str(config_path)})
        plugins_path = root / "server-plugins.yaml"
        inherited_plugins = daemon_env.get("SKYPILOT_SERVER_PLUGINS_CONFIG")
        if inherited_plugins and Path(inherited_plugins) != plugins_path:
            raise IsolatedApiError("isolated SkyPilot API cannot inherit unverified server plugins")
        daemon_env["SKYPILOT_SERVER_PLUGINS_CONFIG"] = str(plugins_path)
        # Retain only hashes of settings that determine executing identity.
        identity_keys = ("HOME", "SKYPILOT_USER_ID", "KUBECONFIG", "NEBIUS_CONFIG_DIR", "NEBIUS_PROFILE",
                         "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_ENDPOINT_URL",
                         "AWS_ENDPOINT_URL_S3", "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE", "AWS_CONFIG_FILE",
                         "AWS_SHARED_CREDENTIALS_FILE", "S3_ENDPOINT_URL", "NEBIUS_S3_ENDPOINT", "NPA_STORAGE_ENDPOINT",
                         "NPA_CONFIG_DIR", "NPA_SKYPILOT_PROJECT", "NPA_S3_BUCKET", "NPA_S3_PREFIX",
                         "NEBIUS_IAM_TOKEN", "NEBIUS_IAM_TOKEN_FILE", "NPA_NEBIUS_IAM_TOKEN", "NPA_NEBIUS_IAM_TOKEN_FILE",
                         "SKYPILOT_GLOBAL_CONFIG", "SKYPILOT_SERVER_PLUGINS_CONFIG", "PYTHONPATH", _ENDPOINT, _MARKER)
        binding = {key: hashlib.sha256(daemon_env.get(key, "").encode()).hexdigest() for key in identity_keys}
        files = _identity_files(daemon_env, config=parsed_config)
        process = _process(record) if record.get("interpreter") else None
        if process:
            if record.get("interpreter") != interpreter or record.get("config_sha256") != config_hash:
                raise IsolatedApiError("running isolated SkyPilot API has a different verified configuration; preserve its jobs before restarting")
            if record["environment_binding"] != binding or record.get("identity_files") != files:
                raise IsolatedApiError("running isolated SkyPilot API has a different executing identity or changed credential configuration")
        else:
            # No process with this marker exists; starting the same persistent
            # API database recovers controller/job identity, never submits again.
            if _session_members(record):
                raise IsolatedApiError("owned SkyPilot API children survived their leader; finish its process-session cleanup before recovery")
            if record.get("environment_binding") and (record["environment_binding"] != binding or record.get("identity_files") != files):
                raise IsolatedApiError("isolated SkyPilot API recovery requires the original executing identity and credential configuration")
            for port in (record["port"], record["queue_port"], record["metrics_port"]):
                with socket.socket() as listener:
                    try:
                        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        listener.bind(("127.0.0.1", port))
                    except OSError:
                        raise IsolatedApiError("isolated SkyPilot API endpoint is occupied by an unowned listener") from None
            with open(plugins_path, "w", opener=lambda p, flags: os.open(p, flags, 0o600)) as handle:
                yaml.safe_dump({"plugins": [{"class": "npa.orchestration.skypilot.local_api_plugin.IsolatedQueuePlugin",
                                            "parameters": {"port": record["queue_port"]}}]}, handle)
            with open(config_path, "wb", opener=lambda p, flags: os.open(p, flags, 0o600)) as handle:
                handle.write(config_bytes)
            record.update(interpreter=interpreter, environment_binding=binding,
                          config_sha256=config_hash, identity_files=files, project_alias=daemon_env.get("NPA_SKYPILOT_PROJECT", ""),
                          runtime_settings={setting: daemon_env[name] for setting, name in _RUNTIME_SETTINGS.items() if daemon_env.get(name)},
                          pid=None, start_ticks=None, state="starting")
            _write(root / "daemon.json", record)
            with open(root / "server.log", "ab", opener=lambda p, flags: os.open(p, flags, 0o600)) as log:
                subprocess.Popen([interpreter, "-m", "sky.server.server", "--host", "127.0.0.1",
                                  "--port", str(record["port"]), "--metrics-port", str(record["metrics_port"])],
                                 env=daemon_env, cwd=cwd, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, start_new_session=True)
        while True:
            process = _process(record)
            if process is None:
                raise IsolatedApiError("owned SkyPilot API exited before readiness; inspect its private server log")
            record.update(process)
            _write(root / "daemon.json", record)
            queue_ready = _listener_owned(record, process, port=record["queue_port"])
            metrics_ready = not daemon_env.get("SKY_API_SERVER_METRICS_ENABLED") or _listener_owned(record, process, port=record["metrics_port"])
            if _listener_owned(record, process) and queue_ready and metrics_ready:
                try:
                    with urlopen(f"{_endpoint(record)}/api/health") as response:
                        health = json.load(response)
                    if health.get("version") != "0.12.2" or str(health.get("status") or "").lower() != "healthy":
                        raise IsolatedApiError("owned SkyPilot API readiness/version evidence is inconsistent")
                    record["state"] = "ready"
                    _write(root / "daemon.json", record)
                    return {"healthy": True, "outcome": "owned_isolated_api", "process_count": 1}
                except (URLError, ConnectionError, json.JSONDecodeError):
                    pass
            time.sleep(0.2)


def owned_daemon_environment(isolated_dir: Path) -> dict[str, str]:
    """Read an owned daemon's exact environment in memory for a cleanup transaction.

    Credentials must never be written to a transaction record or log. PID,
    lifetime, marker, executable, config and identity-file checks precede and
    follow the read; no foreign or ambiguous process may supply credentials.
    """
    root = Path(isolated_dir).absolute() / "local-api"
    with _locked(root):
        record = _read(root)
        if not record or not record.get("interpreter"):
            raise IsolatedApiError("controller transaction requires an owned source API")
        process = _process(record)
        if not process or not _listener_owned(record, process):
            raise IsolatedApiError("controller transaction source API is not verified ready")
        try:
            raw = (Path("/proc") / str(process["pid"]) / "environ").read_bytes()
        except OSError:
            raise IsolatedApiError("controller transaction source identity is unreadable") from None
        environment = dict(
            entry.decode().split("=", 1) for entry in raw.split(b"\0") if b"=" in entry
        )
        if _process(record) != process:
            raise IsolatedApiError("controller transaction source API changed during inspection")
        return environment


def stop_isolated_api(isolated_dir: Path) -> None:
    """Stop only the owned local process group, after callers finish cloud jobs."""
    root = Path(isolated_dir).absolute() / "local-api"
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
            _session_members(record)
            try:
                os.killpg(record["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
            while _session_members(record):
                time.sleep(0.2)
        record.update(state="stopped", pid=None, start_ticks=None)
        _write(root / "daemon.json", record)
