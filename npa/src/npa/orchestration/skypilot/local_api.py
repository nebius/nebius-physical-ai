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


def _identity_files(environment: Mapping[str, str], *, config: Mapping[str, Any] | None = None) -> dict[str, str]:
    home = Path(environment.get("HOME") or "").expanduser()
    kube_paths = [Path(item).expanduser() for item in environment.get("KUBECONFIG", "").split(os.pathsep) if item]
    paths = [*kube_paths, home / ".aws" / "config", home / ".aws" / "credentials"]
    for directory in (home / ".nebius", Path(environment.get("NEBIUS_CONFIG_DIR") or home / ".nebius"),
                      Path(environment.get("NPA_CONFIG_DIR") or home / ".npa")):
        paths.extend(directory / name for name in ("config.yaml", "credentials.yaml", "credentials.json",
                     "NEBIUS_IAM_TOKEN.txt", "NEBIUS_TENANT_ID.txt", "NEBIUS_DOMAIN.txt"))
    for key in ("AWS_CONFIG_FILE", "AWS_SHARED_CREDENTIALS_FILE", "NPA_NEBIUS_IAM_TOKEN_FILE", "NEBIUS_IAM_TOKEN_FILE"):
        if environment.get(key):
            paths.append(Path(environment[key]).expanduser())
    if config is None:
        config_path = Path(environment.get("SKYPILOT_GLOBAL_CONFIG") or "")
        config = _yaml_document(config_path.read_text()) if config_path.is_file() else {}
    config = config or {}
    workspace = (config.get("workspaces") or {}).get(config.get("active_workspace") or "default") or {}
    native = workspace.get("nebius") or {}
    if native.get("credentials_file_path"):
        paths.append(Path(str(native["credentials_file_path"]).replace("~", str(home), 1)))
    # Fingerprint referenced files for only the selected kube context(s), never
    # private keys belonging to an unrelated context in a merged kubeconfig.
    allowed = (config.get("kubernetes") or {}).get("allowed_contexts") or []
    for kube_path in kube_paths:
        if not kube_path.is_file():
            continue
        kube = _yaml_document(kube_path.read_text())
        if not isinstance(kube, dict):
            continue
        contexts = [item.get("context", {}) for item in kube.get("contexts", [])
                    if item.get("name") in (allowed or [kube.get("current-context")])]
        users = {item.get("user") for item in contexts}
        clusters = {item.get("cluster") for item in contexts}
        for key, selected, settings in (("users", users, ("client-key", "client-certificate", "tokenFile")),
                                        ("clusters", clusters, ("certificate-authority",))):
            for item in kube.get(key, []):
                if item.get("name") not in selected:
                    continue
                details = item.get("user" if key == "users" else "cluster") or {}
                for setting in settings:
                    if details.get(setting):
                        referenced = Path(details[setting]).expanduser()
                        paths.append(referenced if referenced.is_absolute() else kube_path.parent / referenced)
    result = {}
    for path in paths:
        try:
            result[str(path.absolute())] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "absent"
        except OSError:
            raise IsolatedApiError("executing credential/configuration file identity cannot be inspected") from None
    return result


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
