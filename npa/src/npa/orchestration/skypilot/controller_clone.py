"""Relocate verified controller metadata inside a disposable Sky state copy.

Executed with the pinned Sky interpreter before the clone API starts. The
original HOME, database, SSH configuration and process identities stay intact.
"""
from __future__ import annotations

import copy
import hmac
import json
import os
from pathlib import Path
import re
import shlex
import sys

import yaml


def _private_path(path: Path, root: Path) -> Path:
    if not path.is_absolute() or not path.is_relative_to(root) or path.resolve() != path:
        raise ValueError("controller clone path is outside its private snapshot")
    return path


def relocate_auth(
    config: dict, *, source_home: Path, clone_home: Path,
) -> tuple[dict, str, Path]:
    """Relocate only canonical managed SSH paths; preserve executable settings."""
    relocated = copy.deepcopy(config)
    auth = relocated.get("auth")
    if not isinstance(auth, dict):
        raise ValueError("controller SSH configuration is missing")
    source_key = auth.get("ssh_private_key")
    if not isinstance(source_key, str):
        raise ValueError("controller managed SSH key path is missing")
    if source_key.startswith("~/"):
        source_key = str(source_home / source_key[2:])
    try:
        relative = Path(source_key).relative_to(source_home)
    except ValueError:
        raise ValueError("controller SSH key is outside its source home") from None
    parts = relative.parts
    if (len(parts) != 5 or parts[:2] != (".sky", "clients")
            or parts[3:] != ("ssh", "sky-key")
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", parts[2])):
        raise ValueError("controller SSH key is not a canonical managed key")
    key = _private_path(clone_home / relative, clone_home)
    auth["ssh_private_key"] = str(key)
    if "ssh_public_key" in auth:
        public = auth["ssh_public_key"]
        if isinstance(public, str) and public.startswith("~/"):
            public = str(source_home / public[2:])
        if public != source_key + ".pub":
            raise ValueError("controller public key is not its managed companion")
        auth["ssh_public_key"] = str(key) + ".pub"
    # Proxy commands and other settings are executable operator configuration.
    # Never rewrite them as strings or leave references into the source HOME.
    for field, value in auth.items():
        if field in {"ssh_private_key", "ssh_public_key", "ssh_proxy_command"}:
            continue
        if isinstance(value, str) and str(source_home) in value:
            raise ValueError("controller authentication retains a source-home reference")
    return relocated, parts[2], key


def verify_proxy_relocation(
    original: str, generated: str, *, source_key: str, clone_key: str,
    source_script: str, clone_script: str, ssh_user: str,
) -> str:
    """Relocate owned paths, preserving Sky's canonical username substitution."""
    old = shlex.split(original)
    expected = shlex.split(generated)
    relocated = list(expected)
    if not isinstance(ssh_user, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*[$]?", ssh_user.strip()):
        raise ValueError("controller proxy SSH username is invalid")
    if len(old) != len(expected):
        raise ValueError("controller proxy does not match the pinned builder")
    # Sky persists both its raw builder output and the credentials API's
    # resolved username. Accept only that exact substitution at the jump-host
    # destination, never a different user or replacements inside other argv.
    placeholder = "skypilot:ssh_user@127.0.0.1"
    destinations = [i for i, value in enumerate(expected) if value == placeholder]
    if len(destinations) != 1:
        raise ValueError("controller proxy has no unique managed destination")
    destination = destinations[0]
    if old[destination] == ssh_user.strip() + "@127.0.0.1":
        expected[destination] = relocated[destination] = old[destination]
    if expected.count("-i") != 1:
        raise ValueError("controller proxy has no unique managed key")
    index = expected.index("-i") + 1
    if index >= len(expected) or expected[index] != clone_key:
        raise ValueError("controller proxy key is not its copied key")
    expected[index] = source_key
    proxies = [i for i, value in enumerate(expected) if value.startswith("ProxyCommand=")]
    if len(proxies) != 1:
        raise ValueError("controller proxy has no unique port-forward script")
    index = proxies[0]
    nested = shlex.split(expected[index].removeprefix("ProxyCommand="))
    if not nested or nested[0] != clone_script:
        raise ValueError("controller proxy script is outside its copy")
    nested[0] = source_script
    old_nested = shlex.split(old[index].removeprefix("ProxyCommand="))
    if not old[index].startswith("ProxyCommand=") or old_nested != nested:
        raise ValueError("controller proxy changes its target or command")
    old[index] = expected[index]
    if old != expected:
        raise ValueError("controller proxy changes its flags or identity")
    return shlex.join(relocated)


def prepare(manifest: dict) -> None:
    root = Path(manifest["clone_root"])
    home = root / "home"
    runtime = root / "sky-runtime"
    source_home = Path(manifest["source_home"])
    if (not source_home.is_absolute() or source_home.resolve() != source_home
            or os.environ.get("HOME") != str(home)
            or os.environ.get("SKY_RUNTIME_DIR") != str(runtime)):
        raise ValueError("controller clone executing paths do not match")
    _private_path(home, root)
    _private_path(runtime / ".sky/state.db", root)
    config_path = _private_path(Path(os.environ.get("SKYPILOT_GLOBAL_CONFIG", "")), root)
    global_config = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(global_config, dict) or global_config.get("db") or os.environ.get("SKYPILOT_DB_CONNECTION_URI"):
        raise ValueError("controller clone cannot use an external database")
    names = manifest["controller_names"]
    context = manifest["context"]
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or not isinstance(context, str) or not context):
        raise ValueError("controller clone requires exact targets")
    # Import Sky only after the cloned paths and database settings are checked.
    # Ordinary npa installations need not install the cloud runtime.
    import sky
    from sky import backends, clouds, global_user_state as state
    from sky.backends import backend_utils
    from sky.provision.kubernetes import utils as kubernetes_utils
    from sky.utils import auth_utils

    if sky.__version__ != "0.12.2":
        raise ValueError("unsupported controller state version")
    prepared = []
    for name in names:
        if not isinstance(name, str) or not re.fullmatch(r"sky-jobs-controller-[A-Za-z0-9-]+", name):
            raise ValueError("controller clone target is invalid")
        record = state.get_cluster_from_name(name, include_user_info=False)
        if record is None:
            raise ValueError("controller clone target is absent")
        handle = record["handle"]
        if (not isinstance(handle, backends.CloudVmRayResourceHandle)
                or handle.cluster_name != name
                or not isinstance(handle.launched_resources.cloud, clouds.Kubernetes)):
            raise ValueError("controller clone target has an unsupported backend")
        stored = handle._cluster_yaml
        if not isinstance(stored, str):
            raise ValueError("controller clone target has no configuration")
        if stored.startswith("~/"):
            relative = Path(stored[2:])
        else:
            try:
                relative = Path(stored).relative_to(source_home)
            except ValueError:
                raise ValueError("controller configuration is outside its source home") from None
        if (relative.parent != Path(".sky/generated")
                or relative.stem != name or relative.suffix not in {".yml", ".yaml"}):
            raise ValueError("controller configuration path is not canonical")
        target = _private_path(home / relative, home)
        # Use the copied YAML/database path only: Sky's fallback must never read
        # or migrate a source file into the transaction.
        config = state.get_cluster_yaml_dict(str(target))
        provider = config.get("provider") or {}
        if (provider.get("type") != "external" or provider.get("context") != context
                or provider.get("module") != "sky.provision.kubernetes"):
            raise ValueError("controller configuration does not match the verified context")
        relocated, user, key = relocate_auth(config, source_home=source_home, clone_home=home)
        public, private, exists = state.get_ssh_keys(user)
        if not exists or not public or not private:
            raise ValueError("controller managed SSH identity is absent")
        for path, expected in [(key, private), (Path(str(key) + ".pub"), public)]:
            _private_path(path, home)
            if not path.is_file() or not hmac.compare_digest(path.read_bytes(), expected.encode()):
                raise ValueError("controller copied SSH identity does not match its database")
        proxy = config["auth"].get("ssh_proxy_command")
        if proxy is not None:
            namespace = provider.get("namespace")
            pod = config.get("cluster_name")
            if (not isinstance(proxy, str) or not isinstance(namespace, str)
                    or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", namespace)
                    or not isinstance(pod, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", pod)):
                raise ValueError("controller proxy target is not explicit")
            script_relative = Path(kubernetes_utils.PORT_FORWARD_PROXY_CMD_PATH.removeprefix("~/"))
            script = _private_path(home / script_relative, home)
            generated = kubernetes_utils.get_ssh_proxy_command(
                pod_name=pod + "-head", private_key_path=str(key), context=context, namespace=namespace,
            )
            relocated["auth"]["ssh_proxy_command"] = verify_proxy_relocation(
                proxy, generated, source_key=str(source_home / key.relative_to(home)),
                clone_key=str(key), source_script=str(source_home / script_relative),
                clone_script=str(script), ssh_user=config["auth"].get("ssh_user"),
            )
        prepared.append((name, handle, target, relocated, key))
    # Validate every selected target before changing any controller records.
    for name, handle, target, config, key in prepared:
        body = yaml.safe_dump(config, sort_keys=False)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with open(target, "w", opener=lambda p, flags: os.open(p, flags, 0o600)) as stream:
            stream.write(body)
        target.chmod(0o600)
        state.set_cluster_yaml(name, body)
        handle.cluster_yaml = str(target)
        state.update_cluster_handle(name, handle)
        # These PIDs belong to the source process namespace. A copied record
        # must never authorize terminating the source tunnel or a reused PID.
        state.set_cluster_skylet_ssh_tunnel_metadata(name, None)
        credentials = backend_utils.ssh_credential_from_yaml(handle.cluster_yaml)
        if credentials["ssh_private_key"] != str(key):
            raise ValueError("controller clone SSH path did not round-trip")
        if not auth_utils.create_ssh_key_files_from_db(str(key)):
            raise ValueError("controller clone SSH identity could not be restored")


def main() -> int:
    os.umask(0o077)
    try:
        prepare(json.loads(Path(sys.argv[1]).read_text()))
    except (OSError, ValueError, KeyError, TypeError, ImportError, AssertionError):
        # Captured private configuration and key values must not become a CLI
        # exception. The parent preserves original state on any nonzero result.
        print("Controller clone metadata verification failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
