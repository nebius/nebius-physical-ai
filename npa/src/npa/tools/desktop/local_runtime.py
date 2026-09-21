"""Install Mac chat controls while keeping existing VS Code sessions and credentials."""

import hashlib
import base64
import hmac
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import socket
import subprocess
import sys
import webbrowser
import uuid
import fcntl
from urllib.request import Request, urlopen

from .local_gateway import gateway_operation
from .local_service import install_service, service_running, tunnel_command
from . import _validate_host

_LABEL = "com.nebius.codex-chat"


def _root():
    return Path.home() / ".local/share/nebius-desktop/local-chat"


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(content)
    temporary.replace(path)


def _binary():
    machine = "macos-aarch64" if platform.machine() == "arm64" else "macos-x86_64"
    extensions = list(Path.home().glob(f".vscode/extensions/openai.chatgpt-*/bin/{machine}/codex"))
    if extensions:
        return str(max(extensions, key=lambda path: path.stat().st_mtime))
    binary = shutil.which("codex")
    if not binary:
        raise RuntimeError("Install Codex and sign in on this Mac first.")
    return binary


def _node():
    binary = shutil.which("node")
    if not binary:
        raise RuntimeError("Install Node.js 22.13 or newer for the Mac adapter.")
    version = subprocess.check_output([binary, "--version"], text=True).strip().lstrip("v")
    if tuple(int(part) for part in version.split(".")[:2]) < (22, 13):
        raise RuntimeError("The Mac adapter requires Node.js 22.13 or newer.")
    return binary


def _gateway_password(info):
    if "password" in info:
        return info["password"]
    access = Path.home() / ".codex-mobile/access.txt"
    if not access.exists():
        raise RuntimeError("The existing mobile gateway's private access file is required for migration.")
    values = dict(line.split(": ", 1) for line in access.read_text().splitlines() if ": " in line)
    password = values.get("Password", "")
    digest = hashlib.scrypt(password.encode(), salt=info["salt"].encode(), n=16384, r=8, p=1, dklen=64)
    if not hmac.compare_digest(digest.hex(), info["password_hash"]):
        raise RuntimeError("The saved mobile login does not match the selected gateway.")
    return password


def _configuration(gateway_host, port, gateway_port):
    root = _root()
    path = root / "config.json"
    if path.exists():
        config = json.loads(path.read_text())
        if gateway_host and config.get("gateway_host") is None:
            return _adopt_gateway(config, gateway_host)
        if gateway_host and config.get("gateway_host") != gateway_host:
            raise ValueError("This local installation already uses another gateway.")
        return config
    info = gateway_operation(gateway_host, "inspect") if gateway_host else None
    origin = info["origin"] if info else f"http://127.0.0.1:{port}"
    _write(root / "password", _gateway_password(info) if info else secrets.token_urlsafe(32))
    return {"backend": "native", "host_label": "your Mac", "node": _node(),
            "binary": _binary(), "codexHome": os.environ.get("CODEX_HOME", str(Path.home() / ".codex")),
            "defaultCwd": str(Path.cwd()), "cwd": str(Path.cwd()), "socket": "",
            "port": port, "gateway_port": gateway_port, "gateway_host": gateway_host,
            "username": info["username"] if info else "developer", "origin": origin,
            "password_file": str(root / "password"),
            "session_secret": info.get("session_secret") if info else None,
            "url": origin + ("/local-chat/" if info and info["kind"] == "desktop" else "/chat/")}


def _adopt_gateway(config, host):
    _check_upgrade(config, "configuration-change")
    info = gateway_operation(host, "inspect")
    password = _gateway_password(info)
    _write(_root() / "password.previous", Path(config["password_file"]).read_text())
    password_path = _root() / "gateway-password"
    _write(password_path, password)
    return {**config, "gateway_host": host, "origin": info["origin"],
            "password_file": str(password_path),
            "username": info["username"], "session_secret": info.get("session_secret"),
            "url": info["origin"] + ("/local-chat/" if info["kind"] == "desktop" else "/chat/")}


def _install_assets(root):
    source = Path(__file__).parent
    names = ["chat_server.py", "chat_rpc.py", "chat_history.py", "chat_models.py",
             "chat_delivery.py", "chat_session.py", "chat_native.py", "chat.html", "chat.css", "chat.js"]
    assets = {name: (source / name).read_text() for name in names}
    assets.update({"native/" + path.name: path.read_text()
                   for path in (source / "native").iterdir() if path.is_file()})
    digest = hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest()
    runtime = root / "versions" / digest
    if (runtime / ".ready").exists():
        return runtime
    runtime.mkdir(parents=True, exist_ok=True)
    for name, content in assets.items():
        _write(runtime / name, content)
    subprocess.run(["npm", "ci", "--ignore-scripts", "--omit=dev", "--no-audit", "--no-fund"],
                   cwd=runtime / "native", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _write(runtime / ".ready", digest)
    return runtime


def setup_local(*, gateway_host=None, port=6091, gateway_port=6091, dry_run=False):
    """Install local Mac chat with an optional existing public HTTPS gateway.

    Args:
        gateway_host: Existing desktop/mobile gateway SSH alias; never the Mac host.
        port: Dedicated Mac loopback listener.
        gateway_port: Dedicated remote loopback tunnel listener.
        dry_run: Validate and describe without touching files or connecting.
    Returns:
        Public status metadata, without passwords or tokens.
    Raises:
        ValueError: The platform or selected ports are unsupported.
        RuntimeError: Prerequisites or the existing gateway are unavailable.
    """
    if platform.system() != "Darwin":
        raise ValueError("--local currently supports macOS; use --ssh-host for Linux desktops.")
    if not all(1024 <= value <= 65535 for value in (port, gateway_port)):
        raise ValueError("Choose unprivileged local and gateway ports.")
    if gateway_host:
        _validate_host(gateway_host)
    if dry_run:
        return {"planned": True, "runtime": "local", "gateway": bool(gateway_host),
                "port": port, "gateway_port": gateway_port}
    root = _root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / "operation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _apply_local(root, gateway_host, port, gateway_port)


def _apply_local(root, gateway_host, port, gateway_port):
    previous = (root / "config.json").read_text() if (root / "config.json").exists() else None
    config = _configuration(gateway_host, port, gateway_port)
    config.setdefault("installation_id", str(uuid.uuid4()))
    if not service_running(_LABEL):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", config["port"]))
    runtime = _install_assets(root)
    if previous and not json.loads(previous).get("native_socket"):
        _check_upgrade(json.loads(previous), runtime)
    config["runtime_path"] = str(runtime)
    config.setdefault("native_socket", str(root / "native.sock"))
    engine = _prepare_engine(config, runtime)
    version = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if previous:
        _write(root / "config.previous.json", previous)
    _write(root / "config.json", json.dumps(config))
    try:
        _install_engine(config, root, engine)
    except (OSError, RuntimeError, subprocess.SubprocessError):
        if previous:
            _write(root / "config.json", previous)
        raise
    install_service(_LABEL, [sys.executable, str(runtime / "chat_server.py"), str(root / "config.json"), version], root)
    _connect_gateway(config, root)
    return local_status()


def _prepare_engine(config, runtime):
    contents = [path.read_bytes() for path in sorted((runtime / "native").iterdir()) if path.is_file()]
    settings = {key: config.get(key) for key in ("binary", "node", "codexHome", "defaultCwd")}
    digest = hashlib.sha256(b"".join(contents) + json.dumps(settings, sort_keys=True).encode()).hexdigest()
    if config.get("native_version") == digest:
        return None
    connection = None
    if service_running(_LABEL + ".engine"):
        from .chat_native import native_connection
        connection = native_connection(config, _root() / "config.json", None)
        try:
            connection.call("runtime/prepareUpdate", {})
        except RuntimeError:
            connection.connection.close()
            raise
    config.update(native_version=digest, native_runtime_path=str(runtime))
    return connection


def _install_engine(config, root, prepared):
    command = [config["node"], str(Path(config["native_runtime_path"]) / "native/bridge.mjs"),
               str(root / "config.json"), "--serve", config["native_version"]]
    try:
        install_service(_LABEL + ".engine", command, root)
    except (OSError, subprocess.SubprocessError):
        if prepared is not None and prepared.alive:
            prepared.call("runtime/cancelUpdate", {})
        raise
    finally:
        if prepared is not None:
            prepared.connection.close()


def _check_upgrade(config, runtime):
    if not service_running(_LABEL) or config.get("runtime_path") == str(runtime):
        return
    password = Path(config["password_file"]).read_text().strip()
    authorization = base64.b64encode(f'{config["username"]}:{password}'.encode()).decode()
    request = Request(f'http://127.0.0.1:{config["port"]}/chat/api/state',
                      headers={"Authorization": "Basic " + authorization})
    with urlopen(request) as response:
        state = json.load(response)
    if state.get("runtime", {}).get("ownedActive"):
        raise RuntimeError("Wait for mobile-owned turns to finish before updating local chat.")


def _connect_gateway(config, root):
    host = config.get("gateway_host")
    if not host:
        return
    command = tunnel_command(host, config["port"], config["gateway_port"])
    install_service(_LABEL + ".tunnel", command, root)
    gateway_operation(host, "configure", origin=config["origin"], port=config["gateway_port"],
                      installation_id=config["installation_id"], username=config["username"],
                      password=Path(config["password_file"]).read_text().strip())


def local_status():
    """Read the local installation and process status without exposing credentials.

    Args:
        None.
    Returns:
        Installed state, public URL, and independent service health.
    Raises:
        OSError: Private configuration cannot be read.
    """
    path = _root() / "config.json"
    if not path.exists():
        return {"installed": False, "runtime": "local"}
    config = json.loads(path.read_text())
    return {"installed": True, "runtime": "local", "url": config["url"],
            "service_running": service_running(_LABEL),
            "engine_running": service_running(_LABEL + ".engine"),
            "tunnel_running": service_running(_LABEL + ".tunnel") if config.get("gateway_host") else None,
            "credentials_file": config.get("password_file", str(_root() / "password")), "username": config["username"]}


def open_local():
    """Open the configured local chat URL.

    Args:
        None.
    Returns:
        The URL without credentials.
    Raises:
        RuntimeError: Local chat has not been installed.
    """
    status = local_status()
    if not status["installed"]:
        raise RuntimeError("Run chat-setup --local first.")
    webbrowser.open(status["url"])
    return status["url"]
