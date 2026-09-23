"""Manage a chat-only route while preserving the gateway's existing applications."""

import json
import base64
from http.client import HTTPConnection, HTTPException
from pathlib import Path
import subprocess
import tempfile
import time
from urllib.parse import urlsplit


def _read_root(path):
    return subprocess.check_output(["sudo", "-n", "cat", path], text=True)


def _origin(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        raise ValueError("The managed gateway must use authenticated public HTTPS.")
    return f"{parsed.scheme}://{parsed.netloc}"


def _inspect():
    desktop = Path.home() / ".local/share/nebius-desktop/public-access.json"
    if desktop.exists():
        config = json.loads(desktop.read_text())
        return {
            "kind": "desktop",
            "origin": _origin(config["url"]),
            "username": config["username"],
            "password": Path(config["password_file"]).read_text().strip(),
        }
    mobile = Path("/etc/codex-mobile/server.json")
    if mobile.exists():
        config = json.loads(_read_root(str(mobile)))
        return {
            "kind": "mobile",
            "origin": _origin(config["origin"]),
            "username": config["username"],
            "salt": config["passwordSalt"],
            "password_hash": config["passwordHash"],
            "session_secret": config["sessionSecret"],
        }
    raise RuntimeError(
        "Configure an authenticated desktop or mobile HTTPS gateway first."
    )


def _install(path, value):
    with tempfile.NamedTemporaryFile(mode="w") as temporary:
        temporary.write(value)
        temporary.flush()
        subprocess.run(
            ["sudo", "-n", "install", "-m", "644", temporary.name, path], check=True
        )


def _configure_mobile(port):
    path = "/etc/caddy/Caddyfile"
    original = _read_root(path)
    candidate = _mobile_routes(original, port)
    if candidate != original:
        _install(path + ".npa-candidate", candidate)
        subprocess.run(
            [
                "sudo", "-n", "/usr/local/bin/caddy", "validate",
                "--config", path + ".npa-candidate", "--adapter", "caddyfile",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _replace_and_reload(path, original, candidate, "caddy")
    _ensure_restart("caddy")


def _mobile_routes(original, port):
    marker = "# npa-local-chat"
    if marker in original:
        if f"127.0.0.1:{port}" not in original:
            raise RuntimeError("An existing local route uses another port.")
        return _mobile_entry_routes(original)
    target = "reverse_proxy 127.0.0.1:8787"
    if original.count(target) != 1:
        raise RuntimeError("The managed mobile gateway route was customized.")
    start = original.index(target)
    end = start + len(target)
    opening = end + len(original[end:]) - len(original[end:].lstrip())
    if original[opening : opening + 1] == "{":
        depth = 0
        for index in range(opening, len(original)):
            depth += (original[index] == "{") - (original[index] == "}")
            if depth == 0:
                end = index + 1
                break
    directive = original[start:end]
    replacement = (
        marker + "\nhandle /chat/* {\n"
        f"reverse_proxy 127.0.0.1:{port}\n"
        "}\nhandle {\n" + directive + "\n}"
    )
    candidate = original[:start] + replacement + original[end:]
    return _mobile_entry_routes(candidate)


def _mobile_entry_routes(original):
    marker = "# npa-local-chat-entry"
    if marker in original:
        return original
    routes = (
        marker + "\n@npa_chat_entry path / /index.html /chat\n"
        "handle @npa_chat_entry {\n"
        "header Cache-Control no-store\nredir * /chat/?{query} 302\n}\n"
        "handle /legacy/ {\nrewrite * /\n"
        "reverse_proxy 127.0.0.1:8787\n}\n"
    )
    return original.replace("# npa-local-chat\n", "# npa-local-chat\n" + routes, 1)


def _ensure_restart(service):
    result = subprocess.run(
        ["systemctl", "show", service, "-p", "Restart", "--value"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip() != "no":
        return
    directory = f"/etc/systemd/system/{service}.service.d"
    subprocess.run(["sudo", "-n", "mkdir", "-p", directory], check=True)
    _install(
        directory + "/npa-local-recovery.conf",
        "[Service]\nRestart=on-failure\nRestartSec=3\n",
    )
    subprocess.run(["sudo", "-n", "systemctl", "daemon-reload"], check=True)


def _configure_desktop(port):
    path = "/etc/npa-desktop/nginx.conf"
    original = _read_root(path)
    marker = "# npa-local-chat"
    if marker in original:
        if f"127.0.0.1:{port}/chat/" not in original:
            raise RuntimeError("An existing local route uses another port.")
        return
    target = "location = /desktop-credentials.json"
    if original.count(target) != 1:
        raise RuntimeError("The managed desktop gateway route was customized.")
    route = (
        marker + "\nlocation /local-chat/ {\n"
        f"proxy_pass http://127.0.0.1:{port}/chat/;\n"
        "proxy_set_header Authorization $http_authorization;\n"
        "proxy_read_timeout 65s;\nclient_max_body_size 16m;\n}\n"
    )
    candidate = original.replace(target, route + target)
    _install(path + ".npa-candidate", candidate)
    subprocess.run(
        ["sudo", "-n", "nginx", "-t", "-c", path + ".npa-candidate"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _replace_and_reload(path, original, candidate, "npa-desktop-gateway")


def _replace_and_reload(path, original, candidate, service):
    _install(path + ".before-local", original)
    _install(path, candidate)
    try:
        subprocess.run(
            ["sudo", "-n", "systemctl", "reload", service],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        _install(path, original)
        subprocess.run(
            ["sudo", "-n", "systemctl", "reload", service],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        raise


def _local_state(port, username, password):
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("Invalid local chat port")
    authorization = base64.b64encode(f"{username}:{password}".encode()).decode()
    # A direct connection cannot follow redirects or use an environment proxy.
    connection = HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request(
            "GET",
            "/chat/api/state",
            headers={"Authorization": "Basic " + authorization},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError("The local chat service did not authenticate its state.")
        return json.load(response)
    finally:
        connection.close()


def _verify_tunnel(config):
    port = config["port"]
    for attempt in range(30):
        try:
            state = _local_state(port, config["username"], config["password"])
            if state.get("installationId") != config["installation_id"]:
                raise RuntimeError("The gateway port belongs to another installation.")
            break
        except (OSError, HTTPException):
            if attempt == 29:
                raise RuntimeError(
                    "The authenticated Mac tunnel did not become ready."
                ) from None
            time.sleep(1)
    listeners = subprocess.check_output(
        ["ss", "-H", "-ltn", "sport", "=", f":{port}"], text=True
    )
    addresses = [line.split()[3] for line in listeners.splitlines()]
    if not addresses or any(
        address not in {f"127.0.0.1:{port}", f"[::1]:{port}"} for address in addresses
    ):
        raise RuntimeError("The SSH gateway listener must bind only to loopback.")


def main(config):
    """Execute a gateway action without printing command output or errors.

    Args:
        config: Validated operation and private metadata.
    Returns:
        None.
    Raises:
        RuntimeError: An existing route cannot be safely adopted.
    """
    info = _inspect()
    if config["action"] == "configure":
        if config["origin"] != info["origin"]:
            raise RuntimeError("Gateway origin changed; inspect it again.")
        port = config["port"]
        if not isinstance(port, int) or not 1024 <= port <= 65535:
            raise ValueError("Invalid gateway port")
        _verify_tunnel(config)
        if info["kind"] == "mobile":
            _configure_mobile(port)
        else:
            _configure_desktop(port)
        info = {"configured": True}
    print(json.dumps(info))
