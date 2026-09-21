"""Manage a chat-only route while preserving the gateway's existing applications."""

import json
import base64
from pathlib import Path
import subprocess
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def _read_root(path):
    return subprocess.check_output(["sudo", "-n", "cat", path], text=True)


def _inspect():
    desktop = Path.home() / ".local/share/nebius-desktop/public-access.json"
    if desktop.exists():
        config = json.loads(desktop.read_text())
        return {"kind": "desktop", "origin": config["url"].rstrip("/"),
                "username": config["username"],
                "password": Path(config["password_file"]).read_text().strip()}
    mobile = Path("/etc/codex-mobile/server.json")
    if mobile.exists():
        config = json.loads(_read_root(str(mobile)))
        return {"kind": "mobile", "origin": config["origin"],
                "username": config["username"], "salt": config["passwordSalt"],
                "password_hash": config["passwordHash"], "session_secret": config["sessionSecret"]}
    raise RuntimeError("Configure an authenticated desktop or mobile HTTPS gateway first.")


def _install(path, value):
    with tempfile.NamedTemporaryFile(mode="w") as temporary:
        temporary.write(value)
        temporary.flush()
        subprocess.run(["sudo", "-n", "install", "-m", "644", temporary.name, path], check=True)


def _configure_mobile(port):
    path = "/etc/caddy/Caddyfile"
    original = _read_root(path)
    marker = "# npa-local-chat"
    if marker in original:
        if f"127.0.0.1:{port}" not in original:
            raise RuntimeError("An existing local route uses another port.")
        _ensure_restart("caddy")
        return
    target = "reverse_proxy 127.0.0.1:8787"
    if original.count(target) != 1:
        raise RuntimeError("The managed mobile gateway route was customized.")
    start = original.index(target)
    end = start + len(target)
    opening = end + len(original[end:]) - len(original[end:].lstrip())
    if original[opening:opening + 1] == "{":
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
    _install(path + ".npa-candidate", candidate)
    subprocess.run(["sudo", "-n", "/usr/local/bin/caddy", "validate", "--config",
                    path + ".npa-candidate", "--adapter", "caddyfile"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _replace_and_reload(path, original, candidate, "caddy")
    _ensure_restart("caddy")


def _ensure_restart(service):
    result = subprocess.run(["systemctl", "show", service, "-p", "Restart", "--value"],
                            check=True, capture_output=True, text=True)
    if result.stdout.strip() != "no":
        return
    directory = f"/etc/systemd/system/{service}.service.d"
    subprocess.run(["sudo", "-n", "mkdir", "-p", directory], check=True)
    _install(directory + "/npa-local-recovery.conf", "[Service]\nRestart=on-failure\nRestartSec=3\n")
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
    route = (marker + "\nlocation /local-chat/ {\n"
             f"proxy_pass http://127.0.0.1:{port}/chat/;\n"
             "proxy_set_header Authorization $http_authorization;\n"
             "proxy_read_timeout 65s;\nclient_max_body_size 16m;\n}\n")
    candidate = original.replace(target, route + target)
    _install(path + ".npa-candidate", candidate)
    subprocess.run(["sudo", "-n", "nginx", "-t", "-c", path + ".npa-candidate"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _replace_and_reload(path, original, candidate, "npa-desktop-gateway")


def _replace_and_reload(path, original, candidate, service):
    _install(path + ".before-local", original)
    _install(path, candidate)
    try:
        subprocess.run(["sudo", "-n", "systemctl", "reload", service], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        _install(path, original)
        subprocess.run(["sudo", "-n", "systemctl", "reload", service], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raise


def _verify_tunnel(config):
    port = config["port"]
    authorization = base64.b64encode(f'{config["username"]}:{config["password"]}'.encode()).decode()
    request = Request(f"http://127.0.0.1:{port}/chat/api/state",
                      headers={"Authorization": "Basic " + authorization})
    for attempt in range(30):
        try:
            with urlopen(request, timeout=2) as response:
                state = json.load(response)
            if state.get("installationId") != config["installation_id"]:
                raise RuntimeError("The gateway port belongs to another installation.")
            break
        except URLError:
            if attempt == 29:
                raise RuntimeError("The authenticated Mac tunnel did not become ready.") from None
            time.sleep(1)
    listeners = subprocess.check_output(["ss", "-H", "-ltn", "sport", "=", f":{port}"], text=True)
    addresses = [line.split()[3] for line in listeners.splitlines()]
    if not addresses or any(address not in {f"127.0.0.1:{port}", f"[::1]:{port}"} for address in addresses):
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
