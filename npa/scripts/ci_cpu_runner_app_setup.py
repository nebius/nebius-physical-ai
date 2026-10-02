"""Complete owner-approved GitHub App registration through a private loopback callback."""

from __future__ import annotations

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import html
import json
import os
from pathlib import Path
import re
import secrets
import threading
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from ci_cpu_runner_auth import _api, _app_jwt, _installation_token, _PERMISSIONS
from ci_cpu_runner_cloud import _load, _save


def _manifest(setup) -> dict:
    return {
        "name": "NPA disposable CI runners",
        "url": f"https://github.com/{setup.repository}",
        "description": "Manage disposable CPU runners for one Workbench repository.",
        "public": False,
        "hook_attributes": {"url": "https://example.invalid/unused", "active": False},
        "redirect_url": setup.url + "/created",
        "setup_url": setup.url + "/installed/" + setup.state,
        "default_permissions": _PERMISSIONS,
        "default_events": [],
    }


def _registration_form(setup) -> str:
    if (setup.root / "github-app-pending.json").exists():
        settings = _load(setup.root / "github-app-pending.json")
        destination = _installation_url(settings)
        return (
            f'<a href="{destination}">Finish installing the existing controller App</a>'
        )
    owner = setup.repository.split("/")[0]
    action = f"https://github.com/organizations/{owner}/settings/apps/new?state={setup.state}"
    manifest = html.escape(json.dumps(_manifest(setup)), quote=True)
    return (
        "<!doctype html><meta charset=utf-8><title>NPA CI controller setup</title>"
        "<h1>Register the CI controller GitHub App</h1>"
        f"<p>Install this private App only on <strong>{html.escape(setup.repository)}</strong>.</p>"
        "<p>Permissions: repository administration write (runner registration), Actions read "
        "(draining), and Actions variables write (routing rollback). No webhook is enabled.</p>"
        "<p>The key is saved privately on this machine for transfer to the cloud controller. "
        "It is never shown in chat, committed, or sent to a worker.</p>"
        f'<form method="post" action="{html.escape(action, quote=True)}">'
        f'<input type="hidden" name="manifest" value="{manifest}">'
        '<button type="submit">Continue to GitHub</button></form>'
    )


def _record_app(setup, query: dict) -> str:
    if not hmac.compare_digest(query.get("state", [""])[0], setup.state):
        raise ValueError("Registration state did not match")
    code = query.get("code", [""])[0]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", code):
        raise ValueError("Missing registration code")
    if (setup.root / "github-app-pending.json").exists():
        return _installation_url(_load(setup.root / "github-app-pending.json"))
    response = _api(f"/app-manifests/{code}/conversions", payload={})
    if response["permissions"] != {**_PERMISSIONS, "metadata": "read"}:
        raise ValueError("Registered App has unexpected permissions")
    key = setup.root / "github-app.pem"
    descriptor = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(response["pem"])
    settings = {
        "app_id": response["id"],
        "private_key_file": str(key),
        "repository": setup.repository,
        "app_slug": response["slug"],
    }
    _save(setup.root / "github-app-pending.json", settings)
    return _installation_url(settings)


def _installation_url(settings: dict) -> str:
    slug = settings.get("app_slug") or _api("/app", token=_app_jwt(settings))["slug"]
    if not re.fullmatch(r"[a-z0-9-]+", slug):
        raise ValueError("GitHub returned an invalid App slug")
    return f"https://github.com/apps/{slug}/installations/new"


def _verify_installation_scope(settings: dict, installation: dict) -> None:
    if installation.get("repository_selection") != "selected":
        raise ValueError("Install the controller App on selected repositories only")
    token = _api(
        f"/app/installations/{installation['id']}/access_tokens",
        token=_app_jwt(settings),
        payload={"permissions": {"metadata": "read"}},
    )["token"]
    repositories = _api("/installation/repositories", token=token)
    names = [item["full_name"].lower() for item in repositories["repositories"]]
    if repositories["total_count"] != 1 or names != [settings["repository"].lower()]:
        raise ValueError(
            "The controller App must be installed only on its one repository"
        )


def _record_installation(setup, query: dict) -> None:
    settings = _load(setup.root / "github-app-pending.json")
    installation = _api(
        f"/repos/{setup.repository}/installation", token=_app_jwt(settings)
    )
    if str(installation["id"]) != query.get("installation_id", [""])[0]:
        raise ValueError("Installation does not match the selected repository")
    if installation["app_id"] != settings["app_id"]:
        raise ValueError("Installation belongs to another App")
    _verify_installation_scope(settings, installation)
    settings["installation_id"] = installation["id"]
    _installation_token(settings, setup.repository)
    _save(setup.root / "github-app.json", settings)


class _SetupHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Callback URLs contain one-use codes; HTTP access logs must omit them.
        pass

    def _respond(self, status, body="", location=None):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; form-action https://github.com; frame-ancestors 'none'",
        )
        if location:
            self.send_header("Location", location)
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        """Handle only the private registration and installation callback routes."""
        setup = self.server.setup
        parsed = urlsplit(self.path)
        if self.headers.get("Host") != urlsplit(setup.url).netloc:
            self._respond(403, "Unexpected host")
            return
        try:
            if parsed.path == "/start/" + setup.state:
                self._respond(200, _registration_form(setup))
            elif parsed.path == "/created":
                destination = _record_app(setup, parse_qs(parsed.query))
                self._respond(303, location=destination)
            elif parsed.path == "/installed/" + setup.state:
                _record_installation(setup, parse_qs(parsed.query))
                self._respond(
                    200,
                    "GitHub App verified. You can close this tab; the cloud migration can continue.",
                )
                threading.Thread(target=self.server.shutdown).start()
            else:
                self._respond(404, "Not found")
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            self._respond(
                400, "Setup could not complete. See the private setup status."
            )
            _save(
                setup.root / "app-setup-error.json",
                {"error_type": type(error).__name__, "message": str(error)},
            )


def _setup_identity(root: Path) -> tuple[int, str]:
    path = root / "app-setup.json"
    if not path.exists():
        return 0, secrets.token_urlsafe(32)
    saved = urlsplit(_load(path)["url"])
    state = saved.path.removeprefix("/start/")
    if saved.hostname != "127.0.0.1" or not re.fullmatch(r"[\w-]{43}", state):
        raise ValueError("Invalid saved loopback setup identity")
    return saved.port, state


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--repository", required=True)
    args = parser.parse_args()
    root = args.state_dir.resolve()
    if root.stat().st_mode & 0o077:
        raise ValueError("Use a private state directory (chmod 700)")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repository):
        raise ValueError("Invalid repository")
    if (root / "github-app.json").exists():
        print("GitHub App is already configured.")
        return
    port, state = _setup_identity(root)
    with HTTPServer(("127.0.0.1", port), _SetupHandler) as server:
        server.setup = SimpleNamespace(
            root=root,
            repository=args.repository,
            state=state,
            url=f"http://127.0.0.1:{server.server_port}",
        )
        url = server.setup.url + "/start/" + server.setup.state
        _save(root / "app-setup.json", {"url": url, "pid": os.getpid()})
        print(
            "GitHub App setup is ready; open the URL saved in private app-setup.json.",
            flush=True,
        )
        server.serve_forever()


if __name__ == "__main__":
    _main()
