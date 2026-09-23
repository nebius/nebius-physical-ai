"""Serve authenticated mobile chat controls through the desktop HTTPS gateway."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
from pathlib import Path
import sys
import threading
from urllib.parse import parse_qs, urlsplit

try:
    from .chat_rpc import CodexConnection
    from .chat_history import owned_elsewhere
    from .chat_models import available_models, model_selection
    from .chat_delivery import Deliveries
    from .chat_session import authorized as _authorized, mobile_session, session_key
    from .chat_pwa import public_asset
except ImportError:
    from chat_rpc import CodexConnection
    from chat_history import owned_elsewhere
    from chat_models import available_models, model_selection
    from chat_delivery import Deliveries
    from chat_session import authorized as _authorized, mobile_session, session_key
    from chat_pwa import public_asset

_SOURCES = ["vscode"]

_STATIC = {
    "/chat/": ("chat.html", "text/html"),
    "/chat/chat.js": ("chat.js", "text/javascript"),
    "/chat/chat.css": ("chat.css", "text/css"),
}


def _update_created_threads(created, message):
    params = message.get("params", {})
    identifier = params.get("threadId")
    if message["method"] in {"turn/started", "turn/completed"}:
        created.pop(identifier, None)
    thread = created.get(identifier)
    if thread is not None and message["method"] == "thread/settings/updated":
        settings = params["threadSettings"]
        thread.update(model=settings["model"], reasoningEffort=settings["effort"])


def _thread_list_entry(thread):
    entry = dict(thread)
    # Native owners can use an entire prompt as a title; polling needs only labels.
    for field in ("name", "preview"):
        value = entry.get(field)
        if isinstance(value, str) and len(value) > 240:
            entry[field] = value[:239] + "…"
    return entry


def _workspace_choices(threads, default):
    choices = {}
    for thread in [*threads, {"cwd": default}]:
        raw = thread.get("cwd")
        if not isinstance(raw, str) or not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute() or not path.is_dir():
            continue
        path = path.resolve()
        key = str(path)
        updated = thread.get("updatedAt") or thread.get("createdAt") or 0
        previous = choices.get(key)
        if previous and previous["lastUsed"] >= updated:
            continue
        choices[key] = {
            "path": key,
            "name": path.name or key,
            "preferred": path.name == "nebius-physical-ai",
            "lastUsed": updated,
        }
    return sorted(
        choices.values(),
        key=lambda choice: (
            not choice["preferred"],
            -choice["lastUsed"],
            choice["path"],
        ),
    )


def _answer_result(request, body):
    method = request["method"]
    if method in {
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
    }:
        decision = body.get("decision")
        if decision not in {"accept", "decline", "cancel"}:
            raise ValueError("Choose an approval decision.")
        return {"decision": decision}
    if method == "item/tool/requestUserInput":
        answers = body.get("answers", {})
        expected = {q["id"] for q in request["params"]["questions"]}
        if set(answers) != expected or any(
            not isinstance(v, str) for v in answers.values()
        ):
            raise ValueError("Answer every question.")
        return {
            "answers": {key: {"answers": [value]} for key, value in answers.items()}
        }
    if method == "item/permissions/requestApproval":
        return {
            "permissions": request["params"]["permissions"]
            if body.get("decision") == "accept"
            else {},
            "scope": "turn",
        }
    raise ValueError("Complete this specialized request in VS Code.")


class ChatHandler(BaseHTTPRequestHandler):
    """Authenticate every route and expose a bounded set of chat operations.

    Args:
        See BaseHTTPRequestHandler.
    Returns:
        None.
    Raises:
        OSError: The HTTP connection fails.
    """

    def log_message(self, *_):
        """Keep prompts, URLs, and credentials out of access logs.

        Args:
            _: Unused log arguments.
        Returns:
            None.
        Raises:
            None.
        """

    def _guard(self, mutation=False):
        config = self.server.config
        password = Path(config["password_file"]).read_text().strip()
        if not mobile_session(
            self.headers.get("Cookie", ""), session_key(config, password)
        ) and not _authorized(
            self.headers.get("Authorization", ""), config["username"], password
        ):
            self._respond(
                401, {"error": "Sign in with your desktop username and password."}
            )
            return False
        origin = self.headers.get("Origin")
        if (origin and origin != config["origin"]) or (
            mutation and origin != config["origin"]
        ):
            self._respond(403, {"error": "Use this page's authenticated origin."})
            return False
        return True

    def _respond(self, status, body, content_type="application/json"):
        raw = json.dumps(body).encode() if content_type == "application/json" else body
        self.send_response(status)
        self.send_header(
            "Content-Type",
            content_type
            if content_type == "image/png"
            else content_type + "; charset=utf-8",
        )
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'",
        )
        if status == 401 and not self.server.config.get("auth_port"):
            self.send_header(
                "WWW-Authenticate", 'Basic realm="Development desktop", charset="UTF-8"'
            )
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        """Read authenticated static assets and session data.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        path = urlsplit(self.path)
        asset = public_asset(path.path)
        if asset:
            body, mime = asset
            self._respond(200, body, mime)
            return
        if not self._guard():
            return
        try:
            if path.path in _STATIC:
                name, mime = _STATIC[path.path]
                self._respond(200, (Path(__file__).parent / name).read_bytes(), mime)
                return
            query = {k: v[0] for k, v in parse_qs(path.query).items()}
            self._respond(200, self._get_api(path.path, query))
        except (ValueError, RuntimeError, KeyError) as error:
            self._respond(400, {"error": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            return

    def _get_api(self, path, query):
        rpc = self.server.rpc
        if path == "/chat/api/models":
            return {"data": available_models(rpc)}
        if path == "/chat/api/modes":
            modes = rpc.call("collaborationMode/list", {})["data"]
            return {
                "data": [
                    {"mode": mode["mode"], "name": mode.get("name", mode["mode"])}
                    for mode in modes
                ]
            }
        if path == "/chat/api/threads":
            return self._list_threads(query)
        if path == "/chat/api/workspaces":
            return self._workspaces()
        if path == "/chat/api/thread":
            return {"thread": self._thread(query["id"])}
        if path == "/chat/api/turns":
            return self._turns(query)
        if path == "/chat/api/events":
            return self._events(query)
        if path == "/chat/api/state":
            return self._state()
        raise ValueError("Unknown chat route.")

    def _state(self):
        rpc = self.server.rpc
        return {
            "connected": rpc.alive,
            "pending": rpc.pending(),
            "cursor": rpc.sequence,
            "cwd": self.server.config["cwd"],
            "instance": rpc.instance,
            "installationId": self.server.config.get("installation_id"),
            "hostLabel": self.server.config.get("host_label", "VDI"),
            "native": self.server.config.get("backend") == "native",
            "runtime": rpc.call("runtime/status", {})
            if self.server.config.get("backend") == "native"
            else {},
        }

    def _workspaces(self):
        threads = list(self.server.created.values())
        cursor = None
        seen = set()
        while True:
            page = self._list_threads({"cursor": cursor})
            threads.extend(page["data"])
            cursor = page.get("nextCursor")
            if not cursor:
                break
            if cursor in seen:
                raise RuntimeError(
                    "Recent paths could not be loaded. Enter a path below."
                )
            seen.add(cursor)
        return {"data": _workspace_choices(threads, self.server.config["cwd"])}

    def _thread(self, identifier):
        created = self.server.created.get(identifier)
        if created is not None:
            return created
        thread = self.server.rpc.call(
            "thread/read", {"threadId": identifier, "includeTurns": False}
        )["thread"]
        return {**thread, "archived": self._archived(thread)}

    @staticmethod
    def _archived(thread):
        return (
            bool(thread.get("archived"))
            or "archived_sessions" in Path(thread.get("path") or "").parts
        )

    def _turns(self, query):
        if query["id"] in self.server.created:
            return {"data": [], "nextCursor": None}
        return self.server.rpc.call(
            "thread/turns/list",
            {
                "threadId": query["id"],
                "limit": 10,
                "sortDirection": "desc",
                "itemsView": "full",
                "cursor": query.get("cursor"),
            },
        )

    def _events(self, query):
        rpc = self.server.rpc
        cursor, events, reset = rpc.updates(int(query.get("after", "0")))
        return {
            "cursor": cursor,
            "events": events,
            "reset": reset,
            "pending": rpc.pending(),
            "connected": rpc.alive,
            "instance": rpc.instance,
        }

    def _list_threads(self, query):
        page = self.server.rpc.call(
            "thread/list",
            {
                "limit": 50,
                "sortKey": "updated_at",
                "sourceKinds": _SOURCES,
                "modelProviders": [],
                "useStateDbOnly": True,
                "cursor": query.get("cursor"),
                "searchTerm": query.get("search"),
                "archived": query.get("archived") == "true",
            },
        )
        return {
            **page,
            "data": [
                {
                    **_thread_list_entry(thread),
                    "archived": query.get("archived") == "true",
                }
                for thread in page["data"]
            ],
        }

    def do_POST(self):
        """Apply explicit, same-origin chat actions.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        if not self._guard(mutation=True):
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16 * 1024 * 1024:
                raise ValueError("Request body is missing or too large.")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Send JSON data.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Send a JSON object.")
            self._respond(200, self._action(urlsplit(self.path).path, body))
        except (ValueError, RuntimeError, KeyError, TypeError) as error:
            self._respond(400, {"error": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            return

    def _action(self, path, body):
        rpc = self.server.rpc
        if path == "/chat/api/resume":
            return self._resume(body["id"])
        if path == "/chat/api/new":
            return self._new_thread(body)
        if path == "/chat/api/send":
            return self._send(body)
        if path == "/chat/api/settings":
            return self._settings(body)
        if path in {"/chat/api/rename", "/chat/api/archive", "/chat/api/unarchive"}:
            return self._manage_thread(path.rsplit("/", 1)[1], body)
        if path == "/chat/api/stop":
            return rpc.call(
                "turn/interrupt", {"threadId": body["id"], "turnId": body["turnId"]}
            )
        if path == "/chat/api/open" and self.server.config.get("backend") == "native":
            return rpc.call("thread/open", {"threadId": body["id"]})
        if path == "/chat/api/answer":
            request = next(
                (r for r in rpc.pending() if str(r["id"]) == str(body["requestId"])),
                None,
            )
            if request is None:
                raise ValueError("This request has already been resolved.")
            rpc.answer(body["requestId"], _answer_result(request, body))
            return {"answered": True}
        raise ValueError("Unknown chat action.")

    def _manage_thread(self, action, body):
        identifier = body["id"]
        params = {"threadId": identifier}
        if action == "rename":
            name = body.get("name")
            if not isinstance(name, str) or not 1 <= len(name.strip()) <= 200:
                raise ValueError("Enter a chat name between 1 and 200 characters.")
            if any(ord(char) < 32 or ord(char) == 127 for char in name):
                raise ValueError("Use a single line for the chat name.")
            params["name"] = name.strip()
        thread = self._thread(identifier)
        if action == "rename" and self._archived(thread):
            raise ValueError("Restore this chat before renaming it.")
        if self._owned_elsewhere(thread):
            raise ValueError("Close this chat in the older Codex client first.")
        if action != "rename" and thread.get("status", {}).get("type") == "active":
            raise ValueError(
                "Wait for Codex to finish or stop the turn before archiving."
            )
        method = "thread/name/set" if action == "rename" else "thread/" + action
        self.server.rpc.call(method, params)
        if action != "rename":
            self.server.attached.discard(identifier)
            self.server.created.pop(identifier, None)
        elif identifier in self.server.created:
            self.server.created[identifier]["name"] = params["name"]
        return {"ok": True}

    def _new_thread(self, body):
        cwd = Path(body.get("cwd") or self.server.config["cwd"]).expanduser()
        if not cwd.is_dir():
            raise ValueError(
                "Choose an existing project directory on the execution host."
            )
        result = self.server.rpc.call("thread/start", {"cwd": str(cwd.resolve())})
        self.server.attached.add(result["thread"]["id"])
        if self.server.config.get("backend") != "native":
            self.server.created[result["thread"]["id"]] = result["thread"]
        return result

    def _settings(self, body):
        resumed = self._resume(body["id"])
        if self._archived(resumed["thread"]):
            raise ValueError("Restore this chat before changing its settings.")
        if resumed.get("externalOwner"):
            raise ValueError("Close this session in the older Codex client first.")
        params = model_selection(self.server.rpc, body, resumed["thread"])
        self.server.rpc.call("thread/settings/update", params)
        created = self.server.created.get(body["id"])
        if created is not None:
            created.update(model=params["model"], reasoningEffort=params["effort"])
            if "serviceTier" in params:
                created["serviceTier"] = params["serviceTier"]
            if "collaborationMode" in params:
                created["collaborationMode"] = params["collaborationMode"]
                created["mode"] = params["collaborationMode"]["mode"]
        return {"model": params["model"], "effort": params["effort"]}

    def _resume(self, identifier):
        created = self.server.created.get(identifier)
        if created is not None:
            return {"thread": created}
        rpc = self.server.rpc
        thread = self._thread(identifier)
        if self._archived(thread):
            return {"thread": thread}
        if self._owned_elsewhere(thread):
            return {"thread": thread, "externalOwner": True}
        if identifier in self.server.attached:
            return {"thread": thread}
        result = rpc.call(
            "thread/resume", {"threadId": identifier, "excludeTurns": True}
        )
        self.server.attached.add(identifier)
        return result

    def _send(self, body):
        thread = self._thread(body["id"])
        if self._archived(thread):
            raise ValueError("Restore this chat before sending a message.")
        if self._owned_elsewhere(thread):
            raise ValueError(
                "This session is open in an older Codex client. Close it there, then refresh to continue here."
            )
        text = body.get("text", "")
        images = body.get("images", [])
        if not isinstance(text, str) or (not text.strip() and not images):
            raise ValueError("Write a message first.")
        if not isinstance(images, list) or any(
            not isinstance(image, str)
            or not re.fullmatch(
                r"data:image/(png|jpeg|webp);base64,[A-Za-z0-9+/=]+", image
            )
            for image in images
        ):
            raise ValueError("Attach PNG, JPEG, or WebP images.")
        return self._deliver(body, text, images)

    def _owned_elsewhere(self, thread):
        if self.server.config.get("backend") == "native":
            return False
        return owned_elsewhere(thread, self.server.config["socket"])

    def _deliver(self, body, text, images):
        params = {"threadId": body["id"], "input": [{"type": "text", "text": text}]}
        params["input"].extend({"type": "image", "url": image} for image in images)
        identifier = body.get("clientUserMessageId")
        if identifier is not None:
            params["clientUserMessageId"] = identifier
            return self.server.deliveries.execute(
                identifier, body, lambda: self._start_or_steer(body, params)
            )
        return self._start_or_steer(body, params)

    def _start_or_steer(self, body, params):
        if body.get("turnId"):
            params["expectedTurnId"] = body["turnId"]
            return self.server.rpc.call("turn/steer", params)
        result = self.server.rpc.call("turn/start", params)
        self.server.created.pop(body["id"], None)
        return result


def main(config_path):
    """Start a loopback-only service behind the authenticated HTTPS gateway.

    Args:
        config_path: Private runtime JSON configuration.
    Returns:
        None.
    Raises:
        OSError: Runtime files or the listener are unavailable.
        RuntimeError: Codex initialization fails.
    """
    config = json.loads(Path(config_path).read_text())
    server = ThreadingHTTPServer(("127.0.0.1", config["port"]), ChatHandler)
    server.config = config
    server.attached = set()
    server.created = {}
    server.deliveries = Deliveries(Path(config_path).parent / "deliveries.sqlite")

    def notify(message):
        _update_created_threads(server.created, message)

    if config.get("backend") == "native":
        try:
            from .chat_native import native_connection
        except ImportError:
            from chat_native import native_connection
        server.rpc = native_connection(config, config_path, notify)
    else:
        server.rpc = CodexConnection(config["socket"], on_notification=notify)

    def disconnected():
        server.rpc.closed.wait()
        server.shutdown()

    threading.Thread(target=disconnected, daemon=True).start()
    server.serve_forever()
    server.server_close()
    raise RuntimeError("Codex disconnected; the service manager will reconnect.")


if __name__ == "__main__":
    main(sys.argv[1])
