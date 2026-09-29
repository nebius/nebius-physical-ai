"""Authenticate desktop and chat browsers independently of the Codex engine."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlsplit

try:
    from .chat_pwa import public_asset
    from .chat_session import authorized, mobile_session, session_cookie, session_key
except ImportError:
    from chat_pwa import public_asset
    from chat_session import authorized, mobile_session, session_cookie, session_key

_LOGIN_ASSETS = {
    "/chat/login": ("chat_login.html", "text/html; charset=utf-8"),
    "/chat/login.js": ("chat_login.js", "text/javascript; charset=utf-8"),
    "/chat/login.css": ("chat_login.css", "text/css; charset=utf-8"),
}


def _destination(value):
    if not value.startswith("/") or value.startswith("//"):
        return "/chat/"
    if any(character in value for character in "\\\r\n\x00"):
        return "/chat/"
    if any(ord(character) < 32 for character in value):
        return "/chat/"
    if urlsplit(value).path not in {"/", "/chat/", "/desktop.html"}:
        return "/chat/"
    return value


class LoginHandler(BaseHTTPRequestHandler):
    """Serve login pages and nginx authorization checks on loopback only.

    Args:
        See BaseHTTPRequestHandler.
    Returns:
        None.
    Raises:
        OSError: The HTTP connection fails.
    """

    def log_message(self, *_):
        """Suppress request paths and login details in access logs.

        Args:
            _: Unused HTTP logger arguments.
        Returns:
            None.
        Raises:
            None.
        """

    def _reply(self, status, body=b"", mime="text/plain; charset=utf-8", **headers):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        )
        for key, value in headers.items():
            self.send_header(key.replace("_", "-"), value)
        self.end_headers()
        self.wfile.write(body)

    def _credentials(self):
        config = self.server.config
        password = Path(config["password_file"]).read_text().strip()
        return config, password, session_key(config, password)

    def do_GET(self):
        """Check authentication or serve only public sign-in and install assets.

        Args:
            None.
        Returns:
            None.
        Raises:
            OSError: Private configuration or the connection is unavailable.
        """
        path = urlsplit(self.path).path
        if path == "/auth/check":
            config, password, secret = self._credentials()
            valid = mobile_session(self.headers.get("Cookie", ""), secret)
            valid |= authorized(
                self.headers.get("Authorization", ""), config["username"], password
            )
            self._reply(204 if valid else 401)
            return
        asset = public_asset(path)
        if asset:
            self._reply(200, *asset)
        elif path in _LOGIN_ASSETS:
            name, mime = _LOGIN_ASSETS[path]
            self._reply(200, (Path(__file__).parent / name).read_bytes(), mime)
        else:
            self._reply(404)

    def do_POST(self):
        """Exchange an origin-checked password form for a persistent session.

        Args:
            None.
        Returns:
            None.
        Raises:
            OSError: Private configuration or the connection is unavailable.
        """
        config, password, secret = self._credentials()
        if urlsplit(self.path).path != "/chat/login":
            self._reply(404)
            return
        if self.headers.get("Origin") != config["origin"]:
            self._reply(403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError("Invalid form size")
            body = parse_qs(self.rfile.read(length).decode(), max_num_fields=4)
        except (ValueError, UnicodeDecodeError):
            self._reply(400)
            return
        self._login(body, config, password, secret)

    def _login(self, body, config, password, secret):
        import base64

        supplied = f"{body.get('username', [''])[0]}:{body.get('password', [''])[0]}"
        header = "Basic " + base64.b64encode(supplied.encode()).decode()
        if not authorized(header, config["username"], password):
            self._reply(401, b"The username or password is incorrect.")
            return
        destination = _destination(body.get("next", ["/chat/"])[0])
        self._reply(303, Location=destination, Set_Cookie=session_cookie(secret))


def main():
    """Run the private authentication listener without connecting to Codex.

    Args:
        None; the config path is passed as the first process argument.
    Returns:
        None.
    Raises:
        OSError: Configuration or the loopback listener is unavailable.
    """
    config = json.loads(Path(sys.argv[1]).read_text())
    server = ThreadingHTTPServer(("127.0.0.1", config["auth_port"]), LoginHandler)
    server.config = config
    server.serve_forever()


if __name__ == "__main__":
    main()
