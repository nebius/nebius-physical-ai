"""Serve optional company sign-in and a live view of personal Workbench access."""

import time
from pathlib import Path

from fastapi import Depends, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from .browser_sessions import SESSION_COOKIE, STATE_COOKIE, BrowserSessions
from .errors import AuthenticationError
from .oidc import OidcProvider


def create_browser(config, verifier):
    """Create the optional browser authentication boundary.

    Args:
        config, verifier: Validated installation and trusted token verifier.
    Returns:
        Browser sessions, or None when browser login is disabled.
    Raises:
        AuthenticationError: Identity-provider discovery fails.
    """
    if config.browser_login is None:
        return None
    provider = OidcProvider(config.identity, config.browser_login)
    return BrowserSessions(provider, verifier)


def install_browser(app, sessions, actor):
    """Register fixed login endpoints and a portal with no embedded credentials.

    Args:
        app, sessions, actor: API, browser sessions, and shared authentication dependency.
    Returns:
        None.
    Raises:
        None.
    """

    @app.get("/")
    def portal():
        return HTMLResponse(Path(__file__).with_name("portal.html").read_text())

    _assets(app)

    @app.get("/auth/login")
    def login():
        state, destination = sessions.begin()
        response = RedirectResponse(destination, status_code=303)
        _cookie(response, STATE_COOKIE, state, 600)
        return response

    @app.get("/auth/callback")
    def callback(request: Request):
        return _callback(sessions, request)

    @app.post("/auth/logout")
    def logout(request: Request, identity=Depends(actor)):
        sessions.authorization(request)
        sessions.discard(request.cookies.get(SESSION_COOKIE))
        response = Response(status_code=204)
        response.delete_cookie(SESSION_COOKIE, path="/", secure=True, httponly=True)
        return response


def _assets(app):
    @app.get("/portal.js")
    def javascript():
        return FileResponse(
            Path(__file__).with_name("portal.js"), media_type="text/javascript"
        )

    @app.get("/portal.css")
    def stylesheet():
        return FileResponse(
            Path(__file__).with_name("portal.css"), media_type="text/css"
        )


def _callback(sessions, request):
    query = request.query_params
    if any(len(query.getlist(key)) != 1 for key in ("state", "code")):
        raise AuthenticationError("login callback is incomplete or ambiguous")
    session, expires = sessions.finish(
        query["state"], request.cookies.get(STATE_COOKIE), query["code"]
    )
    sessions.discard(request.cookies.get(SESSION_COOKIE))
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(STATE_COOKIE, path="/", secure=True, httponly=True)
    _cookie(response, SESSION_COOKIE, session, max(0, int(expires - time.time())))
    return response


def _cookie(response, name, value, duration):
    response.set_cookie(
        name,
        value,
        secure=True,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=duration,
    )


def secure_response(response):
    """Prevent caching and cross-origin embedding of private identity and run data.

    Args:
        response: Team API or browser response.
    Returns:
        Response with explicit privacy and browser security headers.
    Raises:
        None.
    """
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    )
    return response
