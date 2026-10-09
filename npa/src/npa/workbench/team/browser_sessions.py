"""Keep one-use login challenges and expiring browser credentials on the server."""

import secrets
import threading
import time

from .errors import AuthenticationError, AuthorizationError

SESSION_COOKIE = "__Host-workbench-session"
STATE_COOKIE = "__Host-workbench-login"


class BrowserSessions:
    """Store opaque browser sessions independently of durable workflow state.

    Args:
        provider, verifier: OIDC code-exchange client and trusted token verifier.
    Returns:
        Thread-safe, process-local browser session storage.
    Raises:
        None.
    """

    def __init__(self, provider, verifier, *, browser=None):
        self.provider, self.verifier = provider, verifier
        self.browser = browser or provider.browser
        self.pending, self.sessions = {}, {}
        self.lock = threading.RLock()

    def begin(self):
        """Create a browser-bound, one-use login challenge.

        Args:
            None.
        Returns:
            State cookie and authorization URL.
        Raises:
            None.
        """
        if self.provider is None:
            raise AuthenticationError("external login is not configured")
        state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
        with self.lock:
            self._prune()
            self.pending[state] = {
                "nonce": nonce,
                "verifier": verifier,
                "expires": time.time() + 600,
            }
        return state, self.provider.authorization_url(state, nonce, verifier)

    def finish(self, state, cookie, code):
        """Consume a bound challenge and accept only the provider's matching login.

        Args:
            state, cookie, code: Callback state, host cookie, and authorization code.
        Returns:
            New opaque session ID and its token expiration.
        Raises:
            AuthenticationError: State, PKCE exchange, token, or nonce is invalid.
        """
        if not state or not cookie or not _matches_secret(state, cookie) or not code:
            raise AuthenticationError("login state could not be verified")
        with self.lock:
            self._prune()
            pending = self.pending.pop(state, None)
        if pending is None:
            raise AuthenticationError("login state has expired or was already used")
        token = self.provider.exchange(code, pending["verifier"])
        claims = self.verifier.claims("Bearer " + token)
        _verify_login_claims(claims, pending["nonce"], self.provider.browser.client_id)
        session = secrets.token_urlsafe(32)
        with self.lock:
            self.sessions[session] = {
                "token": token,
                "expires": float(claims["exp"]),
                "csrf": secrets.token_urlsafe(32),
                "display_name": _display_name(claims),
            }
        return session, float(claims["exp"])

    def authorization(self, request):
        """Resolve a session to its token and require same-origin CSRF proof on writes.

        Args:
            request: HTTP request carrying the opaque session cookie.
        Returns:
            Server-held bearer authorization value.
        Raises:
            AuthenticationError, AuthorizationError: Session or CSRF proof is invalid.
        """
        session = self._authorized_session(request)
        if "token" not in session:
            raise AuthenticationError("this session uses a local access key")
        return "Bearer " + session["token"]

    def actor(self, request):
        """Resolve a browser session while rechecking its original credential.

        Args:
            request: Cookie-authenticated request with CSRF proof on writes.
        Returns:
            Current verified account identity.
        Raises:
            AuthenticationError, AuthorizationError: Session or credential is invalid.
        """
        session = self._authorized_session(request)
        if "key_id" in session:
            return self.verifier.accounts.key_actor(session["key_id"])
        return self.verifier.verify("Bearer " + session["token"])

    def sign_in_key(self, credential):
        """Exchange a local access key for an opaque, revocable browser session.

        Args:
            credential: Secret supplied once in an HTTPS request body.
        Returns:
            Random session cookie and expiration, without retaining the access key.
        Raises:
            AuthenticationError: Local keys are disabled or the credential is invalid.
        """
        if not hasattr(self.verifier, "accounts"):
            raise AuthenticationError("local accounts are not configured")
        actor, key_id = self.verifier.accounts.authenticate(credential)
        session, expires = secrets.token_urlsafe(32), time.time() + 8 * 60 * 60
        with self.lock:
            self._prune()
            self.sessions[session] = {
                "key_id": key_id,
                "expires": expires,
                "csrf": secrets.token_urlsafe(32),
                "display_name": actor.display_name or actor.subject,
            }
        return session, expires

    def _authorized_session(self, request):
        session = self._session(request)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = self.browser.public_url.rstrip("/")
            if request.headers.get("origin") != origin or not _matches_secret(
                request.headers.get("x-workbench-csrf", ""), session["csrf"]
            ):
                raise AuthorizationError("same-origin session proof is required")
        return session

    def details(self, request):
        """Return display identity and CSRF proof after session authentication.

        Args:
            request: Authenticated browser request.
        Returns:
            Verified display name and CSRF proof for same-origin writes.
        Raises:
            AuthenticationError: Session is absent or expired.
        """
        session = self._session(request)
        return {key: session[key] for key in ("display_name", "csrf")}

    def discard(self, session):
        """Invalidate the exact browser session without affecting admitted runs.

        Args:
            session: Opaque session cookie, possibly absent.
        Returns:
            None.
        Raises:
            None.
        """
        with self.lock:
            self.sessions.pop(session, None)

    def _session(self, request):
        with self.lock:
            self._prune()
            session = self.sessions.get(request.cookies.get(SESSION_COOKIE))
        if session is None:
            raise AuthenticationError("sign in to Workbench")
        return session

    def _prune(self):
        now = time.time()
        for collection in (self.pending, self.sessions):
            for key in list(collection):
                if collection[key]["expires"] <= now:
                    del collection[key]


def _verify_login_claims(claims, nonce, client):
    received = claims.get("nonce")
    if not _matches_secret(received, nonce):
        raise AuthenticationError("login nonce could not be verified")
    audiences = claims["aud"]
    if (
        isinstance(audiences, list)
        and len(audiences) > 1
        and claims.get("azp") != client
    ):
        raise AuthenticationError("login authorized party could not be verified")
    if "azp" in claims and claims["azp"] != client:
        raise AuthenticationError("login authorized party could not be verified")


def _display_name(claims):
    for key in ("preferred_username", "name", "sub"):
        if isinstance(claims.get(key), str) and claims[key]:
            return claims[key]


def _matches_secret(first, second):
    if not isinstance(first, str) or not isinstance(second, str):
        return False
    return secrets.compare_digest(first.encode(), second.encode())
