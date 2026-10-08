"""Perform an HTTPS OpenID Connect authorization-code exchange with PKCE."""

import base64
import hashlib
import stat
from urllib.parse import urlencode, urlsplit

import httpx

from .errors import AuthenticationError


class OidcProvider:
    """Use configured discovery and client credentials without following redirects.

    Args:
        identity, browser: Administrator-selected issuer and browser client.
        http: Optional transport for protocol tests.
    Returns:
        An OIDC authorization-code client.
    Raises:
        AuthenticationError: Discovery or client-secret configuration is invalid.
    """

    def __init__(self, identity, browser, *, http=None):
        self.identity, self.browser = identity, browser
        self.http = http or httpx.Client(follow_redirects=False)
        self.redirect_uri = browser.public_url.rstrip("/") + "/auth/callback"
        self.metadata = self._discover()
        self.secret = _secret(browser.client_secret_file)

    def close(self):
        """Release identity-provider HTTP connections.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        self.http.close()

    def authorization_url(self, state, nonce, verifier):
        """Build a fixed-callback authorization URL using an S256 PKCE challenge.

        Args:
            state, nonce, verifier: Independent cryptographically random values.
        Returns:
            Provider authorization URL without tokens or client secrets.
        Raises:
            None.
        """
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        parameters = {
            "response_type": "code",
            "client_id": self.browser.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.browser.scopes),
            "state": state,
            "nonce": nonce,
            "code_challenge_method": "S256",
            "code_challenge": challenge.rstrip(b"=").decode(),
            "prompt": "login",
        }
        return self.metadata["authorization_endpoint"] + "?" + urlencode(parameters)

    def exchange(self, code, verifier):
        """Exchange one authorization code without exposing credentials to callers.

        Args:
            code, verifier: Returned code and server-held PKCE verifier.
        Returns:
            Provider-issued ID token for signature and nonce validation.
        Raises:
            AuthenticationError: Exchange fails or omits the ID token.
        """
        data = {
            "grant_type": "authorization_code",
            "client_id": self.browser.client_id,
            "redirect_uri": self.redirect_uri,
            "code": code,
            "code_verifier": verifier,
        }
        auth = (self.browser.client_id, self.secret) if self.secret else None
        payload = self._json(
            "POST", self.metadata["token_endpoint"], data=data, auth=auth
        )
        token = payload.get("id_token")
        if not isinstance(token, str) or not token:
            raise AuthenticationError("identity provider did not return an ID token")
        return token

    def _discover(self):
        document = self._json(
            "GET",
            self.identity.issuer.rstrip("/") + "/.well-known/openid-configuration",
        )
        if (
            document.get("issuer") != self.identity.issuer
            or document.get("jwks_uri") != self.identity.jwks_url
        ):
            raise AuthenticationError(
                "identity discovery does not match configured issuer and keys"
            )
        for field in ("authorization_endpoint", "token_endpoint"):
            _https_endpoint(document.get(field))
        if "S256" not in document.get("code_challenge_methods_supported", []):
            raise AuthenticationError("identity provider must support S256 PKCE")
        return document

    def _json(self, method, url, **kwargs):
        try:
            response = self.http.request(method, url, **kwargs)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AuthenticationError("identity provider request failed") from exc
        if not isinstance(payload, dict):
            raise AuthenticationError("identity provider returned an invalid response")
        return payload


def _https_endpoint(value):
    parsed = urlsplit(value) if isinstance(value, str) else None
    if (
        parsed is None
        or parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.query
        or parsed.fragment
    ):
        raise AuthenticationError(
            "identity endpoints must use HTTPS without credentials"
        )


def _secret(path):
    if path is None:
        return None
    try:
        if (
            not stat.S_ISREG(path.stat().st_mode)
            or stat.S_IMODE(path.stat().st_mode) != 0o600
        ):
            raise AuthenticationError(
                "browser client secret must be a private mode-0600 file"
            )
        secret = path.read_text().strip()
    except OSError as exc:
        raise AuthenticationError("browser client secret could not be read") from exc
    if not secret:
        raise AuthenticationError("browser client secret is empty")
    return secret
