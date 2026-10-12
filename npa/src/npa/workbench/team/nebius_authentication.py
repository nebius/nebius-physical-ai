"""Verify Nebius human bearer tokens through the fixed IAM profile endpoint."""

from __future__ import annotations

import httpx

from .errors import AuthenticationError
from .models import Actor, NebiusIdentity

NEBIUS_ISSUER = "https://api.nebius.cloud"
PROFILE_ENDPOINT = f"{NEBIUS_ISSUER}/iam/v1/profiles"
PROFILE_TIMEOUT_SECONDS = 5.0


class NebiusProfileVerifier:
    """Resolve a native Nebius human token to one configured tenant membership.

    Args:
        identity: Administrator-selected Nebius tenant and optional federation.
        client: Optional fixed-endpoint HTTP client for deterministic tests.
    Returns:
        A verifier that contacts Nebius IAM for every non-local bearer request.
    Raises:
        None.
    """

    def __init__(self, identity: NebiusIdentity, *, client: httpx.Client | None = None):
        """Prepare a verifier with no redirect, proxy, or ambient-token fallback.

        Args:
            identity: Native provider configuration selected at service startup.
            client: Optional test client; production uses verified TLS by default.
        Returns:
            None.
        Raises:
            None.
        """
        self.identity = identity
        self.http = client or httpx.Client(
            timeout=PROFILE_TIMEOUT_SECONDS,
            verify=True,
            follow_redirects=False,
            trust_env=False,
        )

    def verify(self, authorization: str) -> Actor:
        """Verify one bearer token with Nebius IAM and return its tenant subject.

        Args:
            authorization: Untrusted HTTP Authorization header.
        Returns:
            Provider-verified actor with no imported cloud groups or roles.
        Raises:
            AuthenticationError: The token, provider response, or membership is invalid.
        """
        token = _bearer_token(authorization)
        try:
            response = self.http.get(
                PROFILE_ENDPOINT, headers={"Authorization": f"Bearer {token}"}
            )
        except httpx.HTTPError as exc:
            raise AuthenticationError("Nebius identity could not be verified") from exc
        if not response.is_success:
            raise AuthenticationError("Nebius identity could not be verified")
        try:
            return _profile_actor(response.json(), self.identity)
        except (TypeError, ValueError) as exc:
            raise AuthenticationError("Nebius identity could not be verified") from exc


def _bearer_token(authorization: str) -> str:
    if not isinstance(authorization, str):
        raise AuthenticationError("a valid bearer token is required")
    scheme, separator, token = authorization.partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not token
        or token != token.strip()
        or any(character.isspace() for character in token)
    ):
        raise AuthenticationError("a valid bearer token is required")
    return token


def _profile_actor(payload: object, identity: NebiusIdentity) -> Actor:
    if not isinstance(payload, dict):
        raise ValueError("profile response must be an object")
    if "serviceAccountProfile" in payload or "anonymousProfile" in payload:
        raise ValueError("profile is not a human user")
    profile = payload.get("userProfile")
    if not isinstance(profile, dict):
        raise ValueError("profile does not contain a user")
    _identifier(profile.get("id"))
    if profile.get("userAccountState") != "ACTIVE":
        raise ValueError("global user is not active")
    _verify_federation(profile, identity)
    membership = _configured_membership(profile.get("tenants"), identity.tenant_id)
    subject = _identifier(membership.get("tenantUserAccountId"))
    if membership.get("tenantUserAccountState") != "ACTIVE":
        raise ValueError("tenant user is not active")
    return Actor(issuer=NEBIUS_ISSUER, subject=subject)


def _verify_federation(profile: dict, identity: NebiusIdentity) -> None:
    if identity.federation_id is None:
        return
    federation = profile.get("federationInfo")
    if not isinstance(federation, dict):
        raise ValueError("federation information is missing")
    if federation.get("federationId") != identity.federation_id:
        raise ValueError("federation does not match")


def _configured_membership(tenants: object, tenant_id: str) -> dict:
    if not isinstance(tenants, list):
        raise ValueError("tenant memberships are missing")
    matches = [
        membership
        for membership in tenants
        if isinstance(membership, dict) and membership.get("tenantId") == tenant_id
    ]
    if len(matches) != 1:
        raise ValueError("configured tenant membership is ambiguous")
    return matches[0]


def _identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character.isspace() or ord(character) < 33 for character in value)
    ):
        raise ValueError("profile identifier is malformed")
    return value
