"""Save verified Workbench connections and resolve credentials without cloud coupling."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .client import TeamClient
from .errors import AuthenticationError, TeamError
from .session_storage import (
    private_record_lock,
    read_private_json,
    remove_private_file,
    write_private_json,
)


class _SignedOutError(AuthenticationError):
    """Distinguish an explicit logout from malformed or missing session state."""


class SavedSession(BaseModel):
    """Describe a verified connection without holding its bearer credential.

    Args:
        endpoint, auth_mode: HTTPS service and local or Nebius authentication.
        nebius_profile: Explicit Nebius CLI profile for native human login.
        nebius_config: Optional isolated Nebius CLI configuration file.
        workspace, cluster: Optional authorized default placement.
        ca_file: Optional administrator-provided TLS CA certificate file.
        verified_subject: Stable Workbench subject verified at login.
    Returns:
        Immutable connection settings safe to display without bearer credentials.
    Raises:
        ValueError: Connection settings or authentication selection is invalid.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    endpoint: str
    auth_mode: Literal["local", "nebius"] = "local"
    nebius_profile: str | None = None
    nebius_config: str | None = None
    workspace: str | None = None
    cluster: str | None = None
    ca_file: str | None = None
    verified_subject: str | None = None

    @field_validator("endpoint")
    @classmethod
    def _endpoint(cls, value):
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or "\\" in value
            or any(character.isspace() for character in value)
        ):
            raise ValueError(
                "team endpoint must use HTTPS without embedded credentials"
            )
        return value.rstrip("/")

    @field_validator("ca_file", "nebius_config")
    @classmethod
    def _certificate(cls, value):
        return str(Path(value).expanduser().absolute()) if value else None

    @model_validator(mode="after")
    def _authentication(self):
        if self.auth_mode == "nebius" and not self.nebius_profile:
            raise ValueError("Nebius login requires an explicit CLI profile")
        if self.auth_mode == "local" and (
            self.nebius_profile is not None or self.nebius_config is not None
        ):
            raise ValueError("local login cannot select a Nebius CLI profile")
        if self.nebius_profile and (
            self.nebius_profile.startswith("-")
            or any(ord(character) < 32 for character in self.nebius_profile)
        ):
            raise ValueError("invalid Nebius CLI profile name")
        return self


def _profile_name(profile):
    if (
        not isinstance(profile, str)
        or profile.lower() == "current"
        or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", profile)
    ):
        raise TeamError(
            "profile must contain only letters, digits, hyphens, underscores"
        )
    return profile


def _valid_token(token):
    if not isinstance(token, str) or not token or token.strip() != token:
        raise AuthenticationError("a personal bearer token is required")
    if "\n" in token or "\r" in token:
        raise AuthenticationError("a personal bearer token is required")
    return token


class SessionStore:
    """Store named private connections, committing local credentials atomically.

    Args:
        root: Override the default NPA_CONFIG_DIR/team directory.
    Returns:
        Private connection store; construction does not access the filesystem.
    Raises:
        None.
    """

    def __init__(self, root: Path | None = None):
        config = os.environ.get("NPA_CONFIG_DIR", "").strip()
        self.root = (
            Path(root)
            if root is not None
            else Path(config or Path.home() / ".npa") / "team"
        )

    def _selected(self, profile):
        if profile is not None:
            return _profile_name(profile)
        try:
            current = read_private_json(self.root, "current.json")
        except FileNotFoundError:
            return "default"
        if not isinstance(current, dict):
            raise TeamError("invalid current session; run npa login again")
        if "profile" in current and current["profile"] is None:
            raise _SignedOutError("signed out; run npa login again")
        return _profile_name(current.get("profile"))

    def selected_profile(self, profile=None) -> str:
        """Resolve an explicit connection name or the active saved selection.

        Args:
            profile: Optional exact saved connection name.
        Returns:
            Validated profile name, defaulting to default for first login.
        Raises:
            TeamError: Name or private selection file is invalid.
        """
        try:
            return self._selected(profile)
        except _SignedOutError:
            return "default"

    def _record(self, profile):
        selected = self._selected(profile)
        try:
            record = read_private_json(self.root, selected + ".json")
        except FileNotFoundError as exc:
            raise AuthenticationError(
                "no saved connection; run npa login first"
            ) from exc
        try:
            if record.get("version") != 1:
                raise ValueError("unsupported session format")
            session = SavedSession.model_validate(record["session"])
            if not session.verified_subject:
                raise ValueError("saved session has no verified account")
            token = record.get("token")
            if session.auth_mode == "local":
                _valid_token(token)
            elif token is not None:
                raise ValueError("native token must not be stored")
            return session, token
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise TeamError("invalid saved connection; run npa login again") from exc

    def save(self, profile, session, *, token=None, make_default=True):
        """Commit a verified connection and optionally make it the active profile.

        Args:
            profile, session: Safe profile name and verified connection settings.
            token: Local credential only; native Nebius tokens must be omitted.
            make_default: Select this profile for subsequent unqualified commands.
        Returns:
            None.
        Raises:
            TeamError: Identity, token, path, or private persistence is invalid.
        """
        selected = _profile_name(profile)
        if not session.verified_subject:
            raise TeamError("verify the account before saving its connection")
        if session.auth_mode == "local":
            _valid_token(token)
        elif token is not None:
            raise TeamError("Nebius tokens must never be saved in a Workbench session")
        record = {"version": 1, "session": session.model_dump(mode="json")}
        if token is not None:
            record["token"] = token
        with private_record_lock(self.root, "profiles.lock"):
            write_private_json(self.root, selected + ".json", record)
            if make_default:
                write_private_json(self.root, "current.json", {"profile": selected})

    def load(self, profile=None) -> SavedSession:
        """Read connection metadata without returning a credential.

        Args:
            profile: Exact profile name or the currently selected profile.
        Returns:
            Immutable saved connection metadata.
        Raises:
            TeamError: The selected session is missing, malformed, or unsafe.
        """
        return self._record(profile)[0]

    def exists(self, profile=None) -> bool:
        """Check for a valid saved connection while retaining unsafe-file errors.

        Args:
            profile: Exact profile name or the currently selected profile.
        Returns:
            Whether the connection exists and is valid.
        Raises:
            TeamError: Existing connection state is malformed or unsafe.
        """
        try:
            self._record(profile)
            return True
        except _SignedOutError:
            return False
        except AuthenticationError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                return False
            raise

    def token(self, profile=None) -> str:
        """Read a saved local key without invoking a cloud login.

        Args:
            profile: Exact profile name or the currently selected profile.
        Returns:
            Local bearer key for the selected connection.
        Raises:
            TeamError: Session is absent, unsafe, or uses native Nebius login.
        """
        session, token = self._record(profile)
        if session.auth_mode != "local":
            raise TeamError("request a fresh token from the saved Nebius CLI profile")
        return token

    def remove(self, profile=None) -> bool:
        """Forget one saved connection and its local credential.

        Args:
            profile: Exact profile name or the currently selected profile.
        Returns:
            Whether a saved profile existed; server-side keys are not revoked.
        Raises:
            TeamError: Private files cannot be safely removed.
        """
        with private_record_lock(self.root, "profiles.lock"):
            try:
                selected = self._selected(profile)
            except _SignedOutError:
                return False
            try:
                active = self._selected(None)
            except _SignedOutError:
                active = None
            if active == selected:
                write_private_json(self.root, "current.json", {"profile": None})
            return remove_private_file(self.root, selected + ".json")


def _workspace_permissions(access):
    workspaces = access.get("workspaces", [])
    if not isinstance(workspaces, list) or any(
        not isinstance(entry, dict)
        or not isinstance(entry.get("name"), str)
        or not entry["name"]
        or not isinstance(entry.get("clusters", {}), dict)
        for entry in workspaces
    ):
        raise TeamError("team service returned invalid workspace permissions")
    return workspaces


def _placement(session, access):
    workspaces = _workspace_permissions(access)
    eligible = [
        entry
        for entry in workspaces
        if entry.get("allocated") and entry.get("role") in {"runner", "admin"}
    ]
    selected = _select_workspace(session, workspaces, eligible)
    cluster = session.cluster
    if selected is None:
        return None, None
    clusters = selected.get("clusters", {})
    if not isinstance(clusters, dict):
        raise TeamError("team service returned invalid cluster permissions")
    if cluster is not None and cluster not in clusters:
        raise TeamError("selected cluster is not allocated in this workspace")
    if cluster is None and len(clusters) == 1:
        cluster = next(iter(clusters))
    return selected["name"], cluster


def _select_workspace(session, workspaces, eligible):
    if session.workspace is not None:
        matches = [
            entry for entry in workspaces if entry.get("name") == session.workspace
        ]
        if len(matches) != 1:
            raise TeamError("selected workspace is not authorized")
        return matches[0]
    if session.cluster is not None:
        eligible = [
            entry for entry in eligible if session.cluster in entry.get("clusters", {})
        ]
        if len(eligible) != 1:
            raise TeamError("select a workspace that allocates the requested cluster")
    return eligible[0] if len(eligible) == 1 else None


def _verified_access(session, token, transport):
    options = {"transport": transport}
    if session.ca_file:
        options["ca_file"] = session.ca_file
    client = TeamClient(session.endpoint, _valid_token(token), **options)
    try:
        access = client.whoami()
    finally:
        client.close()
    if (
        not isinstance(access, dict)
        or not isinstance(access.get("subject"), str)
        or not access["subject"]
    ):
        raise TeamError("team service did not return a verified account")
    return access


def _retain_placement(session, access, store, profile):
    if session.workspace is None and session.cluster is not None:
        return session
    if not store.exists(profile):
        return session
    saved = store.load(profile)
    if (
        saved.endpoint != session.endpoint
        or saved.verified_subject != access["subject"]
    ):
        return session
    workspace = session.workspace if session.workspace is not None else saved.workspace
    cluster = session.cluster
    if cluster is None and workspace == saved.workspace:
        cluster = saved.cluster
    return session.model_copy(update={"workspace": workspace, "cluster": cluster})


def login_session(session, token, *, profile="default", store=None, transport=None):
    """Verify a login and placement before persisting any connection or local key.

    Args:
        session, token: Requested connection and credential to verify online.
        profile, store: Named connection and optional private store override.
        transport: Optional deterministic HTTP transport for tests.
    Returns:
        SavedSession and the verified identity/access response.
    Raises:
        TeamError: Authentication, requested placement, or private persistence fails.
    """
    _profile_name(profile)
    access = _verified_access(session, token, transport)
    store = store or SessionStore()
    session = _retain_placement(session, access, store, profile)
    workspace, cluster = _placement(session, access)
    saved = session.model_copy(
        update={
            "workspace": workspace,
            "cluster": cluster,
            "verified_subject": access["subject"],
        }
    )
    store.save(profile, saved, token=token if saved.auth_mode == "local" else None)
    return saved, access


def resolve_session(
    *,
    profile=None,
    endpoint=None,
    token=None,
    token_supplier: Callable[..., str] | None = None,
    store=None,
):
    """Resolve one connection snapshot without sending saved keys to another endpoint.

    Args:
        profile, endpoint: Saved profile and optional explicit endpoint override.
        token: Explicit credential, required to use a different endpoint.
        token_supplier: Fetch a fresh native token for the saved Nebius profile.
        store: Optional private session store override.
    Returns:
        SavedSession and its current bearer credential.
    Raises:
        TeamError: Connection is absent, unsafe, or would cross endpoint boundaries.
    """
    session, saved_token = (store or SessionStore())._record(profile)
    if endpoint is not None:
        explicit = SavedSession(endpoint=endpoint)
        if explicit.endpoint != session.endpoint:
            if token is None:
                raise TeamError(
                    "endpoint differs from saved login; provide its own credential or run npa login"
                )
            return explicit, _valid_token(token)
    if token is not None:
        return session, _valid_token(token)
    if session.auth_mode == "local":
        return session, saved_token
    if token_supplier is None:
        raise TeamError("native login needs its saved Nebius CLI credential provider")
    options = (
        {"config_file": Path(session.nebius_config)} if session.nebius_config else {}
    )
    return session, _valid_token(token_supplier(session.nebius_profile, **options))


def verify_session_identity(session: SavedSession, access: dict):
    """Require refreshed credentials to retain the account verified during login.

    Args:
        session, access: Saved connection and current authenticated whoami response.
    Returns:
        None.
    Raises:
        AuthenticationError: The verified account changed; a new login is required.
    """
    if session.verified_subject and (
        not isinstance(access, dict)
        or access.get("subject") != session.verified_subject
    ):
        raise AuthenticationError("account changed since login; run npa login again")
