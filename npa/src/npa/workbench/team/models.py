"""Validate administrator-owned workspaces, allocations, and public requests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import UUID

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    model_validator,
)

from .errors import TeamError

Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]{0,47}$")]
GpuCount = Annotated[StrictInt, Field(ge=0)]


def _absolute_path(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("team configuration paths must be absolute")
    return path


AbsolutePath = Annotated[Path, AfterValidator(_absolute_path)]


class Contract(BaseModel):
    """Reject unknown fields in team configuration and public request bodies.

    Args:
        **data: Keyword values for the annotated contract fields below.
    Returns:
        A Contract instance.
    Raises:
        ValidationError: Fields or cross-field policy constraints are invalid.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class IdentityProvider(Contract):
    """Describe the single trusted JWT issuer for this service installation.

    Args:
        **data: Keyword values for the annotated contract fields below.
    Returns:
        A IdentityProvider instance.
    Raises:
        ValidationError: Fields or cross-field policy constraints are invalid.
    """

    issuer: str
    audience: str = Field(min_length=1)
    jwks_url: str
    groups_claim: str = "groups"
    algorithms: tuple[Literal["RS256", "ES256"], ...] = ("RS256",)

    @model_validator(mode="after")
    def validate_urls(self):
        """Require authenticated transport for administrator-selected identity URLs.

        Args:
            None.
        Returns:
            The validated contract instance.
        Raises:
            ValueError: A policy constraint is violated.
        """
        for value in (self.issuer, self.jwks_url):
            parsed = urlsplit(value)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username:
                raise ValueError("identity issuer and JWKS URL must use HTTPS")
            if parsed.fragment:
                raise ValueError("identity URLs cannot include fragments")
        if not self.algorithms:
            raise ValueError("at least one signing algorithm is required")
        return self


class Grant(Contract):
    """Assign a workspace role to an external subject or group.

    Args:
        **data: Keyword values for the annotated contract fields below.
    Returns:
        A Grant instance.
    Raises:
        ValidationError: Fields or cross-field policy constraints are invalid.
    """

    kind: Literal["subject", "group"]
    value: str = Field(min_length=1)
    role: Literal["reader", "runner", "admin"] = "runner"


class StorageGrant(Contract):
    """Reference a separately scoped storage principal without embedding its key.

    Args:
        **data: Keyword values for the annotated contract fields below.
    Returns:
        A StorageGrant instance.
    Raises:
        ValidationError: Fields or cross-field policy constraints are invalid.
    """

    endpoint: str
    bucket: str = Field(min_length=3, pattern=r"^[a-z0-9][a-z0-9.-]+[a-z0-9]$")
    prefix: str = Field(min_length=1)
    credentials_file: AbsolutePath
    principal: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_scope(self):
        """Reject ambiguous object boundaries and insecure storage endpoints.

        Args:
            None.
        Returns:
            The validated contract instance.
        Raises:
            ValueError: A policy constraint is violated.
        """
        parsed = urlsplit(self.endpoint)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username:
            raise ValueError("storage endpoint must use HTTPS without credentials")
        parts = self.prefix.strip("/").split("/")
        if self.prefix.startswith("/") or any(p in ("", ".", "..") for p in parts):
            raise ValueError("storage prefix must be an unambiguous relative path")
        if any(c in self.prefix for c in ("\\", "%", "?", "#")):
            raise ValueError("storage prefix contains unsupported characters")
        return self


class Allocation(Contract):
    """Assign one external subject fixed capacity and storage in a workspace.

    Args:
        **data: Keyword values for the annotated contract fields below.
    Returns:
        A Allocation instance.
    Raises:
        ValidationError: Fields or cross-field policy constraints are invalid.
    """

    subject: str = Field(min_length=1)
    gpu_limit: GpuCount
    clusters: dict[Name, GpuCount]
    storage: StorageGrant
    shared_inputs: tuple[str, ...] = ()
    workload_secrets_file: AbsolutePath | None = None
    source_s3_uri: str = ""

    @model_validator(mode="after")
    def validate_total(self):
        """Keep an individual's cluster allocations within their explicit total.

        Args:
            None.
        Returns:
            The validated contract instance.
        Raises:
            ValueError: A policy constraint is violated.
        """
        if sum(self.clusters.values()) > self.gpu_limit:
            raise ValueError("cluster allocations exceed the individual's GPU limit")
        if self.source_s3_uri:
            from .storage import authorize_uri

            try:
                authorize_uri(self.source_s3_uri, self.storage, self.shared_inputs)
            except TeamError as exc:
                raise ValueError(
                    "runtime source must use an allocated input scope"
                ) from exc
        return self


class Cluster(Contract):
    """Describe an explicitly enrolled Kubernetes connection.

    Args:
        **data: Keyword values for the annotated contract fields below.
    Returns:
        A Cluster instance.
    Raises:
        ValidationError: Fields or cross-field policy constraints are invalid.
    """

    context: str = Field(min_length=1)
    kubeconfig: AbsolutePath


class Workspace(Contract):
    """Group grants, fixed cluster budgets, and personal execution allocations.

    Args:
        **data: Keyword values for the annotated contract fields below.
    Returns:
        A Workspace instance.
    Raises:
        ValidationError: Fields or cross-field policy constraints are invalid.
    """

    grants: tuple[Grant, ...]
    gpu_limit: GpuCount
    gpu_limits: dict[Name, GpuCount]
    allocations: tuple[Allocation, ...]
    disabled: bool = False

    @model_validator(mode="after")
    def validate_allocations(self):
        """Prevent overlapping identities and multiplication of GPU allocations.

        Args:
            None.
        Returns:
            The validated contract instance.
        Raises:
            ValueError: A policy constraint is violated.
        """
        subjects = [item.subject for item in self.allocations]
        if sum(self.gpu_limits.values()) > self.gpu_limit:
            raise ValueError("cluster budgets exceed the workspace GPU limit")
        if len(subjects) != len(set(subjects)):
            raise ValueError("workspace allocations contain duplicate subjects")
        assigned = {name: 0 for name in self.gpu_limits}
        for item in self.allocations:
            for cluster, count in item.clusters.items():
                if cluster not in assigned:
                    raise ValueError("allocation names a cluster outside the workspace")
                assigned[cluster] += count
        if any(count > self.gpu_limits[name] for name, count in assigned.items()):
            raise ValueError(
                "personal GPU allocations exceed a workspace cluster limit"
            )
        return self


class TeamConfig(Contract):
    """Describe an opt-in team installation without changing local operator mode.

    Args:
        **data: Keyword values for the annotated contract fields below.
    Returns:
        A TeamConfig instance.
    Raises:
        ValidationError: Fields or cross-field policy constraints are invalid.
    """

    api_version: Literal["npa.team/v1"] = "npa.team/v1"
    identity: IdentityProvider | None = None
    account_namespace: UUID | None = None
    clusters: dict[Name, Cluster]
    workspaces: dict[Name, Workspace]
    state_dir: AbsolutePath
    sky_endpoint: str
    sky_python: AbsolutePath
    disabled_subjects: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_references(self):
        """Ensure all placement and storage boundaries have explicit owners.

        Args:
            None.
        Returns:
            The validated contract instance.
        Raises:
            ValueError: A policy constraint is violated.
        """
        if self.identity is None and self.account_namespace is None:
            raise ValueError(
                "configure local accounts or an external identity provider"
            )
        for workspace in self.workspaces.values():
            if set(workspace.gpu_limits) - self.clusters.keys():
                raise ValueError("workspace names an unenrolled cluster")
        _validate_storage_boundaries(self.workspaces)
        endpoint = urlsplit(self.sky_endpoint)
        if endpoint.scheme not in ("http", "https") or not endpoint.hostname:
            raise ValueError("SkyPilot endpoint must be an explicit HTTP(S) URL")
        if endpoint.username or endpoint.query or endpoint.fragment:
            raise ValueError("SkyPilot credentials cannot be embedded in its URL")
        if endpoint.hostname not in ("127.0.0.1", "::1"):
            raise ValueError("the private SkyPilot endpoint must bind to loopback")
        if endpoint.port != 46580 or endpoint.path not in ("", "/"):
            raise ValueError(
                "native managed jobs require the private SkyPilot port 46580"
            )
        return self

    @property
    def principal_issuer(self) -> str:
        """Return the permanent ownership domain independent of login credentials.

        Args:
            None.
        Returns:
            Local account domain, or the legacy external ownership domain.
        Raises:
            None.
        """
        if self.account_namespace is not None:
            return f"urn:npa:team:{self.account_namespace}"
        return self.identity.issuer


class Actor(Contract):
    """Carry identity claims only after issuer, signature, and audience verification.

    Args:
        **data: Keyword values for the annotated contract fields below.
    Returns:
        A Actor instance.
    Raises:
        ValidationError: Fields or cross-field policy constraints are invalid.
    """

    issuer: str
    subject: str
    groups: frozenset[str] = frozenset()
    display_name: str | None = None


class SubmitRequest(Contract):
    """Submit an NPA workflow into one authorized workspace and cluster.

    Args:
        **data: Keyword values for the annotated contract fields below.
    Returns:
        A SubmitRequest instance.
    Raises:
        ValidationError: Fields or cross-field policy constraints are invalid.
    """

    workspace: Name
    cluster: Name
    idempotency_key: str = Field(min_length=1, max_length=128)
    workflow: dict


def load_config(path: Path) -> TeamConfig:
    """Load administrator-owned team configuration.

    Args:
        path: Explicit configuration file.
    Returns:
        Validated team configuration.
    Raises:
        TeamError: Configuration is unreadable or invalid.
    """
    try:
        return TeamConfig.model_validate(yaml.safe_load(path.read_text()))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise TeamError("team configuration is unreadable or invalid") from exc


def execution_name(issuer: str, workspace: str, subject: str) -> str:
    """Derive a stable namespace from external identity and workspace.

    Args:
        issuer, workspace, subject: Trusted identity and durable workspace name.
    Returns:
        A Kubernetes-safe name without personal information.
    Raises:
        None.
    """
    identity = json.dumps([issuer, workspace, subject], separators=(",", ":"))
    return "npa-team-" + hashlib.sha256(identity.encode()).hexdigest()[:24]


def _validate_storage_boundaries(workspaces):
    scopes = []
    for workspace in workspaces.values():
        scopes.extend(item.storage for item in workspace.allocations)
    for index, scope in enumerate(scopes):
        for other in scopes[index + 1 :]:
            if scope.principal == other.principal:
                raise ValueError(
                    "personal allocations must use distinct storage principals"
                )
            if (scope.endpoint, scope.bucket) != (other.endpoint, other.bucket):
                continue
            raise ValueError("personal allocations must use distinct storage buckets")
