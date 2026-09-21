"""Validate operator-owned specialist profiles and keep secrets out of configuration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Operation(BaseModel):
    """An operator-authorized command selected by name, never by model-supplied argv.

    Args:
        argv: Fixed arguments; {python}, {workspace}, {task_id}, {run_id} expand locally.
        description: What this operation does and how to interpret its result.
        pass_env: Explicit environment names to forward to the command.
    Returns:
        Validated command policy.
    Raises:
        ValueError: Arguments are empty or a placeholder is unsupported.
    """

    model_config = ConfigDict(extra="forbid")
    argv: list[str] = Field(min_length=1)
    description: str = Field(min_length=1)
    pass_env: list[str] = Field(default_factory=list)

    @field_validator("argv")
    @classmethod
    def _argv(cls, values):
        import string

        for value in values:
            if not value or "\0" in value:
                raise ValueError("command arguments must be nonempty")
            for _, name, format_spec, conversion in string.Formatter().parse(value):
                if name is not None and name not in {
                    "python",
                    "workspace",
                    "task_id",
                    "run_id",
                }:
                    raise ValueError("unsupported command placeholder")
                if format_spec or conversion:
                    raise ValueError("command format specifications are unsupported")
        return values


class Profile(BaseModel):
    """Bind a specialist to one explicit endpoint, workspace and tool policy.

    Args:
        name, description, instructions: Specialist identity and role.
        model, base_url, key_env, model_options: Explicit inference configuration.
        workspace, read_paths, write_paths, operations: Operator-owned grants.
    Returns:
        Validated profile without credential values.
    Raises:
        ValueError: Endpoint, scopes, identity or model options are unsafe.
    """

    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    description: str = Field(min_length=1)
    instructions: str = ""
    model: str = Field(min_length=1)
    base_url: str = "https://api.tokenfactory.nebius.com/v1"
    key_env: str = Field(
        default="NEBIUS_TOKEN_FACTORY_KEY", pattern=r"^[A-Z][A-Z0-9_]*$"
    )
    model_options: dict = Field(default_factory=dict)
    workspace: Path
    read_paths: list[str] = Field(default_factory=list)
    write_paths: list[str] = Field(default_factory=list)
    operations: dict[str, Operation] = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def _endpoint(cls, value):
        url = urlsplit(value)
        local = url.hostname in {"localhost", "127.0.0.1", "::1"}
        if (
            not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("endpoint must not contain credentials, query or fragment")
        if url.scheme != "https" and not (local and url.scheme == "http"):
            raise ValueError("endpoint requires HTTPS, except on loopback")
        return value.rstrip("/")

    @field_validator("read_paths", "write_paths")
    @classmethod
    def _scopes(cls, values):
        for value in values:
            path = PurePosixPath(value)
            if (
                not value
                or path.is_absolute()
                or ".." in path.parts
                or ".git" in path.parts
            ):
                raise ValueError("scopes must be relative paths outside .git")
            if value in {".", "/"}:
                raise ValueError(
                    "grant explicit source directories, not the workspace root"
                )
        return values

    @field_validator("model_options")
    @classmethod
    def _options(cls, value):
        allowed = {"reasoning_effort", "chat_template_kwargs", "max_tokens"}
        if value.keys() - allowed:
            raise ValueError(
                "model_options can set reasoning/template options or max_tokens"
            )
        return value


class TeamConfig(BaseModel):
    """Configure one host's durable team without bundling secrets or remote services.

    Args:
        state_directory: Private storage outside every editable workspace.
        profiles: Specialists with disjoint workspaces.
        default_profile: Fallback for automatic routing.
        router: Explicit assignment by default, optionally Jev.
        token_env: Bearer credential required by the HTTP service.
    Returns:
        Validated team configuration.
    Raises:
        ValueError: Names, workspace ownership or state placement conflict.
    """

    model_config = ConfigDict(extra="forbid")
    state_directory: Path
    profiles: list[Profile] = Field(min_length=1)
    default_profile: str
    router: Literal["explicit", "jev"] = "explicit"
    token_env: str = "NPA_SPECIALISTS_TOKEN"

    @model_validator(mode="after")
    def _layout(self):
        names = [profile.name for profile in self.profiles]
        if (
            len(set(names)) != len(names)
            or self.default_profile not in names
            or "none" in names
        ):
            raise ValueError("profile names must be unique with a valid default")
        roots = [profile.workspace.expanduser().resolve() for profile in self.profiles]
        state = self.state_directory.expanduser().resolve()
        for index, root in enumerate(roots):
            if Path(__file__).resolve().is_relative_to(root):
                raise ValueError("install the controller outside editable workspaces")
            if state.is_relative_to(root) or root.is_relative_to(state):
                raise ValueError("state and workspace directories must be separate")
            for other in roots[index + 1 :]:
                if root.is_relative_to(other) or other.is_relative_to(root):
                    raise ValueError("specialists require disjoint workspaces")
        return self

    def profile(self, name: str) -> Profile:
        """Resolve an exact configured specialist.

        Args: name: Specialist name.
        Returns: The matching profile.
        Raises: ValueError: The profile does not exist.
        """
        for profile in self.profiles:
            if profile.name == name:
                return profile
        raise ValueError("unknown specialist")


def load_config(path: str | Path) -> TeamConfig:
    """Read a JSON operator configuration without reading credential values.

    Args: path: Local configuration file.
    Returns: Validated team configuration with absolute directories.
    Raises: ValueError, OSError: The file or its configuration is invalid.
    """
    source = Path(path).expanduser().resolve(strict=True)
    config = TeamConfig.model_validate_json(source.read_text())
    config.state_directory = config.state_directory.expanduser().resolve()
    for profile in config.profiles:
        profile.workspace = profile.workspace.expanduser().resolve(strict=True)
    return config


def fingerprint(profile: Profile) -> str:
    """Bind durable tasks to the exact model and tool policy that created them.

    Args: profile: Validated profile.
    Returns: SHA-256 of the canonical profile.
    Raises: None.
    """
    body = json.dumps(profile.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


def private_directory(path: Path) -> Path:
    """Create private runtime storage without following a final symlink.

    Args: path: Operator-selected state directory.
    Returns: Absolute private directory.
    Raises: ValueError, OSError: Storage is shared, a symlink or owned by another user.
    """
    if path.is_symlink():
        raise ValueError("state directory must not be a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("state directory must be owned by this user and mode 0700")
    return path.resolve()
