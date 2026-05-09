"""Environment-file rendering helpers."""

from __future__ import annotations

import re
from typing import Any, Mapping

ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_env_name(name: str) -> str:
    if not ENV_NAME_RE.fullmatch(name):
        raise ValueError(f"Invalid environment variable name: {name!r}")
    return name


def shell_quote_env_value(value: Any) -> str:
    """Return a single-quoted shell literal for an env-file value."""
    text = str(value)
    return "'" + text.replace("'", "'\\''") + "'"


def render_shell_env_file(env: Mapping[str, Any], *, export: bool = False) -> str:
    """Render shell-sourceable env lines with no interpolation risk."""
    prefix = "export " if export else ""
    lines: list[str] = []
    for key, value in env.items():
        if value is None:
            continue
        name = validate_env_name(str(key))
        lines.append(f"{prefix}{name}={shell_quote_env_value(value)}")
    return "\n".join(lines) + ("\n" if lines else "")


def render_docker_env_file(env: Mapping[str, Any]) -> str:
    """Render Docker --env-file content without shell interpolation."""
    lines: list[str] = []
    for key, value in env.items():
        if value is None:
            continue
        name = validate_env_name(str(key))
        text = str(value)
        if "\n" in text or "\r" in text:
            raise ValueError(f"Environment variable {name!r} contains a newline")
        lines.append(f"{name}={text}")
    return "\n".join(lines) + ("\n" if lines else "")


def redact_value(value: Any) -> str:
    text = str(value)
    if not text:
        return ""
    return f"{text[:4]}****"


def redacted_env(env: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(key): redact_value(value)
        for key, value in env.items()
        if value is not None
    }


def render_redacted_env_file(env: Mapping[str, Any]) -> str:
    return render_shell_env_file(redacted_env(env))
