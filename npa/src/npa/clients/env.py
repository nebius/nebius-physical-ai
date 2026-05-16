"""Environment-file rendering helpers."""

from __future__ import annotations

import difflib
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


def merge_env_file_content(current: str, updates: Mapping[str, Any]) -> str:
    """Return env-file content with updates applied while preserving other lines."""
    clean_updates = {
        validate_env_name(str(key)): str(value)
        for key, value in updates.items()
        if value is not None and str(value)
    }
    lines = current.splitlines()
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0] if "=" in line else ""
        if key in clean_updates:
            if key not in seen:
                out.append(f"{key}={clean_updates[key]}")
                seen.add(key)
            continue
        out.append(line)
    for key, value in clean_updates.items():
        if key not in seen:
            out.append(f"{key}={value}")
    return "\n".join(out).rstrip() + ("\n" if out else "")


def _redact_env_file_content(content: str) -> str:
    lines: list[str] = []
    for line in content.splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            lines.append(line)
            continue
        key, value = line.split("=", 1)
        lines.append(f"{key}={redact_value(value)}")
    return "\n".join(lines) + ("\n" if lines else "")


def render_redacted_env_diff(current: str, proposed: str) -> str:
    """Return a unified diff with env values redacted."""
    diff = difflib.unified_diff(
        _redact_env_file_content(current).splitlines(keepends=True),
        _redact_env_file_content(proposed).splitlines(keepends=True),
        fromfile="current",
        tofile="proposed",
    )
    return "".join(diff)
