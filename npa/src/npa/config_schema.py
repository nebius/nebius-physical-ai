"""Shared schema for strictly validated NPA configuration sections."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SKYPILOT_RUNTIME_CONFIG_KEYS = frozenset(
    {"sky_bin", "global_config_path", "isolated_config_dir"}
)
SKYPILOT_METADATA_CONFIG_KEYS = frozenset({"controller_owner"})
SKYPILOT_CONFIG_KEYS = SKYPILOT_RUNTIME_CONFIG_KEYS | SKYPILOT_METADATA_CONFIG_KEYS

# Every section listed here is rejected on unknown keys by at least one reader.
# Config writers and those readers both consume this registry so their contracts
# cannot drift independently.
STRICT_CONFIG_SECTION_KEYS: Mapping[str, frozenset[str]] = {
    "skypilot": SKYPILOT_CONFIG_KEYS,
}


def unknown_config_keys(section: str, value: Mapping[str, Any]) -> list[str]:
    """Return keys outside the shared allowlist for a strict section."""

    allowed = STRICT_CONFIG_SECTION_KEYS.get(section)
    if allowed is None:
        return []
    return sorted(str(key) for key in set(value) - allowed)
