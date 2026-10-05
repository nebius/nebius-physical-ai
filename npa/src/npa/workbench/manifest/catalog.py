"""Project catalog: name@version -> Descriptor.

Registration is the core of onboarding. Adding a tool means adding one
descriptor file — no code changes, no registration edits elsewhere.
"""

from __future__ import annotations

from pathlib import Path

from .schema import Descriptor, load_descriptor


class Catalog:
    def __init__(self, descriptor_dir: str | Path):
        self.descriptor_dir = Path(descriptor_dir)
        self._entries: dict[str, Descriptor] = {}
        self.refresh()

    def refresh(self) -> None:
        self._entries.clear()
        for path in sorted(self.descriptor_dir.glob("*.yaml")):
            desc = load_descriptor(path)
            if desc.id in self._entries:
                raise ValueError(f"duplicate descriptor id {desc.id}")
            self._entries[desc.id] = desc

    def get(self, name: str, version: str) -> Descriptor:
        key = f"{name}@{version}"
        if key not in self._entries:
            raise KeyError(f"unknown tool {key}; known: {sorted(self._entries)}")
        return self._entries[key]

    def list(self) -> list[str]:
        return sorted(self._entries)
