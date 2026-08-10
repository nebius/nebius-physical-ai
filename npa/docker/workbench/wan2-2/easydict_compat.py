"""Small Apache-2.0 attribute-access mapping used by Wan configuration files."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class EasyDict(dict[str, Any]):
    """A dict whose string keys are also available as attributes."""

    def __init__(self, mapping: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        for key, value in dict(mapping or {}, **kwargs).items():
            self[key] = value

    @classmethod
    def _convert(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and not isinstance(value, cls):
            return cls(value)
        if isinstance(value, list):
            return [cls._convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._convert(item) for item in value)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, self._convert(value))

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
