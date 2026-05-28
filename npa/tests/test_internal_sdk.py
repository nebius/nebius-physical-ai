"""Unit tests for `npa._sdk` helper utilities."""

# NOTE: intentionally no `from __future__ import annotations` so that
# parameter annotations stay as live classes (needed by enum coercion).

from enum import Enum
from typing import Optional

import pytest

from npa._sdk import call_cli_callback, make_cli_wrapper


class _Color(str, Enum):
    red = "red"
    blue = "blue"


def test_call_cli_callback_invokes_callback_with_plain_defaults() -> None:
    def callback(name: str = "default", count: int = 3) -> tuple[str, int]:
        return name, count

    assert call_cli_callback(callback) == ("default", 3)


def test_call_cli_callback_accepts_overrides() -> None:
    def callback(name: str = "default", count: int = 3) -> tuple[str, int]:
        return name, count

    assert call_cli_callback(callback, name="custom", count=7) == ("custom", 7)


def test_call_cli_callback_rejects_unexpected_kwargs() -> None:
    def callback(name: str = "x") -> str:
        return name

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        call_cli_callback(callback, bogus=1)


def test_call_cli_callback_raises_for_missing_required_args() -> None:
    def callback(name: str, count: int = 1) -> tuple[str, int]:
        return name, count

    with pytest.raises(TypeError, match="missing required argument"):
        call_cli_callback(callback)


def test_call_cli_callback_unwraps_typer_option_defaults() -> None:
    class FakeOption:
        def __init__(self, default):
            self.default = default

    def callback(name=FakeOption("from-option"), count=FakeOption(5)):
        return name, count

    assert call_cli_callback(callback) == ("from-option", 5)


def test_call_cli_callback_treats_ellipsis_default_as_required() -> None:
    class FakeOption:
        default = ...

    def callback(value=FakeOption()):
        return value

    with pytest.raises(TypeError, match="missing required argument.*value"):
        call_cli_callback(callback)


def test_call_cli_callback_coerces_enum_values() -> None:
    def callback(color: _Color = _Color.red) -> _Color:
        return color

    assert call_cli_callback(callback, color="blue") is _Color.blue
    assert call_cli_callback(callback, color=_Color.red) is _Color.red


def test_call_cli_callback_preserves_none_for_optional() -> None:
    def callback(value: Optional[int] = None) -> object:
        return value

    assert call_cli_callback(callback) is None
    assert call_cli_callback(callback, value=None) is None


def test_make_cli_wrapper_lazily_imports_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    module = types.ModuleType("npa._sdk_lazy_target")

    def add(a: int = 1, b: int = 2) -> int:
        return a + b

    module.add = add
    sys.modules["npa._sdk_lazy_target"] = module
    try:
        wrapper = make_cli_wrapper("npa._sdk_lazy_target", "add", doc="Add two numbers.")

        assert wrapper.__doc__ == "Add two numbers."
        assert wrapper.__name__ == "add"
        assert wrapper(a=10, b=5) == 15
    finally:
        sys.modules.pop("npa._sdk_lazy_target", None)


def test_make_cli_wrapper_strips_cmd_suffix_from_name() -> None:
    import sys
    import types

    module = types.ModuleType("npa._sdk_test_target")

    def example_cmd(value: int = 1) -> int:
        return value * 2

    module.example_cmd = example_cmd
    sys.modules["npa._sdk_test_target"] = module
    try:
        wrapper = make_cli_wrapper("npa._sdk_test_target", "example_cmd", doc="d")
        assert wrapper.__name__ == "example"
        assert wrapper(value=5) == 10
    finally:
        sys.modules.pop("npa._sdk_test_target", None)
