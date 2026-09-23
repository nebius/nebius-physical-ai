"""Hold `isolate_home_config` to its own stated contract.

That fixture repoints module-level constants which were computed from the
operator's home directory at import time, and its docstring asks that new ones
be added to the list. Nothing enforced it, so a constant added later kept
pointing at the real `~/.npa` and the unit suite read whatever the developer or
workbench VM happened to have there.

This checks the property directly rather than the list: during a unit test, no
loaded module may expose a `.npa` path outside the test's temporary HOME.
"""

from __future__ import annotations

import os
import pwd
import sys
from pathlib import Path

import pytest


def _account_home() -> Path:
    """Return the real account home, which `$HOME` cannot mask.

    The fixture redirects `HOME`, so `Path.home()` answers with the temporary
    directory and cannot be used to ask whether a constant escaped. The passwd
    database is not redirected and gives the operator's actual home.

    Returns:
        The invoking account's home directory.
    """

    return Path(pwd.getpwuid(os.getuid()).pw_dir)


# Only modules already imported are inspected, which keeps the check honest.
# Importing them here instead would load them *after* HOME is redirected, so
# their constants would resolve to the temporary home and the leak this guard
# exists to catch would be invisible.
_PRODUCT_PREFIX = "npa."


def _npa_path_attributes() -> list[tuple[str, Path]]:
    """Return every ``.npa`` path constant exposed by a loaded product module.

    Returns:
        ``(dotted name, value)`` for each module-level ``Path`` under ``.npa``.
    """

    found: list[tuple[str, Path]] = []
    for module_name, module in sorted(sys.modules.items()):
        if not module_name.startswith(_PRODUCT_PREFIX) or module is None:
            continue
        for attribute, value in vars(module).items():
            if not isinstance(value, Path) or ".npa" not in str(value):
                continue
            found.append((f"{module_name}.{attribute}", value))
    return found


def test_no_loaded_module_points_at_the_real_npa_directory() -> None:
    """Fail when a home-derived constant escaped the fixture's repoint list.

    The test compares against the account home rather than the redirected
    `HOME`. A constant left over from an earlier test's temporary home is a
    separate and much smaller problem, and failing on it here would report a
    cross-test staleness bug as a credential leak.

    Returns:
        None.
    Raises:
        AssertionError: A loaded module exposes a `.npa` path under the real
            account home, so a unit test could read or write the operator's
            actual configuration and credentials.
    """

    account_home = _account_home()
    escaped = [
        f"  {name} = {value}"
        for name, value in _npa_path_attributes()
        if value.is_relative_to(account_home)
    ]

    assert not escaped, (
        "these constants resolve under the real account home, so a unit test "
        "reads the operator's real ~/.npa:\n"
        + "\n".join(escaped)
        + "\n\nRepoint each one in isolate_home_config in npa/tests/conftest.py."
    )


def test_guard_catches_a_constant_left_pointing_at_the_real_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove the guard fails when a constant does escape.

    Args:
        monkeypatch: Used to plant an escaped constant on a loaded module.
    Returns:
        None.
    """

    import npa.clients.config as config

    monkeypatch.setattr(
        config, "_ESCAPED_FIXTURE", _account_home() / ".npa", raising=False
    )

    with pytest.raises(AssertionError, match="real ~/.npa"):
        test_no_loaded_module_points_at_the_real_npa_directory()


def test_credentials_path_and_config_path_are_inside_the_test_home() -> None:
    """Pin the two paths that would leak real secrets if they were missed.

    Returns:
        None.
    """

    import npa.clients.config as config
    import npa.clients.credentials as credentials

    home = Path(os.environ["HOME"])
    for name, value in (
        ("clients.config.CONFIG_PATH", config.CONFIG_PATH),
        ("clients.config.NPA_CONFIG_DIR", config.NPA_CONFIG_DIR),
        ("clients.credentials.CREDENTIALS_PATH", credentials.CREDENTIALS_PATH),
        ("clients.credentials.NPA_CONFIG_DIR", credentials.NPA_CONFIG_DIR),
    ):
        assert value.is_relative_to(home), f"{name} resolves to {value}"
