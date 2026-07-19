"""Unit tests for shared project/workbench config resolution guards."""

from __future__ import annotations

import pytest


def test_ambiguous_project_without_default_errors_with_aliases():
    from npa.clients.config import ConfigError, _resolve_project_section

    yml = {"projects": {"proj-a": {"workbenches": {}}, "proj-b": {"workbenches": {}}}}
    with pytest.raises(ConfigError, match="-p/--project"):
        _resolve_project_section(yml, None)


def test_single_project_without_default_resolves_unambiguously():
    from npa.clients.config import _resolve_project_section

    yml = {"projects": {"only": {"workbenches": {"wb": {"x": 1}}}}}
    assert _resolve_project_section(yml, None) == {"workbenches": {"wb": {"x": 1}}}


def test_ambiguous_workbench_without_default_errors_with_aliases():
    from npa.clients.config import ConfigError, _resolve_workbench_in_project

    proj = {"workbenches": {"wb-a": {"x": 1}, "wb-b": {"x": 2}}}
    with pytest.raises(ConfigError, match="-n/--name"):
        _resolve_workbench_in_project(proj, None, {})


def test_single_workbench_without_default_resolves_unambiguously():
    from npa.clients.config import _resolve_workbench_in_project

    proj = {"workbenches": {"only": {"x": 1}}}
    assert _resolve_workbench_in_project(proj, None, {}) == {"x": 1}
