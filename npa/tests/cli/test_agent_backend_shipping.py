"""Focused tests for the shipped agent-backend bootstrap seam."""

from __future__ import annotations

import re

import pytest

from npa.agent_backend.shipping import (
    SHIPPED_BACKEND_MODULES,
    render_shipped_backend_install,
    shipped_backend_module_source,
)


def test_rendered_install_contains_exact_compilable_module_sources() -> None:
    script = render_shipped_backend_install()
    assert script.count("sudo mkdir -p /opt/npa-agent/agent_backend") == 1
    for name in SHIPPED_BACKEND_MODULES:
        match = re.search(
            rf"cat <<'PY' \| sudo tee /opt/npa-agent/agent_backend/{name}\.py "
            r">/dev/null\n(?P<body>.*?)\nPY\n?",
            script,
            flags=re.DOTALL,
        )
        assert match, f"missing shipped module {name}"
        source = shipped_backend_module_source(name)
        assert match.group("body").rstrip("\n") == source.rstrip("\n")
        compile(match.group("body"), f"agent_backend/{name}.py", "exec")


def test_unknown_shipped_module_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown shipped agent-backend module"):
        shipped_backend_module_source("not_in_manifest")
