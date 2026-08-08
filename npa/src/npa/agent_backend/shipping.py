"""Render the importable agent-backend package installed during bootstrap.

Keeping the shipped-module manifest and heredoc generation here avoids growing
the already-large CLI bootstrap template.  The CLI supplies one rendered shell
fragment; the remote backend still imports the same standalone Python modules.
"""

from __future__ import annotations

from pathlib import Path


SHIPPED_BACKEND_MODULES = (
    "memory",
    "actions",
    "semantic_router",
    "sim2real_loop",
    "retrieval",
    "trace",
    "foxglove",
    "canonical_mcap",
    "foxglove_cloud",
    "foxglove_routes",
    "artifact_routes",
)

_HEREDOC_MARKER = "PY"


def shipped_backend_module_source(name: str) -> str:
    """Return the full source for one module in the shipped manifest."""

    if name not in SHIPPED_BACKEND_MODULES:
        raise ValueError(f"unknown shipped agent-backend module: {name}")
    return (Path(__file__).parent / f"{name}.py").read_text(encoding="utf-8")


def render_shipped_backend_install(install_root: str = "/opt/npa-agent") -> str:
    """Render shell that installs every shipped module as an importable file."""

    package_root = f"{install_root.rstrip('/')}/agent_backend"
    chunks = [
        f"sudo mkdir -p {package_root}\n",
        f"printf '' | sudo tee {package_root}/__init__.py >/dev/null\n",
    ]
    for name in SHIPPED_BACKEND_MODULES:
        source = shipped_backend_module_source(name)
        if f"\n{_HEREDOC_MARKER}\n" in source:
            raise ValueError(f"{name}.py contains reserved heredoc marker")
        chunks.append(
            f"cat <<'{_HEREDOC_MARKER}' | sudo tee "
            f"{package_root}/{name}.py >/dev/null\n"
            f"{source}\n{_HEREDOC_MARKER}\n"
        )
    return "".join(chunks).rstrip("\n")
