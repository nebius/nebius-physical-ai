"""Fail-closed compatibility surface for Genesis' optional TetGen import.

The public Sim2Real Envgen image intentionally excludes the AGPL ``tetgen``
distribution. Genesis imports that module eagerly even for rigid-body scenes,
although it only constructs ``TetGen`` for deformable tetrahedralization. The
canonical pipeline does not advertise or invoke that capability.
"""

from __future__ import annotations


class TetGen:
    """Reject tetrahedralization in the capability-reduced public runtime."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError(
            "TetGen tetrahedralization is not available in the public Sim2Real "
            "Envgen runtime; use a separately licensed private extension"
        )
