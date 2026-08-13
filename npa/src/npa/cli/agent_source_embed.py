"""Helpers for embedding dependency-light modules in the remote agent backend."""

from __future__ import annotations

import ast
from pathlib import Path


def embedded_module_source(path: Path) -> str:
    """Read a module and remove declarations invalid inside a concatenated file."""

    raw = path.read_text(encoding="utf-8")
    tree = ast.parse(raw, filename=str(path))
    removable: list[ast.AST] = []
    for index, node in enumerate(tree.body):
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            removable.append(node)
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            removable.append(node)
            continue
        break

    lines = raw.splitlines(keepends=True)
    for node in removable:
        start = int(node.lineno) - 1
        end = int(getattr(node, "end_lineno", node.lineno))
        for line_number in range(start, end):
            # Preserve line numbering for diagnostics in the concatenated
            # backend while removing syntax-invalid future declarations.
            lines[line_number] = "\n" if lines[line_number].endswith("\n") else ""
    return "".join(lines)
