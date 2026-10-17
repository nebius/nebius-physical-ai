"""Remove a token-shaped documentation literal from scikit-image's grass helper.

The pinned scikit-image distribution includes one JWT-shaped string in a
standalone documentation expression in ``skimage.data._fetchers.grass``.  It
is not a runtime credential, but it must not be committed to the image layer.
This script changes only that non-semantic expression and fails closed if the
upstream module's structure changes.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import subprocess


TOKEN_PATTERN = re.compile(r"\beyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")


class ScikitImageSanitizationError(RuntimeError):
    """The pinned module no longer has the one scoped literal we expect."""


def _source_offset(lines: list[str], line: int, column: int) -> int:
    return sum(len(item) for item in lines[: line - 1]) + column


def _grass_expression_with_token(tree: ast.Module, source: str) -> ast.Expr:
    grass_definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "grass"
    ]
    if len(grass_definitions) != 1:
        raise ScikitImageSanitizationError("expected exactly one grass helper")
    body = grass_definitions[0].body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    lines = source.splitlines(keepends=True)
    matches: list[ast.Expr] = []
    for statement in body:
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        start = _source_offset(lines, statement.lineno, statement.col_offset)
        end = _source_offset(lines, statement.end_lineno, statement.end_col_offset)
        if TOKEN_PATTERN.search(source[start:end]):
            matches.append(statement)
    if len(matches) != 1:
        raise ScikitImageSanitizationError(
            "expected exactly one token-shaped standalone grass documentation expression"
        )
    return matches[0]


def sanitize_fetchers_module(path: Path) -> int:
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    target = _grass_expression_with_token(ast.parse(source), source)
    start = _source_offset(lines, target.lineno, target.col_offset)
    end = _source_offset(lines, target.end_lineno, target.end_col_offset)
    replacement, substitutions = TOKEN_PATTERN.subn(
        "[redacted-token]", source[start:end]
    )
    if substitutions != 1:
        raise ScikitImageSanitizationError(
            "expected exactly one token-shaped value in the scoped grass expression"
        )
    sanitized = source[:start] + replacement + source[end:]
    ast.parse(sanitized)
    path.write_text(sanitized)
    return substitutions


def _fetchers_path(python: Path) -> Path:
    result = subprocess.run(
        [
            str(python),
            "-c",
            "from pathlib import Path; import skimage.data._fetchers as module; print(Path(module.__file__).resolve())",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ScikitImageSanitizationError(
            "could not locate the installed scikit-image module"
        )
    path = Path(result.stdout.strip())
    if (
        path.name != "_fetchers.py"
        or path.parent.name != "data"
        or path.parent.parent.name != "skimage"
    ):
        raise ScikitImageSanitizationError(
            "installed scikit-image module path is unexpected"
        )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    sanitize_fetchers_module(_fetchers_path(args.python))
    print("SCIKIT_IMAGE_GRASS_DOCUMENTATION_SANITIZED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
