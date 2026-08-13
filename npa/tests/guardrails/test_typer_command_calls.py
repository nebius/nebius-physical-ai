"""Guardrail: Typer commands called as functions must resolve their defaults.

Typer stores each option's default as a ``typer.models.OptionInfo`` sentinel and
relies on Click to replace it with a concrete value before the command runs. A
command invoked directly from Python (``npa agent setup`` -> ``fresh-setup`` ->
``deploy``) skips that step, so every omitted option arrives as an
``OptionInfo``. Those sentinels are silently stringified into Terraform vars
(``server_port="<typer.models.OptionInfo object at 0x...>"``) and every boolean
flag becomes truthy regardless of its declared default.

This guard walks the CLI source, finds Typer commands that are called as plain
functions from elsewhere in the same module, and requires the *callee* to carry
``@resolve_typer_defaults`` (see ``npa/src/npa/cli/_typer_defaults.py``).
"""

from __future__ import annotations

import ast
from pathlib import Path

CLI_ROOT = Path(__file__).resolve().parents[2] / "src" / "npa" / "cli"
DECORATOR_NAME = "resolve_typer_defaults"


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return a flat list of decorator "names" for *node*.

    Handles ``@resolve_typer_defaults``, ``@app.command("x")`` and
    ``@app.command`` alike by reducing each decorator to its dotted source
    fragment.
    """

    names: list[str] = []
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        names.append(ast.unparse(target))
    return names


def _is_typer_command(names: list[str]) -> bool:
    return any(name.endswith(".command") or name.endswith(".callback") for name in names)


def _iter_functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _cli_modules() -> list[Path]:
    return sorted(p for p in CLI_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def test_typer_commands_called_as_functions_resolve_defaults() -> None:
    violations: list[str] = []

    for path in _cli_modules():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - never expected in tracked source
            continue

        commands: dict[str, list[str]] = {}
        for func in _iter_functions(tree):
            names = _decorator_names(func)
            if _is_typer_command(names):
                commands[func.name] = names
        if not commands:
            continue

        for func in _iter_functions(tree):
            for node in ast.walk(func):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                callee = node.func.id
                if callee not in commands or callee == func.name:
                    continue
                if DECORATOR_NAME in commands[callee]:
                    continue
                violations.append(
                    f"{path.relative_to(CLI_ROOT.parents[3])}:{node.lineno}: "
                    f"{func.name}() calls the Typer command {callee}() as a plain "
                    f"function, but {callee}() is not decorated with "
                    f"@{DECORATOR_NAME}"
                )

    assert not violations, (
        "Typer commands invoked as Python functions leak unresolved option "
        "defaults (typer.models.OptionInfo) into downstream code.\n"
        "Fix by adding `@resolve_typer_defaults` (from npa.cli._typer_defaults) "
        "directly under the callee's `@app.command(...)`, or by passing every "
        "argument explicitly.\n\n" + "\n".join(violations)
    )


def test_guard_detects_an_undecorated_command(tmp_path) -> None:
    """The guard must actually fire — pin its detection logic on a sample."""

    sample = tmp_path / "sample.py"
    sample.write_text(
        "import typer\n"
        "app = typer.Typer()\n"
        "\n"
        '@app.command("inner")\n'
        "def inner_cmd(port: int = typer.Option(1, '--port')) -> None:\n"
        "    pass\n"
        "\n"
        '@app.command("outer")\n'
        "def outer_cmd() -> None:\n"
        "    inner_cmd()\n",
        encoding="utf-8",
    )
    tree = ast.parse(sample.read_text(encoding="utf-8"))

    commands = {
        func.name: _decorator_names(func)
        for func in _iter_functions(tree)
        if _is_typer_command(_decorator_names(func))
    }
    assert set(commands) == {"inner_cmd", "outer_cmd"}
    assert DECORATOR_NAME not in commands["inner_cmd"]

    calls = [
        node.func.id
        for func in _iter_functions(tree)
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in commands
        and node.func.id != func.name
    ]
    assert calls == ["inner_cmd"]


def test_known_cross_command_calls_are_covered() -> None:
    """Sanity: the agent chain the guard exists for is actually detected."""

    tree = ast.parse((CLI_ROOT / "agent.py").read_text(encoding="utf-8"))
    commands = {
        func.name: _decorator_names(func)
        for func in _iter_functions(tree)
        if _is_typer_command(_decorator_names(func))
    }
    for name in ("deploy_cmd", "fresh_setup_cmd", "setup_cmd", "destroy_cmd"):
        assert name in commands, f"{name} is no longer a Typer command"
        assert DECORATOR_NAME in commands[name], f"{name} lost @{DECORATOR_NAME}"
