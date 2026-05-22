"""Contract tests for the hello-world CI smoke test.

These tests verify that the scaffold file keeps its documented purpose and stays intentionally small, so it remains a CI/pytest discovery sentinel rather than growing hidden behavior.
"""

import ast
from pathlib import Path


SMOKE_TEST_PATH = Path(__file__).with_name("test_hello_world.py")


def _module_tree():
    return ast.parse(SMOKE_TEST_PATH.read_text(encoding="utf-8"))


def test_hello_world_smoke_contract():
    tree = _module_tree()
    docstring = ast.get_docstring(tree) or ""
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

    assert SMOKE_TEST_PATH.exists()
    assert "CI/test-infrastructure smoke test" in docstring
    assert "pytest can discover" in docstring
    assert "execute a test function end-to-end" in docstring
    assert [function.name for function in functions] == ["test_hello_world"]


def test_hello_world_has_no_hidden_dependencies_or_logic():
    tree = _module_tree()
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))

    assert imports == []
    assert len(function.args.args) == 0
    assert len(function.decorator_list) == 0
    assert len(function.body) == 1
    assert isinstance(function.body[0], ast.Assert)
    assert isinstance(function.body[0].test, ast.Constant)
    assert function.body[0].test.value is True
