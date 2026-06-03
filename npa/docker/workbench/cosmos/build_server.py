"""Generate the embedded Cosmos FastAPI server during Docker builds."""

from __future__ import annotations

import ast
import os
from importlib import metadata
from pathlib import Path
from typing import Any


COSMOS_CLI_SOURCE = Path("/opt/npa/src/npa/cli/cosmos/__init__.py")
OUTPUT_PATH = Path("/opt/cosmos/server.py")
_GENERATOR_CONSTANTS = {
    "COSMOS_HOME",
    "COSMOS_DATA_HOME",
    "COSMOS_MODEL_DIR",
    "DEFAULT_MODEL",
}


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    if isinstance(node, ast.AnnAssign):
        return {node.target.id} if isinstance(node.target, ast.Name) else set()
    return {target.id for target in node.targets if isinstance(target, ast.Name)}


def _load_generator(source_path: Path) -> dict[str, Any]:
    source = source_path.read_text()
    module = ast.parse(source, filename=str(source_path))
    body: list[ast.stmt] = []

    for node in module.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if _assigned_names(node) & _GENERATOR_CONSTANTS:
                body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "_build_server_py":
            body.append(node)

    namespace: dict[str, Any] = {}
    extracted = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(extracted)
    exec(compile(extracted, str(source_path), "exec"), namespace)
    return namespace


def main() -> None:
    expected = os.environ["COSMOS_VERSION"]
    actual = metadata.version("cosmos-predict2")
    if actual != expected:
        raise RuntimeError(f"expected cosmos-predict2 {expected}, found {actual}")

    namespace = _load_generator(COSMOS_CLI_SOURCE)
    server_py = namespace["_build_server_py"](namespace["DEFAULT_MODEL"])
    OUTPUT_PATH.write_text(server_py)
    print(f"COSMOS_BUILD_IMPORT_OK {actual}")


if __name__ == "__main__":
    main()
