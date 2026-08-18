"""No live test may accept a vendor's terms on the operator's behalf.

Some vendor terms cannot be accepted by us: NVIDIA's CUDA runtime terms and
Isaac's EULA are decisions a person makes. A container that refuses until those
are accepted is the mechanism, and a live test that exports them for convenience
makes every subsequent run technically valid and legally meaningless.

LTX's four `NPA_LTX_*` declaration variables used to be listed here. They are
gone from the repo entirely: the LTX-2.x agreement binds by conduct, so a local
`ACCEPT=YES` never formed it, and the entity/use answers were unverifiable
self-certification. The gated-repository entitlement replaced them, and a token
is a credential rather than an answer — `HF_TOKEN` is deliberately not scanned
for here. `NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS` stays, because that one really
is an acceptance, of a different vendor's terms.

This scans the live tier by AST rather than by grep, because the shapes matter
more than the spelling. The version this replaces lived inside the one file it
guarded and matched only `os.environ[...] = ` and `.setenv(...)`, so the
shape a live test would most plausibly reach for — a dict literal passed as
`env=` to `subprocess.run` — went straight through it.

Unit tests are deliberately out of scope. Building a synthetic declaration to
check that the gate parses it is exactly what those tests are for, and none of
them can reach a vendor's servers.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

E2E = Path(__file__).resolve().parents[1] / "e2e"

#: Variables that record a human's acceptance of a vendor's terms.
DECLARATION_ENVS = frozenset(
    {
        "NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS",
        "NPA_WAN_ACCEPT_NVIDIA_RUNTIME_TERMS",
        "OMNI_KIT_ACCEPT_EULA",
        "ISAACSIM_ACCEPT_EULA",
    }
)

#: The names such a variable would plausibly be imported under, so aliasing it
#: does not hide the write.
DECLARATION_ALIASES = frozenset(
    {"ACCEPT_ENV", "ACCEPT_EULA_ENV", "EULA_ENV", "NVIDIA_ACCEPT_ENV"}
)

ENV_WRITE_METHODS = frozenset({"setenv", "setdefault", "putenv", "update"})


def _names_a_declaration(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and node.value in DECLARATION_ENVS:
        return str(node.value)
    if isinstance(node, ast.Name) and node.id in DECLARATION_ALIASES:
        return node.id
    if isinstance(node, ast.Attribute) and node.attr in DECLARATION_ALIASES:
        return node.attr
    return None


def _writes(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Subscript):
                name = _names_a_declaration(target.slice)
                if name:
                    found.append(f"{name} assigned")

        # `monkeypatch.setenv(ACCEPT_ENV, ...)`, `env.setdefault(...)`, ...
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ENV_WRITE_METHODS
            and node.args
        ):
            name = _names_a_declaration(node.args[0])
            if name:
                found.append(f"{name} set via {node.func.attr}()")

        # `{ACCEPT_ENV: "YES", ...}` anywhere — the shape the old check missed.
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if key is None:
                    continue
                name = _names_a_declaration(key)
                if name:
                    found.append(f"{name} in a dict literal")
    return found


@pytest.mark.parametrize(
    "path", sorted(E2E.glob("test_*.py")), ids=lambda path: path.name
)
def test_a_live_test_never_declares_on_the_operators_behalf(path: Path) -> None:
    written = _writes(path)

    assert written == [], (
        f"{path.name} sets a licence declaration ({sorted(set(written))}). A live "
        "test that declares makes every run it performs technically valid and "
        "legally meaningless: only the operator may answer these. Read them from "
        "the environment and skip when they are absent, as the container does."
    )


def test_the_scan_would_notice_each_shape_it_claims_to_cover() -> None:
    """Without this the parametrized test above passes on an empty matcher."""

    import tempfile

    shapes = (
        'import os\nos.environ["NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS"] = "YES"\n',
        'monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")\n',
        'env = {}\nenv.setdefault("ISAACSIM_ACCEPT_EULA", "YES")\n',
        'run(env={"NPA_WAN_ACCEPT_NVIDIA_RUNTIME_TERMS": "YES"})\n',
        'from x import ACCEPT_ENV\nrun(env={ACCEPT_ENV: "YES"})\n',
    )
    with tempfile.TemporaryDirectory() as tmp:
        for index, source in enumerate(shapes):
            probe = Path(tmp) / f"probe_{index}.py"
            probe.write_text(source, encoding="utf-8")
            assert _writes(probe), f"shape {index} slipped through: {source!r}"

        clean = Path(tmp) / "clean.py"
        clean.write_text(
            "import os\n"
            'value = os.environ.get("NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS", "")\n'
            'assert value == "YES"\n',
            encoding="utf-8",
        )
        assert _writes(clean) == [], "reading an acceptance must stay allowed"
