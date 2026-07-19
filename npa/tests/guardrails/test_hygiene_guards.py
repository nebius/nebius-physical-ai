from __future__ import annotations

import ast
from pathlib import Path
import warnings

from npa.guardrails.skypilot import (
    scan_for_forbidden_teardown,
    skypilot_launching_scripts_missing_sigterm,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _workflow_and_script_paths() -> list[Path]:
    workflow_dir = REPO_ROOT / "npa" / "workflows" / "workbench" / "skypilot"
    script_dir = REPO_ROOT / "npa" / "scripts"
    return sorted(workflow_dir.glob("*.yaml")) + sorted(script_dir.glob("*.py"))


def _test_paths() -> list[Path]:
    root = REPO_ROOT / "npa" / "tests"
    return sorted(root.rglob("test_*.py")) + sorted(root.rglob("conftest.py"))


def test_no_unsupported_skypilot_down_or_autodown() -> None:
    hits = scan_for_forbidden_teardown(_workflow_and_script_paths())
    assert not hits, "\n".join(f"{hit.path}:{hit.line_number}: {hit.line}" for hit in hits)


def test_teardown_guard_catches_broken_fixture(tmp_path: Path) -> None:
    bad = tmp_path / "bad.sh"
    bad.write_text("sky launch --down task.yaml\n", encoding="utf-8")

    hits = scan_for_forbidden_teardown([bad])

    assert len(hits) == 1
    assert hits[0].line_number == 1


def test_skypilot_launching_scripts_without_sigterm_are_warned() -> None:
    missing = skypilot_launching_scripts_missing_sigterm(sorted((REPO_ROOT / "npa" / "scripts").glob("*.py")))
    for path in missing:
        warnings.warn(
            f"SkyPilot-launching script lacks an explicit SIGTERM hook: {path.relative_to(REPO_ROOT)}",
            UserWarning,
            stacklevel=1,
        )


def test_gpu_tests_skip_only_on_explicit_env_flags() -> None:
    violations: list[str] = []
    for path in _test_paths():
        violations.extend(_cuda_skip_violations(path))
    assert not violations, "\n".join(violations)


def test_gpu_skip_guard_catches_broken_fixture(tmp_path: Path) -> None:
    bad = tmp_path / "test_bad_gpu_skip.py"
    bad.write_text(
        "import pytest\n"
        "import torch\n\n"
        "@pytest.mark.skipif(not torch.cuda.is_available(), reason='no local GPU')\n"
        "def test_gpu():\n"
        "    pass\n",
        encoding="utf-8",
    )

    violations = _cuda_skip_violations(bad)

    assert violations
    assert "local CUDA" in violations[0]


def _cuda_skip_violations(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _is_pytest_mark_skipif(node) and _mentions_local_cuda(node, source):
            violations.append(f"{path}:{node.lineno}: GPU skip depends on local CUDA")
        if _is_pytest_skip(node):
            parent = parents.get(node)
            while parent is not None:
                if isinstance(parent, ast.If) and _mentions_local_cuda(parent.test, source):
                    violations.append(f"{path}:{node.lineno}: GPU skip depends on local CUDA")
                    break
                parent = parents.get(parent)
    return violations


def _is_pytest_mark_skipif(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "skipif"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "mark"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "pytest"
    )


def _is_pytest_skip(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "skip"
        and isinstance(func.value, ast.Name)
        and func.value.id == "pytest"
    )


def _mentions_local_cuda(node: ast.AST, source: str) -> bool:
    segment = ast.get_source_segment(source, node) or ""
    return "cuda.is_available" in segment or "torch.cuda" in segment


def test_shipped_examples_use_registry_placeholder_not_first_party_id() -> None:
    """Shipped BYO examples must not bake in the first-party registry ID.

    Resolver-owned defaults (npa.deploy.images, the image manifests, and ops
    scripts) may reference the concrete `npa-workbench` registry; committed
    example YAMLs and cookbooks must use the `<your-registry-id>` placeholder
    so external users never pull against a registry they cannot access.
    """
    from npa.deploy.images import DEFAULT_CONTAINER_REGISTRY_ID

    example_roots = [
        REPO_ROOT / "npa" / "workflows",
        REPO_ROOT / "docs" / "workbench" / "cookbooks",
        REPO_ROOT / "docs" / "demos",
    ]
    offenders: list[str] = []
    for root in example_roots:
        for path in sorted(root.rglob("*")):
            if path.suffix not in {".yaml", ".yml", ".md", ".json"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if DEFAULT_CONTAINER_REGISTRY_ID in text:
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        "Concrete first-party registry ID found in shipped examples; "
        "use the <your-registry-id> placeholder instead: " + ", ".join(offenders)
    )


def test_monolith_modules_do_not_grow() -> None:
    """Size ratchet for the largest modules.

    These files are already big enough to resist review; new functionality
    belongs in new modules, not appended here. If a change legitimately grows
    one (e.g. mechanical refactor prep), lower other entries or split the file
    and tighten the cap — never raise a cap to make room for features.
    """
    caps = {
        "npa/src/npa/cli/agent.py": 10_100,
        "npa/src/npa/workflows/sim2real_loop.py": 5_800,
        "npa/src/npa/workflows/sim2real/engine.py": 5_600,
        "npa/src/npa/cli/groot/__init__.py": 4_400,
        "npa/src/npa/cli/fiftyone/__init__.py": 4_250,
        "npa/src/npa/cli/cosmos/__init__.py": 4_050,
        "npa/src/npa/cli/isaac_lab/__init__.py": 3_500,
    }
    over = []
    for rel_path, cap in caps.items():
        lines = sum(1 for _ in (REPO_ROOT / rel_path).open())
        if lines > cap:
            over.append(f"{rel_path}: {lines} lines > cap {cap}")
    assert not over, "Monolith size ratchet exceeded — split, don't grow:\n" + "\n".join(over)


def test_no_silent_except_exception_pass() -> None:
    """`except Exception: pass` hides real failures; log at debug or narrow it."""
    offenders = []
    for path in sorted((REPO_ROOT / "npa" / "src").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ExceptHandler)
                and isinstance(node.type, ast.Name)
                and node.type.id == "Exception"
                and len(node.body) == 1
                and isinstance(node.body[0], ast.Pass)
            ):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, (
        "Silent `except Exception: pass` found; log the exception at debug "
        "level or narrow the except type:\n" + "\n".join(offenders)
    )
