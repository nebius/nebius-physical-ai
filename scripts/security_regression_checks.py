"""Prove the security gate rejects synthetic regressions using real scanners."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from security_dependencies import scan_dependencies
from security_gate import regressions
from security_source import scan_source


def _write_fixture(root: Path, relative: str, content: str) -> Path:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination


def _source_fixture(root: Path) -> None:
    _write_fixture(root, "candidate.py", "eval(user_expression)  # nosec\n")
    _write_fixture(root, ".github/workflows/candidate.yml", """\
name: Synthetic security regression
on: pull_request
permissions: {}
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - run: echo "${{ github.event.pull_request.title }}" # zizmor: ignore[template-injection]
""")


def _dependency_fixture(root: Path) -> None:
    _write_fixture(root, "requirements.txt", "requests==2.19.1\n")
    hashed_pin = "requests==2.19.1 --hash=sha256:" + "0" * 64 + "\n"
    _write_fixture(root, "hashed-requirements.txt", hashed_pin)
    _write_fixture(root, "commented-requirements.txt", "requests==2.19.1\t# synthetic pin\n")
    lock = {
        "name": "synthetic-security-regression", "version": "1.0.0",
        "lockfileVersion": 3, "requires": True,
        "packages": {
            "": {"dependencies": {"minimist": "1.2.0"}},
            "node_modules/minimist": {"version": "1.2.0"},
        },
    }
    _write_fixture(root, "package-lock.json", json.dumps(lock))


def _require_finding(findings: list[dict], scanner: str, rule: str) -> None:
    matched = [item for item in findings if item["scanner"] == scanner and item["rule"] == rule]
    if not matched:
        raise AssertionError(f"Real {scanner} scan missed synthetic {rule} regression")
    if regressions([], matched) != matched:
        raise AssertionError(f"Gate did not reject {scanner} regression")


def _check_source(root: Path, output: Path) -> dict:
    _source_fixture(root)
    baseline = scan_source(root, output / "source-initial")
    _require_finding(baseline, "bandit", "B307")
    _require_finding(baseline, "zizmor", "template-injection")
    if regressions(baseline, baseline):
        raise AssertionError("Existing source findings must be tolerated")
    _write_fixture(root, "candidate.py", "eval(user_expression)  # nosec\n" * 2)
    duplicate = scan_source(root, output / "source-duplicate")
    _require_finding(regressions(baseline, duplicate), "bandit", "B307")
    (root / "candidate.py").rename(root / "moved.py")
    moved = scan_source(root, output / "source-moved")
    _require_finding(regressions(duplicate, moved), "bandit", "B307")
    return {"initial_findings": len(baseline), "duplicate_rejected": True,
            "moved_rejected": True, "existing_tolerated": True,
            "inline_suppressions_ignored": True}


def _check_parse_failure(root: Path, output: Path) -> None:
    _write_fixture(root, "broken.py", "def incomplete(\n")
    try:
        scan_source(root, output / "source-parse-failure")
    except RuntimeError as error:
        if "parse" not in str(error).lower() and "scan" not in str(error).lower():
            raise
    else:
        raise AssertionError("Invalid Python must fail the gate instead of disappearing")


def _check_dependencies(root: Path, output: Path) -> dict:
    _dependency_fixture(root)
    findings = scan_dependencies(root, output / "dependencies", output / "cache")
    declarations = (("requirements.txt", "requests"), ("package-lock.json", "minimist"),
                    ("hashed-requirements.txt", "requests"), ("commented-requirements.txt", "requests"))
    for manifest, package in declarations:
        matched = [item for item in findings if item["path"] == manifest
                   and item["identity"].startswith(f"{package}==")]
        if not matched or regressions([], matched) != matched:
            raise AssertionError(f"Gate missed known vulnerable synthetic {package} dependency")
    if regressions(findings, findings):
        raise AssertionError("Existing dependency findings must be tolerated")
    return {"findings": len(findings), "python_rejected": True,
            "npm_rejected": True, "existing_tolerated": True}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Private reports directory outside the checkout")
    return parser.parse_args()


def _private_output(directory: Path) -> Path:
    output = directory.resolve()
    repository = Path.cwd().resolve()
    if output == repository or repository in output.parents:
        raise ValueError("Keep security regression reports outside the checkout")
    os.umask(0o077)
    output.mkdir(parents=True, exist_ok=True)
    metadata = output.stat()
    if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise ValueError("Use an owner-only security regression report directory")
    return output


def main() -> int:
    """Run safe scanner workloads and retain a machine-readable proof report.

    Args:
        None; configuration comes from command-line arguments.
    Returns:
        Zero after all real scanner and comparison assertions pass.
    Raises:
        AssertionError: A scanner or comparator fails to reject a fixture.
        OSError: Private evidence cannot be created.
        RuntimeError: An actual scanner cannot complete.
        ValueError: The report directory is inside the checkout or not private.
    """
    output = _private_output(_arguments().output_dir)
    with tempfile.TemporaryDirectory(prefix="synthetic-", dir=output) as directory:
        root = Path(directory)
        source = _check_source(root / "source", output)
        dependencies = _check_dependencies(root / "dependencies", output)
        _check_parse_failure(root / "invalid", output)
    summary = {"source": source, "dependencies": dependencies,
               "parse_failure_rejected": True, "fixtures_executed_or_installed": False}
    (output / "regression-proof.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Real scanner regression checks passed: Python, Actions, Python/npm dependencies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
