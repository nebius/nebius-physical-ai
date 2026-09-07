"""Verify differential security findings and snapshot/report failure boundaries."""

from __future__ import annotations

import importlib
import io
import json
import subprocess
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def security_modules(monkeypatch: pytest.MonkeyPatch):
    """Import the security gate from the checked-out trusted scripts.

    Args:
        monkeypatch: Isolates the scripts import path.
    Returns:
        The gate and dependency adapter modules.
    Raises:
        ImportError: A required checked-out script cannot be loaded.
    """
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("security_gate"), importlib.import_module("security_dependencies")


def _finding(path: str = "module.py", line: int = 4) -> dict:
    return {"scanner": "bandit", "path": path, "rule": "B307",
            "identity": "synthetic-expression", "line": line, "message": "Synthetic finding"}


@pytest.mark.parametrize("baseline_count,candidate_count,added_count", [
    (1, 1, 0), (1, 0, 0), (0, 1, 1), (1, 2, 1), (2, 1, 0),
])
def test_comparison_counts_occurrences(security_modules, baseline_count, candidate_count, added_count):
    """Count duplicate findings without treating removals as regressions.

    Args:
        security_modules: Checked-out gate modules.
        baseline_count: Number of baseline occurrences.
        candidate_count: Number of candidate occurrences.
        added_count: Expected unmatched occurrences.
    Returns:
        None.
    Raises:
        AssertionError: Duplicate or removal handling changes.
    """
    gate, _ = security_modules
    findings = gate.regressions([_finding()] * baseline_count, [_finding()] * candidate_count)
    assert len(findings) == added_count


def test_comparison_tolerates_line_movement_but_rejects_path_movement(security_modules):
    """Bind findings to their file while tolerating unrelated inserted lines.

    Args:
        security_modules: Checked-out gate modules.
    Returns:
        None.
    Raises:
        AssertionError: Moving vulnerable code to another file is tolerated.
    """
    gate, _ = security_modules
    assert gate.regressions([_finding()], [_finding(line=40)]) == []
    moved = _finding(path="different.py")
    assert gate.regressions([_finding()], [moved]) == [moved]


@pytest.mark.parametrize("field", ["scanner", "rule", "identity"])
def test_comparison_rejects_changed_finding_identity(security_modules, field):
    """Reject changed source expressions and scanner/rule identities.

    Args:
        security_modules: Checked-out gate modules.
        field: Identity component changed by the candidate.
    Returns:
        None.
    Raises:
        AssertionError: A changed identity cancels a baseline occurrence.
    """
    gate, _ = security_modules
    changed = dict(_finding(), **{field: "changed"})
    assert gate.regressions([_finding()], [changed]) == [changed]


@pytest.mark.parametrize("nested", [False, True])
def test_working_snapshot_rejects_symlink_escapes(security_modules, monkeypatch, tmp_path, nested):
    """Reject leaf and parent-directory symlinks before copying outside bytes.

    Args:
        security_modules: Checked-out gate modules.
        monkeypatch: Supplies the tracked path inventory.
        tmp_path: Isolated public synthetic files.
        nested: Whether a parent directory carries the symlink.
    Returns:
        None.
    Raises:
        AssertionError: Snapshot materialization follows an external symlink.
    """
    gate, _ = security_modules
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "fixture.py").write_text("synthetic = True\n")
    relative = "nested/fixture.py" if nested else "fixture.py"
    link = source / ("nested" if nested else "fixture.py")
    link.symlink_to(outside if nested else outside / "fixture.py", target_is_directory=nested)
    monkeypatch.setattr(gate, "_git", lambda *arguments: relative.encode() + b"\0")
    with pytest.raises(ValueError, match="symlink|outside|contain"):
        gate._snapshot_working(source, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot" / relative).exists()


def test_working_snapshot_preserves_scannable_files(security_modules, monkeypatch, tmp_path):
    """Copy candidate files while allowing deleted files and known aliases.

    Args:
        security_modules: Checked-out gate modules.
        monkeypatch: Supplies the tracked path inventory.
        tmp_path: Isolated source and snapshot directories.
    Returns:
        None.
    Raises:
        AssertionError: A real source is omitted or alias handling regresses.
    """
    gate, _ = security_modules
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("synthetic = True\n")
    paths = b"module.py\0deleted.py\0.agents/skills\0.claude/skills\0"
    monkeypatch.setattr(gate, "_git", lambda *arguments: paths)
    destination = tmp_path / "snapshot"
    gate._snapshot_working(source, destination)
    assert (destination / "module.py").read_text() == "synthetic = True\n"
    assert not (destination / "deleted.py").exists()


def _blob_batch(digest: str, content: bytes) -> bytes:
    header = f"{digest} blob {len(content)}\n".encode()
    return header + content + b"\n"


@pytest.mark.parametrize("mode,kind", [("120000", "blob"), ("160000", "commit")])
def test_revision_snapshot_rejects_unsupported_source_entries(security_modules, monkeypatch, tmp_path, mode, kind):
    """Fail closed on source symlinks and submodules outside the policy.

    Args:
        security_modules: Checked-out gate modules.
        monkeypatch: Supplies a synthetic Git tree entry.
        tmp_path: Isolated destination directory.
        mode: Unsupported tree entry mode.
        kind: Source object type reported by Git.
    Returns:
        None.
    Raises:
        AssertionError: An unsupported source entry is accepted.
    """
    gate, _ = security_modules
    entry = f"{mode} {kind} synthetic-digest\tmodule.py\0".encode()
    monkeypatch.setattr(gate, "_git", lambda *arguments: entry)
    with pytest.raises(ValueError, match="Unsupported source entry"):
        gate._tree_entries(tmp_path, "base")


def test_revision_snapshot_rejects_blob_path_traversal(security_modules, tmp_path):
    """Reject source object paths that escape the destination directory.

    Args:
        security_modules: Checked-out gate modules.
        tmp_path: Isolated destination directory.
    Returns:
        None.
    Raises:
        AssertionError: A path traversal writes source bytes outside the snapshot.
    """
    gate, _ = security_modules
    stream = io.BytesIO(_blob_batch("synthetic-digest", b"synthetic = True\n"))
    with pytest.raises(ValueError, match="escapes"):
        gate._write_blobs([("../outside.py", "synthetic-digest")], stream, tmp_path / "snapshot")
    assert not (tmp_path / "outside.py").exists()


@pytest.mark.parametrize("payload", [b"other-digest blob 1\nx\n",
                                   b"synthetic-digest commit 1\nx\n",
                                   b"synthetic-digest blob 100\nshort\n"])
def test_revision_snapshot_rejects_incomplete_blob_stream(security_modules, tmp_path, payload):
    """Abort when Git returns a mismatched or incomplete source object.

    Args:
        security_modules: Checked-out gate modules.
        tmp_path: Isolated snapshot directory.
        payload: Synthetic invalid Git batch response.
    Returns:
        None.
    Raises:
        AssertionError: Incorrect bytes are accepted as a complete source object.
    """
    gate, _ = security_modules
    with pytest.raises(ValueError, match="source object"):
        gate._write_blobs([("module.py", "synthetic-digest")], io.BytesIO(payload), tmp_path)
    assert not (tmp_path / "module.py").exists()


def test_revision_snapshot_preserves_export_ignored_and_substituted_bytes(security_modules, monkeypatch, tmp_path):
    """Materialize vulnerable blobs despite candidate export attributes.

    Args:
        security_modules: Checked-out gate modules.
        monkeypatch: Supplies exact Git tree objects without another checkout.
        tmp_path: Isolated snapshot output directory.
    Returns:
        None.
    Raises:
        AssertionError: Export attributes can hide or rewrite scanned source.
    """
    gate, _ = security_modules
    attributes = b"vulnerable.py export-ignore export-subst\n"
    vulnerable = b"eval(user_expression)\n# $Format:%H$\n"
    tree = b"100644 blob attribute-digest\t.gitattributes\0"
    tree += b"100644 blob source-digest\tvulnerable.py\0"
    replies = iter([b"synthetic-commit\n", tree])
    monkeypatch.setattr(gate, "_git", lambda *arguments: next(replies))
    batch = _blob_batch("attribute-digest", attributes) + _blob_batch("source-digest", vulnerable)
    def _read_objects(command, **arguments):
        assert command[-2:] == ["cat-file", "--batch"]
        assert arguments["input"] == b"attribute-digest\nsource-digest\n"
        return subprocess.CompletedProcess(command, 0, stdout=batch)
    monkeypatch.setattr(gate.subprocess, "run", _read_objects)
    destination = tmp_path / "snapshot"
    assert gate._snapshot_revision(tmp_path, "base", destination) == "synthetic-commit"
    assert (destination / ".gitattributes").read_bytes() == attributes
    assert (destination / "vulnerable.py").read_bytes() == vulnerable


@pytest.mark.parametrize("report", [{}, {"SchemaVersion": 1, "Results": []},
                                     {"SchemaVersion": 2, "Results": None}])
def test_dependency_report_rejects_incomplete_schema(security_modules, report):
    """Reject reports that cannot establish a completed vulnerability scan.

    Args:
        security_modules: Checked-out gate modules.
        report: Incomplete public synthetic report.
    Returns:
        None.
    Raises:
        AssertionError: An invalid report is accepted as a clean scan.
    """
    _, dependencies = security_modules
    with pytest.raises(ValueError):
        dependencies._findings(report, {})


def test_dependency_report_rejects_unknown_inventory_target(security_modules):
    """Reject findings attributed to an input that was never inventoried.

    Args:
        security_modules: Checked-out gate modules.
    Returns:
        None.
    Raises:
        AssertionError: Unknown report targets silently disappear.
    """
    _, dependencies = security_modules
    report = {"SchemaVersion": 2, "Results": [{"Target": "unknown", "Vulnerabilities": []}]}
    with pytest.raises((KeyError, ValueError)):
        dependencies._findings(report, {})


@pytest.mark.parametrize("section", ["dependencies", "devDependencies", "optionalDependencies"])
def test_npm_dependencies_require_a_lock(security_modules, tmp_path, section):
    """Reject new npm dependencies that have no auditable package lock.

    Args:
        security_modules: Checked-out gate modules.
        tmp_path: Isolated synthetic npm manifest directory.
        section: npm dependency scope to validate.
    Returns:
        None.
    Raises:
        AssertionError: Dependencies without resolved package versions are accepted.
    """
    _, dependencies = security_modules
    (tmp_path / "package.json").write_text(json.dumps({section: {"minimist": "1.2.0"}}))
    with pytest.raises(ValueError, match="requires a package lock"):
        dependencies._validate_npm_manifests(tmp_path)


@pytest.mark.parametrize("section", ["dependencies", "devDependencies", "optionalDependencies"])
def test_npm_manifest_and_lock_must_match(security_modules, tmp_path, section):
    """Reject changed package declarations hidden behind an unchanged lock.

    Args:
        security_modules: Checked-out gate modules.
        tmp_path: Isolated synthetic npm manifest directory.
        section: npm dependency scope to validate.
    Returns:
        None.
    Raises:
        AssertionError: A stale package lock conceals changed dependencies.
    """
    _, dependencies = security_modules
    (tmp_path / "package.json").write_text(json.dumps({section: {"minimist": "1.2.0"}}))
    lock = {"lockfileVersion": 3, "packages": {"": {section: {"minimist": "1.2.8"}}}}
    (tmp_path / "package-lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="out of sync"):
        dependencies._validate_npm_manifests(tmp_path)


def test_npm_lock_must_resolve_direct_dependencies(security_modules, tmp_path):
    """Reject a lock with matching declarations but no resolved package entry.

    Args:
        security_modules: Checked-out gate modules.
        tmp_path: Isolated synthetic npm manifest directory.
    Returns:
        None.
    Raises:
        AssertionError: Removing resolved packages can conceal vulnerabilities.
    """
    _, dependencies = security_modules
    project = {"dependencies": {"minimist": "1.2.0"}}
    (tmp_path / "package.json").write_text(json.dumps(project))
    lock = {"lockfileVersion": 3, "packages": {"": project}}
    (tmp_path / "package-lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="resolved|missing|omitted"):
        dependencies._validate_npm_manifests(tmp_path)


def test_dependency_scan_rejects_invalid_json(security_modules, monkeypatch, tmp_path):
    """Reject truncated scanner JSON instead of producing a clean result.

    Args:
        security_modules: Checked-out gate modules.
        monkeypatch: Replaces only the external scanner process.
        tmp_path: Isolated synthetic requirement and output files.
    Returns:
        None.
    Raises:
        AssertionError: Malformed scanner output is treated as success.
    """
    _, dependencies = security_modules
    root = tmp_path / "source"
    root.mkdir()
    (root / "requirements.txt").write_text("requests==2.19.1\n")
    def _truncated_report(arguments, directory):
        (directory / "trivy.json").write_text('{"SchemaVersion":')
    monkeypatch.setattr(dependencies, "_scanner_versions", lambda output: None)
    monkeypatch.setattr(dependencies, "_run", _truncated_report)
    with pytest.raises((ValueError, RuntimeError, json.JSONDecodeError)):
        dependencies.scan_dependencies(root, tmp_path / "reports", tmp_path / "cache")


@pytest.mark.parametrize("packages", [[], [{"Name": "requests", "Version": "2.19.0"}]])
def test_dependency_coverage_rejects_dropped_or_changed_packages(security_modules, tmp_path, packages):
    """Require scanner inventory to contain each declared package version.

    Args:
        security_modules: Checked-out gate modules.
        tmp_path: Isolated dependency inventory.
        packages: Incomplete or mismatched scanner package observations.
    Returns:
        None.
    Raises:
        AssertionError: Dropped declared packages produce a clean report.
    """
    _, dependencies = security_modules
    source = tmp_path / "inputs/requirements.txt"
    source.parent.mkdir()
    source.write_text("requests==2.19.1\n")
    report = {"Results": [{"Target": "requirements.txt", "Packages": packages}]}
    with pytest.raises(ValueError, match="omitted"):
        dependencies._validate_coverage(report, {"requirements.txt": "requirements.txt"}, tmp_path)


def test_dependency_coverage_rejects_missing_manifest(security_modules, tmp_path):
    """Reject an empty scanner result when pinned dependencies were present.

    Args:
        security_modules: Checked-out gate modules.
        tmp_path: Isolated dependency inventory.
    Returns:
        None.
    Raises:
        AssertionError: A manifest silently omitted by Trivy is accepted.
    """
    _, dependencies = security_modules
    source = tmp_path / "inputs/requirements.txt"
    source.parent.mkdir()
    source.write_text("requests==2.19.1\n")
    with pytest.raises(ValueError, match="omitted"):
        dependencies._validate_coverage({"Results": []}, {"requirements.txt": "requirements.txt"}, tmp_path)


def test_dependency_resolution_failure_does_not_reuse_partial_output(security_modules, monkeypatch, tmp_path):
    """Propagate resolver failure without treating partial pins as validated.

    Args:
        security_modules: Checked-out gate modules.
        monkeypatch: Replaces the external resolver process.
        tmp_path: Isolated resolver directories.
    Returns:
        None.
    Raises:
        AssertionError: Failed dependency resolution produces a usable inventory.
    """
    _, dependencies = security_modules
    output = tmp_path / "output"
    output.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    def _failed_resolver(arguments, directory):
        raise RuntimeError("synthetic uv failure")
    monkeypatch.setattr(dependencies, "_run", _failed_resolver)
    project = {"dependencies": ["requests>=2"], "optional-dependencies": {"dev": []}}
    with pytest.raises(RuntimeError, match="synthetic uv failure"):
        dependencies._resolve_project(project, output, cache)


def test_dependency_process_failure_is_actionable(security_modules, monkeypatch, tmp_path):
    """Make failed scanner and resolver exit codes abort with a report pointer.

    Args:
        security_modules: Checked-out gate modules.
        monkeypatch: Supplies an external process failure.
        tmp_path: Private scanner log directory.
    Returns:
        None.
    Raises:
        AssertionError: A failed external process is accepted.
    """
    _, dependencies = security_modules
    result = subprocess.CompletedProcess(["uv"], 7)
    monkeypatch.setattr(dependencies.subprocess, "run", lambda *arguments, **keywords: result)
    with pytest.raises(RuntimeError, match="commands.log"):
        dependencies._run(["uv", "pip", "compile"], tmp_path)


@pytest.mark.parametrize("private", [True, False])
def test_regression_reports_require_owner_only_directory(security_modules, monkeypatch, tmp_path, private):
    """Require private existing output directories before writing scanner reports.

    Args:
        security_modules: Establishes the checked-out scripts import path.
        monkeypatch: Preserves the test process file-creation mask.
        tmp_path: Isolated existing report directory.
        private: Whether the output directory has owner-only permissions.
    Returns:
        None.
    Raises:
        AssertionError: Reports can be written into a publicly readable directory.
    """
    regression = importlib.import_module("security_regression_checks")
    monkeypatch.setattr(regression.os, "umask", lambda mode: 0o077)
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    output.chmod(0o700 if private else 0o755)
    if private:
        assert regression._private_output(output) == output.resolve()
        return
    with pytest.raises(ValueError, match="owner-only"):
        regression._private_output(output)


def test_regression_reports_use_checkout_not_trusted_policy_location(security_modules, monkeypatch, tmp_path):
    """Allow external trusted policy scripts without treating all temporary files as source.

    Args:
        security_modules: Establishes the checked-out scripts import path.
        monkeypatch: Simulates a runner's separate trusted policy directory.
        tmp_path: Isolated checkout, trusted policy, and output locations.
    Returns:
        None.
    Raises:
        AssertionError: External policy placement blocks private reports or permits source writes.
    """
    regression = importlib.import_module("security_regression_checks")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(regression, "__file__", str(tmp_path / "trusted-policy/security_regression_checks.py"))
    monkeypatch.setattr(regression.os, "umask", lambda mode: 0o077)
    output = tmp_path / "reports"
    output.mkdir(mode=0o700)
    assert regression._private_output(output) == output.resolve()
    with pytest.raises(ValueError, match="outside the checkout"):
        regression._private_output(checkout / "reports")
    assert not (checkout / "reports").exists()


@pytest.mark.parametrize("declaration", [
    "requests==2.19.1 --hash=sha256:" + "0" * 64,
    "requests==2.19.1 --hash sha256:" + "0" * 64,
    "requests==2.19.1\t# retained exact pin",
    "requests==2.19.1 \\\n        --hash=sha256:" + "0" * 64,
    "requests==2.19.1 --config-settings=setting=value",
])
def test_python_pins_survive_requirement_options(security_modules, tmp_path, declaration):
    """Retain vulnerable pins written with supported pip options and comments.

    Args:
        security_modules: Checked-out gate and dependency modules.
        tmp_path: Isolated dependency inventory directories.
        declaration: Valid pip syntax that must not hide an exact version.
    Returns:
        None.
    Raises:
        AssertionError: A declared exact dependency disappears from scanning.
    """
    _, dependencies = security_modules
    root = tmp_path / "source"
    root.mkdir()
    (root / "requirements.txt").write_text(declaration + "\n")
    output = tmp_path / "report"
    inventory = dependencies._inventory(root, output, tmp_path / "cache")
    target = next(iter(inventory))
    assert dependencies._expected_packages(output / "inputs" / target) == {("requests", "2.19.1")}


@pytest.mark.parametrize("section", ["dependencies", "devDependencies", "optionalDependencies"])
def test_npm_lock_rejects_changed_exact_resolution(security_modules, tmp_path, section):
    """Reject a safe lock version that conceals a vulnerable exact declaration.

    Args:
        security_modules: Checked-out gate and dependency modules.
        tmp_path: Isolated synthetic npm manifests.
        section: Dependency scope carrying the contradictory exact version.
    Returns:
        None.
    Raises:
        AssertionError: A stale resolved lock version conceals the declared pin.
    """
    _, dependencies = security_modules
    project = {section: {"minimist": "1.2.0"}}
    (tmp_path / "package.json").write_text(json.dumps(project))
    lock = {"lockfileVersion": 3, "packages": {
        "": project, "node_modules/minimist": {"version": "1.2.8"},
    }}
    (tmp_path / "package-lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="contradicts its exact version pin"):
        dependencies._validate_npm_manifests(tmp_path)
    lock["packages"]["node_modules/minimist"]["version"] = "1.2.0"
    (tmp_path / "package-lock.json").write_text(json.dumps(lock))
    dependencies._validate_npm_manifests(tmp_path)


@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped", ""])
def test_required_security_check_propagates_scanner_failure(monkeypatch, result):
    """Fail the required runtime check unless its isolated scanner job passed.

    Args:
        monkeypatch: Supplies a GitHub job-result observation to the shell step.
        result: Completed scanner status, including missing and skipped work.
    Returns:
        None.
    Raises:
        AssertionError: Scanner failure can become a passing required check.
    """
    workflow_path = Path(__file__).resolve().parents[3] / ".github/workflows/security-regression.yml"
    workflow = yaml.safe_load(workflow_path.read_text())
    job = workflow["jobs"]["security-regression"]
    assert job["needs"] == "security-scanners"
    assert job["if"] == "${{ always() }}"
    assert "continue-on-error" not in job
    prerequisite = job["steps"][0]
    assert prerequisite["env"] == {"SCANNER_RESULT": "${{ needs.security-scanners.result }}"}
    assert "if" not in prerequisite and "continue-on-error" not in prerequisite
    monkeypatch.setenv("SCANNER_RESULT", result)
    completed = subprocess.run(["bash", "-e", "-c", prerequisite["run"]], check=False)
    assert (completed.returncode == 0) == (result == "success")
