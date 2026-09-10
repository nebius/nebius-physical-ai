"""Keep NCore publication subprocess output and evidence outside the checkout."""

import hashlib
from contextlib import contextmanager
from importlib.machinery import PathFinder, SourceFileLoader
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

from image_byte_scan import core as W

ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / "npa/.venv/bin/python"
CONTEXT = ("npa/src/npa", "npa/docker/workbench/ncore")
SOURCE_PATHS = ("npa/docker/workbench/ncore", "npa/scripts/image_byte_scan", "npa/scripts/ncore_publication",
                "npa/src/npa/guardrails/ncore_attribution.py",
                "npa/src/npa/guardrails/__init__.py", "npa/src/npa/guardrails/confidentiality.py",
                "npa/src/npa/deploy/images.py", "npa/src/npa/deploy/publish_public.py",
                "npa/src/npa/deploy/ncore_selected_sbom.py", "npa/src/npa/_public_https.py",
                "npa/src/npa/deploy/ncore_component_scan.py", "npa/src/npa/deploy/ncore_component_inventory.py",
                "npa/src/npa/deploy/ncore_component_advisories.py",
                "npa/src/npa/deploy/ncore_component_sources.py",
                "npa/src/npa/deploy/ncore_karamel_source.py",
                "npa/src/npa/__init__.py", "npa/src/npa/deploy/__init__.py",
                "npa/src/npa/workbench/__init__.py", "npa/src/npa/workbench/gpu_classes.py",
                "npa/src/npa/workflows/ncore_runtime.py", "npa/tests/conftest.py", "npa/pyproject.toml",
                "npa/scripts/publish_ncore_oci.py", "npa/scripts/scan_image_bytes.py",
                "npa/scripts/scan_image_omniverse_payload.py",
                "npa/tests/docker/test_image_byte_go_build.py",
                "npa/tests/docker/test_ncore_public_attribution.py",
                "npa/tests/docker/test_packaging_contract.py",
                "npa/tests/docker/test_ncore_image_contract.py",
                "npa/tests/deploy/test_ncore_component_scan.py",
                "npa/tests/deploy/test_ncore_karamel_source.py",
                "npa/docker/workbench/packaging-contract.yaml", ".gitleaks.toml",
                ".trivyignore", ".github/workflows/publish-public-images.yml")
SOURCE_GUARDS = ("npa/tests/docker/test_packaging_contract.py",
                 "npa/tests/docker/test_ncore_image_contract.py",
                 "npa/tests/deploy/test_ncore_component_scan.py",
                 "npa/tests/deploy/test_ncore_karamel_source.py")


def run(argv, output, *, env=None, input_bytes=None, cwd=None):
    """Execute a gate with private output and sanitized failure text.

    Args:
        argv: Argument vector without credentials.
        output: New private stdout path; stderr uses the .stderr suffix.
        env: Explicit subprocess environment, or sanitized current environment.
        input_bytes: Optional stdin bytes.
        cwd: Working directory; defaults to the reviewed checkout.
    Returns:
        None.
    Raises:
        OSError: A process or private output cannot be opened.
        ValueError: The subprocess failed.
    """
    environment = {key: value for key, value in (os.environ if env is None else env).items()
                   if not key.startswith(("PYTHON", "PYTEST"))}
    environment.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    if Path(argv[0]).absolute() == PYTHON:
        cache = output.with_suffix(output.suffix + ".python-cache")
        cache.mkdir(mode=0o700)
        environment["PYTHONPYCACHEPREFIX"] = str(cache)
    with output.open("xb") as stdout, output.with_suffix(output.suffix + ".stderr").open("xb") as stderr:
        result = subprocess.run(argv, cwd=ROOT if cwd is None else cwd, env=environment, input=input_bytes,
                                stdout=stdout, stderr=stderr, check=False)
    W.require(result.returncode == 0, "publication_subprocess_failed_see_private_evidence")


def run_byte_scanner(argv, output):
    """Run the raw image scanner and return only its documented verdict status.

    Args:
        argv: Exact raw-scanner argument vector without credentials.
        output: New private stdout path; stderr uses the .stderr suffix.
    Returns:
        Zero for a clean scan or one for a completed scan with findings/failure.
    Raises:
        OSError: Private output or process execution fails.
        ValueError: The scanner returns an undocumented status.
    """
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("PYTHON", "PYTEST"))}
    environment.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    cache = output.with_suffix(output.suffix + ".python-cache")
    cache.mkdir(mode=0o700)
    environment["PYTHONPYCACHEPREFIX"] = str(cache)
    with output.open("xb") as stdout, output.with_suffix(output.suffix + ".stderr").open("xb") as stderr:
        result = subprocess.run(argv, cwd=ROOT, env=environment, stdout=stdout,
                                stderr=stderr, check=False)
    W.require(result.returncode in {0, 1}, "image_byte_scanner_unexpected_exit")
    return result.returncode


def file_sha(path):
    """Hash a file without retaining its contents.

    Args:
        path: File to hash.
    Returns:
        SHA256 hex digest.
    Raises:
        OSError: The file cannot be read.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    """Create private evidence without replacing prior evidence.

    Args:
        path: New file path.
        value: JSON value.
    Returns:
        None.
    Raises:
        OSError: The file exists or cannot be written.
    """
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def committed_source(sha):
    """Require the full checked-out SHA and unchanged gate/build source closure.

    Args:
        sha: Exact reviewed commit.
    Returns:
        SHA256 of the complete committed build-context archive.
    Raises:
        ValueError: Source identity or the relevant worktree differs.
        subprocess.CalledProcessError: Git cannot read the source.
    """
    import re

    W.require(re.fullmatch("[0-9a-f]{40}", sha), "full_source_sha_required")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    W.require(head == sha, "source_sha_must_equal_head")
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *SOURCE_PATHS], cwd=ROOT)
    W.require(not changed, "committed_build_and_gate_source_required")
    _verify_imported_sources(sha)
    # git archive exports tracked files and their committed modes, including the
    # entire NPA source tree. Generated catalog/cache/private inputs are absent.
    archive = subprocess.check_output(["git", "archive", sha, *CONTEXT], cwd=ROOT)
    return hashlib.sha256(archive).hexdigest()


def _verify_imported_sources(sha):
    paths = set()
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).absolute()
        _expected_import_origin(name, path)
        if name == "npa" or name.startswith("npa."):
            W.require(path.is_relative_to(ROOT / "npa/src/npa"), "host_npa_import_outside_checkout")
        if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / "npa/.venv"):
            W.require(path.resolve() == path and path.suffix == ".py", "host_source_import_identity")
            paths.add(path)
    for path in sorted(paths):
        result = subprocess.run(["git", "show", f"{sha}:{path.relative_to(ROOT).as_posix()}"],
                                cwd=ROOT, capture_output=True, check=False)
        W.require(result.returncode == 0 and W.sha(result.stdout) == file_sha(path),
                  "committed_host_import_required")


class _CommittedNpaLoader(SourceFileLoader):
    def __init__(self, name, path, source):
        super().__init__(name, str(path))
        self.source = source

    def get_code(self, fullname):
        return compile(self.source, self.path, "exec", dont_inherit=True)


class _CommittedNpaImports:
    def __init__(self, sha):
        self.sha = sha

    def find_spec(self, fullname, path=None, target=None):
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None or not isinstance(spec.origin, str):
            W.require(fullname.split(".")[0] not in _import_roots(), "host_repository_module_missing")
            return None
        source = Path(spec.origin).absolute()
        _expected_import_origin(fullname, source)
        if not source.is_relative_to(ROOT) or source.is_relative_to(ROOT / "npa/.venv"):
            return None
        W.require(spec is not None and isinstance(spec.loader, SourceFileLoader), "host_npa_source_loader_required")
        W.require(source.resolve() == source and source.suffix == ".py", "host_source_import_identity")
        relative = source.relative_to(ROOT).as_posix()
        result = subprocess.run(["git", "show", f"{self.sha}:{relative}"], cwd=ROOT,
                                capture_output=True, check=False)
        W.require(result.returncode == 0 and W.sha(result.stdout) == file_sha(source),
                  "committed_host_import_required")
        # Compile the compared Git blob directly, so a stale or replaced .pyc
        # cannot substitute executable bytes after source verification.
        spec.loader = _CommittedNpaLoader(fullname, source, result.stdout)
        return spec


def _import_roots():
    return {
        "npa": ROOT / "npa/src/npa",
        "ncore_publication": ROOT / "npa/scripts/ncore_publication",
        "image_byte_scan": ROOT / "npa/scripts/image_byte_scan",
        "scan_image_omniverse_payload": ROOT / "npa/scripts/scan_image_omniverse_payload.py",
    }


def _expected_import_origin(name, path):
    expected = _import_roots().get(name.split(".")[0])
    if expected is not None:
        code = "host_npa_import_outside_checkout" if name.split(".")[0] == "npa" else "host_repository_import_outside_checkout"
        W.require(path == expected or path.is_relative_to(expected), code)


@contextmanager
def committed_npa_imports(sha):
    """Load runtime repository imports only from compared committed Python blobs.

    Args:
        sha: Reviewed commit, checked before importing the NPA gate closure.
    Returns:
        Context manager enforcing imports through the whole CLI operation.
    Raises:
        ValueError: NPA was preloaded, resolves outside this checkout or differs.
    """
    committed_source(sha)
    W.require(not any(name == "npa" or name.startswith("npa.") for name in sys.modules),
              "host_npa_import_before_source_authorization")
    finder = _CommittedNpaImports(sha)
    sys.meta_path.insert(0, finder)
    try:
        yield
        _verify_imported_sources(sha)
    finally:
        sys.meta_path.remove(finder)


def guard_snapshot(directory, sha):
    """Export the complete committed guard import, conftest and data closure.

    Args:
        directory: Private gate evidence directory.
        sha: Full reviewed commit.
    Returns:
        Snapshot directory and archive SHA256.
    Raises:
        ValueError, OSError, tarfile.TarError: Export or safe extraction fails.
    """
    archive = directory / "source-guards.tar"
    run(["git", "archive", sha], archive, env=public_environment())
    snapshot = directory / "source-guards-tree"
    snapshot.mkdir(mode=0o700)
    with tarfile.open(archive) as stream:
        stream.extractall(snapshot, filter="data")
    return snapshot, file_sha(archive)


def guard_command(snapshot, directory):
    """Build a site-startup-free invocation of the committed pytest runner.

    Args:
        snapshot: Complete committed source export.
        directory: Private evidence and temporary-file root.
    Returns:
        Interpreter argv with fixed import paths and no ambient Python hooks.
    Raises:
        None.
    """
    packages = PYTHON.parent.parent / "lib/python3.12/site-packages"
    paths = [str(snapshot / "npa/src"), str(snapshot / "npa/scripts"), str(packages)]
    script = (f"import sys; sys.path[:0] = {paths!r}; "
              "from ncore_publication.process import _execute_source_guards; "
              f"_execute_source_guards({str(snapshot)!r}, {str(directory)!r})")
    return [str(PYTHON), "-I", "-S", "-B", "-c", script]


class _GuardExecution:
    def __init__(self):
        self.collected = []
        self.deselected = []
        self.reports = []

    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]

    def pytest_deselected(self, items):
        self.deselected.extend(item.nodeid for item in items)

    def pytest_runtest_logreport(self, report):
        self.reports.append({"nodeid": report.nodeid, "when": report.when,
                             "outcome": report.outcome, "wasxfail": hasattr(report, "wasxfail")})


def _execute_source_guards(snapshot, directory):
    import pytest

    snapshot, directory = Path(snapshot), Path(directory)
    recorder = _GuardExecution()
    code = pytest.main(["-q", "-c", str(snapshot / "npa/pyproject.toml"), "-o", "addopts=",
                       "--rootdir", str(snapshot), "--confcutdir", str(snapshot),
                       "-p", "no:cacheprovider", "--basetemp", str(directory / "guard-tmp"),
                       *SOURCE_GUARDS], plugins=[recorder])
    receipt = {"exit_code": int(code), "collected": recorder.collected,
               "deselected": recorder.deselected, "reports": recorder.reports}
    write_json(directory / "source-guards.json", receipt)
    verify_guard_execution(receipt)


def verify_guard_execution(receipt):
    """Require every mandatory guard's setup, test and teardown to pass.

    Args:
        receipt: Actual pytest collection and execution reports.
    Returns:
        None.
    Raises:
        ValueError: Tests were missing, deselected, skipped or not executed.
    """
    collected = receipt["collected"]
    W.require(receipt["exit_code"] == 0 and not receipt["deselected"] and collected
              and len(set(collected)) == len(collected)
              and {node.split("::", 1)[0] for node in collected} == set(SOURCE_GUARDS),
              "mandatory_guard_collection_required")
    expected = {(node, phase) for node in collected for phase in ("setup", "call", "teardown")}
    reports = receipt["reports"]
    actual = {(report["nodeid"], report["when"]) for report in reports}
    W.require(len(reports) == len(expected) and actual == expected
              and all(report["outcome"] == "passed" and not report["wasxfail"] for report in reports),
              "mandatory_guards_must_execute_and_pass")


def public_environment():
    """Return an environment without private denylist and runtime credentials.

    Args:
        None.
    Returns:
        Minimal environment for build/runtime/scanner subprocesses.
    Raises:
        None.
    """
    names = ("PATH", "HOME", "DOCKER_CONFIG", "DOCKER_HOST", "DOCKER_CONTEXT",
             "SSL_CERT_FILE", "SSL_CERT_DIR", "LANG")
    return {name: os.environ[name] for name in names if name in os.environ}
