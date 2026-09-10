"""Prove source population, executed guards, config history and keyring refusals."""

import copy
import importlib.util
import io
import json
import os
import runpy
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))

from image_byte_scan import core as W  # noqa: E402
from ncore_publication import cli, gates, process, provenance  # noqa: E402
from ncore_publication import bootstrap  # noqa: E402

SHA = "a" * 40
KEYRING = "usr/share/keyrings/debian-archive-keyring.gpg"


@pytest.mark.parametrize("status,log_exists", [(0, True), (19, True), (19, False)])
def test_bootstrap_diagnostics_preserve_status_and_log_before_container_removal(tmp_path, status, log_exists):
    log = tmp_path / "upstream-apt.log"
    if log_exists:
        log.write_text("retained upstream diagnostic\n")
    script = bootstrap._diagnostic_trap("local log=" + str(log) + "\n")
    result = subprocess.run(["bash", "-c", script + f"exit {status}\n"], capture_output=True, check=False)
    assert result.returncode == status
    assert result.stderr == (b"retained upstream diagnostic\n" if status and log_exists else b"")


def test_actual_buildx_metadata_mode_is_restricted_without_changing_bytes(tmp_path):
    tmp_path.chmod(0o700)
    path = tmp_path / "buildx.json"
    raw = b'{"containerimage.digest":"sha256:' + b"a" * 64 + b'"}'
    path.write_bytes(raw)
    path.chmod(0o644)
    with W.authorized_roots(tmp_path, ROOT):
        binding = cli._build_metadata(path)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.read_bytes() == raw
    assert binding["sha256"] == W.sha(raw)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "writable"])
def test_build_metadata_rejects_unsafe_outputs_without_chmod(tmp_path, kind):
    tmp_path.chmod(0o700)
    original = tmp_path / "original"
    original.write_bytes(b"metadata")
    original.chmod(0o644)
    path = tmp_path / "buildx.json"
    if kind == "symlink":
        path.symlink_to(original)
    elif kind == "hardlink":
        os.link(original, path)
    elif kind == "fifo":
        os.mkfifo(path, 0o600)
    else:
        path.write_bytes(b"metadata")
        path.chmod(0o666)
    with W.authorized_roots(tmp_path, ROOT), pytest.raises((ValueError, OSError)):
        cli._build_metadata(path)
    assert original.stat().st_mode & 0o777 == 0o644


def test_build_metadata_refuses_a_link_added_during_permission_change(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    path = tmp_path / "buildx.json"
    path.write_bytes(b"metadata")
    path.chmod(0o644)
    chmod = os.fchmod

    def add_link(fd, mode):
        chmod(fd, mode)
        os.link(path, tmp_path / "second-link")

    monkeypatch.setattr(cli.os, "fchmod", add_link)
    with W.authorized_roots(tmp_path, ROOT), pytest.raises(ValueError, match="build_metadata_changed"):
        cli._build_metadata(path)


def _tar(path, files):
    with tarfile.open(path, "w") as archive:
        for name, raw in files:
            member = tarfile.TarInfo(name)
            if raw is None:
                member.type = tarfile.SYMTYPE
                member.linkname = "/elsewhere"
                archive.addfile(member)
            else:
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))


def _shipped():
    return [(name, b"reviewed") for name in (
        "usr/share/doc/npa-ncore/recipes/Dockerfile", "usr/share/doc/npa-ncore/runtime-lock.json",
        "opt/ncore/base-sources/recipes/base-source-lock.json", "opt/npa/src/npa/__init__.py",
        *provenance.NATIVE_SOURCES,
    )] + [("usr/share/doc/npa-ncore/npa-source-sha", (SHA + "\n").encode())]


@pytest.mark.parametrize("native", provenance.NATIVE_SOURCES)
@pytest.mark.parametrize("change", ["missing", "changed", "symlink", "duplicate"])
def test_every_privileged_native_file_is_required_and_source_bound(tmp_path, monkeypatch, native, change):
    files = [(name, raw) for name, raw in _shipped() if name != native]
    if change != "missing":
        files.append((native, None if change == "symlink" else b"modified" if change == "changed" else b"reviewed"))
    if change == "duplicate":
        files.append((native, b"reviewed"))
    monkeypatch.setattr(provenance, "_committed_digest", lambda *_: W.sha(b"reviewed"))
    path = tmp_path / "rootfs.tar"
    _tar(path, files)
    with pytest.raises(ValueError):
        provenance.shipped_source(path, SHA)


def test_native_population_is_exact_and_downloader_maps_to_its_real_source(tmp_path, monkeypatch):
    assert provenance._source_path("opt/ncore/native/_public_https.py") == ROOT / "npa/src/npa/_public_https.py"
    monkeypatch.setattr(provenance, "_committed_digest", lambda *_: W.sha(b"reviewed"))
    path = tmp_path / "rootfs.tar"
    _tar(path, _shipped())
    assert provenance.shipped_source(path, SHA) == 9
    _tar(path, _shipped() + [("opt/ncore/native/extra.py", b"unreviewed")])
    with pytest.raises(ValueError, match="unexpected_native_source_population"):
        provenance.shipped_source(path, SHA)


def test_required_native_files_do_not_replace_the_existing_npa_source_population(tmp_path, monkeypatch):
    path = tmp_path / "rootfs.tar"
    _tar(path, [(name, raw) for name, raw in _shipped() if name != "opt/npa/src/npa/__init__.py"])
    monkeypatch.setattr(provenance, "_committed_digest", lambda *_: W.sha(b"reviewed"))
    with pytest.raises(ValueError, match="shipped_source_population_missing"):
        provenance.shipped_source(path, SHA)


@pytest.mark.parametrize("relative", ["npa/__init__.py", "npa/deploy/__init__.py",
                                     "npa/workbench/__init__.py", "npa/workbench/gpu_classes.py",
                                     "npa/previously_unlisted_import.py"])
def test_actual_imported_source_closure_rejects_changed_bytes(tmp_path, monkeypatch, relative):
    path = tmp_path / "npa/src" / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"reviewed")
    module = SimpleNamespace(__file__=str(path))
    monkeypatch.setattr(process, "ROOT", tmp_path)
    monkeypatch.setattr(process.sys, "modules", {"npa.imported": module})
    monkeypatch.setattr(process.subprocess, "run", lambda *_, **__: subprocess.CompletedProcess([], 0, b"reviewed"))
    process._verify_imported_sources(SHA)
    (tmp_path / "unrelated.py").write_bytes(b"dirty unrelated source")
    process._verify_imported_sources(SHA)
    path.write_bytes(b"changed gate input")
    with pytest.raises(ValueError, match="committed_host_import_required"):
        process._verify_imported_sources(SHA)


def test_host_npa_import_cannot_resolve_to_another_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(process, "ROOT", tmp_path / "checkout")
    monkeypatch.setattr(process.sys, "modules", {"npa": SimpleNamespace(__file__=str(tmp_path / "elsewhere.py"))})
    with pytest.raises(ValueError, match="host_npa_import_outside_checkout"):
        process._verify_imported_sources(SHA)


@pytest.mark.parametrize("name,relative", [
    ("npa.example", "npa/src/npa/example.py"),
    ("scan_image_omniverse_payload", "npa/scripts/scan_image_omniverse_payload.py"),
    ("ncore_publication.example", "npa/scripts/ncore_publication/example.py"),
])
def test_repository_loader_compiles_compared_commit_without_reading_cached_bytecode(tmp_path, monkeypatch, name, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"result = 'reviewed'\n")
    spec = SimpleNamespace(origin=str(path), loader=process.SourceFileLoader(name, str(path)))
    monkeypatch.setattr(process, "ROOT", tmp_path)
    monkeypatch.setattr(process.PathFinder, "find_spec", lambda *args: spec)
    monkeypatch.setattr(process.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess([], 0, b"result = 'reviewed'\n"))
    verified = process._CommittedNpaImports(SHA).find_spec(name)
    path.write_bytes(b"raise RuntimeError('changed after comparison')\n")
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, verified.loader))
    verified.loader.exec_module(module)
    assert module.result == "reviewed"
    with pytest.raises(ValueError, match="committed_host_import_required"):
        process._CommittedNpaImports(SHA).find_spec(name)


def test_npa_cannot_be_preloaded_before_import_authorization(monkeypatch):
    monkeypatch.setattr(process, "committed_source", lambda sha: None)
    monkeypatch.setattr(process.sys, "modules", {"npa": SimpleNamespace()})
    with pytest.raises(ValueError, match="host_npa_import_before_source_authorization"):
        with process.committed_npa_imports(SHA):
            pytest.fail("preloaded package accepted")


def _execution():
    nodes = [path + "::test_required" for path in process.SOURCE_GUARDS]
    return {"exit_code": 0, "collected": nodes, "deselected": [], "reports": [
        {"nodeid": node, "when": phase, "outcome": "passed", "wasxfail": False}
        for node in nodes for phase in ("setup", "call", "teardown")]}


@pytest.mark.parametrize("change", [None, "collect-only", "skip", "xfail", "no-call", "no-teardown",
                                    "deselected", "missing-module", "duplicate", "failed"])
def test_source_guards_require_every_test_to_execute_all_phases(change):
    receipt = _execution()
    if change == "collect-only":
        receipt["reports"] = []
    elif change == "skip":
        receipt["reports"][1]["outcome"] = "skipped"
    elif change == "xfail":
        receipt["reports"][1]["wasxfail"] = True
    elif change in {"no-call", "no-teardown"}:
        receipt["reports"] = [report for report in receipt["reports"] if report["when"] != change[3:]]
    elif change == "deselected":
        receipt["deselected"] = ["hidden::test"]
    elif change == "missing-module":
        receipt["collected"].pop()
    elif change == "duplicate":
        receipt["reports"].append(copy.copy(receipt["reports"][0]))
    elif change == "failed":
        receipt["exit_code"] = 1
    if change is None:
        process.verify_guard_execution(receipt)
    else:
        with pytest.raises(ValueError):
            process.verify_guard_execution(receipt)


def test_python_and_pytest_controls_are_removed_from_subprocesses(tmp_path, monkeypatch):
    controls = {"PYTHONPATH": "untrusted", "PYTHONHOME": "untrusted", "PYTEST_ADDOPTS": "--collect-only",
                "PYTEST_PLUGINS": "untrusted", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "0"}
    for key, value in controls.items():
        monkeypatch.setenv(key, value)
    observed = []

    def execute(argv, **kwargs):
        observed.append(kwargs["env"])
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(process.subprocess, "run", execute)
    process.run(["test"], tmp_path / "inherited.log")
    process.run(["test"], tmp_path / "explicit.log", env=controls)
    for environment in observed:
        assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert not set(controls).difference({"PYTEST_DISABLE_PLUGIN_AUTOLOAD"}) & set(environment)
    argv = process.guard_command(tmp_path / "snapshot", tmp_path)
    assert argv[1:5] == ["-I", "-S", "-B", "-c"]
    assert "snapshot/npa/src" in argv[-1] and "site-packages" in argv[-1]


def test_python_subprocess_gets_its_own_empty_cache_namespace(tmp_path, monkeypatch):
    observed = []
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", "untrusted-existing-cache")
    monkeypatch.setattr(process.subprocess, "run", lambda *args, **kwargs:
                        observed.append(kwargs["env"]) or subprocess.CompletedProcess(args, 0))
    for name in ("first.log", "second.log"):
        process.run([str(process.PYTHON), "-c", "pass"], tmp_path / name)
    paths = [Path(environment["PYTHONPYCACHEPREFIX"]) for environment in observed]
    assert paths[0] != paths[1]
    assert all(path.parent == tmp_path and path.stat().st_mode & 0o777 == 0o700 for path in paths)
    assert all(not list(path.iterdir()) for path in paths)
    assert all(environment["PYTHONDONTWRITEBYTECODE"] == "1" for environment in observed)


def test_entrypoint_isolates_import_cache_before_loading_publication_modules(monkeypatch):
    observed = []

    def main():
        path = Path(sys.pycache_prefix)
        observed.append(path)
        assert path.is_dir() and not list(path.iterdir())
        assert path.stat().st_mode & 0o777 == 0o700 and sys.dont_write_bytecode
        return 7

    monkeypatch.setattr(sys, "pycache_prefix", sys.pycache_prefix)
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    monkeypatch.setitem(sys.modules, "ncore_publication.cli", SimpleNamespace(main=main))
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(ROOT / "npa/scripts/publish_ncore_oci.py"), run_name="__main__")
    assert result.value.code == 7 and len(observed) == 1
    assert not observed[0].exists()


def test_guard_snapshot_uses_only_committed_files_including_config_and_conftests(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture.tar"
    _tar(fixture, [("npa/tests/conftest.py", b"committed conftest"),
                   ("npa/pyproject.toml", b"committed configuration"),
                   ("npa/src/npa/transitive.py", b"committed transitive import")])

    def archive(argv, output, **kwargs):
        assert argv == ["git", "archive", SHA]
        output.write_bytes(fixture.read_bytes())

    monkeypatch.setattr(process, "run", archive)
    snapshot, digest = process.guard_snapshot(tmp_path, SHA)
    assert digest == process.file_sha(fixture)
    assert (snapshot / "npa/tests/conftest.py").read_bytes() == b"committed conftest"
    assert not (snapshot / "npa/tests/docker/conftest.py").exists()
    assert (snapshot / "npa/src/npa/transitive.py").read_bytes() == b"committed transitive import"


@pytest.mark.parametrize("change", [None, "install", "digest", "missing", "empty", "bad-entry"])
def test_payload_history_classifies_exact_original_config(tmp_path, change):
    config = {"history": [{"created_by": "COPY public source /opt/ncore"}, {"created_by": "WORKDIR /workspace"}]}
    if change == "install":
        config["history"].append({"created_by": "RUN pip install isaacsim"})
    elif change == "missing":
        del config["history"]
    elif change == "empty":
        config["history"] = []
    elif change == "bad-entry":
        config["history"].append({"created_by": None})
    raw = json.dumps(config).encode()
    digest = "sha256:" + W.sha(raw)
    _tar(tmp_path / "inspection.tar", [(digest[7:] + ".json", b"{}" if change == "digest" else raw)])
    graph = {"image_config_digest": digest}
    if change is None:
        gates._payload_history(tmp_path, graph)
        receipt = json.loads((tmp_path / "payload-history.json").read_bytes())
        assert receipt["entries_classified"] == 2 and receipt["valid"] is True
        assert receipt["config_digest"] == digest and receipt["complete"] is True
    else:
        with pytest.raises(ValueError):
            gates._payload_history(tmp_path, graph)
    if change == "install":
        receipt = json.loads((tmp_path / "payload-history.json").read_bytes())
        assert receipt["history_hits"] and receipt["valid"] is False
        assert receipt["entries_classified"] == 3


@pytest.mark.parametrize("change", [None, "missing", "changed", "symlink", "duplicate"])
def test_keyring_extraction_requires_exact_single_regular_locked_member(tmp_path, change):
    files = [] if change == "missing" else [("./" + KEYRING, None if change == "symlink" else b"public keys")]
    if change == "duplicate":
        files.append((KEYRING, b"public keys"))
    archive = tmp_path / "data.tar"
    _tar(archive, files)
    target = tmp_path / "prepared/keyring.gpg"
    with W.authorized_roots(tmp_path, ROOT):
        if change is None:
            cli._extract_keyring(archive, target, W.sha(b"public keys"))
            assert target.read_bytes() == b"public keys" and target.stat().st_mode & 0o777 == 0o600
        else:
            with pytest.raises(ValueError):
                cli._extract_keyring(archive, target, W.sha(b"different" if change == "changed" else b"public keys"))
            assert not target.exists()


def test_keyring_package_is_authenticated_before_any_extraction(tmp_path, monkeypatch):
    from npa import _public_https

    calls = []
    monkeypatch.setattr(_public_https, "download_public_https", lambda url, output, **kwargs: output.write(b"wrong package"))
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="locked_keyring_package_digest"):
        cli._prepare_keyring(tmp_path, tmp_path / "keyring/keyring.gpg")
    assert calls == []


@pytest.mark.parametrize("action", ["build", "check", "publish"])
def test_absent_keyring_fails_before_build_or_scanner_gates(tmp_path, action):
    args = SimpleNamespace(analysis_root=tmp_path, action=action, keyring=None)
    with W.authorized_roots(tmp_path, ROOT), pytest.raises((ValueError, OSError)):
        cli._keyring_input(args)
    assert args.keyring == tmp_path / "keyring/debian-archive-keyring.gpg"


def test_keyring_preparation_precedes_expensive_native_preparation(tmp_path, monkeypatch):
    def absent(*args):
        raise ValueError("keyring unavailable")

    monkeypatch.setattr(cli, "_prepare_keyring", absent)
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: pytest.fail("native preparation reached"))
    args = SimpleNamespace(analysis_root=tmp_path, keyring=tmp_path / "keyring.gpg")
    with pytest.raises(ValueError, match="keyring unavailable"):
        cli._prepare(args)


def test_ci_passes_prepared_keyring_to_prepare_build_and_publish():
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/publish-public-images.yml").read_text())
    steps = workflow["jobs"]["build-development"]["steps"]
    commands = [step["run"] for step in steps if "publish_ncore_oci.py" in step.get("run", "")]
    assert len(commands) == 3
    assert all('--keyring "' in command and "/keyring/debian-archive-keyring.gpg" in command for command in commands)


def _base_material():
    base = json.loads((ROOT / "npa/docker/workbench/ncore/base-source-lock.json").read_bytes())["base_image"]
    reference, digest = base.split("@")
    return {"uri": "pkg:docker/" + reference.replace(":", "@") + "?platform=linux%2Famd64&digest=" + digest,
            "digest": {"sha256": digest[7:]}}


@pytest.mark.parametrize("change", [None, "unrelated", "digest", "uri-digest", "platform", "duplicate", "query", "builder", "missing-generator"])
@pytest.mark.parametrize("kind", sorted(provenance.SLSA))
def test_provenance_requires_locked_base_material(kind, change):
    item = _base_material()
    if change == "unrelated":
        item["uri"] = "pkg:docker/unapproved@1"
    elif change == "digest":
        item["digest"]["sha256"] = "e" * 64
    elif change == "uri-digest":
        item["uri"] = item["uri"].split("&digest=")[0] + "&digest=sha256:" + "e" * 64
    elif change == "platform":
        item["uri"] = item["uri"].replace("amd64", "arm64")
    elif change == "query":
        item["uri"] += "&platform=linux%2Famd64"
    materials = [item, item] if change == "duplicate" else [item]
    if change != "missing-generator":
        generator, generator_digest = provenance._sbom_generator()
        materials.append({"uri": "pkg:" + generator, "digest": {"sha256": generator_digest[7:]}})
    parameters = {"args": {"build-arg:SOURCE_SHA": SHA}}
    if kind.endswith("v0.2"):
        predicate = {"buildType": "https://mobyproject.org/buildkit@v1",
                     "invocation": {"parameters": parameters}, "materials": materials}
    else:
        predicate = {"buildDefinition": {"buildType": provenance.BUILDKIT_TYPES[kind],
                     "externalParameters": {"request": parameters}, "resolvedDependencies": materials}}
    if change == "builder":
        predicate.get("buildDefinition", predicate)["buildType"] = "unapproved-builder"
    if change is None:
        provenance._source_parameters(kind, predicate, SHA)
    else:
        with pytest.raises(ValueError):
            provenance._source_parameters(kind, predicate, SHA)


def test_only_the_committed_frontend_may_accompany_the_locked_base_material():
    frontend = {"uri": "pkg:docker/docker/dockerfile@1.7", "digest": {"sha256": "b" * 64}}
    provenance._base_material([_base_material(), frontend])
    with pytest.raises(ValueError, match="unexpected_provenance_material"):
        provenance._base_material([_base_material(), frontend, frontend])
    frontend["uri"] = "pkg:docker/unapproved@1"
    with pytest.raises(ValueError, match="unexpected_provenance_material"):
        provenance._base_material([_base_material(), frontend])


@pytest.mark.parametrize("change", [None, "digest", "duplicate", "unrelated"])
def test_real_buildkit_sbom_material_is_pinned_and_never_a_wildcard(change):
    reference, digest = provenance._sbom_generator()
    scanner = {"uri": "pkg:" + reference, "digest": {"sha256": digest[7:]}}
    materials = [_base_material(), scanner]
    if change == "digest":
        scanner["digest"]["sha256"] = "a" * 64
    elif change == "duplicate":
        materials.append(scanner)
    elif change == "unrelated":
        scanner["uri"] = "pkg:docker/unreviewed/scanner@stable-1"
    if change is None:
        provenance._base_material(materials)
        build = (ROOT / "npa/docker/workbench/ncore/build.sh").read_text()
        assert '--attest "type=sbom,generator=$SBOM_GENERATOR"' in build
    else:
        with pytest.raises(ValueError):
            provenance._base_material(materials)
