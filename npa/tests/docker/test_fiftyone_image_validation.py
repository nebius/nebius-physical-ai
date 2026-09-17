"""Exercise exact-image validation, offline phases, owned cleanup and public evidence."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import sys
import tarfile
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/fiftyone"
IMAGE_ID = "sha256:" + "1" * 64
REVISION = "2" * 40


@pytest.fixture
def modules(monkeypatch):
    loaded = []
    for name in ("validation_checks", "validation_docker", "validate_image"):
        spec = importlib.util.spec_from_file_location(name, IMAGE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return SimpleNamespace(checks=loaded[0], docker=loaded[1], host=loaded[2])


@pytest.mark.parametrize("mismatch", [None, "source", "receipt"])
def test_installed_npa_import_uses_fresh_interpreter(
    modules, tmp_path, monkeypatch, mismatch
):
    source = ROOT / "npa"
    receipt = tmp_path / "source-receipt"
    receipt.write_text(str(source) if mismatch != "receipt" else "different-source")
    monkeypatch.setattr(modules.checks, "_SOURCE", source)
    monkeypatch.setattr(modules.checks, "_RECEIPT", receipt)
    monkeypatch.setattr(modules.checks, "_OUTPUT", tmp_path)
    monkeypatch.setattr(modules.checks, "_PYTHON", sys.executable)
    # A pre-install interpreter can retain a namespace that lacks the new editable hook.
    monkeypatch.setitem(sys.modules, "npa", SimpleNamespace(__file__=None))
    if mismatch == "source":
        monkeypatch.setattr(modules.checks, "_SOURCE", tmp_path / "different-source")
    if mismatch:
        message = "another source" if mismatch == "source" else "receipt differs"
        with pytest.raises(RuntimeError, match=message):
            modules.checks._installed_npa_version("installed")
    else:
        assert modules.checks._installed_npa_version("installed")
        assert modules.checks._installed_npa_version("post")
        assert (tmp_path / "post-npa-import.exit.json").is_file()
    result = json.loads((tmp_path / "installed-npa-import.stdout").read_text())
    assert Path(result["file"]).resolve().is_relative_to(source / "src")
    assert (
        json.loads((tmp_path / "installed-npa-import.exit.json").read_text())[
            "exit_code"
        ]
        == 0
    )


def _image():
    return {
        "Id": IMAGE_ID,
        "Os": "linux",
        "Architecture": "amd64",
        "Config": {
            "User": "ubuntu",
            "Entrypoint": ["/opt/npa/docker/workbench/fiftyone/entrypoint.sh"],
            "Labels": {
                "org.opencontainers.image.revision": REVISION,
                "org.nebius.npa.skypilot-bootstrap-contract": "skypilot-0.12.2-v1",
            },
        },
    }


def _archive(path, files):
    with tarfile.open(path, "w") as archive:
        for name, content, mode in files:
            entry = tarfile.TarInfo("npa/" + name)
            entry.size, entry.mode = len(content), mode
            archive.addfile(entry, io.BytesIO(content))


@pytest.fixture
def source_fixture(tmp_path, modules):
    archive = tmp_path / "source.tar"
    files = [
        ("docker/workbench/fiftyone/" + name, (IMAGE / name).read_bytes(), 0o644)
        for name in (
            "validate_image.py",
            "validation_docker.py",
            "validation_checks.py",
        )
    ]
    files += [("src/run.sh", b"synthetic executable\n", 0o755)]
    _archive(archive, files)
    source = tmp_path / "source"
    records = modules.host._unpack_source(archive, source)
    (tmp_path / "source.json").write_text(json.dumps({"files": records}))
    commands = modules.docker._Commands(tmp_path, ROOT)
    return commands, source


def test_source_binding_and_editable_directory_permissions(modules, source_fixture):
    commands, source = source_fixture
    modules.host._bind_validator(commands, source)
    records = json.loads((commands.root / "validator-source.json").read_text())
    assert len(records) == 3
    assert all(
        path.stat().st_mode & 0o007 == 0o007 for path in [source, source / "src"]
    )
    assert source.stat().st_uid == os.getuid()
    assert (source / "src/run.sh").stat().st_mode & 0o111
    (source / "src/generated.egg-info").mkdir()
    modules.host._verify_source(commands, source)


def _validator_archive_files():
    return [
        ("docker/workbench/fiftyone/" + name, (IMAGE / name).read_bytes(), 0o644)
        for name in (
            "validate_image.py",
            "validation_docker.py",
            "validation_checks.py",
        )
    ]


@pytest.mark.parametrize("mask", [0o022, 0o077])
def test_actual_source_staging_preserves_private_parent_and_cross_uid_readability(
    modules, tmp_path, monkeypatch, mask
):
    calls = []

    def git(name, argv):
        calls.append(argv)
        if name == "source-revision":
            return REVISION.encode()
        assert name == "source-archive" and argv[:2] == ["git", "archive"]
        _archive(Path(argv[argv.index("-o") + 1]), _validator_archive_files())
        return b""

    arguments = argparse.Namespace(
        image_id=IMAGE_ID,
        revision=REVISION,
        source_root=ROOT,
        output_path=tmp_path / "private",
    )
    previous = os.umask(mask)
    try:
        commands = modules.host._prepare(arguments)
        monkeypatch.setattr(commands, "run", git)
        source, checks = modules.host._source_inputs(commands, REVISION)
    finally:
        os.umask(previous)
    assert commands.root.stat().st_mode & 0o777 == 0o700
    assert checks.stat().st_mode & 0o777 == 0o755
    assert (checks / "validation_checks.py").stat().st_mode & 0o777 == 0o644
    assert source.stat().st_mode & 0o777 == 0o777
    assert len(calls) == 2 and all(argv[0] == "git" for argv in calls)
    modules.host._verify_source(commands, source)


@pytest.mark.parametrize("change", ["content", "mode", "symlink"])
def test_dirty_executing_validator_refused(
    modules, source_fixture, monkeypatch, change
):
    commands, source = source_fixture
    executing = commands.root / "executing"
    executing.mkdir()
    for name in ("validate_image.py", "validation_docker.py", "validation_checks.py"):
        (executing / name).write_bytes((IMAGE / name).read_bytes())
        (executing / name).chmod(0o644)
    changed = executing / "validation_checks.py"
    if change == "content":
        changed.write_text("# different validator\n")
    elif change == "mode":
        changed.chmod(0o755)
    else:
        changed.unlink()
        changed.symlink_to(IMAGE / "validation_checks.py")
    monkeypatch.setattr(modules.host, "__file__", str(executing / "validate_image.py"))
    with pytest.raises(modules.docker._ValidationError, match="Executing validator"):
        modules.host._bind_validator(commands, source)


@pytest.mark.parametrize("change", ["content", "executable", "symlink"])
def test_original_source_change_refused(modules, source_fixture, change):
    commands, source = source_fixture
    path = source / "src/run.sh"
    if change == "content":
        path.write_text("changed")
    elif change == "executable":
        path.chmod(0o644)
    else:
        (source / "generated-link").symlink_to(path)
    with pytest.raises(modules.docker._ValidationError, match="source"):
        modules.host._verify_source(commands, source)


@pytest.mark.parametrize(
    "member_type", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE]
)
def test_source_archive_rejects_special_members(modules, tmp_path, member_type):
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as stream:
        entry = tarfile.TarInfo("npa/member")
        entry.type = member_type
        stream.addfile(entry)
    with pytest.raises(modules.docker._ValidationError, match="regular files"):
        modules.host._unpack_source(archive, tmp_path / "source")


@pytest.mark.parametrize(
    "mutation", ["revision", "user", "entrypoint", "bootstrap", "id"]
)
def test_image_contract_rejects_changed_identity(modules, tmp_path, mutation):
    observed = _image()
    if mutation == "revision":
        observed["Config"]["Labels"]["org.opencontainers.image.revision"] = "3" * 40
    elif mutation == "bootstrap":
        observed["Config"]["Labels"].clear()
    elif mutation == "user":
        observed["Config"]["User"] = "root"
    elif mutation == "entrypoint":
        observed["Config"]["Entrypoint"] = ["bash"]
    else:
        observed["Id"] = "sha256:" + "4" * 64
    commands = SimpleNamespace(run=lambda *args: json.dumps([observed]).encode())
    arguments = argparse.Namespace(image_id=IMAGE_ID, revision=REVISION)
    with pytest.raises(modules.docker._ValidationError):
        modules.host._image_contract(commands, arguments)


class _DockerSimulation:
    def __init__(self, modules, root):
        self.modules, self.root = modules, root
        self.commands, self.containers = [], {}
        self.fail_phase = self.fail_cleanup = ""
        self.private_environment = []

    def __call__(self, argv, **kwargs):
        self.commands.append(argv)
        self.private_environment.append(kwargs["env"])
        result, output = self._dispatch(argv)
        kwargs["stdout"].write(output)
        return SimpleNamespace(returncode=result)

    def _dispatch(self, argv):
        operation = argv[1]
        if operation == "inspect":
            record = _image() if argv[2] == IMAGE_ID else self._find(argv[2])
            return 0, json.dumps([record]).encode()
        if operation == "create":
            return self._create(argv)
        if operation in {"start", "exec"}:
            return self._payload(argv)
        if operation == "network":
            self._find(argv[-1])["NetworkSettings"]["Networks"] = {}
        elif operation == "stop":
            self._find(argv[-1])["State"] = {
                "Running": False,
                "Pid": 0,
                "ExitCode": 137,
            }
        elif operation == "rm":
            if self.fail_cleanup:
                return 1, b""
            del self.containers[argv[-1]]
        elif operation != "ps":
            raise AssertionError(f"Unexpected Docker operation {operation}")
        return 0, b""

    def _find(self, identity):
        matches = [
            row
            for key, row in self.containers.items()
            if key == identity or row["Name"] == "/" + identity
        ]
        assert len(matches) == 1
        return matches[0]

    def _create(self, argv):
        identity = str(len(self.containers) + 5) * 64
        name = argv[argv.index("--name") + 1]
        nonce = argv[argv.index("--label") + 1].split("=", 1)[1]
        network = argv[argv.index("--network") + 1]
        mounts = self._mounts(argv)
        environment = [
            argv[index + 1] for index, value in enumerate(argv) if value == "--env"
        ]
        self.containers[identity] = {
            "Id": identity,
            "Name": "/" + name,
            "Image": IMAGE_ID,
            "Created": "synthetic creation lifetime",
            "Mounts": mounts,
            "Config": {
                "Image": IMAGE_ID,
                "Labels": {"npa.fiftyone.validation": nonce},
                "Env": environment,
            },
            "State": {"Running": False, "Pid": 0, "ExitCode": 0},
            "NetworkSettings": {"Networks": {network: {}}},
        }
        return 0, identity.encode()

    def _mounts(self, argv):
        mounts = []
        for index, value in enumerate(argv):
            if value != "--mount":
                continue
            parts = argv[index + 1].split(",")
            fields = dict(item.split("=", 1) for item in parts if "=" in item)
            mounts.append(
                {
                    "Type": fields["type"],
                    "Source": fields["src"],
                    "Destination": fields["dst"],
                    "RW": "readonly" not in parts,
                }
            )
        return mounts

    def _payload(self, argv):
        if argv[1] == "start" and "--attach" not in argv:
            self._find(argv[-1])["State"] = {"Running": True, "Pid": 9}
            return 0, b""
        phase = "bare" if argv[1] == "start" else argv[-1]
        if phase == "initial-install":
            identity = next(value for value in argv if value in self.containers)
            mounts = self.containers[identity]["Mounts"]
            source = next(
                row for row in mounts if row["Destination"].endswith("/npa-src")
            )
            receipt = next(
                row for row in mounts if row["Destination"].endswith("/npa-src-root")
            )
            Path(receipt["Source"]).write_text(source["Destination"])
        output = self.root / ("bare" if phase == "bare" else "post")
        checks = [
            {"name": name, "ok": True, "result": {}}
            for name, _ in self.modules.checks._phase_checks(phase)
        ]
        status = "passed"
        if phase == self.fail_phase:
            checks[-1]["ok"], status = False, "failed"
        (output / f"{phase}.json").write_text(
            json.dumps({"phase": phase, "status": status, "checks": checks})
        )
        return (0 if status == "passed" else 17), b""


@pytest.fixture
def execution(tmp_path, modules, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    checks = tmp_path / "checks"
    checks.mkdir()
    (tmp_path / "source.tar").write_bytes(b"synthetic source archive")
    (tmp_path / "source.json").write_text(json.dumps({"files": []}))
    commands = modules.docker._Commands(tmp_path, ROOT)
    simulation = _DockerSimulation(modules, tmp_path)
    monkeypatch.setattr(modules.docker.subprocess, "run", simulation)
    monkeypatch.setattr(modules.host, "_source_inputs", lambda *args: (source, checks))
    arguments = argparse.Namespace(image_id=IMAGE_ID, revision=REVISION)
    return commands, arguments, simulation


def test_actual_orchestration_keeps_functional_phases_offline_and_cleans_exact_containers(
    modules, execution
):
    commands, arguments, simulation = execution
    result = modules.host._validate(commands, arguments)
    assert result["status"] == "passed" and len(result["cleanup"]) == 2
    assert [item["shutdown_exit_code"] for item in result["cleanup"]] == [137, 0]
    assert simulation.containers == {}
    operations = simulation.commands
    disconnect = next(
        index for index, argv in enumerate(operations) if argv[1] == "network"
    )
    install = next(
        index for index, argv in enumerate(operations) if argv[-1] == "initial-install"
    )
    post = next(
        index for index, argv in enumerate(operations) if argv[-1] == "post-install"
    )
    assert install < disconnect < post
    creates = [argv for argv in operations if argv[1] == "create"]
    assert len(creates) == 2 and creates[0][creates[0].index("--network") + 1] == "none"
    assert all("--pull=never" in argv and IMAGE_ID in argv for argv in creates)
    assert not any(
        set(argv) & {"--privileged", "--user", "--entrypoint", "-p", "--publish"}
        for argv in creates
    )
    assert len([argv for argv in operations if argv[1] == "rm"]) == 2


@pytest.mark.parametrize("phase", ["bare", "initial-install", "post-install"])
def test_payload_failure_stops_pipeline_and_preserves_exact_exit(
    modules, execution, phase
):
    commands, arguments, simulation = execution
    simulation.fail_phase = phase
    with pytest.raises(
        modules.docker._ValidationError, match="validation failed"
    ) as error:
        modules.host._validate(commands, arguments)
    assert isinstance(error.value.__cause__, modules.docker._ValidationError)
    result = json.loads((commands.root / "result.json").read_text())
    assert "exited 17" in result["payload_failure"]
    assert simulation.containers == {} and result["status"] == "failed"
    assert (
        not any(argv[-1] == "post-install" for argv in simulation.commands)
        if phase != "post-install"
        else True
    )


def test_cleanup_failure_preserves_payload_cause_and_private_evidence(
    modules, execution
):
    commands, arguments, simulation = execution
    simulation.fail_phase, simulation.fail_cleanup = "post-install", "remove refused"
    with pytest.raises(modules.docker._ValidationError) as error:
        modules.host._validate(commands, arguments)
    assert "post-install exited 17" in str(error.value.__cause__)
    result = json.loads((commands.root / "result.json").read_text())
    assert (
        result["cleanup_errors"]
        and "post-install exited 17" in result["payload_failure"]
    )
    assert (
        json.loads((commands.root / "public-summary.json").read_text())[
            "owned_cleanup"
        ]["verified"]
        is False
    )


@pytest.mark.parametrize("field", ["label", "id", "name", "image", "lifetime"])
def test_cleanup_refuses_changed_owned_identity_without_stop_or_remove(
    modules, tmp_path, field
):
    container = modules.docker._Container(
        "bare", "fixture", IMAGE_ID, "nonce", "7" * 64, "created"
    )
    row = {
        "Id": container.container_id,
        "Name": "/fixture",
        "Image": IMAGE_ID,
        "Created": "created",
        "Config": {"Image": IMAGE_ID, "Labels": {"npa.fiftyone.validation": "nonce"}},
        "State": {"Running": True, "Pid": 9},
    }
    if field == "label":
        row["Config"]["Labels"].clear()
    else:
        key = {"id": "Id", "name": "Name", "image": "Image", "lifetime": "Created"}[
            field
        ]
        row[key] = "changed"
    calls = []

    def command(name, argv):
        calls.append(argv)
        return json.dumps([row]).encode()

    commands = SimpleNamespace(root=tmp_path, run=command)
    with pytest.raises(modules.docker._ValidationError):
        modules.docker._retire(commands, container)
    assert len(calls) == 1 and calls[0][1] == "inspect"


@pytest.mark.parametrize(
    "change", ["missing", "failed", "wrong-order", "malformed", "wrong-phase"]
)
def test_phase_receipt_cannot_hide_incomplete_payload(modules, tmp_path, change):
    checks = [
        {"name": name, "ok": True} for name, _ in modules.checks._phase_checks("bare")
    ]
    record = {"phase": "bare", "status": "passed", "checks": checks}
    if change == "missing":
        checks.pop()
    elif change == "failed":
        checks[-1]["ok"] = False
    elif change == "wrong-order":
        checks.reverse()
    elif change == "malformed":
        checks[-1] = None
    else:
        record["phase"] = "post-install"
    (tmp_path / "bare.json").write_text(json.dumps(record))
    with pytest.raises(modules.docker._ValidationError):
        modules.host._require_phase(tmp_path, "bare")


def test_commands_do_not_inherit_cloud_registry_or_python_selectors(
    modules, tmp_path, monkeypatch
):
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
        "DOCKER_HOST",
        "PYTHONPATH",
    ):
        monkeypatch.setenv(name, "synthetic-private-value")
    commands = modules.docker._Commands(tmp_path, ROOT)
    environment = commands._environment()
    assert set(environment) == {"PATH", "HOME", "LANG", "DOCKER_CONFIG"}
    assert environment["DOCKER_CONFIG"] == str(tmp_path / "docker-config")
    assert "synthetic-private-value" not in json.dumps(environment)


def test_setup_is_exact_current_literal_without_import_or_overlay(
    modules, tmp_path, monkeypatch
):
    from npa.orchestration.npa_workflow.skypilot_render import default_npa_setup

    monkeypatch.setattr(modules.checks, "_SOURCE", ROOT / "npa")
    monkeypatch.setattr(modules.checks, "_OUTPUT", tmp_path)
    result = modules.checks._extract_setup()
    assert (tmp_path / "default-npa-setup.sh").read_text() == default_npa_setup()
    assert result["extraction"] == "unchanged AST literal return"


@pytest.mark.parametrize("requirement", ["paramiko>=3,<4", "paramiko<5", "paramiko>=6"])
def test_actual_dependency_requirement_rejects_incompatible_security_floor(
    modules, monkeypatch, requirement
):
    requirements = {
        "fiftyone": ["voxel51-eta>=0.17,<0.18"],
        "voxel51-eta": [requirement],
    }
    monkeypatch.setattr(modules.checks.metadata, "requires", requirements.get)
    versions = {
        "fiftyone": "1.21.0",
        "voxel51-eta": "0.17.0",
        "paramiko": "5.0.0",
        "datasets": "5.0.1",
        "pillow": "12.3.0",
        "starlette": "1.3.1",
    }
    with pytest.raises(RuntimeError, match="declared requirements"):
        modules.checks._compatibility(versions)


def test_compatible_dependencies_and_normal_identifier_values_remain(
    modules, monkeypatch
):
    requirements = {
        "fiftyone": ["voxel51-eta>=0.17,<0.18"],
        "voxel51-eta": ["paramiko>=3,<6"],
    }
    monkeypatch.setattr(modules.checks.metadata, "requires", requirements.get)
    versions = {
        "fiftyone": "1.21.0",
        "voxel51-eta": "0.17.0",
        "paramiko": "5.0.0",
        "datasets": "5.0.1",
        "pillow": "12.3.0",
        "starlette": "1.3.1",
    }
    assert modules.checks._compatibility(versions)["voxel51-eta"] == ["paramiko<6,>=3"]


def test_dataset_emitted_files_are_contained_without_testing_vendor_exploitation(
    modules, tmp_path
):
    root = tmp_path / "dataset"
    root.mkdir()
    contained = root / "part.arrow"
    contained.write_bytes(b"synthetic fixture")
    outside = tmp_path / "different-fixture.arrow"
    outside.write_bytes(b"different synthetic fixture")
    assert modules.checks._contained_files(root, [contained.name, str(contained)]) == [
        contained,
        contained,
    ]
    with pytest.raises(RuntimeError, match="escaped its fixture"):
        modules.checks._contained_files(root, [str(outside)])
    with pytest.raises(RuntimeError, match="no data files"):
        modules.checks._contained_files(root, [])


def test_publication_orders_runtime_gate_before_login_and_push_and_uploads_only_summary():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/publish-public-images.yml").read_text()
    )
    steps = next(
        job["steps"]
        for job in workflow["jobs"].values()
        if any(step.get("id") == "fiftyone-validation" for step in job.get("steps", []))
    )
    gate = next(
        index
        for index, step in enumerate(steps)
        if step.get("id") == "fiftyone-validation"
    )
    push = next(index for index, step in enumerate(steps) if step.get("id") == "push")
    login = max(
        index
        for index, step in enumerate(steps[:push])
        if "docker/login-action" in step.get("uses", "")
    )
    assert gate < login < push
    assert steps[gate]["if"] == "matrix.tool == 'fiftyone'"
    assert '--image-id "$image_id"' in steps[gate]["run"]
    upload = next(
        step
        for step in steps
        if step.get("name") == "Retain only the public FiftyOne validation summary"
    )
    assert (
        upload["with"]["path"]
        == "${{ runner.temp }}/fiftyone-validation/public-summary.json"
    )
    assert "always()" in upload["if"] and upload["with"]["if-no-files-found"] == "error"
    assert upload["uses"] == (
        "actions/upload-artifact@330a01c490aca151604b8cf639adc76d48f6c5d4"
    )


@pytest.mark.parametrize(
    "filename", ["verify_source.py", "source.json", "version.json", "SOURCE.md"]
)
def test_annex_must_match_committed_metadata_before_helper_execution(
    modules, tmp_path, monkeypatch, filename
):
    source, installed = tmp_path / "source", tmp_path / "installed"
    committed = source / "docker/workbench/fiftyone"
    committed.mkdir(parents=True)
    installed.mkdir()
    names = {
        "verify_source.py": "verify_source.py",
        "source.json": "mongodb-source.json",
        "version.json": "mongodb-source-version.json",
        "SOURCE.md": "SOURCE.md",
    }
    for baked, original in names.items():
        data = (IMAGE / original).read_bytes()
        (installed / baked).write_bytes(data)
        (committed / original).write_bytes(data)
    monkeypatch.setattr(modules.checks, "_SOURCE", source)
    monkeypatch.setattr(modules.checks, "_SMOKES", installed)
    monkeypatch.setattr(modules.checks, "_NOTICES", installed)
    modules.checks._annex_source_files()
    (installed / filename).write_text(
        "self-consistent alternate source is not this commit"
    )
    with pytest.raises(RuntimeError, match="source binding differs"):
        modules.checks._annex_source_files()


@pytest.mark.parametrize(
    "arguments,pid",
    [
        (["literal argument", "--forwarded-option"], 2),
        (["literal", "argument", "--forwarded-option"], 1),
        (["literal argument", "--forwarded-option"], 1),
    ],
)
def test_actual_entrypoint_contract_requires_exec_and_literal_arguments(
    modules, monkeypatch, arguments, pid
):
    monkeypatch.setattr(modules.checks.sys, "argv", ["checks", "bare", *arguments])
    monkeypatch.setattr(modules.checks.os, "getpid", lambda: pid)
    if pid == 1 and len(arguments) == 2:
        assert modules.checks._entrypoint()["arguments_preserved"] is True
    else:
        with pytest.raises(RuntimeError, match="Entrypoint"):
            modules.checks._entrypoint()


@pytest.mark.parametrize(
    "filename,total", [("smoke_env.py", 3), ("smoke_functional.py", 5)]
)
@pytest.mark.parametrize("reported", ["complete", "incomplete", "missing"])
def test_smoke_exit_zero_still_requires_complete_real_check_summary(
    modules, tmp_path, monkeypatch, filename, total, reported
):
    def command(name, argv):
        text = "ordinary diagnostic\n"
        if reported != "missing":
            count = total if reported == "complete" else total - 1
            text += f"SUMMARY: {count}/{total} checks passed\n"
        (tmp_path / f"{name}.stdout").write_text(text)
        return {"exit_code": 0}

    monkeypatch.setattr(modules.checks, "_OUTPUT", tmp_path)
    monkeypatch.setattr(modules.checks, "_command", command)
    if reported == "complete":
        assert modules.checks._smoke("smoke", filename)["checks_passed"] == total
    else:
        with pytest.raises(RuntimeError, match="complete checks"):
            modules.checks._smoke("smoke", filename)


def test_container_command_retains_payload_exit_and_start_end_without_console_dump(
    modules, tmp_path, monkeypatch
):
    monkeypatch.setattr(modules.checks, "_OUTPUT", tmp_path)

    def failed(argv, **kwargs):
        kwargs["stderr"].write(b"private synthetic diagnostic\n")
        return SimpleNamespace(returncode=12)

    monkeypatch.setattr(modules.checks.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="pip exited 12"):
        modules.checks._command("pip", ["fixture-executable"])
    record = json.loads((tmp_path / "pip.exit.json").read_text())
    assert record["exit_code"] == 12 and record["started_at"] <= record["ended_at"]
    assert (tmp_path / "pip.stderr").read_text() == "private synthetic diagnostic\n"


def _public_receipt_fixture(sensitive):
    return [
        {
            "name": "bare-packages",
            "ok": True,
            "result": {
                "versions": {
                    "fiftyone": "1.21.0",
                    "private-package": sensitive,
                    "npa": sensitive,
                },
                "environment": sensitive,
            },
        },
        {
            "name": "bare-functional",
            "ok": True,
            "result": {"checks_passed": 5, "checks_total": 5, "log": sensitive},
        },
        {
            "name": "mongodb-source-annex",
            "ok": True,
            "result": {
                "source_delivery": "verified",
                "source_members": 44555,
                "version": "7.0.40",
                "path": sensitive,
            },
        },
        {"name": sensitive, "ok": True, "result": {}},
    ]


def test_public_summary_uses_only_explicit_package_gate_and_annex_fields(
    modules, tmp_path
):
    commands = modules.docker._Commands(tmp_path, ROOT)
    output = tmp_path / "bare"
    output.mkdir()
    sensitive = "synthetic-private-diagnostic"
    checks = _public_receipt_fixture(sensitive)
    (output / "bare.json").write_text(
        json.dumps({"checks": checks, "diagnostic": sensitive})
    )
    result = {
        "status": "failed",
        "image_id": IMAGE_ID,
        "revision": REVISION,
        "source_archive_sha256": "8" * 64,
        "cleanup": [],
        "cleanup_errors": [sensitive],
        "payload_failure": sensitive,
    }
    summary = modules.host._public_results(commands, result)
    assert sensitive not in json.dumps(summary)
    assert summary["package_versions"] == {"fiftyone": "1.21.0"}
    assert summary["source_annex"] == {
        "verified": True,
        "source_members": 44555,
        "version": "7.0.40",
    }
    assert summary["owned_cleanup"]["verified"] is False
    assert summary["phases"][0]["checks"][1]["checks_total"] == 5


@pytest.mark.parametrize(
    "networks", [{"bridge": {}}, {"none": {}, "unexpected": {}}, None]
)
def test_unknown_or_remaining_network_attachment_blocks_functional_work(
    modules, tmp_path, networks
):
    container = modules.docker._Container(
        "bare", "fixture", IMAGE_ID, "nonce", "7" * 64, "created"
    )
    row = {
        "Id": container.container_id,
        "Name": "/fixture",
        "Image": IMAGE_ID,
        "Created": "created",
        "Config": {"Image": IMAGE_ID, "Labels": {"npa.fiftyone.validation": "nonce"}},
        "NetworkSettings": {"Networks": networks},
    }
    commands = SimpleNamespace(run=lambda *args: json.dumps([row]).encode())
    with pytest.raises(modules.docker._ValidationError, match="external network"):
        modules.host._offline(commands, container, "offline")


def test_unresolved_create_response_still_cleans_only_nonce_owned_container(
    modules, execution, monkeypatch
):
    commands, arguments, simulation = execution
    original = simulation._create

    def partial(argv):
        _, output = original(argv)
        return 17, output

    monkeypatch.setattr(simulation, "_create", partial)
    with pytest.raises(modules.docker._ValidationError) as error:
        modules.host._validate(commands, arguments)
    assert "bare-create exited 17" in str(error.value.__cause__)
    assert simulation.containers == {}
    assert len([argv for argv in simulation.commands if argv[1] == "rm"]) == 1
    assert not any(argv[1] in {"start", "exec"} for argv in simulation.commands)


def test_cli_private_failure_retains_original_reason_without_public_disclosure(
    modules, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "private"

    def fail(*args):
        raise RuntimeError("synthetic-private-credential-like-diagnostic")

    monkeypatch.setattr(modules.host, "_validate", fail)
    assert (
        modules.host.main(
            [
                "--image-id",
                IMAGE_ID,
                "--revision",
                REVISION,
                "--source-root",
                str(ROOT),
                "--output-path",
                str(output),
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out) == {"status": "failed"}
    assert "synthetic-private" in (output / "failure.json").read_text()
    assert "synthetic-private" not in (output / "public-summary.json").read_text()


@pytest.mark.parametrize("origin_header,status", [(None, 200), ("*", 200), (None, 503)])
def test_benign_cors_uses_actual_selected_app_and_checks_normal_response(
    modules, monkeypatch, origin_header, status
):
    from starlette.applications import Starlette
    from starlette.responses import Response
    from starlette.routing import Route

    async def root(request):
        headers = (
            {}
            if origin_header is None
            else {"Access-Control-Allow-Origin": origin_header}
        )
        return Response("synthetic app root", status_code=status, headers=headers)

    app = Starlette(routes=[Route("/", root)])
    monkeypatch.setitem(
        sys.modules,
        "fiftyone",
        SimpleNamespace(
            config=SimpleNamespace(default_app_address="127.0.0.1", allowed_origins="")
        ),
    )
    monkeypatch.setitem(sys.modules, "fiftyone.server.app", SimpleNamespace(app=app))
    if origin_header is None and status == 200:
        assert modules.checks._app_access()["response_transport"] == "in-process ASGI"
    else:
        with pytest.raises(RuntimeError, match="App"):
            modules.checks._app_access()


def _change_mount_observation(row, change):
    mounts = row["Mounts"]
    if change == "missing":
        mounts.pop()
    elif change == "duplicate":
        mounts[-1] = dict(mounts[0])
    elif change == "type":
        mounts[1]["Type"] = "volume"
    elif change == "source":
        mounts[1]["Source"] += "-different"
    elif change == "checks-writable":
        mounts[0]["RW"] = True
    elif change == "source-readonly":
        mounts[1]["RW"] = False
    elif change == "receipt-readonly":
        mounts[2]["RW"] = False
    elif change == "receipt-source":
        mounts[2]["Source"] = mounts[1]["Source"]
    elif change == "destination":
        mounts[1]["Destination"] += "-different"
    else:
        row["Config"]["Env"] = ["NPA_FIFTYONE_VALIDATION_CONTRACT=wrong"]


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "type",
        "source",
        "checks-writable",
        "source-readonly",
        "receipt-readonly",
        "receipt-source",
        "destination",
        "selector",
    ],
)
def test_mount_mismatch_refuses_before_payload_and_cleans_only_owned_container(
    modules, execution, monkeypatch, change
):
    commands, arguments, simulation = execution
    original = simulation._create

    def create(argv):
        result, identity = original(argv)
        _change_mount_observation(simulation.containers[identity.decode()], change)
        return result, identity

    monkeypatch.setattr(simulation, "_create", create)
    with pytest.raises(modules.docker._ValidationError, match="validation failed"):
        modules.host._validate(commands, arguments)
    assert not any(argv[1] in {"start", "exec"} for argv in simulation.commands)
    assert simulation.containers == {}
    assert len([argv for argv in simulation.commands if argv[1] == "rm"]) == 1


def _producer_mount_destinations():
    from npa.orchestration.npa_workflow.skypilot_render import default_npa_setup

    lines = default_npa_setup().splitlines()
    writer = next(line for line in lines if line.startswith("npa_record_src_root()"))
    tokens = shlex.split(writer)
    receipt = tokens[tokens.index(">") + 1].rstrip(";")
    calls = [
        shlex.split(line)
        for line in lines
        if line.strip().startswith("npa_record_src_root ")
    ]
    mounted = [call[1] for call in calls if Path(call[1]).name == "npa-src"]
    assert len(mounted) == 1
    source = mounted[0]
    installs = [
        shlex.split(line)
        for line in lines
        if line.strip().startswith("npa_pip_install -e ")
    ]
    assert ["npa_pip_install", "-e", source] in installs
    return source, receipt


def test_receipts_use_unpredictable_private_backing_and_same_canonical_setup_mounts(
    modules, execution, monkeypatch
):
    commands, arguments, simulation = execution
    commands.root.chmod(0o700)
    original = simulation._create
    backing = []
    source_destination, receipt_destination = _producer_mount_destinations()

    def create(argv):
        result, identity = original(argv)
        mounts = simulation.containers[identity.decode()]["Mounts"]
        source = next(row for row in mounts if row["Destination"] == source_destination)
        receipt = next(
            row for row in mounts if row["Destination"] == receipt_destination
        )
        path = Path(receipt["Source"])
        assert path.parent.parent == commands.root and path.read_bytes() == b""
        assert (
            path.name.startswith("npa-source-receipt-") and path.name != "npa-src-root"
        )
        assert path.stat().st_mode & 0o777 == 0o666
        assert source["RW"] is receipt["RW"] is True
        backing.append(path)
        return result, identity

    monkeypatch.setattr(simulation, "_create", create)
    assert modules.host._validate(commands, arguments)["status"] == "passed"
    assert len(set(backing)) == 2 and commands.root.stat().st_mode & 0o777 == 0o700
    assert [path.read_text() for path in backing] == ["", source_destination]
    for role in ("bare", "post"):
        proof = json.loads((commands.root / f"{role}-receipt-after.json").read_text())
        assert proof["same_backing_file"] and proof["contents_match_setup"]


@pytest.mark.parametrize("change", ["replaced", "symlink", "content"])
def test_receipt_tampering_blocks_further_phases_and_preserves_owned_cleanup(
    modules, execution, monkeypatch, change
):
    commands, arguments, simulation = execution
    original = simulation._payload

    def payload(argv):
        result = original(argv)
        record = json.loads((commands.root / "bare-receipt.json").read_text())
        path = Path(record["path"])
        if change == "content":
            path.write_text("different setup")
        else:
            saved = path.with_suffix(".original")
            path.rename(saved)
            if change == "symlink":
                path.symlink_to(saved)
            else:
                path.write_bytes(b"")
        return result

    monkeypatch.setattr(simulation, "_payload", payload)
    with pytest.raises(modules.docker._ValidationError) as failure:
        modules.host._validate(commands, arguments)
    assert "receipt" in str(failure.value.__cause__)
    assert simulation.containers == {}
    assert not any(argv[-1] == "initial-install" for argv in simulation.commands)


@pytest.fixture
def mount_contract(modules, tmp_path, monkeypatch):
    checks, source, receipt = (
        tmp_path / "validation",
        tmp_path / "npa-src",
        tmp_path / "npa-src-root",
    )
    checks.mkdir()
    source.mkdir()
    receipt.write_bytes(b"")
    path = checks / "bare-mounts.json"
    contract = {
        "schema": "npa.fiftyone.mounts.v1",
        "checks": {"Destination": str(checks), "RW": False},
        "source": {"Type": "bind", "Destination": str(source), "RW": True},
        "receipt": {"Type": "bind", "Destination": str(receipt), "RW": True},
    }
    path.write_text(json.dumps(contract))
    monkeypatch.setenv("NPA_FIFTYONE_VALIDATION_CONTRACT", str(path))
    monkeypatch.setattr(
        modules.checks,
        "Path",
        lambda value: checks if value == "/validation" else Path(value),
    )
    return path, contract, source, receipt


def test_checks_select_verified_source_and_empty_receipt(modules, mount_contract):
    _, _, source, receipt = mount_contract
    modules.checks._configure_mounts()
    assert modules.checks._SOURCE == source and modules.checks._RECEIPT == receipt
    assert receipt.read_bytes() == b""


@pytest.fixture
def receipt_ownership(modules, tmp_path, monkeypatch):
    receipt = tmp_path / "npa-src-root"
    receipt.write_bytes(b"")
    observed = list(receipt.lstat())
    observed[4:6] = [1001, 1001]
    calls = []
    original = Path.lstat

    def inspect(path, *args, **kwargs):
        if path == receipt:
            return os.stat_result(observed)
        return original(path, *args, **kwargs)

    def chown(name, argv):
        calls.append((name, argv))
        observed[4:6] = [1000, 1000]
        return {"exit_code": 0}

    monkeypatch.setattr(Path, "lstat", inspect)
    monkeypatch.setattr(modules.checks.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(modules.checks.os, "getegid", lambda: 1000)
    monkeypatch.setattr(modules.checks, "_RECEIPT", receipt)
    monkeypatch.setattr(modules.checks, "_command", chown)
    return receipt, observed, calls


@pytest.mark.parametrize("same_owner", [True, False])
def test_initial_receipt_matches_runtime_owner_before_unchanged_setup(
    modules, receipt_ownership, same_owner
):
    receipt, observed, calls = receipt_ownership
    if same_owner:
        observed[4:6] = [1000, 1000]
    before = receipt.lstat()
    assert modules.checks._source_receipt_owner() == {
        "owner_matches_runtime": True,
        "same_backing_file": True,
    }
    assert receipt.read_bytes() == b"" and receipt.lstat().st_ino == before.st_ino
    expected = [
        "sudo",
        "-n",
        "chown",
        "--no-dereference",
        "--",
        "1000:1000",
        str(receipt),
    ]
    assert calls == ([] if same_owner else [("install-source-receipt-owner", expected)])
    names = [name for name, _ in modules.checks._install_checks()]
    assert names.index("initial-source-absence") < names.index(
        "install-source-receipt-owner"
    )
    assert names.index("install-source-receipt-owner") < names.index(
        "initial-default-npa-setup"
    )


@pytest.mark.parametrize("change", ["owner", "inode", "contents", "chown-failed"])
def test_receipt_ownership_failure_stops_before_setup(
    modules, receipt_ownership, monkeypatch, change
):
    receipt, observed, _ = receipt_ownership

    def unsuccessful(name, argv):
        if change == "chown-failed":
            raise RuntimeError("ownership command exited 1")
        if change != "owner":
            observed[4:6] = [1000, 1000]
        if change == "inode":
            observed[1] += 1
        if change == "contents":
            receipt.write_text("unexpected setup receipt")

    monkeypatch.setattr(modules.checks, "_command", unsuccessful)
    with pytest.raises(RuntimeError, match="receipt|ownership command"):
        modules.checks._source_receipt_owner()


@pytest.mark.parametrize("change", ["root", "populated", "symlink"])
def test_receipt_ownership_refuses_ineligible_input_without_chown(
    modules, receipt_ownership, monkeypatch, change
):
    receipt, observed, calls = receipt_ownership
    if change == "root":
        monkeypatch.setattr(modules.checks.os, "geteuid", lambda: 0)
    elif change == "populated":
        receipt.write_text("already populated")
    else:
        observed[0] = 0o120777
    with pytest.raises(RuntimeError, match="receipt|non-root"):
        modules.checks._source_receipt_owner()
    assert calls == []


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "schema",
        "writable-checks",
        "source-mode",
        "receipt-mode",
        "source-name",
        "receipt-name",
        "receipt-directory",
    ],
)
def test_checks_refuse_missing_or_incompatible_mount_contract(
    modules, mount_contract, monkeypatch, change
):
    path, contract, _, receipt = mount_contract
    if change == "missing":
        monkeypatch.delenv("NPA_FIFTYONE_VALIDATION_CONTRACT")
    elif change == "schema":
        contract["schema"] = "unknown"
    elif change == "writable-checks":
        contract["checks"]["RW"] = True
    elif change in {"source-mode", "receipt-mode"}:
        contract[change.split("-")[0]]["RW"] = False
    elif change in {"source-name", "receipt-name"}:
        contract[change.split("-")[0]]["Destination"] += "-different"
    else:
        receipt.unlink()
        receipt.mkdir()
    path.write_text(json.dumps(contract))
    with pytest.raises(RuntimeError):
        modules.checks._configure_mounts()


def test_temp_writability_uses_selected_root_and_removes_own_fixtures(
    modules, tmp_path, monkeypatch
):
    selected = tmp_path / "selected-temporary-root"
    selected.mkdir()
    monkeypatch.setattr(modules.checks.tempfile, "gettempdir", lambda: str(selected))
    original = modules.checks.tempfile.TemporaryDirectory
    observed = []

    def temporary(*, prefix, dir):
        observed.append(Path(dir))
        return original(prefix=prefix, dir=selected)

    monkeypatch.setattr(modules.checks.tempfile, "TemporaryDirectory", temporary)
    assert str(selected) in modules.checks._writable_paths()
    assert selected in observed and len(observed) == 5
    assert list(selected.iterdir()) == []
