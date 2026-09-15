"""Exercise exact-image validation, offline phases, owned cleanup and public evidence."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
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
        self.containers[identity] = {
            "Id": identity,
            "Name": "/" + name,
            "Image": IMAGE_ID,
            "Created": "synthetic creation lifetime",
            "Config": {"Image": IMAGE_ID, "Labels": {"npa.fiftyone.validation": nonce}},
            "State": {"Running": False, "Pid": 0, "ExitCode": 0},
            "NetworkSettings": {"Networks": {network: {}}},
        }
        return 0, identity.encode()

    def _payload(self, argv):
        if argv[1] == "start" and "--attach" not in argv:
            self._find(argv[-1])["State"] = {"Running": True, "Pid": 9}
            return 0, b""
        phase = "bare" if argv[1] == "start" else argv[-1]
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
