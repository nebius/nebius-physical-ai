"""Exercise the real FiftyOne runtime with synthetic fixtures inside owned containers."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import traceback
from importlib import metadata
from pathlib import Path
from datetime import datetime, timezone


_SOURCE: Path | None = None
_RECEIPT: Path | None = None
_OUTPUT = Path("/evidence")
_PYTHON = "/opt/fiftyone/venv/bin/python"
_SMOKES = Path("/opt/npa/docker/workbench/fiftyone")
_NOTICES = Path("/opt/fiftyone/mongodb-notices")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command(name: str, argv: list[str]) -> dict:
    record = {"argv": argv, "started_at": datetime.now(timezone.utc).isoformat()}
    _json(_OUTPUT / f"{name}.command.json", record)
    with (_OUTPUT / f"{name}.stdout").open("wb") as stdout:
        with (_OUTPUT / f"{name}.stderr").open("wb") as stderr:
            result = subprocess.run(argv, stdout=stdout, stderr=stderr, check=False)
    record.update(
        exit_code=result.returncode, ended_at=datetime.now(timezone.utc).isoformat()
    )
    _json(_OUTPUT / f"{name}.exit.json", record)
    _require(result.returncode == 0, f"{name} exited {result.returncode}")
    return {
        "exit_code": result.returncode,
        "stdout_sha256": _sha(_OUTPUT / f"{name}.stdout"),
    }


def _version_requirements() -> dict[str, str]:
    dockerfile = (_SOURCE / "docker/workbench/fiftyone/Dockerfile").read_text()
    versions = {}
    names = {
        "fiftyone": "FIFTYONE",
        "datasets": "DATASETS",
        "pillow": "PILLOW",
        "boto3": "BOTO3",
        "pyarrow": "PYARROW",
        "huggingface-hub": "HUGGINGFACE_HUB",
    }
    for package, argument in names.items():
        found = re.findall(
            rf"^ARG {argument}_VERSION=([^\s]+)$", dockerfile, re.MULTILINE
        )
        _require(len(found) == 1, f"Source must declare one {package} version")
        versions[package] = found[0]
    found = re.findall(r'"paramiko==([^"\s]+)"', dockerfile)
    _require(len(found) == 1, "Source must pin one Paramiko version")
    versions["paramiko"] = found[0]
    return versions


def _compatibility(versions: dict[str, str]) -> dict:
    from packaging.requirements import Requirement
    from packaging.version import Version

    dependencies = {}
    for package, dependency in (
        ("fiftyone", "voxel51-eta"),
        ("voxel51-eta", "paramiko"),
    ):
        requirements = [
            Requirement(value) for value in metadata.requires(package) or []
        ]
        selected = [value for value in requirements if value.name.lower() == dependency]
        _require(bool(selected), f"{package} must declare {dependency} compatibility")
        _require(
            all(Version(versions[dependency]) in value.specifier for value in selected),
            f"Installed {dependency} violates {package}'s declared requirements",
        )
        dependencies[package] = [str(value) for value in selected]
    floors = {
        "fiftyone": "1.21.0",
        "datasets": "5.0.1",
        "pillow": "12.3.0",
        "paramiko": "5.0.0",
        "starlette": "1.3.1",
    }
    for name, minimum in floors.items():
        _require(
            Version(versions[name]) >= Version(minimum),
            f"{name} is below the security floor",
        )
    return dependencies


def _packages(phase: str) -> dict:
    _require(os.geteuid() != 0, "The actual runtime must be non-root")
    _require(
        sys.executable == _PYTHON, "Checks must use the image's vendor interpreter"
    )
    wanted = _version_requirements()
    versions = {
        name: metadata.version(name)
        for name in set(wanted)
        | {
            "voxel51-eta",
            "fiftyone-brain",
            "fiftyone-db",
            "numpy",
            "scikit-learn",
            "starlette",
        }
    }
    for name, expected in wanted.items():
        _require(
            versions[name] == expected,
            f"Installed {name} does not match the source version",
        )
    dependencies = _compatibility(versions)
    _built_scripts()
    if phase != "bare":
        versions["npa"] = _installed_npa_version(phase)
    installed = {
        dist.metadata["Name"]: dist.version for dist in metadata.distributions()
    }
    _json(_OUTPUT / f"{phase}-packages.json", installed)
    return {
        "versions": versions,
        "dependency_requirements": dependencies,
        "uid": os.geteuid(),
    }


def _built_scripts() -> None:
    for filename in (
        "smoke_env.py",
        "smoke_functional.py",
        "entrypoint.sh",
        "verify_source.py",
    ):
        _require(
            (_SMOKES / filename).read_bytes()
            == (_SOURCE / "docker/workbench/fiftyone" / filename).read_bytes(),
            f"Built {filename} differs from the source archive",
        )


def _installed_npa_version(phase: str) -> str:
    # Editable installation adds startup hooks that this pre-install process has not loaded.
    name = f"{phase}-npa-import"
    _command(
        name,
        [
            _PYTHON,
            "-c",
            "import json, npa; from importlib import metadata; "
            "print(json.dumps({'file': npa.__file__, 'version': metadata.version('npa')}))",
        ],
    )
    installed = json.loads((_OUTPUT / f"{name}.stdout").read_text())
    filename = installed["file"]
    _require(
        isinstance(filename, str)
        and Path(filename).resolve().is_relative_to(_SOURCE / "src"),
        "NPA came from another source",
    )
    _require(
        _RECEIPT.read_text() == str(_SOURCE),
        "NPA source receipt differs",
    )
    return installed["version"]


def _configure_mounts() -> None:
    global _SOURCE, _RECEIPT
    selected = os.environ.get("NPA_FIFTYONE_VALIDATION_CONTRACT")
    _require(bool(selected), "The owned container mount contract is missing")
    path = Path(selected)
    _require(
        path.parent == Path("/validation") and not path.is_symlink(),
        "Mount contract must come from the read-only validation mount",
    )
    contract = json.loads(path.read_text())
    _require(
        contract.get("schema") == "npa.fiftyone.mounts.v1", "Unsupported mount contract"
    )
    _require(
        contract["checks"]["Destination"] == str(path.parent)
        and contract["checks"]["RW"] is False,
        "Mount contract was not bound read-only",
    )
    source, receipt = contract["source"], contract["receipt"]
    _require(
        source["Type"] == receipt["Type"] == "bind"
        and source["RW"] is receipt["RW"] is True,
        "Source and receipt must use the verified owned bind mounts",
    )
    _SOURCE, _RECEIPT = Path(source["Destination"]), Path(receipt["Destination"])
    _require(
        _SOURCE.name == "npa-src"
        and _RECEIPT.name == "npa-src-root"
        and _SOURCE.parent == _RECEIPT.parent,
        "Mounts do not match the unchanged setup producer contract",
    )
    _require(
        _SOURCE.is_absolute() and _SOURCE.is_dir() and not _SOURCE.is_symlink(),
        "Owned source mount is unavailable",
    )
    _require(
        stat.S_ISREG(_RECEIPT.lstat().st_mode),
        "Owned receipt backing file is not regular",
    )


def _initial_source() -> dict:
    _require(
        shutil.which("npa") is None,
        "NPA is already installed; initial setup cannot be tested",
    )
    _require(importlib.util.find_spec("npa") is None, "NPA is already importable")
    _require(
        not Path("/opt/nebius-physical-ai/npa").exists(),
        "A different baked NPA source is present",
    )
    _require(
        not Path("/opt/npa/pyproject.toml").exists(),
        "A thin baked NPA installation is present",
    )
    _require(
        (_SOURCE / "pyproject.toml").is_file(), "Exact NPA source archive is missing"
    )
    _require(
        _RECEIPT is not None and _RECEIPT.read_bytes() == b"",
        "Initial source receipt is not empty",
    )
    return {"initial_install_required": True}


def _source_receipt_owner() -> dict:
    owner = (os.geteuid(), os.getegid())
    _require(owner[0] != 0, "Source setup must use the non-root runtime user")
    before = _RECEIPT.lstat()
    _require(
        stat.S_ISREG(before.st_mode) and _RECEIPT.read_bytes() == b"",
        "Initial source receipt must be an empty regular file",
    )
    # Sticky /tmp can reject shell redirection to a foreign-owned file even at 0666.
    if (before.st_uid, before.st_gid) != owner:
        _command(
            "install-source-receipt-owner",
            [
                "sudo",
                "-n",
                "chown",
                "--no-dereference",
                "--",
                f"{owner[0]}:{owner[1]}",
                str(_RECEIPT),
            ],
        )
    after = _RECEIPT.lstat()
    _require(
        stat.S_ISREG(after.st_mode)
        and (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
        "Owned source receipt backing file changed during ownership setup",
    )
    _require(
        (after.st_uid, after.st_gid) == owner and _RECEIPT.read_bytes() == b"",
        "Initial source receipt ownership or contents differ",
    )
    return {"owner_matches_runtime": True, "same_backing_file": True}


def _extract_setup() -> dict:
    source = _SOURCE / "src/npa/orchestration/npa_workflow/skypilot_render.py"
    tree = ast.parse(source.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "default_npa_setup"
    ]
    _require(len(functions) == 1, "Expected exactly one default_npa_setup producer")
    returns = [node for node in functions[0].body if isinstance(node, ast.Return)]
    _require(len(returns) == 1, "Setup producer does not have one literal return")
    setup = ast.literal_eval(returns[0].value)
    _require(isinstance(setup, str), "Setup producer is not a literal shell program")
    destination = _OUTPUT / "default-npa-setup.sh"
    destination.write_text(setup)
    result = {
        "renderer_sha256": _sha(source),
        "setup_sha256": _sha(destination),
        "extraction": "unchanged AST literal return",
    }
    _json(_OUTPUT / "default-npa-setup-provenance.json", result)
    return result


def _entrypoint() -> dict:
    _require(
        sys.argv[2:] == ["literal argument", "--forwarded-option"],
        "Entrypoint altered arguments",
    )
    _require(os.getpid() == 1, "Entrypoint did not exec the supplied command")
    return {"arguments_preserved": True, "exec_pid_one": True}


def _writable_paths() -> list[str]:
    paths = [
        Path.home(),
        Path(tempfile.gettempdir()),
        Path("/opt/fiftyone/db"),
        Path("/opt/fiftyone/datasets"),
        Path("/opt/fiftyone/smoke"),
    ]
    for path in paths:
        with tempfile.TemporaryDirectory(
            prefix="npa-validation-", dir=path
        ) as directory:
            probe = Path(directory) / "fixture.txt"
            probe.write_text("synthetic writable-path fixture\n")
            _require(
                probe.read_text() == "synthetic writable-path fixture\n",
                "Writable path roundtrip failed",
            )
    return [str(path) for path in paths]


def _ssh_state() -> dict:
    _command("ssh-generate-ephemeral-keys", ["sudo", "-n", "ssh-keygen", "-A"])
    _command("ssh-effective-config", ["sudo", "-n", "sshd", "-T"])
    config = (_OUTPUT / "ssh-effective-config.stdout").read_text().splitlines()
    _require(
        "passwordauthentication no" in config and "permitrootlogin no" in config,
        "SSH must reject password and root logins",
    )
    _command("ssh-start", ["sudo", "-n", "service", "ssh", "start"])
    try:
        _command("ssh-restart", ["sudo", "-n", "service", "ssh", "restart"])
        with socket.create_connection(("127.0.0.1", 22)) as connection:
            _require(
                connection.recv(128).startswith(b"SSH-2.0-"),
                "SSH did not serve its protocol banner",
            )
    finally:
        _command("ssh-stop", ["sudo", "-n", "service", "ssh", "stop"])
    try:
        with socket.create_connection(("127.0.0.1", 22)):
            raise RuntimeError("SSH listener remains after the owned service stop")
    except ConnectionRefusedError:
        return {"start_restart_banner_stop": True, "password_root_login_disabled": True}


def _bootstrap_behavior() -> dict:
    _require(os.geteuid() != 0, "Bootstrap must begin as the non-root image user")
    for executable in ("sudo", "service", "rsync", "ssh", "ssh-keygen", "sshd"):
        _require(
            shutil.which(executable) is not None,
            f"Missing bootstrap prerequisite: {executable}",
        )
    _command("sudo-root-check", ["sudo", "-n", "id", "-u"])
    _require(
        (_OUTPUT / "sudo-root-check.stdout").read_text().strip() == "0",
        "Passwordless sudo failed",
    )
    writable = _writable_paths()
    with tempfile.TemporaryDirectory(prefix="npa-rsync-fixture-") as directory:
        source, target = Path(directory) / "source", Path(directory) / "target"
        source.write_bytes(b"synthetic rsync fixture\n")
        _command("rsync-roundtrip", ["rsync", "-a", str(source), str(target)])
        _require(
            target.read_bytes() == source.read_bytes(), "Rsync changed its fixture"
        )
    return {"writable_paths": writable, "rsync_roundtrip": True, "ssh": _ssh_state()}


def _app_access() -> dict:
    import fiftyone as fo
    from fiftyone.server.app import app
    from starlette.testclient import TestClient

    _require(
        fo.config.default_app_address == "127.0.0.1", "App default must be loopback"
    )
    _require(
        not fo.config.allowed_origins, "Public image must keep same-origin defaults"
    )
    # The unchanged functional smoke proves TCP startup; this checks the real middleware without a second startup race.
    with TestClient(app) as client:
        response = client.get("/", headers={"Origin": "https://example.test"})
    _require(response.status_code == 200, "App did not return its normal root response")
    _require(
        response.headers.get("Access-Control-Allow-Origin") is None,
        "Default App root response granted cross-origin access",
    )
    return {
        "loopback_default": True,
        "same_origin_default": True,
        "root_http_status": response.status_code,
        "response_transport": "in-process ASGI",
    }


def _source_annex() -> dict:
    import fiftyone.db as database

    _annex_source_files()
    _command(
        "mongodb-source-annex",
        [
            _PYTHON,
            str(_SMOKES / "verify_source.py"),
            "--mongod",
            str(Path(database.FIFTYONE_DB_BIN_DIR) / "mongod"),
            "--notices-dir",
            str(_NOTICES),
        ],
    )
    result = json.loads((_OUTPUT / "mongodb-source-annex.stdout").read_text())
    _require(
        result.get("source_delivery") == "verified",
        "MongoDB source delivery was not verified",
    )
    _require(
        isinstance(result.get("source_members"), int) and result["source_members"] > 0,
        "MongoDB source annex has no verified members",
    )
    return {
        name: result[name] for name in ("version", "source_members", "source_delivery")
    }


def _annex_source_files() -> None:
    source = _SOURCE / "docker/workbench/fiftyone"
    selected = [
        (_SMOKES / "verify_source.py", "verify_source.py"),
        (_NOTICES / "source.json", "mongodb-source.json"),
        (_NOTICES / "version.json", "mongodb-source-version.json"),
        (_NOTICES / "SOURCE.md", "SOURCE.md"),
    ]
    for installed, filename in selected:
        _require(
            installed.read_bytes() == (source / filename).read_bytes(),
            f"Installed MongoDB source binding differs from committed {filename}",
        )


def _dataset_builder_identity() -> dict:
    name = "datasets/packaged_modules/folder_based_builder/folder_based_builder.py"
    member = Path(metadata.distribution("datasets").locate_file(name))
    digest = _sha(member)
    expected = "2dd8a0c4b98a173f8545b1af0de4be4fb8074064f75c60d0d55bc0c52a7bae31"
    _require(
        member.stat().st_size == 25184 and digest == expected,
        "Dataset folder builder differs from the reviewed 5.0.1 wheel member",
    )
    return {
        "member": name,
        "sha256": digest,
        "bytes": member.stat().st_size,
        "coverage": "static installed-source identity only",
    }


def _dataset_roundtrip() -> dict:
    from datasets import Dataset, load_from_disk

    rows = {
        "id": [1, 2],
        "caption": ["synthetic first sample", "synthetic second sample"],
    }
    with tempfile.TemporaryDirectory(prefix="npa-dataset-fixture-") as directory:
        root = Path(directory) / "dataset"
        Dataset.from_dict(rows).save_to_disk(str(root))
        state = json.loads((root / "state.json").read_text())
        files = _contained_files(
            root, [item["filename"] for item in state["_data_files"]]
        )
        restored = load_from_disk(str(root), keep_in_memory=False)
        _require(
            restored.to_dict() == rows, "Synthetic dataset roundtrip changed its rows"
        )
        _contained_files(root, [item["filename"] for item in restored.cache_files])
        return {
            "rows": len(rows["id"]),
            "emitted_files": len(files),
            "roundtrip": True,
            "emitted_file_containment": True,
        }


def _contained_files(root: Path, filenames: list[str]) -> list[Path]:
    _require(bool(filenames), "Synthetic dataset emitted no data files")
    selected = []
    for filename in filenames:
        path = Path(filename)
        path = path if path.is_absolute() else root / path
        _require(
            path.resolve().is_relative_to(root.resolve()),
            "Emitted dataset file escaped its fixture",
        )
        _require(
            path.is_file() and not path.is_symlink(),
            "Emitted dataset file is not a regular fixture",
        )
        selected.append(path)
    return selected


def _record(name: str, operation, records: list[dict]) -> None:
    try:
        result = operation()
    except Exception as error:
        (_OUTPUT / f"{name}.traceback").write_text(traceback.format_exc())
        records.append({"name": name, "ok": False, "error_type": type(error).__name__})
        raise
    records.append({"name": name, "ok": True, "result": result})


def _smoke(name: str, filename: str) -> dict:
    result = _command(name, [_PYTHON, str(_SMOKES / filename)])
    output = (_OUTPUT / f"{name}.stdout").read_text()
    summaries = re.findall(
        r"^SUMMARY: (\d+)/(\d+) checks passed$", output, re.MULTILINE
    )
    expected = 3 if filename == "smoke_env.py" else 5
    _require(
        summaries == [(str(expected), str(expected))],
        "Real smoke did not report its complete checks",
    )
    result.update(checks_passed=expected, checks_total=expected)
    return result


def _phase_checks(phase: str) -> list[tuple]:
    if phase == "initial-install":
        return _install_checks()
    return _functional_checks(phase)


def _install_checks() -> list[tuple]:
    return [
        ("initial-source-absence", _initial_source),
        ("install-source-receipt-owner", _source_receipt_owner),
        (
            "install-before-pip-check",
            lambda: _command(
                "install-before-pip-check", [_PYTHON, "-m", "pip", "check"]
            ),
        ),
        ("default-setup-extraction", _extract_setup),
        (
            "initial-default-npa-setup",
            lambda: _command(
                "initial-default-npa-setup",
                ["bash", "-x", str(_OUTPUT / "default-npa-setup.sh")],
            ),
        ),
        (
            "install-pip-check",
            lambda: _command("install-pip-check", [_PYTHON, "-m", "pip", "check"]),
        ),
        ("install-packages", lambda: _packages("installed")),
        ("npa-version", lambda: _command("npa-version", ["npa", "--version"])),
        (
            "npa-fiftyone-command",
            lambda: _command(
                "npa-fiftyone-command",
                ["npa", "workbench", "fiftyone", "curate-augmented", "--help"],
            ),
        ),
    ]


def _functional_checks(phase: str) -> list[tuple]:
    prefix = "bare" if phase == "bare" else "post"
    checks = [
        (
            f"{prefix}-pip-check",
            lambda: _command(f"{prefix}-pip-check", [_PYTHON, "-m", "pip", "check"]),
        ),
        (f"{prefix}-packages", lambda: _packages(prefix)),
        (
            f"{prefix}-environment",
            lambda: _smoke(f"{prefix}-environment", "smoke_env.py"),
        ),
        (
            f"{prefix}-functional",
            lambda: _smoke(f"{prefix}-functional", "smoke_functional.py"),
        ),
        (f"{prefix}-app-access", _app_access),
        (f"{prefix}-dataset-builder-identity", _dataset_builder_identity),
        (f"{prefix}-dataset-roundtrip-containment", _dataset_roundtrip),
    ]
    if phase == "bare":
        checks = [
            ("entrypoint", _entrypoint),
            ("initial-source-absence", _initial_source),
            ("mongodb-source-annex", _source_annex),
        ] + checks
        checks.append(("skypilot-bootstrap", _bootstrap_behavior))
    checks.append(
        (
            f"{prefix}-final-pip-check",
            lambda: _command(
                f"{prefix}-final-pip-check", [_PYTHON, "-m", "pip", "check"]
            ),
        )
    )
    return checks


def main() -> int:
    """Run one container-side validation phase and retain its exact result.

    Args:
        None. The phase comes from the process arguments.
    Returns:
        Zero when every phase check succeeds; one on failure.
    Raises:
        None. Original tracebacks and failed checks remain in the evidence directory.
    """
    records = []
    phase = sys.argv[1] if len(sys.argv) > 1 else ""
    status = "failed"
    try:
        _require(
            phase in {"bare", "initial-install", "post-install"},
            "Unknown validation phase",
        )
        _configure_mounts()
        for name, operation in _phase_checks(phase):
            _record(name, operation, records)
        status = "passed"
    except Exception:
        (_OUTPUT / f"{phase}-failure.traceback").write_text(traceback.format_exc())
    _json(
        _OUTPUT / f"{phase}.json", {"phase": phase, "status": status, "checks": records}
    )
    print(json.dumps({"phase": phase, "status": status, "checks": len(records)}))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
