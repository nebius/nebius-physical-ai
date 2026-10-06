"""Exercise private qualification boundaries with synthetic bytes and local children."""

from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/private_image_qualification.py"
SPEC = importlib.util.spec_from_file_location("private_qualification", SOURCE)
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


def _policy_receipt():
    return {
        "schema": "npa.image-byte-confidentiality-policy.v1",
        "policy_sha256": "b" * 64,
        "customer": "configured",
        "infra": "not_configured",
    }


@pytest.fixture
def private_root(tmp_path):
    directory = tmp_path.resolve() / "private"
    directory.mkdir(mode=0o700)
    return directory


@pytest.fixture
def export(private_root, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: private_root)
    payload = b"synthetic OCI archive bytes; not executable"
    manifest = {
        "schema_version": Q.MANIFEST_SCHEMA,
        "archive_sha256": Q._sha(payload),
        "archive_bytes": len(payload),
        "expected_image_id": "sha256:" + "a" * 64,
        "workspace_bytes": 8192,
    }
    data = Q._json_bytes(manifest)
    selector = Q._sha(data)
    directory = private_root / Q.EXPORT_ROOT / selector
    directory.mkdir(parents=True, mode=0o700)
    Q._write(directory / "manifest.json", data)
    Q._write(directory / "image.tar", payload)
    return directory, selector, manifest, payload


def test_manifest_binds_exact_serialized_bytes(export):
    directory, selector, expected, _ = export
    data = (directory / "manifest.json").read_bytes()
    assert Q._manifest(data, selector) == expected
    with pytest.raises(Q._QualificationError, match="manifest_digest"):
        Q._manifest(data + b" ", selector)


@pytest.mark.parametrize(
    "field,value",
    [
        ("archive_sha256", "../../image.tar"),
        ("expected_image_id", "$(echo injected)"),
        ("archive_bytes", True),
        ("archive_bytes", 0),
        ("workspace_bytes", -1),
        ("path", "/unrelated/private/file"),
    ],
)
def test_manifest_rejects_untrusted_shape(export, field, value):
    manifest = {**export[2], field: value}
    data = Q._json_bytes(manifest)
    with pytest.raises(Q._QualificationError):
        Q._manifest(data, Q._sha(data))


def test_manifest_rejects_duplicate_keys_and_oversized_metadata(export):
    data = (export[0] / "manifest.json").read_bytes().strip()
    duplicate = data[:-1] + b',"archive_bytes":1}'
    with pytest.raises(Q._QualificationError, match="duplicate_json_key"):
        Q._manifest(duplicate, Q._sha(duplicate))
    large = data + b" " * Q.MANIFEST_BYTES
    with pytest.raises(Q._QualificationError, match="manifest_digest"):
        Q._manifest(large, Q._sha(large))


def test_exact_copy_rejects_truncation_excess_and_wrong_digest():
    for payload, size, digest, code in (
        (b"abc", 4, Q._sha(b"abc"), "truncated"),
        (b"abcd", 3, Q._sha(b"abc"), "oversized"),
        (b"abc", 3, Q._sha(b"other"), "digest"),
    ):
        with pytest.raises(Q._QualificationError, match=code):
            Q._copy_exact(io.BytesIO(payload), io.BytesIO(), size, digest)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "public"])
def test_remote_archive_rejects_aliases_and_permissions(export, kind, monkeypatch):
    directory, selector, _, payload = export
    image = directory / "image.tar"
    if kind == "symlink":
        image.rename(directory / "target")
        image.symlink_to("target")
    elif kind == "hardlink":
        os.link(image, directory / "target")
    else:
        image.chmod(0o644)
    output = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", type("Output", (), {"buffer": output})())
    with pytest.raises((OSError, Q._QualificationError)):
        Q._remote_fetch(selector, "archive")
    assert output.getvalue() != payload


def test_remote_path_component_symlink_is_rejected(export):
    directory, selector, _, _ = export
    moved = directory.with_name("moved")
    directory.rename(moved)
    directory.symlink_to(moved, target_is_directory=True)
    with pytest.raises(OSError):
        Q._remote_manifest(selector)


def test_input_mutation_and_replacement_are_detected(private_root):
    path = private_root / "image.tar"
    Q._write(path, b"before")
    before = path.stat()
    with pytest.raises(Q._QualificationError, match="input_changed"):
        with Q._private_input(path) as (stream, _info):
            assert stream.read() == b"before"
            path.write_bytes(b"after!")
            # Exercise metadata identity independently of the filesystem clock tick.
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
    with pytest.raises(Q._QualificationError, match="input_changed|input_replaced"):
        with Q._private_input(path):
            path.rename(private_root / "old")
            Q._write(path, b"after!")


def test_remote_fetch_preserves_exact_bytes_without_interpreting_them(
    export, monkeypatch
):
    output = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", type("Output", (), {"buffer": output})())
    Q._remote_fetch(export[1], "archive")
    assert output.getvalue() == export[3]


def test_frozen_archive_hardlink_preserves_original_without_copy(export, monkeypatch):
    directory, selector, _, payload = export
    image = directory / "image.tar"
    original = directory.parent / "original-archive"
    image.chmod(0o400)
    os.link(image, original)
    before = Q._fingerprint(original.stat())
    output = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", type("Output", (), {"buffer": output})())
    Q._remote_fetch(selector, "archive")
    assert output.getvalue() == payload == original.read_bytes()
    assert before == Q._fingerprint(original.stat())
    assert image.stat().st_ino == original.stat().st_ino
    with pytest.raises(Q._QualificationError, match="input_not_private_regular"):
        with Q._private_input(image):
            pass


def test_hardlink_alias_mutation_fails_even_after_restoring_permissions(export):
    image = export[0] / "image.tar"
    alias = export[0] / "alias"
    image.chmod(0o400)
    os.link(image, alias)
    before = image.stat()
    with pytest.raises(Q._QualificationError, match="input_changed"):
        with Q._private_input(image, frozen_archive=True) as (stream, _info):
            stream.read()
            alias.chmod(0o600)
            alias.write_bytes(b"x" * export[2]["archive_bytes"])
            alias.chmod(0o400)
            os.utime(alias, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))


def test_remote_stream_rehashes_same_size_alias_mutation_on_held_descriptor(
    export, monkeypatch
):
    image = export[0] / "image.tar"
    image.chmod(0o400)
    alias = export[0] / "alias"
    os.link(image, alias)
    original_metadata = Q._fingerprint(image.stat())
    # Simulate unchanged metadata; only the real kernel-byte recheck can reject.
    monkeypatch.setattr(Q, "_fingerprint", lambda _info: original_metadata)

    class MutatingOutput(io.BytesIO):
        def write(self, data):
            alias.chmod(0o600)
            alias.write_bytes(b"x" * export[2]["archive_bytes"])
            alias.chmod(0o400)
            return super().write(data)

    output = MutatingOutput()
    monkeypatch.setattr(sys, "stdout", type("Output", (), {"buffer": output})())
    with pytest.raises(Q._QualificationError, match="input_changed"):
        Q._remote_fetch(export[1], "archive")
    assert output.getvalue() == export[3]
    assert alias.read_bytes() != export[3]


def test_fetch_hashes_actual_local_child_stream(private_root, monkeypatch):
    payload = b"synthetic binary\x00content"
    monkeypatch.setattr(
        Q,
        "_remote_command",
        lambda *_args: [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(" + repr(payload) + ")",
        ],
    )
    target = private_root / "archive"
    Q._fetch([], "a" * 64, "archive", target, size=len(payload), digest=Q._sha(payload))
    assert target.read_bytes() == payload
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    with pytest.raises(Q._QualificationError, match="transfer_digest"):
        Q._fetch(
            [],
            "a" * 64,
            "archive",
            private_root / "bad",
            size=len(payload),
            digest="0" * 64,
        )


def test_nonzero_sender_exit_does_not_qualify_exact_bytes(private_root, monkeypatch):
    monkeypatch.setattr(
        Q,
        "_remote_command",
        lambda *_args: [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'abc'); sys.exit(1)",
        ],
    )
    with pytest.raises(Q._QualificationError, match="ssh_transfer_failed"):
        Q._fetch(
            [], "a" * 64, "archive", private_root / "bad", size=3, digest=Q._sha(b"abc")
        )


def test_remote_shell_command_roundtrips_quoted_source_and_arguments(monkeypatch):
    special = "$(touch bad);'\nprivate"
    command = Q._remote_command(["ssh", "synthetic-host"], "fetch", special, "manifest")
    arguments = shlex.split(command[-1])
    assert arguments[:3] == ["/usr/bin/python3", "-I", "-c"]
    assert arguments[3] == SOURCE.read_text()
    assert arguments[4:] == ["remote", "fetch", special, "manifest"]


def test_ssh_uses_pinned_host_trust_and_scrubbed_child_environment(
    private_root, monkeypatch
):
    values = {
        "DEV_VM_SSH_HOST": "example.invalid",
        "DEV_VM_SSH_USER": "synthetic",
        "DEV_VM_SSH_PRIVATE_KEY": "synthetic-key",
        "DEV_VM_SSH_KNOWN_HOSTS": "synthetic-host-key",
        "CUSTOMER_DENYLIST": "synthetic-customer-policy",
        "INFRA_DENYLIST": "synthetic-infra-policy",
        "NEBIUS_TOKEN": "synthetic-token",
        "SSH_AUTH_SOCK": "/synthetic/socket",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    arguments = Q._ssh_arguments(private_root)
    assert "StrictHostKeyChecking=yes" in arguments
    assert "GlobalKnownHostsFile=/dev/null" in arguments
    assert arguments[1:3] == ["-F", "/dev/null"]
    assert not (set(values) & set(Q._clean_environment()))
    assert set(values) & set(Q._clean_environment(policy=True)) == {
        "CUSTOMER_DENYLIST",
        "INFRA_DENYLIST",
    }
    assert stat.S_IMODE((private_root / "ssh-key").stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "key,value",
    [
        ("DEV_VM_SSH_HOST", "-oProxyCommand=echo"),
        ("DEV_VM_SSH_HOST", "host\ncommand"),
        ("DEV_VM_SSH_USER", "user;command"),
        ("DEV_VM_SSH_PORT", "22;command"),
    ],
)
def test_ssh_configuration_rejects_option_and_shell_injection(
    private_root, monkeypatch, key, value
):
    monkeypatch.setenv("DEV_VM_SSH_HOST", "example.invalid")
    monkeypatch.setenv("DEV_VM_SSH_USER", "synthetic")
    monkeypatch.setenv(key, value)
    with pytest.raises(Q._QualificationError):
        Q._ssh_arguments(private_root)


def test_capacity_refuses_before_any_archive_transfer(
    private_root, export, monkeypatch
):
    monkeypatch.setattr(
        Q.shutil, "disk_usage", lambda _path: type("Disk", (), {"free": 1})()
    )
    with pytest.raises(Q._QualificationError, match="insufficient_capacity"):
        Q._capacity(private_root, export[2])
    receipt = json.loads((private_root / "capacity.json").read_bytes())
    assert (
        receipt["required_free_bytes"]
        == export[2]["archive_bytes"] + export[2]["workspace_bytes"]
    )
    assert receipt["available_bytes"] == 1


def test_scanner_commands_keep_policy_on_authorization_boundary(
    private_root, export, monkeypatch
):
    commands = []
    monkeypatch.setattr(
        Q, "_execute", lambda command, **kwargs: commands.append((command, kwargs)) or 0
    )
    scanner = private_root / "reviewed-source"
    Q._prepare_scanner(scanner, private_root)
    Q._scan(scanner, private_root, export[2])
    assert len(commands) == 7
    assert [
        index for index, (_, kwargs) in enumerate(commands) if kwargs.get("policy")
    ] == [0, 5]
    assert commands[4][0][1].endswith("image_byte_scan/oci_verification.py")
    assert "--expected-image-id" in commands[4][0]
    assert "--policy-mode" in commands[5][0] and "ci-regex" in commands[5][0]
    assert commands[6][0][1].endswith("scan_image_bytes.py")


def test_child_stdout_and_stderr_never_reach_public_output(capsys):
    command = [
        sys.executable,
        "-c",
        "import sys; print('synthetic-private-stdout'); print('synthetic-private-stderr', file=sys.stderr); sys.exit(1)",
    ]
    assert Q._execute(command) == 1
    assert capsys.readouterr() == ("", "")


def test_receipt_bundle_has_exact_allowlist_and_no_policy_or_authorization(
    private_root,
):
    for name in Q.RECEIPT_FILES:
        path = private_root / name
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        Q._write(path, b"synthetic-private-receipt")
    for name in (
        "authorization/confidentiality.json",
        "authorization/authorization.json",
        "image.tar",
        "unexpected.json",
    ):
        path = private_root / name
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        Q._write(path, b"synthetic-policy-must-stay-on-runner")
    path, size, digest = Q._bundle(private_root)
    assert size == path.stat().st_size
    assert digest == Q._sha(path.read_bytes())
    assert b"synthetic-policy-must-stay-on-runner" not in path.read_bytes()
    with tarfile.open(path) as archive:
        assert archive.getnames() == list(Q.RECEIPT_FILES)


def test_remote_receipt_is_opaque_private_and_cannot_overwrite(export, monkeypatch):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        member = tarfile.TarInfo("../../escape")
        member.size = 3
        archive.addfile(member, io.BytesIO(b"abc"))
    data = payload.getvalue()
    monkeypatch.setattr(sys, "stdin", type("Input", (), {"buffer": io.BytesIO(data)})())
    Q._remote_store(export[1], "123-1", len(data), Q._sha(data))
    target = Path.home() / Q.RECEIPT_ROOT / export[1] / "123-1/result.tar"
    assert target.read_bytes() == data
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not (Path.home() / "escape").exists()
    with pytest.raises(FileExistsError):
        Q._remote_store(export[1], "123-1", len(data), Q._sha(data))


def test_report_summary_never_promotes_incomplete_or_unjoined_scan(
    private_root, export
):
    directory = private_root / "scan"
    directory.mkdir(mode=0o700)
    report = {
        "archive_sha256": export[2]["archive_sha256"],
        "expected_image_id": export[2]["expected_image_id"],
        "records": 1,
        "scanned_bytes": 10,
        "regular_files": 1,
        "regular_bytes": 10,
        "findings": 0,
        "valid": True,
        "complete": True,
        "helper_joined": True,
        "private_untrusted_field": "must-not-be-public",
        "confidentiality_policy": _policy_receipt(),
    }
    for complete, joined, returncode, expected in (
        (True, True, 0, "passed"),
        (False, True, 0, "not-qualified"),
        (True, False, 0, "not-qualified"),
        (True, True, 1, "not-qualified"),
    ):
        path = directory / "report.json"
        path.unlink(missing_ok=True)
        Q._write(
            path,
            Q._json_bytes({**report, "complete": complete, "helper_joined": joined}),
        )
        summary = Q._scan_summary(private_root, export[2], export[1], returncode)
        assert summary["status"] == expected
        assert "must-not-be-public" not in json.dumps(summary)
        assert summary["confidentiality_policy"] == {
            "policy_sha256": "b" * 64,
            "customer": "configured",
            "infra": "not_configured",
        }


def test_source_binding_requires_exact_clean_commit(private_root, monkeypatch):
    repository = private_root / "repository"
    repository.mkdir(mode=0o700)
    environment = {
        **Q._clean_environment(),
        "GIT_AUTHOR_NAME": "Synthetic",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Synthetic",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }

    def git(*arguments):
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            env=environment,
            check=True,
            capture_output=True,
        ).stdout

    git("init", "--quiet")
    (repository / "trusted.py").write_text("pass\n")
    git("add", "trusted.py")
    git("commit", "--quiet", "-m", "Synthetic source")
    with pytest.raises(Q._QualificationError, match="scanner_source_revision"):
        Q._source_binding(repository)
    monkeypatch.setattr(
        Q, "SCANNER_REVISION", git("rev-parse", "HEAD").decode().strip()
    )
    Q._source_binding(repository)
    (repository / "trusted.py").write_text("changed\n")
    with pytest.raises(Q._QualificationError, match="scanner_source_dirty"):
        Q._source_binding(repository)


def test_workflow_exposes_only_digest_and_runs_reviewed_sources():
    workflow = ROOT.parent / ".github/workflows/private-image-qualification.yml"
    value = yaml.safe_load(workflow.read_text())
    triggers = value.get("on", value.get(True))
    assert set(triggers) == {"workflow_dispatch"}
    assert set(triggers["workflow_dispatch"]["inputs"]) == {"manifest_sha256"}
    assert value["permissions"] == {"contents": "read"}
    job = value["jobs"]["qualify"]
    assert "github.event.repository.default_branch" in job["if"]
    assert "github.event_name == 'workflow_dispatch'" in job["if"]
    scanner = next(
        step for step in job["steps"] if step.get("with", {}).get("path") == "scanner"
    )
    assert scanner["with"]["ref"] == Q.SCANNER_REVISION
    assert all("upload-artifact" not in step.get("uses", "") for step in job["steps"])


def test_parent_directory_replacement_is_detected(private_root):
    parent = private_root / "parent"
    parent.mkdir(mode=0o700)
    Q._write(parent / "input", b"immutable")
    with pytest.raises(Q._QualificationError, match="parent_replaced"):
        with Q._private_input(parent / "input") as (stream, _info):
            assert stream.read() == b"immutable"
            parent.rename(private_root / "old-parent")
            parent.mkdir(mode=0o700)


def test_output_parent_replacement_does_not_claim_durable_write(private_root):
    parent = private_root / "parent"
    parent.mkdir(mode=0o700)
    with pytest.raises(Q._QualificationError, match="parent_replaced"):
        with Q._private_output(parent / "output") as stream:
            stream.write(b"receipt")
            parent.rename(private_root / "old-parent")
            parent.mkdir(mode=0o700)


def test_argument_errors_never_echo_supplied_values(capsys):
    assert Q._main(["run", "--synthetic-private-input"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err) == {
        "status": "failed",
        "failure_code": "invalid_arguments",
        "failure_stage": "interface",
        "exception_class": "_QualificationError",
    }


def test_synthetic_transport_executes_remote_reader_and_retains_exact_receipt(
    export, private_root, monkeypatch, capsys
):
    monkeypatch.setenv("HOME", str(private_root))
    runner = private_root / "runner"
    runner.mkdir(mode=0o700)
    # The shell consumes the same reviewed remote command as SSH; no network or provider CLI.
    local_transport = [
        sys.executable,
        "-c",
        "import subprocess,sys; sys.exit(subprocess.call(sys.argv[-1], shell=True))",
    ]
    monkeypatch.setattr(Q, "_prepare_scanner", lambda *_args: None)

    def synthetic_scan(_scanner, root, manifest):
        assert (root / "image.tar").read_bytes() == export[3]
        directory = root / "scan"
        directory.mkdir(mode=0o700)
        Q._write(
            directory / "report.json",
            Q._json_bytes(
                {
                    "archive_sha256": manifest["archive_sha256"],
                    "expected_image_id": manifest["expected_image_id"],
                    "records": 1,
                    "scanned_bytes": len(export[3]),
                    "regular_files": 1,
                    "regular_bytes": len(export[3]),
                    "findings": 0,
                    "valid": True,
                    "complete": True,
                    "helper_joined": True,
                    "confidentiality_policy": _policy_receipt(),
                }
            ),
        )
        return 0

    monkeypatch.setattr(Q, "_scan", synthetic_scan)
    assert Q._qualify(private_root, runner, local_transport, export[1], "456-1") == 0
    receipt = private_root / Q.RECEIPT_ROOT / export[1] / "456-1/result.tar"
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert output[-1]["status"] == "passed"
    assert output[-1]["receipt_sha256"] == Q._sha(receipt.read_bytes())
    assert output[-1]["receipt_bytes"] == receipt.stat().st_size
    assert str(private_root) not in json.dumps(output)
    with tarfile.open(receipt) as archive:
        assert "manifest.json" in archive.getnames()
        assert "scan/report.json" in archive.getnames()
        assert "image.tar" not in archive.getnames()


def test_private_receipt_transport_failure_prevents_success(
    private_root, export, monkeypatch
):
    Q._write(private_root / "summary.json", b'{"status":"passed"}')
    monkeypatch.setattr(
        Q,
        "_remote_command",
        lambda *_args: [sys.executable, "-c", "raise SystemExit(1)"],
    )
    with pytest.raises(Q._QualificationError, match="private_receipt_not_retained"):
        Q._retain([], private_root, export[1], "789-1")


def test_private_failure_receipt_distinguishes_capacity_and_hides_exception_text(
    export, private_root, monkeypatch, capsys
):
    monkeypatch.setenv("HOME", str(private_root))
    runner = private_root / "failed-runner"
    runner.mkdir(mode=0o700)
    local_transport = [
        sys.executable,
        "-c",
        "import subprocess,sys; sys.exit(subprocess.call(sys.argv[-1], shell=True))",
    ]

    def fail_preparation(*_args):
        raise PermissionError("synthetic-private-policy-and-path")

    monkeypatch.setattr(Q, "_prepare_scanner", fail_preparation)
    assert Q._qualify(private_root, runner, local_transport, export[1], "987-1") == 1
    output = capsys.readouterr().out
    summary = json.loads(output.splitlines()[-1])
    assert summary["failure_stage"] == "scanner-preparation"
    assert summary["exception_class"] == "PermissionError"
    assert summary["failure_code"] == "operation_failed"
    assert "synthetic-private-policy-and-path" not in output
    receipt = private_root / Q.RECEIPT_ROOT / export[1] / "987-1/result.tar"
    with tarfile.open(receipt) as archive:
        retained = json.load(archive.extractfile("summary.json"))
    assert retained["exception_class"] == "PermissionError"
    assert "synthetic-private-policy-and-path" not in json.dumps(retained)


def test_unknown_error_text_is_not_a_public_diagnostic():
    failure = Q._failure(Q._QualificationError("synthetic-private-text"), "image-scan")
    assert failure["failure_code"] == "qualification_failed"
    assert "synthetic-private-text" not in json.dumps(failure)


def test_run_keeps_credentials_outside_every_scanner_analysis_root(
    private_root, monkeypatch
):
    for name, value in {
        "DEV_VM_SSH_HOST": "example.invalid",
        "DEV_VM_SSH_USER": "synthetic",
        "DEV_VM_SSH_PRIVATE_KEY": "synthetic-key",
        "DEV_VM_SSH_KNOWN_HOSTS": "synthetic-host-key",
    }.items():
        monkeypatch.setenv(name, value)
    observed = {}
    monkeypatch.setattr(
        Q, "_source_binding", lambda _path: observed.update(source_bound=True)
    )

    def qualify(_scanner, analysis, ssh, _selector, _run):
        assert observed["source_bound"]
        key = Path(ssh[ssh.index("-i") + 1])
        assert key.read_text().strip() == "synthetic-key"
        assert not key.parent.is_relative_to(analysis)
        assert not analysis.is_relative_to(key.parent)
        assert not (analysis / "ssh-key").exists()
        assert not (analysis / "known-hosts").exists()
        assert list(analysis.rglob("*")) == []
        observed.update(analysis=analysis, credentials=key.parent)
        return 0

    monkeypatch.setattr(Q, "_qualify", qualify)
    args = SimpleNamespace(
        manifest_sha256="a" * 64,
        run_id="123-1",
        scanner_root=private_root / "scanner",
        work_parent=private_root,
    )
    assert Q._run(args) == 0
    assert not observed["analysis"].exists()
    assert not observed["credentials"].exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "other-schema"),
        ("policy_sha256", "synthetic-private-policy"),
        ("policy_sha256", "b" * 63),
        ("policy_sha256", None),
        ("customer", "not_configured"),
        ("customer", None),
        ("infra", "synthetic-private-status"),
        ("infra", True),
    ],
)
def test_policy_receipt_refuses_invalid_or_missing_coverage(field, value):
    receipt = {**_policy_receipt(), field: value}
    with pytest.raises(Q._QualificationError, match="policy_receipt_invalid"):
        Q._policy_summary({"confidentiality_policy": receipt})


def test_policy_summary_accepts_only_fixed_statuses_and_digest():
    receipt = {
        **_policy_receipt(),
        "infra": "configured",
        "pattern": "synthetic-private-pattern",
    }
    assert Q._policy_summary({"confidentiality_policy": receipt}) == {
        "policy_sha256": "b" * 64,
        "customer": "configured",
        "infra": "configured",
    }
    with pytest.raises(Q._QualificationError, match="policy_receipt_invalid"):
        Q._policy_summary({})


def test_missing_report_has_distinct_bounded_error(private_root, export):
    with pytest.raises(Q._QualificationError, match="report_missing") as caught:
        Q._scan_summary(private_root, export[2], export[1], 1)
    failure = Q._failure(caught.value, "scan-report")
    assert failure["failure_code"] == "report_missing"
    assert str(private_root) not in json.dumps(failure)


def test_ignored_bytecode_is_rejected_before_native_children(private_root, monkeypatch):
    repository = private_root / "ignored-repository"
    repository.mkdir(mode=0o700)
    environment = {
        **Q._clean_environment(),
        "GIT_AUTHOR_NAME": "Synthetic",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Synthetic",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }

    def git(*arguments):
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            env=environment,
            check=True,
            capture_output=True,
        ).stdout

    git("init", "--quiet")
    (repository / ".gitignore").write_text("__pycache__/\n")
    git("add", ".gitignore")
    git("commit", "--quiet", "-m", "Synthetic ignored cache")
    monkeypatch.setattr(
        Q, "SCANNER_REVISION", git("rev-parse", "HEAD").decode().strip()
    )
    Q._source_binding(repository)
    cache = repository / "__pycache__"
    cache.mkdir()
    (cache / "core.cpython-312.pyc").write_bytes(b"synthetic-unexecuted-bytecode")
    assert not git("status", "--porcelain")
    monkeypatch.setattr(
        Q,
        "_ssh_arguments",
        lambda *_args: pytest.fail(
            "source refusal must precede credentials and native children"
        ),
    )
    args = SimpleNamespace(
        manifest_sha256="a" * 64,
        run_id="123-1",
        scanner_root=repository,
        work_parent=private_root,
    )
    with pytest.raises(Q._QualificationError, match="scanner_source_dirty"):
        Q._run(args)


@pytest.mark.parametrize(
    "diagnostic,code",
    [
        (
            "private-user@private-host: Permission denied (publickey).",
            "ssh_authentication_failed",
        ),
        (
            "Host key verification failed for private-host",
            "ssh_host_verification_failed",
        ),
        (
            "ssh: connect to host private-host: Connection refused",
            "ssh_connection_failed",
        ),
        (
            json.dumps(
                {
                    "status": "failed",
                    "exception_class": "FileNotFoundError",
                    "path": "private-path",
                }
            ),
            "remote_export_missing",
        ),
        (
            json.dumps(
                {
                    "status": "failed",
                    "exception_class": "PermissionError",
                    "path": "private-path",
                }
            ),
            "remote_export_permissions",
        ),
        (
            json.dumps(
                {
                    "status": "failed",
                    "exception_class": "_QualificationError",
                    "failure_code": "directory_not_private",
                }
            ),
            "remote_interface_failed",
        ),
        ("unclassified private diagnostic", "ssh_transfer_failed"),
    ],
)
def test_manifest_reports_failed_child_transport_before_digest(
    private_root, monkeypatch, capsys, diagnostic, code
):
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write(" + repr(diagnostic) + "); sys.exit(255)",
    ]
    monkeypatch.setattr(Q, "_remote_command", lambda *_args: command)
    target = private_root / "manifest"
    with pytest.raises(Q._QualificationError) as raised:
        Q._fetch([], "a" * 64, "manifest", target)
    assert raised.value.code == code
    assert "private" not in json.dumps(Q._failure(raised.value, "interface"))
    assert capsys.readouterr() == ("", "")
    assert target.read_bytes() == b""


@pytest.mark.parametrize(
    "sender_exit,expected", [(0, None), (1, "ssh_transfer_failed")]
)
def test_manifest_requires_successful_sender_even_with_exact_bytes(
    export, private_root, monkeypatch, sender_exit, expected
):
    data = (export[0] / "manifest.json").read_bytes()
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write("
        + repr(data)
        + "); sys.exit("
        + str(sender_exit)
        + ")",
    ]
    monkeypatch.setattr(Q, "_remote_command", lambda *_args: command)
    target = private_root / "received-manifest"
    if expected:
        with pytest.raises(Q._QualificationError, match=expected):
            Q._fetch([], export[1], "manifest", target)
        assert target.read_bytes() == b""
    else:
        Q._fetch([], export[1], "manifest", target)
        assert target.read_bytes() == data


def test_successful_manifest_sender_still_requires_matching_digest(
    private_root, monkeypatch
):
    command = [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'{}')"]
    monkeypatch.setattr(Q, "_remote_command", lambda *_args: command)
    with pytest.raises(Q._QualificationError, match="manifest_digest"):
        Q._fetch([], "a" * 64, "manifest", private_root / "wrong-manifest")


def test_oversized_manifest_sender_is_reaped_without_waiting_for_full_stream(
    private_root, monkeypatch
):
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(b'x' * (8 * 1024 * 1024))",
    ]
    monkeypatch.setattr(Q, "_remote_command", lambda *_args: command)
    original = subprocess.Popen
    children = []

    def tracked(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(Q.subprocess, "Popen", tracked)
    with pytest.raises(Q._QualificationError, match="manifest_size"):
        Q._fetch([], "a" * 64, "manifest", private_root / "oversized-manifest")
    assert len(children) == 1
    assert children[0].returncode is not None
