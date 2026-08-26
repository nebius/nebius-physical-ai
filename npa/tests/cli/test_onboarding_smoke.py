"""Guards for the documented first-time-user onboarding path.

These tests defend the copy-pasteable quickstart so docs that "look right"
cannot silently rot: the setup guidance must stay placeholder-only (public
hygiene), and the advertised first real success must keep working offline.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.vlm_eval import DEFAULT_MODEL, DEFAULT_SAMPLE_BENCHMARK_PATH


runner = CliRunner()

# Matches any dotted-quad IPv4 literal, e.g. 203.0.113.10.
_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")


def test_setup_guidance_points_to_single_configure_entrypoint() -> None:
    """Onboarding guidance should not require separate nebius CLI steps first."""
    for command in ("configure", "init"):
        result = runner.invoke(app, [command])
        assert result.exit_code == 0
        lowered = result.output.lower()
        assert "npa configure --interactive" in result.output
        assert "nebius profile create" not in lowered
        assert "get-access-token" not in lowered


def test_setup_guidance_contains_no_raw_ip_address() -> None:
    """Setup guidance must use placeholders, never a literal host/IP."""
    for command in ("configure", "init"):
        result = runner.invoke(app, [command])
        assert result.exit_code == 0
        match = _IPV4.search(result.output)
        assert match is None, (
            f"`npa {command}` guidance leaks a literal IP {match.group(0)!r}; "
            "use a placeholder such as <your-byovm-host> instead."
        )
        assert "<your-byovm-host>" in result.output


def test_known_project_configure_is_non_interactive_and_reuses_storage(
    monkeypatch, tmp_path
) -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    from npa.clients.storage_validation import StorageProbeResult

    config_path = tmp_path / ".npa" / "config.yaml"
    credentials_path = tmp_path / ".npa" / "credentials.yaml"
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", credentials_path)
    credentials_module.write_credentials_file(
        {
            "storage": {
                "aws_access_key_id": "existing-access",
                "aws_secret_access_key": "existing-secret",
                "endpoint_url": "https://storage.eu-north1.nebius.cloud",
                "bucket": "s3://existing-bucket/",
            },
            "storage_setup": {
                "version": 1,
                "projects": {
                    "project-known": {
                        "status": "complete",
                        "bucket_name": "existing-bucket",
                    }
                },
            },
        }
    )
    monkeypatch.setattr("npa.clients.nebius.get_iam_token", lambda: "iam-token")
    monkeypatch.setattr("npa.clients.nebius.set_profile_project", lambda *a, **k: True)
    monkeypatch.setattr(
        "npa.clients.storage_validation.probe_storage_write",
        lambda **kwargs: StorageProbeResult(
            True,
            "ok",
            "Writable S3 verified with a cleaned write/delete probe.",
            cleanup_attempted=True,
            cleanup_succeeded=True,
        ),
    )

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError(
            "known-project configure must not discover, prompt, or provision"
        )

    monkeypatch.setattr("npa.cli.main._provision_object_storage", _must_not_run)
    monkeypatch.setattr("npa.clients.nebius.list_tenants", _must_not_run)
    monkeypatch.setattr("npa.clients.nebius.list_projects_in_tenant", _must_not_run)
    monkeypatch.setattr("npa.clients.nebius._list_access_key_metadata", _must_not_run)
    monkeypatch.setattr("npa.clients.nebius.ensure_access_key", _must_not_run)

    result = runner.invoke(
        app,
        [
            "configure",
            "--no-interactive",
            "--provision",
            "--tenant-id",
            "tenant-known",
            "--project-id",
            "project-known",
            "--region",
            "eu-north1",
            "--project-alias",
            "paidf-prod",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Reusing health-verified" in result.output
    assert (
        "Project and writable object storage configuration is health-verified."
        in result.output
    )
    saved = yaml.safe_load(config_path.read_text())
    project = saved["projects"]["paidf-prod"]
    assert saved["default_project"] == "paidf-prod"
    assert project["project_id"] == "project-known"
    assert project["tenant_id"] == "tenant-known"
    assert project["region"] == "eu-north1"
    assert "container_registry" not in project
    assert config_module.resolve_container_registry("paidf-prod") == (
        "ghcr.io/nebius/nebius-physical-ai"
    )


def test_known_project_configure_requires_complete_identity_flags() -> None:
    result = runner.invoke(
        app,
        [
            "configure",
            "--no-interactive",
            "--tenant-id",
            "tenant-known",
            "--project-id",
            "project-known",
        ],
        env={"COLUMNS": "240"},
    )

    assert result.exit_code != 0
    assert "Known-project configure requires" in result.output
    assert "--region" in result.output
    assert "--project-alias" in result.output


def test_known_project_configure_forwards_new_bucket_class_and_size(
    monkeypatch, tmp_path
) -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml"
    )
    monkeypatch.setattr("npa.clients.nebius.get_iam_token", lambda: "iam-token")
    monkeypatch.setattr("npa.clients.nebius.set_profile_project", lambda *a, **k: True)
    captured: dict[str, object] = {}

    def fake_provision(*_args, **kwargs):
        captured.update(kwargs)
        return {
            "aws_access_key_id": "access",
            "aws_secret_access_key": "secret",
            "endpoint_url": "https://storage.invalid",
            "bucket": "s3://derived-bucket/",
            "_validated": "true",
        }

    monkeypatch.setattr("npa.cli.main._provision_object_storage", fake_provision)
    result = runner.invoke(
        app,
        [
            "configure",
            "--no-interactive",
            "--provision",
            "--tenant-id",
            "tenant-known",
            "--project-id",
            "project-known",
            "--region",
            "us-central1",
            "--project-alias",
            "fleet-a",
            "--bucket-storage-class",
            "enhanced",
            "--bucket-size-gb",
            "100",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["interactive"] is False
    assert captured["bucket_storage_class"] == "enhanced_throughput"
    assert captured["bucket_max_size_bytes"] == 100 * 1024**3


def test_known_project_bucket_options_reject_invalid_values() -> None:
    common = [
        "configure",
        "--no-interactive",
        "--provision",
        "--tenant-id",
        "tenant-known",
        "--project-id",
        "project-known",
        "--region",
        "us-central1",
        "--project-alias",
        "fleet-a",
    ]
    result = runner.invoke(app, [*common, "--bucket-storage-class", "mystery"])
    assert result.exit_code != 0
    assert "must be standard, enhanced, or intelligent" in result.output

    result = runner.invoke(app, [*common, "--bucket-size-gb", "not-a-number"])
    assert result.exit_code != 0
    assert "must be a finite non-negative number" in result.output


def test_noninteractive_storage_selection_never_prints_prompt_language(
    monkeypatch, capsys
) -> None:
    from types import SimpleNamespace

    from npa.cli.main import _provision_object_storage
    from npa.clients.storage_validation import StorageProbeResult

    fake_nebius = SimpleNamespace(
        bucket_exists=lambda _project, _bucket: False,
        normalize_bucket_storage_class=lambda _value: "standard",
        NebiusError=RuntimeError,
        is_permission_denied=lambda _message: False,
    )
    monkeypatch.setattr(
        "npa.cli.main._generated_configure_bucket_name",
        lambda _tenant, _project: "derived-bucket",
    )
    monkeypatch.setattr(
        "npa.clients.storage_setup.provision_storage",
        lambda **_kwargs: (
            {
                "nebius_api_key": "access",
                "nebius_secret_key": "secret",
                "s3_bucket": "derived-bucket",
                "s3_endpoint": "https://storage.invalid",
            },
            StorageProbeResult(True, "ok", "verified"),
        ),
    )

    result = _provision_object_storage(
        fake_nebius,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("prompted")),
        project_id="project",
        tenant_id="tenant",
        region="eu-north1",
        interactive=False,
    )

    output = capsys.readouterr().out
    assert result is not None
    assert (
        "Object storage (non-interactive): generated fresh name 'derived-bucket'"
        in output
    )
    assert "enter a bucket name" not in output.lower()
    assert "press Enter" not in output


def test_noninteractive_storage_creation_forwards_requested_shape(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from npa.cli.main import _provision_object_storage
    from npa.clients.storage_validation import StorageProbeResult

    fake_nebius = SimpleNamespace(
        bucket_exists=lambda _project, _bucket: False,
        normalize_bucket_storage_class=lambda value: (
            "enhanced_throughput" if value == "enhanced_throughput" else "standard"
        ),
        NebiusError=RuntimeError,
        is_permission_denied=lambda _message: False,
    )
    captured: dict[str, object] = {}

    def fake_provision(**kwargs):
        captured.update(kwargs)
        return (
            {
                "nebius_api_key": "access",
                "nebius_secret_key": "secret",
                "s3_bucket": "derived-bucket",
                "s3_endpoint": "https://storage.invalid",
            },
            StorageProbeResult(True, "ok", "verified"),
        )

    monkeypatch.setattr("npa.clients.storage_setup.provision_storage", fake_provision)
    result = _provision_object_storage(
        fake_nebius,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("prompted")),
        project_id="project",
        tenant_id="tenant",
        region="us-central1",
        existing_bucket="derived-bucket",
        interactive=False,
        bucket_storage_class="enhanced_throughput",
        bucket_max_size_bytes=100 * 1024**3,
    )

    assert result is not None
    assert captured["bucket_storage_class"] == "enhanced_throughput"
    assert captured["bucket_max_size_bytes"] == 100 * 1024**3


def test_noninteractive_storage_reuse_rejects_shape_mismatch(monkeypatch) -> None:
    from types import SimpleNamespace

    from npa.cli.main import _provision_object_storage

    fake_nebius = SimpleNamespace(
        bucket_exists=lambda _project, _bucket: True,
        get_bucket_by_name=lambda _project, _bucket: {
            "spec": {"default_storage_class": "standard", "max_size_bytes": "1"}
        },
        normalize_bucket_storage_class=lambda value: str(value).lower(),
        NebiusError=RuntimeError,
        is_permission_denied=lambda _message: False,
    )

    def must_not_provision(**_kwargs):
        raise AssertionError("mismatched existing bucket must not be provisioned")

    monkeypatch.setattr(
        "npa.clients.storage_setup.provision_storage", must_not_provision
    )
    result = _provision_object_storage(
        fake_nebius,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("prompted")),
        project_id="project",
        tenant_id="tenant",
        region="us-central1",
        existing_bucket="derived-bucket",
        interactive=False,
        bucket_storage_class="enhanced_throughput",
        bucket_max_size_bytes=100 * 1024**3,
    )

    assert result is None


def test_npa_version_emits_no_syntax_warning(tmp_path) -> None:
    """The README verify step `npa --version` must be warning-clean.

    It once printed ``SyntaxWarning: invalid escape sequence '\\s'`` from an
    embedded f-string in ``npa/src/npa/cli/agent.py`` (which `npa --version`
    imports). Reproduce the first-run experience in a subprocess with a fresh
    bytecode cache so every ``npa`` module compiles from source and any escape
    warning would actually surface, then assert none originates from the
    package. Runs the module entrypoint directly so it works without ``npa``
    being on ``PATH``.
    """
    env = dict(os.environ)
    # Fresh, isolated bytecode cache -> our modules recompile from source, so a
    # stray invalid escape re-appears here instead of being masked by a warm
    # ``__pycache__``. Show every SyntaxWarning rather than the default "once".
    env["PYTHONPYCACHEPREFIX"] = str(tmp_path / "pycache")
    # Invalid escapes are a SyntaxWarning on >=3.12 but a DeprecationWarning on
    # 3.10/3.11 (which CI runs and Ubuntu 22.04 ships), so surface both or the
    # check is blind exactly where new users hit it.
    env["PYTHONWARNINGS"] = "always::SyntaxWarning,always::DeprecationWarning"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.argv = ['npa', '--version']; "
            "from npa.cli.main import app_entry; app_entry()",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    combined = proc.stdout + proc.stderr
    # An invalid-escape warning attributable to the shipped package (path under
    # src/npa/); third-party dependency warnings live under site-packages and are
    # ignored. Match both categories so the guard works on the whole 3.10+ matrix.
    package_warnings = [
        line
        for line in combined.splitlines()
        if ("SyntaxWarning" in line or "invalid escape sequence" in line)
        and re.search(r"[\\/]src[\\/]npa[\\/]", line)
    ]
    assert not package_warnings, "npa emitted invalid-escape warning(s):\n" + "\n".join(
        package_warnings
    )
    assert "invalid escape sequence" not in combined, combined
    assert proc.returncode == 0, combined
    assert "npa" in proc.stdout


def test_npa_version_fast_path_skips_heavy_imports() -> None:
    """``npa --version`` must not import the full command tree.

    The console-script entry (``npa.cli.entry:main``) answers a bare version
    request before importing ``npa.cli.main``, which transitively pulls in heavy
    dependencies (boto3, paramiko, rerun, numpy). Guard that this stays fast so
    ``npa --version`` does not regress back to a multi-hundred-millisecond
    import. Runs in a subprocess so the check sees a clean interpreter.
    """
    probe = (
        "import sys; sys.argv = ['npa', '--version']; "
        "from npa.cli.entry import main; main(); "
        "heavy = [m for m in "
        "('npa.cli.main', 'boto3', 'paramiko', 'rerun', 'numpy') "
        "if m in sys.modules]; "
        "print('HEAVY:' + ','.join(heavy))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "npa " in proc.stdout
    assert "HEAVY:\n" in proc.stdout or proc.stdout.rstrip().endswith("HEAVY:"), (
        "npa --version fast path imported heavy modules: " + proc.stdout
    )


def test_cosmos2_capability_path_skips_the_platform_command_tree() -> None:
    """The purpose-built Cosmos image can run its CLI without platform extras."""

    probe = """
import importlib.abc
import os
import sys

blocked = {"httpx", "kubernetes", "paramiko", "rerun"}

class BlockPlatformExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in blocked:
            raise RuntimeError(f"platform-only import attempted: {fullname}")
        return None

os.environ["NPA_SKIP_EAGER_IMPORTS"] = "1"
sys.meta_path.insert(0, BlockPlatformExtras())
sys.argv = ["npa", "workbench", "cosmos2", "transfer", "--help"]
from npa.cli.entry import main
main()
"""
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "--segmentation-mode" in proc.stdout


def test_cosmos2_capability_module_entry_dispatches_the_mounted_path() -> None:
    """The image-authored ``python -m`` wrapper must execute the fast entry."""

    env = dict(os.environ)
    env["NPA_SKIP_EAGER_IMPORTS"] = "1"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "npa.cli.entry",
            "workbench",
            "cosmos2",
            "transfer",
            "--help",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "--control-asset" in proc.stdout
    assert "--segmentation-mode" in proc.stdout


def test_cosmos2_capability_path_consumes_mounted_command_name() -> None:
    """The standalone image entrypoint must parse options after ``transfer``."""

    probe = """
import sys
from npa.cli.workbench import cosmos2 as cli

captured = {}

def fake_app(*, args, prog_name):
    captured["args"] = args
    captured["prog_name"] = prog_name

cli.app = fake_app
sys.argv = [
    "npa", "workbench", "cosmos2", "transfer",
    "--input-uri", "local:///input", "--output-uri", "local:///output",
]
from npa.cli.entry import main
main()
print(captured)
"""
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "'args': ['--input-uri', 'local:///input'" in proc.stdout
    assert "'prog_name': 'npa workbench cosmos2 transfer'" in proc.stdout


def test_quickstart_first_success_fixture_is_packaged() -> None:
    """The fixture the quickstart points at must ship inside the package."""
    assert DEFAULT_SAMPLE_BENCHMARK_PATH.exists(), (
        "Quickstart first-success benchmark fixture is missing: "
        f"{DEFAULT_SAMPLE_BENCHMARK_PATH}"
    )


def test_quickstart_benchmark_command_produces_real_result(tmp_path) -> None:
    """Run the exact documented first-success command end to end, offline."""
    output_path = tmp_path / "vlm-eval-benchmark.json"

    result = runner.invoke(
        app,
        [
            "workbench",
            "vlm-eval",
            "benchmark",
            "--dataset",
            str(DEFAULT_SAMPLE_BENCHMARK_PATH),
            "--output",
            str(output_path),
            "--backend",
            "stub",
            "--thresholds",
            "0.5,0.8,0.9",
            "--rubrics",
            "default,strict",
            "--models",
            DEFAULT_MODEL,
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # A real scoring pass over the shipped labeled rollout set, no GPU or creds.
    assert payload["best_config"]["metrics"]["accuracy"] == 1.0
    assert json.loads(output_path.read_text(encoding="utf-8"))["item_count"] == 4
