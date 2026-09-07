"""Exercise the installed Fleet storage CLI and its shared SDK implementation."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner
import yaml

from npa.cli.fleet.storage import verify_storage_cmd
from npa.cli.main import app
from npa.fleet import storage_verification
from npa.lifecycle_intent import OperationIntent, current_intent
from npa.sdk import fleet as fleet_sdk

runner = CliRunner()


@pytest.fixture
def spec_path(tmp_path):
    path = tmp_path / "fleet.yaml"
    path.write_text(yaml.safe_dump({
        "apiVersion": "npa.fleet/v0.0.1", "name": "storage-cli-test",
        "projects": [{"name": "team", "clusters": [{
            "name": "training", "enable_filestore": True, "cpu_nodes": {"count": 1},
        }]}],
    }))
    return path


def _report(*, passed=True, skipped=0):
    return {"passed": passed, "selected_clusters": 1, "verified_clusters": 1 - skipped,
            "skipped_clusters": skipped, "cpu_workers": 1 - skipped, "gpu_workers": 0,
            "requested_gibibytes": 1024 if not skipped else 0, "clusters": [],
            "evidence_sha256": "a" * 64}


@pytest.mark.parametrize("flag", ["--output", "--output-format"])
def test_json_has_one_document_and_mutation_boundary(spec_path, monkeypatch, flag):
    observed = []
    def verify(spec, **kwargs):
        observed.append(current_intent())
        return _report()
    monkeypatch.setattr(storage_verification, "verify_storage", verify)
    result = runner.invoke(app, ["fleet", "verify-storage", "--spec", str(spec_path), flag, "json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == _report()
    assert observed == [OperationIntent.MUTATE]


def test_cli_passes_all_selectors_to_the_shared_implementation(spec_path, monkeypatch, tmp_path):
    observed = []
    def verify(spec, **kwargs):
        observed.append(kwargs)
        return _report()
    monkeypatch.setattr(storage_verification, "verify_storage", verify)
    result = runner.invoke(app, [
        "fleet", "verify-storage", "--spec", str(spec_path), "--only-projects", "team, other",
        "--only-clusters", "training, inference", "--project-prefix", "custom-",
        "--profile", "operator-test", "--evidence-dir", str(tmp_path), "--output", "json",
    ])
    assert result.exit_code == 0
    assert observed == [{"only_projects": ["team", "other"],
                         "only_clusters": ["training", "inference"],
                         "project_prefix": "custom-", "profile": "operator-test",
                         "evidence_dir": tmp_path}]


def test_disabled_filesystem_result_is_reported(spec_path, monkeypatch):
    monkeypatch.setattr(storage_verification, "verify_storage", lambda *args, **kwargs: _report(skipped=1))
    result = runner.invoke(app, ["fleet", "verify-storage", "--spec", str(spec_path)])
    assert result.exit_code == 0
    assert "Explicitly disabled filesystem targets: 1" in result.stdout


def test_failed_verification_exits_nonzero_with_structured_report(spec_path, monkeypatch):
    monkeypatch.setattr(storage_verification, "verify_storage", lambda *args, **kwargs: _report(passed=False))
    result = runner.invoke(app, ["fleet", "verify-storage", "--spec", str(spec_path), "--output", "json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout) == _report(passed=False)


@pytest.mark.parametrize("error", [ValueError, RuntimeError, OSError])
def test_operation_error_does_not_disclose_private_diagnostics(spec_path, monkeypatch, error):
    def verify(*args, **kwargs):
        raise error("private-provider-receipt")
    monkeypatch.setattr(storage_verification, "verify_storage", verify)
    result = runner.invoke(app, ["fleet", "verify-storage", "--spec", str(spec_path), "--output", "json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["passed"] is False
    assert "private-provider-receipt" not in result.output


@pytest.mark.parametrize("contents", ["apiVersion: private-invalid-value\n", "private-invalid-value: [\n"])
def test_bad_spec_produces_sanitized_json(spec_path, contents):
    spec_path.write_text(contents)
    result = runner.invoke(app, ["fleet", "verify-storage", "--spec", str(spec_path), "--output", "json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["failures"] == ["storage_verification_unavailable"]
    assert "private-invalid-value" not in result.output


def test_direct_command_call_resolves_typer_defaults(spec_path, monkeypatch, capsys):
    observed = []
    def verify(spec, **kwargs):
        observed.append(kwargs)
        return _report()
    monkeypatch.setattr(storage_verification, "verify_storage", verify)
    verify_storage_cmd(spec_path=spec_path)
    assert observed == [{"only_projects": None, "only_clusters": None,
                         "project_prefix": None, "profile": None, "evidence_dir": None}]
    assert "verification passed" in capsys.readouterr().out


def test_help_and_format_validation_are_installed(spec_path):
    result = runner.invoke(app, ["fleet", "verify-storage", "--help"])
    assert result.exit_code == 0
    assert "--spec" in result.stdout
    assert "--only-projects" in result.stdout
    result = runner.invoke(app, ["fleet", "verify-storage", "--spec", str(spec_path), "--output", "yaml"])
    assert result.exit_code == 2


def test_sdk_delegates_to_the_same_domain_implementation(monkeypatch):
    observed = []
    def verify(spec, **kwargs):
        observed.append((spec, kwargs))
        return _report()
    monkeypatch.setattr(storage_verification, "verify_storage", verify)
    assert fleet_sdk.verify_storage("spec", only_projects=["team"], profile="operator-test") == _report()
    assert observed == [("spec", {"only_projects": ["team"], "only_clusters": None,
                                 "project_prefix": None, "profile": "operator-test",
                                 "evidence_dir": None})]
    assert "verify_storage" in fleet_sdk.__all__
