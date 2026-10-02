"""Exercise challenge onboarding without credentials, remote services or GPUs."""

import hashlib
import json
from pathlib import Path
import subprocess

import pytest
from typer.core import TyperGroup
from typer.main import get_command
from typer.testing import CliRunner
import yaml

from npa.cli.main import app
from npa.cli.workbench.workflow.challenge import app as challenge_app
from npa.clients.storage import StoragePreconditionFailed
from npa.orchestration.npa_workflow.challenges import behavior, publication
from npa.orchestration.npa_workflow.challenges.config import (
    ChallengeSetup,
    starter_config,
)
from npa.sdk.workbench.workflow_challenge import check_setup, initialize, prepare
from npa.workflows.behavior_challenge import evaluator_versions


def _git(directory, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(directory), *arguments], text=True, stderr=subprocess.PIPE
    ).strip()


def _upstream_checkout(tmp_path):
    source = tmp_path / "upstream"
    registry = source / "docs/challenge/task_data.json"
    registry.parent.mkdir(parents=True)
    tasks = ["turning_on_radio"] + [f"fixture_task_{index}" for index in range(99)]
    registry.write_text(json.dumps({"tasks": [{"id": task} for task in tasks]}))
    _git(source, "init")
    _git(source, "add", ".")
    _git(
        source,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "Synthetic task registry",
    )
    revision = _git(source, "rev-parse", "HEAD")
    return revision


@pytest.fixture
def setup_file(tmp_path, monkeypatch):
    revision = _upstream_checkout(tmp_path)
    monkeypatch.setitem(evaluator_versions.UPSTREAM_COMMITS, "fixture", revision)
    values = starter_config()
    values.update(
        project="fixture",
        infra="k8s/fixture",
        artifact_root="s3://example-bucket/private/evaluations",
    )
    values["source"] = {"checkout": "upstream", "revision": revision}
    values["runtime"].update(
        image="registry.example.com/operator/runtime@sha256:" + "a" * 64,
        assets_claim="fixture-assets",
        accelerator="L40S:1",
        data_root="/fixture-data",
    )
    values["policy"].update(
        host="policy.fixture", checkpoint_sha256="b" * 64, runbook="policy.md"
    )
    (tmp_path / "policy.md").write_text(
        "Synthetic policy runbook for onboarding tests.\n"
    )
    path = tmp_path / "setup.yaml"
    path.write_text(yaml.safe_dump(values))
    return path


def test_starter_reports_missing_fields_without_cloud_calls(tmp_path):
    result = initialize(tmp_path / "private")
    path = Path(result["config"])
    checked = check_setup(path)
    assert checked["status"] == "blocked"
    assert checked["gpu_readiness"] == "not-checked"
    assert any(issue.startswith("runtime.image:") for issue in checked["issues"])
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_fresh_directory_preparation_preserves_official_cases(
    setup_file, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    checked = check_setup(setup_file)
    assert checked["status"] == "ready-to-prepare"
    assert checked["planned_cases"] == 10
    result = prepare(Path("setup.yaml"), Path("kit"), "fixture-dev")
    assert result["gpu_readiness"] == "not-checked"
    assert not result["inputs_published"]
    plan = json.loads((tmp_path / "kit/plan.json").read_text())
    assert [case["index"] for case in plan["cases"]] == list(range(10, 20))
    assert [case["instance_id"] for case in plan["cases"]] == list(range(311, 321))
    assert not plan["eligible_for_reporting"]
    assert result["checkpoint_identity"] == "operator-declared"
    for filename, digest in result["files"].items():
        path = tmp_path / "kit" / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert path.stat().st_mode & 0o777 == 0o600
    workflow = yaml.safe_load((tmp_path / "kit/workflow.yaml").read_text())
    simulator = workflow["resources"]["simulator"]
    mounts = simulator["kubernetes"]["pod_config"]["spec"]["containers"][0][
        "volumeMounts"
    ]
    assert (
        next(
            mount["mountPath"] for mount in mounts if mount["name"] == "behavior-assets"
        )
        == "/fixture-data"
    )
    assert workflow["config"]["prefix"] == "private/evaluations/fixture-dev"
    execution = json.loads((tmp_path / "kit/execution-plan.json").read_text())
    assert "/fixture-data" in json.dumps(execution)
    assert "{{config." not in json.dumps(execution)


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "report"),
        ("challenge", "unimplemented"),
        ("unknown", True),
        ("artifact_root", "s3://example-bucket/../escape"),
        ("artifact_root", "s3://example-bucket"),
        ("project", "{{config.bucket}}"),
        ("runtime.accelerator", "B200:1"),
        ("runtime.accelerator", "L40S:8"),
        ("runtime.image", "registry.example.com/image:latest"),
        ("runtime.image", "registry.example.com/image@sha256:" + "0" * 64),
        ("runtime.assets_claim", "../escape"),
        ("runtime.data_root", "relative"),
        ("source.revision", "main"),
        ("policy.port", "8000"),
        ("policy.host", "policy.example.invalid"),
        ("policy.checkpoint_sha256", "0" * 64),
    ],
)
def test_invalid_setup_blocks_before_writing(setup_file, tmp_path, field, value):
    values = yaml.safe_load(setup_file.read_text())
    target = values
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    setup_file.write_text(yaml.safe_dump(values))
    result = check_setup(setup_file)
    assert result["status"] == "blocked"
    assert any(issue.startswith(field + ":") for issue in result["issues"])
    with pytest.raises(ValueError):
        prepare(setup_file, tmp_path / "kit", "fixture")
    assert not (tmp_path / "kit").exists()


def test_dirty_source_is_rejected(setup_file):
    registry = setup_file.parent / "upstream/docs/challenge/task_data.json"
    registry.write_text(registry.read_text() + "\n")
    assert check_setup(setup_file)["status"] == "blocked"


@pytest.mark.parametrize("contents", ["", "[]", "[invalid yaml"])
def test_malformed_setup_is_actionable(tmp_path, contents):
    path = tmp_path / "setup.yaml"
    path.write_text(contents)
    result = check_setup(path)
    assert result["status"] == "blocked" and result["issues"]


def test_existing_kit_and_symlink_are_never_overwritten(setup_file, tmp_path):
    prepare(setup_file, tmp_path / "kit", "fixture")
    original = (tmp_path / "kit/manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare(setup_file, tmp_path / "kit", "fixture")
    (tmp_path / "alias").symlink_to(tmp_path / "kit", target_is_directory=True)
    with pytest.raises(FileExistsError):
        initialize(tmp_path / "alias")
    assert (tmp_path / "kit/manifest.json").read_bytes() == original


def _parse_command(argv):
    command = get_command(app)
    context = command.make_context("npa", argv[1:])
    while isinstance(command, TyperGroup):
        name, command, arguments = command.resolve_command(
            context, context._protected_args + context.args
        )
        context = command.make_context(name, arguments, parent=context)
    return context


def test_generated_commands_parse_against_real_cli(setup_file, tmp_path):
    prepare(setup_file, tmp_path / "kit", "fixture")
    commands = json.loads((tmp_path / "kit/commands.json").read_text())
    for argv in commands.values():
        context = _parse_command(argv)
        assert not context.args
        assert context.params["project"] == "fixture"
    assert _parse_command(commands["submit_preflight"]).params["plan_only"] is True
    assert _parse_command(commands["submit"]).params["runtime"] is True
    for name in ("status", "logs", "artifacts", "cancel"):
        assert _parse_command(commands[name]).params["workflow_s3_uri"] == (
            "s3://example-bucket/private/evaluations/fixture/npa-workflow"
        )


def test_cli_returns_one_json_document_and_failure_exit(tmp_path, setup_file):
    runner = CliRunner()
    initialized = runner.invoke(
        challenge_app, ["init", "--directory", str(tmp_path / "private")]
    )
    assert initialized.exit_code == 0
    config = json.loads(initialized.stdout)["config"]
    blocked = runner.invoke(challenge_app, ["check", "--config", config])
    assert blocked.exit_code == 1
    assert json.loads(blocked.stdout)["status"] == "blocked"
    prepared = runner.invoke(
        challenge_app,
        [
            "prepare",
            "--config",
            str(setup_file),
            "--directory",
            str(tmp_path / "kit"),
            "--run-id",
            "fixture",
        ],
    )
    assert prepared.exit_code == 0, prepared.output
    assert json.loads(prepared.stdout)["planned_cases"] == 10


def test_installed_distribution_template_fallback(monkeypatch):
    expected, digest = behavior.workflow_template()
    packaged = Path(__file__).resolve().parents[4] / "workflows"
    # A wheel supplies importlib.resources rather than a neighboring Git checkout.
    monkeypatch.setattr(behavior, "catalog_files", lambda root: {})
    monkeypatch.setattr(behavior, "files", lambda package: packaged)
    document, actual_digest = behavior.workflow_template()
    assert document == expected and actual_digest == digest


class MemoryStorage:
    def __init__(self):
        self.objects = {}

    def put_bytes_conditional(self, payload, uri, *, if_none_match):
        assert if_none_match
        if uri in self.objects:
            raise StoragePreconditionFailed("already exists")
        self.objects[uri] = payload

    def read_bytes_with_etag(self, uri):
        return (self.objects[uri], "etag") if uri in self.objects else None


def test_publication_is_repeatable_and_refuses_changed_bytes(setup_file, monkeypatch):
    setup = ChallengeSetup.model_validate(yaml.safe_load(setup_file.read_text()))
    storage = MemoryStorage()
    monkeypatch.setattr(publication, "_storage_for", lambda *_: storage)
    payloads = {"recipe.json": b"recipe", "policy.md": b"policy", "ignored": b"private"}
    publication.publish_inputs(setup, "s3://example-bucket/inputs/", payloads)
    publication.publish_inputs(setup, "s3://example-bucket/inputs/", payloads)
    assert len(storage.objects) == 2
    payloads["policy.md"] = b"changed"
    with pytest.raises(ValueError, match="conflicts"):
        publication.publish_inputs(setup, "s3://example-bucket/inputs/", payloads)
    assert storage.objects["s3://example-bucket/inputs/policy.md"] == b"policy"


def test_publication_readback_mismatch_fails(setup_file, monkeypatch):
    setup = ChallengeSetup.model_validate(yaml.safe_load(setup_file.read_text()))
    storage = MemoryStorage()
    monkeypatch.setattr(publication, "_storage_for", lambda *_: storage)
    monkeypatch.setattr(
        storage, "read_bytes_with_etag", lambda _: (b"corrupted", "etag")
    )
    with pytest.raises(ValueError, match="readback differs"):
        publication.publish_inputs(
            setup, "s3://example-bucket/inputs/", {"recipe.json": b"recipe"}
        )


def test_publication_scope_failure_preserves_local_kit(
    setup_file, tmp_path, monkeypatch
):
    from npa.clients.config import StorageConfig
    from npa.execution_preflight import ExecutionPreflightError

    calls = []

    def saved_storage(project, **options):
        calls.append((project, options))
        return StorageConfig(
            "example-bucket",
            "https://storage.example.com",
            "fixture-key",
            "fixture-secret",
        )

    def reject_scope(target, *, verify_cluster):
        assert not verify_cluster
        raise ExecutionPreflightError("scope", "fixture ownership mismatch")

    monkeypatch.setattr(publication, "resolve_project_storage", saved_storage)
    monkeypatch.setattr(
        publication, "resolve_execution_target", lambda **kwargs: kwargs
    )
    monkeypatch.setattr(publication, "verify_execution_target", reject_scope)
    monkeypatch.setattr(
        publication,
        "StorageClient",
        lambda **_: pytest.fail("must not write after scope failure"),
    )
    with pytest.raises(ExecutionPreflightError):
        prepare(setup_file, tmp_path / "kit", "fixture", publish=True)
    assert calls == [
        ("fixture", {"include_shared_credentials": False, "include_environment": False})
    ]
    assert (tmp_path / "kit/recipe.json").exists()
    assert "fixture-secret" not in (tmp_path / "kit/manifest.json").read_text()


@pytest.mark.parametrize("run_id", ["../escape", "", "UPPER", "x" * 64, "has spaces"])
def test_invalid_run_identity_has_no_side_effects(setup_file, tmp_path, run_id):
    with pytest.raises(ValueError, match="run_id"):
        prepare(setup_file, tmp_path / "kit", run_id, publish=True)
    assert not (tmp_path / "kit").exists()


def test_missing_policy_runbook_is_actionable(setup_file):
    (setup_file.parent / "policy.md").unlink()
    checked = check_setup(setup_file)
    assert checked["status"] == "blocked"
    assert checked["issues"][0].startswith("policy.runbook:")
