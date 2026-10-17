"""Exercise standalone staging with real scoped configuration and ledger writes."""

import hashlib

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients import config
from npa.orchestration.npa_workflow import submission_state
from npa.orchestration.npa_workflow.src_staging import SrcStagingError

runner = CliRunner()
STAGED_URI = "s3://unit-bucket/source/reviewed/"


def _seed_config():
    config.write_config(
        {
            "default_project": "other",
            "projects": {
                "selected": {"src_s3_uri": "s3://unit-bucket/prior/"},
                "other": {"src_s3_uri": "s3://unit-bucket/other/"},
            },
        }
    )
    return config.CONFIG_PATH.read_bytes(), config.CONFIG_PATH.stat()


def _stage_argv(prefix, flags):
    return [
        "workbench",
        "workflow",
        "stage-src",
        "--bucket",
        "unit-bucket",
        "--project",
        "selected",
        "--run-id",
        "scoped-source",
        *(["--prefix", prefix] if prefix else []),
        *flags,
    ]


@pytest.mark.parametrize("prefix", ["", "custom-source"])
@pytest.mark.parametrize("flags", [[], ["--persist"], ["--no-persist"]])
def test_stage_src_persistence_controls_real_config_and_ledger(mocker, prefix, flags):
    before, original_stat = _seed_config()
    staged = mocker.patch(
        "npa.orchestration.npa_workflow.src_staging.stage_npa_source",
        return_value=STAGED_URI,
    )

    result = runner.invoke(app, _stage_argv(prefix, flags))

    assert result.exit_code == 0, result.output
    staged.assert_called_once()
    assert f"export NPA_SRC_S3_URI={STAGED_URI}" in result.output
    after = config.CONFIG_PATH.read_bytes()
    document = yaml.safe_load(after)
    assert document["projects"]["other"]["src_s3_uri"] == "s3://unit-bucket/other/"
    if flags == ["--no-persist"]:
        assert after == before
        assert hashlib.sha256(after).digest() == hashlib.sha256(before).digest()
        current_stat = config.CONFIG_PATH.stat()
        assert (current_stat.st_ino, current_stat.st_mtime_ns) == (
            original_stat.st_ino,
            original_stat.st_mtime_ns,
        )
    else:
        assert document["projects"]["selected"]["src_s3_uri"] == STAGED_URI
        assert after != before
    if prefix:
        receipt = submission_state.inspect_submission_state("selected", "scoped-source")
        assert receipt.outcome == "found"
        assert receipt.payload["source"] == {"status": "verified", "uri": STAGED_URI}
        assert "launch" not in receipt.payload


@pytest.mark.parametrize("prefix", ["", "custom-source"])
def test_no_persist_does_not_create_missing_config(mocker, prefix):
    assert not config.CONFIG_PATH.exists()
    mocker.patch(
        "npa.orchestration.npa_workflow.src_staging.stage_npa_source",
        return_value=STAGED_URI,
    )

    result = runner.invoke(app, _stage_argv(prefix, ["--no-persist"]))

    assert result.exit_code == 0, result.output
    assert not config.CONFIG_PATH.exists()
    assert f"export NPA_SRC_S3_URI={STAGED_URI}" in result.output


@pytest.mark.parametrize("prefix", ["", "custom-source"])
@pytest.mark.parametrize("flags", [[], ["--no-persist"]])
def test_stage_failure_cannot_persist_or_claim_verified_source(mocker, prefix, flags):
    before, _ = _seed_config()
    mocker.patch(
        "npa.orchestration.npa_workflow.src_staging.stage_npa_source",
        side_effect=SrcStagingError("source manifest verification failed"),
    )

    result = runner.invoke(app, _stage_argv(prefix, flags))

    assert result.exit_code == 1
    assert "source manifest verification failed" in result.output
    assert "export NPA_SRC_S3_URI=" not in result.output
    assert config.CONFIG_PATH.read_bytes() == before
    assert not submission_state.submission_state_path(
        "selected", "scoped-source"
    ).exists()


def test_no_persist_keeps_custom_prefix_ledger_failure_visible(mocker):
    before, _ = _seed_config()
    mocker.patch(
        "npa.orchestration.npa_workflow.src_staging.stage_npa_source",
        return_value=STAGED_URI,
    )
    mocker.patch.object(
        submission_state,
        "update_submission_state",
        side_effect=ValueError("existing submission receipt is unavailable"),
    )

    result = runner.invoke(app, _stage_argv("custom-source", ["--no-persist"]))

    assert result.exit_code == 1
    assert "existing submission receipt is unavailable" in result.output
    assert "export NPA_SRC_S3_URI=" not in result.output
    assert config.CONFIG_PATH.read_bytes() == before
