"""The LTX-2.5 licensing surface as a workflow sees it.

`npa workbench ltx2 stamp` runs in one container and `... gate` runs in another,
with only an S3 object between them. These tests exercise that hop over local
paths, and pin the two behaviours a workflow depends on: an undeclared run
cannot produce a manifest, and a gate that cannot reach a permissive answer
exits non-zero instead of falling through to the trainer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.ltx2 import gate as gate_module
from npa.workbench.ltx2.licensing import (
    ACCEPT_ENV,
    COMMERCIAL_AGREEMENT_ENV,
    ENTITY_CLASS_ENV,
    LtxLicenseError,
    PROVENANCE_SCHEMA,
    USE_CLASS_ENV,
)

runner = CliRunner()

NON_COMMERCIAL = {
    ACCEPT_ENV: "YES",
    ENTITY_CLASS_ENV: "commercial",
    USE_CLASS_ENV: "non-commercial",
}
COMMERCIAL = {
    ACCEPT_ENV: "YES",
    ENTITY_CLASS_ENV: "commercial",
    USE_CLASS_ENV: "commercial",
    COMMERCIAL_AGREEMENT_ENV: "CUA-2026-0001",
}


def _declare(monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
    for name in (ACCEPT_ENV, ENTITY_CLASS_ENV, USE_CLASS_ENV, COMMERCIAL_AGREEMENT_ENV):
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


class TestStamp:
    def test_a_declared_run_writes_a_manifest_next_to_its_video(
        self, tmp_path: Path
    ) -> None:
        result = gate_module.stamp_run(
            run_id="run-1",
            outputs=["s3://bucket/run-1/ltx2_5_text_to_video.mp4"],
            manifest_uri=str(tmp_path),
            env=NON_COMMERCIAL,
        )

        written = json.loads(Path(result.manifest_uri).read_text(encoding="utf-8"))
        assert written["schema"] == PROVENANCE_SCHEMA
        assert written["run_id"] == "run-1"
        assert written["outputs"][0]["uri"].endswith(".mp4")
        assert written["restrictions"]["derived_model_training"] == "non-commercial-only"

    def test_an_undeclared_run_cannot_produce_a_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(LtxLicenseError):
            gate_module.stamp_run(
                run_id="run-1",
                outputs=[],
                manifest_uri=str(tmp_path),
                env={},
            )

        assert not list(tmp_path.iterdir()), "a refusal must not leave an artifact"

    def test_the_cli_refuses_with_the_shared_configuration_exit_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _declare(monkeypatch, {})

        result = runner.invoke(
            app,
            [
                "workbench",
                "ltx2",
                "stamp",
                "--run-id",
                "run-1",
                "--manifest-uri",
                str(tmp_path),
            ],
        )

        # 78 is EX_CONFIG, the same code the container's own gate uses.
        assert result.exit_code == 78
        assert not list(tmp_path.iterdir())


class TestGate:
    def _stamp(self, tmp_path: Path, env: dict[str, str]) -> str:
        return gate_module.stamp_run(
            run_id="run-1",
            outputs=["s3://bucket/run-1/video.mp4"],
            manifest_uri=str(tmp_path),
            env=env,
        ).manifest_uri

    def test_non_commercial_output_may_reach_a_trainer(self, tmp_path: Path) -> None:
        manifest = self._stamp(tmp_path, NON_COMMERCIAL)

        result = gate_module.gate_run(
            manifest_uri=manifest, consumer="LeRobot policy training"
        )

        assert result.decision.allowed
        assert "Attachment A(18)" in result.decision.reason

    def test_commercial_output_may_not(self, tmp_path: Path) -> None:
        manifest = self._stamp(tmp_path, COMMERCIAL)

        result = gate_module.gate_run(
            manifest_uri=manifest, consumer="LeRobot policy training"
        )

        assert not result.decision.allowed
        assert "Attachment A(18)" in result.decision.reason

    def test_an_absent_manifest_denies_rather_than_reading_as_permission(
        self, tmp_path: Path
    ) -> None:
        result = gate_module.gate_run(
            manifest_uri=str(tmp_path / "never-written.json"), consumer="trainer"
        )

        assert not result.decision.allowed

    def test_a_corrupt_manifest_denies(self, tmp_path: Path) -> None:
        broken = tmp_path / "ltx2_provenance.json"
        broken.write_text("{ this is not json", encoding="utf-8")

        result = gate_module.gate_run(manifest_uri=str(tmp_path), consumer="trainer")

        assert not result.decision.allowed

    def test_a_foreign_schema_denies(self, tmp_path: Path) -> None:
        (tmp_path / "ltx2_provenance.json").write_text(
            json.dumps(
                {
                    "schema": "some.other.provenance.v1",
                    "restrictions": {"derived_model_training": "non-commercial-only"},
                }
            ),
            encoding="utf-8",
        )

        result = gate_module.gate_run(manifest_uri=str(tmp_path), consumer="trainer")

        assert not result.decision.allowed, (
            "a permissive-looking disposition under an unknown schema must not pass"
        )

    def test_the_report_records_the_decision_either_way(self, tmp_path: Path) -> None:
        manifest = self._stamp(tmp_path, COMMERCIAL)
        reports = tmp_path / "reports"

        result = gate_module.gate_run(
            manifest_uri=manifest, consumer="trainer", report_uri=str(reports)
        )

        payload = json.loads(Path(result.report_uri).read_text(encoding="utf-8"))
        assert payload["schema"] == gate_module.GATE_SCHEMA
        assert payload["allowed"] is False
        assert payload["consumer"] == "trainer"


class TestGateCli:
    def test_a_denial_exits_non_zero_so_the_workflow_state_fails(
        self, tmp_path: Path
    ) -> None:
        gate_module.stamp_run(
            run_id="run-1", outputs=[], manifest_uri=str(tmp_path), env=COMMERCIAL
        )

        result = runner.invoke(
            app,
            [
                "workbench",
                "ltx2",
                "gate",
                "--manifest-uri",
                str(tmp_path),
                "--consumer",
                "LeRobot policy training",
            ],
        )

        assert result.exit_code != 0

    def test_an_allowance_exits_zero_and_prints_the_disposition(
        self, tmp_path: Path
    ) -> None:
        gate_module.stamp_run(
            run_id="run-1", outputs=[], manifest_uri=str(tmp_path), env=NON_COMMERCIAL
        )

        result = runner.invoke(
            app,
            [
                "workbench",
                "ltx2",
                "gate",
                "--manifest-uri",
                str(tmp_path),
                "--consumer",
                "LeRobot policy training",
            ],
        )

        assert result.exit_code == 0
        assert json.loads(result.stdout)["derived_model_training"] == (
            "non-commercial-only"
        )

    def test_terms_needs_no_declaration_and_names_the_dated_agreement(self) -> None:
        result = runner.invoke(app, ["workbench", "ltx2", "terms", "--output", "json"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["license"]["date"] == "2026-08-11"
        assert payload["license"]["osi_approved"] is False
        assert payload["runtime_fetch"]["baked_into_image"] is False


class TestTheStampReconcilesWithTheRun:
    """`stamp` runs in a different container than the generation it describes.

    Deriving the declaration from `os.environ` alone left a seam exactly where
    the chain of custody must not have one: if the operator's environment drifted
    between states, or secret forwarding differed, the manifest the gate trusts
    would record the CPU state's opinion rather than what the GPU run did.
    """

    def _declaration(self, path: Path, *, use_class: str) -> Path:
        """What `ltx-runtime provenance` writes inside the generation container."""

        from npa.workbench.ltx2.licensing import LicenseDeclaration, ProvenanceRecord

        record = ProvenanceRecord(
            declaration=LicenseDeclaration(
                entity_class="community", use_class=use_class
            ),
            run_id="generation",
        )
        path.write_text(json.dumps(record.as_dict()), encoding="utf-8")
        return path

    def test_a_matching_declaration_stamps(self, tmp_path: Path) -> None:
        declaration = self._declaration(
            tmp_path / "ltx2_5_declaration.json", use_class="non-commercial"
        )

        result = gate_module.stamp_run(
            run_id="r1",
            outputs=[str(tmp_path / "v.mp4")],
            manifest_uri=str(tmp_path / "manifest.json"),
            env={
                ACCEPT_ENV: "YES",
                ENTITY_CLASS_ENV: "community",
                USE_CLASS_ENV: "non-commercial",
            },
            declaration_uri=str(declaration),
        )

        assert result.manifest["operator_declaration"]["use_class"] == "non-commercial"

    def test_a_declaration_that_drifted_between_states_refuses(
        self, tmp_path: Path
    ) -> None:
        """The whole point: the run said commercial, this state says otherwise."""

        declaration = self._declaration(
            tmp_path / "ltx2_5_declaration.json", use_class="commercial"
        )

        with pytest.raises(LtxLicenseError) as excinfo:
            gate_module.stamp_run(
                run_id="r1",
                outputs=[str(tmp_path / "v.mp4")],
                manifest_uri=str(tmp_path / "manifest.json"),
                env={
                    ACCEPT_ENV: "YES",
                    ENTITY_CLASS_ENV: "community",
                    USE_CLASS_ENV: "non-commercial",
                },
                declaration_uri=str(declaration),
            )

        message = str(excinfo.value)
        assert "disagrees with the one the generation ran under" in message
        assert "use_class" in message
        assert not (tmp_path / "manifest.json").exists(), (
            "a refused stamp must not leave a manifest behind"
        )

    def test_an_unreadable_declaration_refuses_rather_than_falling_back(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(LtxLicenseError) as excinfo:
            gate_module.stamp_run(
                run_id="r1",
                outputs=[],
                manifest_uri=str(tmp_path / "manifest.json"),
                env={
                    ACCEPT_ENV: "YES",
                    ENTITY_CLASS_ENV: "community",
                    USE_CLASS_ENV: "non-commercial",
                },
                declaration_uri=str(tmp_path / "absent.json"),
            )

        assert "Cannot read the generation's own declaration" in str(excinfo.value)


class TestTheGateIsBoundToItsArtifacts:
    """A decision about a document is not a decision about the bytes."""

    def _manifest(self, path: Path, *, outputs: list[str]) -> Path:
        from npa.workbench.ltx2.licensing import LicenseDeclaration, ProvenanceRecord

        record = ProvenanceRecord(
            declaration=LicenseDeclaration(
                entity_class="community", use_class="non-commercial"
            ),
            run_id="run-a",
            outputs=tuple(outputs),
        )
        path.write_text(json.dumps(record.as_dict()), encoding="utf-8")
        return path

    def test_a_manifest_that_covers_the_artifact_clears_it(self, tmp_path: Path) -> None:
        video = "s3://bucket/run-a/ltx2_5_text_to_video.mp4"
        manifest = self._manifest(tmp_path / "manifest.json", outputs=[video])

        result = gate_module.gate_run(
            manifest_uri=str(manifest), consumer="trainer", artifacts=[video]
        )

        assert result.decision.allowed is True
        assert result.as_dict()["artifacts"] == [video]

    def test_another_runs_manifest_cannot_clear_these_bytes(
        self, tmp_path: Path
    ) -> None:
        """Without this, a permissive manifest from run A clears run B's video."""

        manifest = self._manifest(
            tmp_path / "manifest.json",
            outputs=["s3://bucket/run-a/ltx2_5_text_to_video.mp4"],
        )

        result = gate_module.gate_run(
            manifest_uri=str(manifest),
            consumer="trainer",
            artifacts=["s3://bucket/run-b/ltx2_5_text_to_video.mp4"],
        )

        assert result.decision.allowed is False
        assert "does not describe" in result.decision.reason

    def test_naming_no_artifacts_still_answers_the_licence_question(
        self, tmp_path: Path
    ) -> None:
        """Callers that only have a manifest keep working; they just prove less."""

        manifest = self._manifest(tmp_path / "manifest.json", outputs=[])

        assert gate_module.gate_run(
            manifest_uri=str(manifest), consumer="trainer"
        ).decision.allowed
