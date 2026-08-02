"""The triage stage, executable and offline-testable.

`tokenfactory-train-triage.yaml` built its prompt in ~45 lines of inline bash + python and then
called `token-factory generate` with `--system-prompt "$(cat …)"`. That shell substitution is
what a `toolRef` argv cannot express, so the stage became one command. Token Factory and storage
are both injected here, so the whole thing is checkable without a network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from npa.workflows.token_factory_combos import (
    DEFAULT_TRIAGE_SYSTEM_PROMPT,
    triage_prompts_uri,
    triage_report_uri,
)
from npa.workflows.token_factory_triage import (
    TEXT_SUFFIXES,
    TriageError,
    build_parser,
    download_textual_artifacts,
    run_triage,
)


@dataclass
class FakeGeneration:
    id: str = "triage"
    prompt: str = "p"
    completion: str = "Summary: the run looks healthy."


@dataclass
class FakeResult:
    status: str = "completed"
    result_uri: str = ""
    model: str = "fake-text-model"
    prompt_count: int = 1
    generations: list = field(default_factory=lambda: [FakeGeneration()])


def _artifacts(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text("policy: act\nsteps: 1\n", encoding="utf-8")
    (root / "train.log").write_text("step 1 loss 0.42\n", encoding="utf-8")
    (root / "metrics.json").write_text(json.dumps({"loss": 0.42}), encoding="utf-8")
    # Binary artifacts must be ignored: a checkpoint tells a text model nothing.
    (root / "model.safetensors").write_bytes(b"\x00\x01\x02")
    (root / "rollout.mp4").write_bytes(b"\x00")
    return root


def test_only_textual_artifacts_are_downloaded(tmp_path: Path) -> None:
    source = _artifacts(tmp_path / "artifacts")
    dest = tmp_path / "dest"
    dest.mkdir()

    fetched = download_textual_artifacts(str(source), dest)

    assert sorted(fetched) == ["config.yaml", "metrics.json", "train.log"]
    assert not (dest / "model.safetensors").exists()
    assert ".mp4" not in TEXT_SUFFIXES


def test_missing_artifacts_directory_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(TriageError, match="not a directory"):
        download_textual_artifacts(str(tmp_path / "nope"), tmp_path)


def test_run_triage_writes_a_report_and_the_prompt_beside_it(tmp_path: Path) -> None:
    source = _artifacts(tmp_path / "artifacts")
    triage = tmp_path / "triage"
    captured: dict = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return FakeResult(result_uri=kwargs["output_path"])

    payload = run_triage(
        artifacts_uri=str(source),
        triage_uri=str(triage),
        job_name="tokenfactory-train-triage",
        generate=fake_generate,
    )

    assert payload["status"] == "completed"
    assert payload["artifact_count"] == 3
    # The report lands where the pure helper says it should, and the prompt is published too —
    # without it the report is unreviewable.
    assert payload["report_uri"].endswith(Path(triage_report_uri(str(triage))).name)
    assert Path(payload["prompts_uri"]) == Path(triage_prompts_uri(str(triage)))
    assert Path(payload["prompts_uri"]).is_file()


def test_the_prompt_carries_the_digest_and_the_default_system_prompt(tmp_path: Path) -> None:
    source = _artifacts(tmp_path / "artifacts")
    captured: dict = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return FakeResult(result_uri=kwargs["output_path"])

    payload = run_triage(
        artifacts_uri=str(source),
        triage_uri=str(tmp_path / "triage"),
        job_name="my-run",
        generate=fake_generate,
    )

    # The system prompt the template `cat`-ed from a file is now passed directly.
    assert captured["system_prompt"] == DEFAULT_TRIAGE_SYSTEM_PROMPT
    # Read the PUBLISHED prompt, not the temp copy: that is the artifact a reviewer sees.
    published = Path(payload["prompts_uri"]).read_text(encoding="utf-8")
    record = json.loads(published.splitlines()[0])
    assert "my-run" in record["prompt"]
    # The digest of the real artifacts reached the prompt.
    assert "train.log" in record["prompt"] or "0.42" in record["prompt"]


def test_a_run_with_no_textual_artifacts_fails_loudly(tmp_path: Path) -> None:
    """A silent empty report would look like a successful triage."""

    source = tmp_path / "artifacts"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"\x00")

    with pytest.raises(TriageError, match="no textual artifacts"):
        run_triage(
            artifacts_uri=str(source),
            triage_uri=str(tmp_path / "triage"),
            job_name="empty",
            generate=lambda **kwargs: pytest.fail("must not call Token Factory"),
        )


@pytest.mark.parametrize("missing", ["--artifacts-uri", "--triage-uri"])
def test_required_uris_are_validated(tmp_path: Path, missing: str) -> None:
    kwargs = {
        "artifacts_uri": str(tmp_path),
        "triage_uri": str(tmp_path / "triage"),
        "job_name": "x",
        "generate": lambda **_: FakeResult(),
    }
    kwargs["artifacts_uri" if missing == "--artifacts-uri" else "triage_uri"] = "   "

    with pytest.raises(TriageError, match="required"):
        run_triage(**kwargs)


def test_parser_requires_both_uris() -> None:
    parser = build_parser()

    assert parser.parse_args(
        ["run", "--artifacts-uri", "s3://b/a/", "--triage-uri", "s3://b/t/"]
    ).command == "run"
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--artifacts-uri", "s3://b/a/"])
