"""Unit tests for the real Cosmos 3 generation runner (no GPU, no network)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from npa.workbench.cosmos.generate import (
    DEFAULT_CHECKPOINT,
    GENERATE_MODES,
    XET_AFFECTED_HF_XET_VERSION,
    XET_AFFECTED_HUGGINGFACE_HUB_VERSION,
    Cosmos3GenerateError,
    _installed_package_versions,
    build_generate_spec,
    check_xet_pin,
    cosmos3_generate_available,
    generate_plan,
    require_model_access,
    resolve_hf_token,
    run_cosmos3_generate,
)

SPEC_YAML = (
    Path(__file__).resolve().parents[2]
    / "workflows/workbench/npa-workflows/cosmos3-generate.yaml"
)


def _fake_runtime(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Create a framework checkout shaped like the one baked into npa-cosmos3."""

    repo = tmp_path / "cosmos-framework"
    (repo / "cosmos_framework" / "scripts").mkdir(parents=True)
    (repo / "cosmos_framework" / "scripts" / "inference.py").write_text("", encoding="utf-8")
    venv_python = repo / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_python.chmod(0o755)
    env = {"COSMOS3_REPO": str(repo), "HF_TOKEN": "hf-secret"}
    return repo, env


def test_generate_modes_cover_the_documented_surface() -> None:
    assert set(GENERATE_MODES) == {
        "image2image",
        "image2video",
        "text2image",
        "text2video",
        "video2video",
    }


def test_build_generate_spec_sets_model_mode_and_omits_unset_overrides() -> None:
    spec = build_generate_spec(prompt="a robot arm", name="sample")

    assert spec == {
        "model_mode": "text2image",
        "name": "sample",
        "prompt": "a robot arm",
    }


def test_build_generate_spec_carries_vision_and_overrides() -> None:
    spec = build_generate_spec(
        mode="image2video",
        prompt="a robot arm pours water",
        name="i2v",
        vision_path="/data/frame.png",
        negative_prompt="blurry",
        num_steps=12,
        guidance=6.5,
    )

    assert spec["model_mode"] == "image2video"
    assert spec["vision_path"] == "/data/frame.png"
    assert spec["negative_prompt"] == "blurry"
    assert spec["num_steps"] == 12
    assert spec["guidance"] == 6.5


@pytest.mark.parametrize("mode", ["image2image", "image2video", "video2video"])
def test_conditioned_modes_require_a_vision_asset(mode: str) -> None:
    with pytest.raises(Cosmos3GenerateError, match="conditions on an input"):
        build_generate_spec(mode=mode, prompt="a robot arm")


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(Cosmos3GenerateError, match="unsupported generate mode"):
        build_generate_spec(mode="reasoner", prompt="a robot arm")


def test_empty_prompt_is_rejected() -> None:
    with pytest.raises(Cosmos3GenerateError, match="prompt must not be empty"):
        build_generate_spec(prompt="   ")


def test_plan_keeps_guardrails_on_by_default(tmp_path: Path) -> None:
    plan = generate_plan(prompt="a robot arm", output_dir=tmp_path)

    assert plan["guardrails"] is True
    assert "--no-guardrails" not in plan["argv"]
    assert plan["checkpoint"] == DEFAULT_CHECKPOINT
    assert plan["argv"][1:3] == ["-m", "cosmos_framework.scripts.inference"]
    assert "--parallelism-preset" in plan["argv"]


def test_plan_disables_guardrails_only_on_explicit_opt_out(tmp_path: Path) -> None:
    plan = generate_plan(prompt="a robot arm", output_dir=tmp_path, no_guardrails=True)

    assert plan["guardrails"] is False
    assert "--no-guardrails" in plan["argv"]


def test_require_model_access_demands_operator_hf_token() -> None:
    with pytest.raises(Cosmos3GenerateError, match="not baked into this image"):
        require_model_access(checkpoint="Cosmos3-Nano", environ={})

    assert require_model_access(
        checkpoint="Cosmos3-Nano", environ={"HF_TOKEN": "hf-secret"}
    ) == {"hf_auth": "configured", "ngc_auth": "skipped"}


def test_require_model_access_missing_token_error_names_the_401_403_diagnostic() -> None:
    """The 401 (bad token) vs 403 (unaccepted license) split, plus a doc pointer."""

    with pytest.raises(Cosmos3GenerateError) as excinfo:
        require_model_access(checkpoint="Cosmos3-Nano", environ={})

    message = str(excinfo.value)
    assert "anonymous" in message
    assert "401" in message
    assert "403" in message
    assert "docs/workbench/cosmos3-access-preflight.md" in message


def test_require_model_access_skips_token_only_when_nothing_is_fetched() -> None:
    """A staged checkpoint alone does not exempt the run.

    Guardrails are on by default and pull the gated guardrail models from Hugging
    Face, so exempting on the checkpoint alone would let the preflight pass and
    the run die mid-inference — the exact failure this check prevents.
    """

    with pytest.raises(Cosmos3GenerateError, match="guardrail"):
        require_model_access(checkpoint="/mnt/checkpoints/cosmos3", environ={})

    result = require_model_access(
        checkpoint="/mnt/checkpoints/cosmos3", guardrails=False, environ={}
    )
    assert result["hf_auth"] == "skipped"


def test_require_model_access_demands_token_for_guardrails_alone() -> None:
    with pytest.raises(Cosmos3GenerateError, match="Cosmos-Guardrail1"):
        require_model_access(checkpoint="s3://bucket/staged-cosmos3", environ={})


def test_run_generate_without_guardrails_needs_no_token_for_staged_weights(
    tmp_path: Path,
) -> None:
    """The exemption the docs promise must actually work end to end."""

    _, env = _fake_runtime(tmp_path)
    env.pop("HF_TOKEN")
    output_dir = tmp_path / "out"

    def fake_runner(argv, **kwargs):
        sample = output_dir / "npa-generate"
        sample.mkdir(parents=True)
        (sample / "vision.jpg").write_bytes(b"y" * 2048)
        return subprocess.CompletedProcess(argv, 0)

    result = run_cosmos3_generate(
        prompt="a robot arm",
        checkpoint="/mnt/checkpoints/cosmos3",
        no_guardrails=True,
        output_dir=output_dir,
        environ=env,
        runner=fake_runner,
    )

    assert result["status"] == "executed"
    assert result["hf_auth"] == "skipped"
    assert result["guardrails"] is False


def test_staged_checkpoint_tilde_is_expanded(tmp_path: Path) -> None:
    """``~`` counts as staged, so it must also be expanded before upstream sees it."""

    plan = generate_plan(
        prompt="a robot arm", checkpoint="~/checkpoints/cosmos3", output_dir=tmp_path
    )

    assert not plan["checkpoint"].startswith("~")
    assert plan["checkpoint"].endswith("/checkpoints/cosmos3")
    assert plan["checkpoint"] in plan["argv"]


def test_require_model_access_enforces_ngc_when_demanded() -> None:
    env = {"HF_TOKEN": "hf-secret", "NPA_COSMOS3_REQUIRE_NGC": "1"}

    with pytest.raises(Cosmos3GenerateError, match="NGC API key"):
        require_model_access(checkpoint="Cosmos3-Nano", environ=env)

    assert require_model_access(
        checkpoint="Cosmos3-Nano", environ={**env, "NGC_API_KEY": "ngc-secret"}
    )["ngc_auth"] == "configured"


def test_resolve_hf_token_honours_the_env_name_override() -> None:
    env = {"NPA_COSMOS3_HF_TOKEN_ENV": "MY_HF", "MY_HF": "token-value"}

    assert resolve_hf_token(env) == "token-value"


def _version_probe_runner(versions: dict[str, str]):
    def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, json.dumps(versions), "")

    return run


def test_check_xet_pin_warns_only_on_the_exact_affected_pair(tmp_path: Path) -> None:
    repo, _ = _fake_runtime(tmp_path)

    warning = check_xet_pin(
        repo,
        runner=_version_probe_runner(
            {
                "huggingface_hub": XET_AFFECTED_HUGGINGFACE_HUB_VERSION,
                "hf-xet": XET_AFFECTED_HF_XET_VERSION,
            }
        ),
    )

    assert "HF_HUB_DISABLE_XET=1" in warning
    assert "xet-core#895" in warning
    assert "docs/workbench/cosmos3-access-preflight.md" in warning


def test_check_xet_pin_is_silent_on_a_newer_unaffected_pair(tmp_path: Path) -> None:
    repo, _ = _fake_runtime(tmp_path)

    warning = check_xet_pin(
        repo,
        runner=_version_probe_runner(
            {"huggingface_hub": "1.26.0", "hf-xet": "1.6.0"}
        ),
    )

    assert warning == ""


def test_check_xet_pin_fails_open_when_the_probe_cannot_run(tmp_path: Path) -> None:
    repo, _ = _fake_runtime(tmp_path)

    def broken_runner(argv, **kwargs):
        raise OSError("no such interpreter")

    assert check_xet_pin(repo, runner=broken_runner) == ""


def test_installed_package_versions_probe_runs_against_a_real_interpreter() -> None:
    """Execute the literal probe string, the one part a fake runner cannot check."""

    versions = _installed_package_versions(
        Path(sys.executable), ("pytest", "no-such-dist-xyz")
    )

    assert versions["pytest"] == pytest.__version__
    assert "no-such-dist-xyz" not in versions


def test_check_xet_pin_probes_the_venv_interpreter_for_both_packages(
    tmp_path: Path,
) -> None:
    repo, _ = _fake_runtime(tmp_path)
    seen: list[list[str]] = []

    def capturing_runner(argv, **kwargs):
        seen.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "{}", "")

    check_xet_pin(repo, runner=capturing_runner)

    argv = seen[0]
    assert argv[0] == str(repo / ".venv" / "bin" / "python")
    assert argv[1] == "-c"
    assert tuple(argv[3:]) == ("huggingface_hub", "hf-xet")


@pytest.mark.parametrize(
    "outcome",
    [
        "nonzero-exit",
        "timeout",
        "malformed-json",
        "json-but-not-an-object",
    ],
)
def test_check_xet_pin_fails_open_on_every_probe_failure_mode(
    tmp_path: Path, outcome: str
) -> None:
    """A warning must never become a gate: every failure returns "" silently."""

    repo, _ = _fake_runtime(tmp_path)

    def runner(argv, **kwargs):
        if outcome == "nonzero-exit":
            return subprocess.CompletedProcess(argv, 1, "", "boom")
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(argv, 30)
        if outcome == "malformed-json":
            return subprocess.CompletedProcess(argv, 0, "not json at all", "")
        return subprocess.CompletedProcess(argv, 0, "[]", "")

    assert check_xet_pin(repo, runner=runner) == ""
    assert _installed_package_versions(
        repo / ".venv" / "bin" / "python", ("huggingface_hub",), runner=runner
    ) == {}


def _generate_with_probe(
    tmp_path: Path,
    output_dir: Path,
    versions: dict[str, str],
    *,
    no_guardrails: bool = False,
) -> None:
    """Drive the real check_xet_pin gate through run_cosmos3_generate's seam."""

    _, env = _fake_runtime(tmp_path)

    def fake_runner(argv, **kwargs):
        sample = output_dir / "npa-generate"
        sample.mkdir(parents=True)
        (sample / "vision.jpg").write_bytes(b"y" * 2048)
        return subprocess.CompletedProcess(argv, 0)

    run_cosmos3_generate(
        prompt="a robot arm",
        output_dir=output_dir,
        environ=env,
        no_guardrails=no_guardrails,
        runner=fake_runner,
        version_probe_runner=_version_probe_runner(versions),
    )


AFFECTED_PAIR = {
    "huggingface_hub": XET_AFFECTED_HUGGINGFACE_HUB_VERSION,
    "hf-xet": XET_AFFECTED_HF_XET_VERSION,
}


def test_run_generate_warns_on_stderr_for_the_affected_xet_pin(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _generate_with_probe(tmp_path, tmp_path / "out", AFFECTED_PAIR)

    assert "HF_HUB_DISABLE_XET=1" in capsys.readouterr().err


def test_run_generate_still_warns_with_guardrails_off(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """xet-core#895 is a Hugging Face transfer bug, not a guardrail one.

    A --no-guardrails run still downloads its checkpoint from Hugging Face, so
    gating the warning on guardrails would hide it from exactly the runs the
    doc tells operators to try first.
    """

    _generate_with_probe(
        tmp_path, tmp_path / "out", AFFECTED_PAIR, no_guardrails=True
    )

    assert "HF_HUB_DISABLE_XET=1" in capsys.readouterr().err


def test_run_generate_is_silent_on_an_unaffected_xet_pin(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _generate_with_probe(
        tmp_path,
        tmp_path / "out",
        {"huggingface_hub": "1.26.0", "hf-xet": "1.6.0"},
    )

    assert "HF_HUB_DISABLE_XET" not in capsys.readouterr().err


def test_availability_is_false_without_the_baked_runtime(tmp_path: Path) -> None:
    assert cosmos3_generate_available({"COSMOS3_REPO": str(tmp_path / "missing")}) is False


def test_run_generate_reports_the_produced_image(tmp_path: Path) -> None:
    repo, env = _fake_runtime(tmp_path)
    output_dir = tmp_path / "out"
    calls: list[list[str]] = []

    def fake_runner(argv, **kwargs):
        calls.append(list(argv))
        sample = output_dir / "sample"
        (sample / "inputs").mkdir(parents=True)
        # The framework copies conditioning assets into inputs/; those must never
        # be reported as the generated result.
        (sample / "inputs" / "source.png").write_bytes(b"x" * 4096)
        (sample / "vision.jpg").write_bytes(b"y" * 2048)
        (sample / "sample_outputs.json").write_text('{"status": "ok"}', encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    result = run_cosmos3_generate(
        prompt="a robot arm",
        output_dir=output_dir,
        name="sample",
        environ=env,
        runner=fake_runner,
    )

    assert result["status"] == "executed"
    assert result["output_kind"] == "image"
    assert result["expected_output_kind"] == "image"
    assert Path(result["output_path"]).name == "vision.jpg"
    assert result["output_bytes"] == 2048
    assert result["guardrails"] is True
    assert result["weights_baked"] is False
    assert result["hf_auth"] == "configured"
    assert result["sample_outputs"] == {"status": "ok"}
    # The input sample the model consumed is written next to the outputs.
    spec = json.loads((output_dir / "sample.json").read_text(encoding="utf-8"))
    assert spec["model_mode"] == "text2image"
    assert calls[0][0] == str(repo / ".venv" / "bin" / "python")


def test_run_generate_reports_video_for_a_video_mode(tmp_path: Path) -> None:
    _, env = _fake_runtime(tmp_path)
    output_dir = tmp_path / "out"

    def fake_runner(argv, **kwargs):
        sample = output_dir / "npa-generate"
        sample.mkdir(parents=True)
        (sample / "vision.mp4").write_bytes(b"v" * 8192)
        return subprocess.CompletedProcess(argv, 0)

    result = run_cosmos3_generate(
        mode="text2video",
        prompt="a robot arm pours water",
        output_dir=output_dir,
        environ=env,
        runner=fake_runner,
    )

    assert result["output_kind"] == "video"
    assert result["expected_output_kind"] == "video"


def test_run_generate_requires_the_runtime(tmp_path: Path) -> None:
    with pytest.raises(Cosmos3GenerateError, match="runtime is not present"):
        run_cosmos3_generate(
            prompt="a robot arm",
            output_dir=tmp_path,
            environ={"COSMOS3_REPO": str(tmp_path / "missing"), "HF_TOKEN": "t"},
            runner=lambda *a, **k: subprocess.CompletedProcess([], 0),
        )


def test_run_generate_requires_the_hf_token_before_launching(tmp_path: Path) -> None:
    repo, env = _fake_runtime(tmp_path)
    env.pop("HF_TOKEN")

    def fail_runner(argv, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("inference must not start without model credentials")

    with pytest.raises(Cosmos3GenerateError, match="not baked into this image"):
        run_cosmos3_generate(
            prompt="a robot arm",
            output_dir=tmp_path / "out",
            environ=env,
            runner=fail_runner,
        )


def test_run_generate_surfaces_inference_failure(tmp_path: Path) -> None:
    _, env = _fake_runtime(tmp_path)

    with pytest.raises(Cosmos3GenerateError, match="exit 2"):
        run_cosmos3_generate(
            prompt="a robot arm",
            output_dir=tmp_path / "out",
            environ=env,
            runner=lambda argv, **k: subprocess.CompletedProcess(argv, 2),
        )


def test_run_generate_fails_when_no_artifact_is_produced(tmp_path: Path) -> None:
    _, env = _fake_runtime(tmp_path)
    output_dir = tmp_path / "out"

    def empty_runner(argv, **kwargs):
        (output_dir / "npa-generate").mkdir(parents=True)
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises(Cosmos3GenerateError, match="no image/video artifact"):
        run_cosmos3_generate(
            prompt="a robot arm",
            output_dir=output_dir,
            environ=env,
            runner=empty_runner,
        )


def test_generate_spec_runs_the_real_toolref() -> None:
    doc = yaml.safe_load(SPEC_YAML.read_text(encoding="utf-8"))

    assert doc["apiVersion"] == "npa.workflow/v0.0.1"
    assert doc["metadata"]["name"] == "cosmos3-generate"
    state = doc["states"]["generate"]
    assert state["toolRef"] == "workbench.cosmos3.generate"
    assert state["resources"] == "gpu"
    assert state["terminal"] is True
    assert doc["config"]["output_uri"].endswith("/generated/")


def test_video_mode_prefers_the_clip_over_a_larger_poster_frame(tmp_path: Path) -> None:
    """Largest-file-wins alone would report a poster frame as the result."""

    _, env = _fake_runtime(tmp_path)
    output_dir = tmp_path / "out"

    def fake_runner(argv, **kwargs):
        sample = output_dir / "npa-generate"
        sample.mkdir(parents=True)
        # The still is deliberately larger than the clip.
        (sample / "poster.jpg").write_bytes(b"p" * 40_000)
        (sample / "vision.mp4").write_bytes(b"v" * 8_000)
        return subprocess.CompletedProcess(argv, 0)

    result = run_cosmos3_generate(
        mode="text2video",
        prompt="a robot arm pours water",
        output_dir=output_dir,
        environ=env,
        runner=fake_runner,
    )

    assert result["output_kind"] == "video"
    assert Path(result["output_path"]).name == "vision.mp4"


def test_video_mode_fails_when_only_an_image_was_produced(tmp_path: Path) -> None:
    """A mode that silently degrades to the wrong medium must not report success."""

    _, env = _fake_runtime(tmp_path)
    output_dir = tmp_path / "out"

    def fake_runner(argv, **kwargs):
        sample = output_dir / "npa-generate"
        sample.mkdir(parents=True)
        (sample / "vision.jpg").write_bytes(b"y" * 2048)
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises(Cosmos3GenerateError, match="should produce a video"):
        run_cosmos3_generate(
            mode="text2video",
            prompt="a robot arm pours water",
            output_dir=output_dir,
            environ=env,
            runner=fake_runner,
        )


def test_image2image_is_a_supported_mode() -> None:
    """It is advertised in the CLI, so it must be spec-buildable and vision-gated."""

    spec = build_generate_spec(
        mode="image2image",
        prompt="repaint the scene at night",
        vision_path="/data/frame.png",
    )

    assert spec["model_mode"] == "image2image"
    assert spec["vision_path"] == "/data/frame.png"
    assert "image2image" in GENERATE_MODES


def test_verify_env_covers_every_advertised_mode() -> None:
    """The build-time graph walk must not verify fewer modes than the CLI offers."""

    verify_env = (
        Path(__file__).resolve().parents[2]
        / "docker/workbench/cosmos3/verify_env.py"
    )
    text = verify_env.read_text(encoding="utf-8")
    match = re.search(r"^MODES = \((?P<body>.*?)\)$", text, flags=re.M | re.S)
    assert match, "verify_env.py must declare a MODES tuple"
    verified = set(re.findall(r'"([a-z0-9]+)"', match.group("body")))
    assert verified == set(GENERATE_MODES), (
        f"verify_env verifies {sorted(verified)} but the CLI advertises "
        f"{sorted(GENERATE_MODES)}"
    )
