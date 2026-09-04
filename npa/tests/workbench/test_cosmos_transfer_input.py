"""Unit tests for optional input-conditioning in the Cosmos Transfer 2.5 runner.

These cover the code path that makes the augment a REAL augmentation of the
caller's input clip (edge control computed on-the-fly from ``video_path``). No
upstream fixture is used.
No GPU / cosmos runtime is touched; the inference subprocess is mocked.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.workbench import cosmos2
from npa.workbench.cosmos import transfer as tx

runner = CliRunner()


def _write_extractor_python(repo: Path, body: str) -> Path:
    """Install a tiny executable used to exercise the real subprocess boundary."""

    executable = repo / ".venv" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    executable.chmod(0o755)
    return executable


def _fake_env(monkeypatch, repo: Path):
    monkeypatch.setattr(tx, "cosmos_transfer_repo", lambda: repo)
    monkeypatch.setattr(tx, "ensure_env", lambda r: Path("/usr/bin/python3"))
    monkeypatch.setenv("HF_TOKEN", "unit-test-placeholder")
    monkeypatch.setattr(tx, "prepare_guardrail_nltk_data", lambda **_kwargs: 0)

    def fake_run(cmd, *args, **kwargs):
        cwd = Path(kwargs["cwd"])
        out = cmd[cmd.index("-o") + 1]
        outdir = cwd / out
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "result.mp4").write_bytes(b"y" * 200_001)
        return None

    monkeypatch.setattr(tx.subprocess, "run", fake_run)


def test_guardrail_nltk_cache_links_are_safely_materialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = tmp_path / "hub"
    package = hub / "models--nvidia--Cosmos-Guardrail1"
    snapshot = package / "snapshots" / tx.GUARDRAIL_REVISION
    tokenizer = snapshot / "blocklist" / "nltk_data" / "tokenizers" / "punkt_tab"
    blob = package / "blobs" / "tokenizer-data"
    tokenizer.mkdir(parents=True)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"pinned tokenizer data")
    link = tokenizer / "collocations.tab"
    link.symlink_to(blob)

    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    monkeypatch.setenv("HF_TOKEN", "unit-test-placeholder")

    assert tx.prepare_guardrail_nltk_data(hf_home=str(tmp_path)) == 1
    assert link.is_file() and link.is_symlink()
    assert link.read_bytes() == b"pinned tokenizer data"
    safe = (
        tmp_path
        / tx.GUARDRAIL_NLTK_MATERIALIZED_DIR
        / tx.GUARDRAIL_REVISION
        / "tokenizers"
        / "punkt_tab"
        / "collocations.tab"
    )
    assert safe.is_file() and not safe.is_symlink()
    assert safe.read_bytes() == b"pinned tokenizer data"
    marker = safe.parents[2] / tx.GUARDRAIL_NLTK_READY_MARKER
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    assert manifest["repo_id"] == tx.GUARDRAIL_REPO
    assert manifest["revision"] == tx.GUARDRAIL_REVISION
    assert manifest["file_count"] == 1
    assert calls == [
        {
            "repo_id": tx.GUARDRAIL_REPO,
            "revision": tx.GUARDRAIL_REVISION,
            "allow_patterns": ["blocklist/nltk_data/**"],
            "cache_dir": hub,
            "token": "unit-test-placeholder",
        }
    ]

    # Verified regular-file materialization is reusable during a transient Hub
    # outage. NLTK consumes this tree and never needs the snapshot/blob links.
    def unavailable_snapshot_download(**_kwargs):
        raise AssertionError("verified cache reuse must not touch Hugging Face")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=unavailable_snapshot_download),
    )
    assert tx.prepare_guardrail_nltk_data(hf_home=str(tmp_path)) == 1


@pytest.mark.parametrize(
    ("exception", "category", "message"),
    [
        (type("RateLimitError", (Exception,), {})(), "rate_limited", "retry later"),
        (
            type("RevisionNotFoundError", (Exception,), {})(),
            "revision_unavailable",
            "do not substitute another revision",
        ),
        (type("OfflineError", (Exception,), {})(), "network_unavailable", "prewarm"),
    ],
)
def test_guardrail_nltk_download_failures_are_typed_and_leave_no_partial_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception: Exception,
    category: str,
    message: str,
) -> None:
    def failed_snapshot_download(**_kwargs):
        raise exception

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=failed_snapshot_download),
    )

    with pytest.raises(tx.GuardrailNLTKDataError, match=message) as caught:
        tx.prepare_guardrail_nltk_data(hf_home=str(tmp_path))

    assert caught.value.category == category
    destination = (
        tmp_path / tx.GUARDRAIL_NLTK_MATERIALIZED_DIR / tx.GUARDRAIL_REVISION
    )
    assert not destination.exists()
    assert not list(destination.parent.glob(f".{tx.GUARDRAIL_REVISION}.*"))


def test_guardrail_nltk_rate_limit_http_status_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HubError(Exception):
        response = types.SimpleNamespace(status_code=429)

    def failed_snapshot_download(**_kwargs):
        raise HubError("secret-bearing upstream response must not be repeated")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=failed_snapshot_download),
    )
    with pytest.raises(tx.GuardrailNLTKDataError, match="rate-limited") as caught:
        tx.prepare_guardrail_nltk_data(hf_home=str(tmp_path))
    assert caught.value.category == "rate_limited"
    assert "secret-bearing" not in str(caught.value)


def test_guardrail_nltk_copy_failure_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = tmp_path / "hub"
    snapshot = (
        hub
        / "models--nvidia--Cosmos-Guardrail1"
        / "snapshots"
        / tx.GUARDRAIL_REVISION
    )
    tokenizer = snapshot / "blocklist" / "nltk_data" / "tokenizers" / "punkt_tab"
    tokenizer.mkdir(parents=True)
    (tokenizer / "a.tab").write_bytes(b"first")
    (tokenizer / "b.tab").write_bytes(b"second")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **_kwargs: str(snapshot)),
    )
    real_copy = shutil.copyfileobj
    copies = 0

    def interrupted_copy(source, destination):  # noqa: ANN001
        nonlocal copies
        copies += 1
        if copies == 2:
            raise OSError("simulated interrupted copy")
        return real_copy(source, destination)

    monkeypatch.setattr(tx.shutil, "copyfileobj", interrupted_copy)
    with pytest.raises(tx.GuardrailNLTKDataError, match="no partial cache") as caught:
        tx.prepare_guardrail_nltk_data(hf_home=str(tmp_path))
    assert caught.value.category == "materialization_failed"

    destination = (
        tmp_path / tx.GUARDRAIL_NLTK_MATERIALIZED_DIR / tx.GUARDRAIL_REVISION
    )
    assert not destination.exists()
    assert not list(destination.parent.glob(f".{tx.GUARDRAIL_REVISION}.*"))


def test_guardrail_nltk_corrupt_or_mismatched_cache_fails_closed_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = tmp_path / "hub"
    snapshot = (
        hub
        / "models--nvidia--Cosmos-Guardrail1"
        / "snapshots"
        / tx.GUARDRAIL_REVISION
    )
    tokenizer = snapshot / "blocklist" / "nltk_data" / "tokenizers" / "punkt_tab"
    tokenizer.mkdir(parents=True)
    (tokenizer / "collocations.tab").write_bytes(b"verified")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **_kwargs: str(snapshot)),
    )
    assert tx.prepare_guardrail_nltk_data(hf_home=str(tmp_path)) == 1

    destination = (
        tmp_path / tx.GUARDRAIL_NLTK_MATERIALIZED_DIR / tx.GUARDRAIL_REVISION
    )
    payload = destination / "tokenizers" / "punkt_tab" / "collocations.tab"
    payload.chmod(0o644)
    payload.write_bytes(b"corrupt")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            snapshot_download=lambda **_kwargs: pytest.fail(
                "an invalid cache must fail closed before network access"
            )
        ),
    )
    with pytest.raises(tx.GuardrailNLTKDataError, match="does not match") as caught:
        tx.prepare_guardrail_nltk_data(hf_home=str(tmp_path))
    assert caught.value.category == "cache_invalid"

    marker = destination / tx.GUARDRAIL_NLTK_READY_MARKER
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    manifest["revision"] = "renamed-revision"
    marker.chmod(0o644)
    marker.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(tx.GuardrailNLTKDataError, match="identity") as identity:
        tx.prepare_guardrail_nltk_data(hf_home=str(tmp_path))
    assert identity.value.category == "cache_invalid"


def test_spec_for_input_video_builds_edge_control(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * 100)
    rel, modality = tx._spec_for_input_video(
        repo,
        input_video=str(clip),
        prompt="rainy night, wet asphalt",
        control="edge",
        control_weight=0.8,
        guidance=4,
        name="run-1",
        seed=1234,
    )
    assert modality == "edge"
    spec = json.loads((repo / rel).read_text())
    assert spec["video_path"] == str(clip.resolve())
    assert spec["prompt"] == "rainy night, wet asphalt"
    assert spec["edge"] == {"control_weight": 0.8}
    assert spec["guidance"] == 4
    assert spec["seed"] == 1234
    # Depth is precomputed-only: this path must never fetch Video Depth Anything.
    depth = tmp_path / "depth.mp4"
    depth.write_bytes(b"depth")
    _rel2, modality2 = tx._spec_for_input_video(
        repo,
        input_video=str(clip),
        prompt="",
        control="depth",
        control_weight=1.0,
        guidance=3,
        name="run-2",
        control_asset=str(depth),
    )
    assert modality2 == "depth"


def test_parse_protected_regions_accepts_labeled_normalized_rectangles() -> None:
    assert tx._parse_protected_regions(
        '[{"id":"identity-bearing-material","bounds":[0.1,0.2,0.8,0.9]}]'
    ) == [(0.1, 0.2, 0.8, 0.9)]


@pytest.mark.parametrize(
    "value",
    ["", "{}", "[]", "[[0,0,2,1]]", '[{"bounds":[0,0,1]}]'],
)
def test_parse_protected_regions_fails_closed_on_invalid_input(value: str) -> None:
    with pytest.raises(tx.ProtectedChromaError):
        tx._parse_protected_regions(value)


def test_preserve_source_chroma_records_effective_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.mp4"
    generated = tmp_path / "generated.mp4"
    source.write_bytes(b"source")
    generated.write_bytes(b"generated")

    def fake_run(command, **kwargs):  # noqa: ANN001
        assert kwargs["check"] is True
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"corrected-video")
            return tx.subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        decoder = command[2]
        assert decoder.count("rect_alpha = np.maximum.reduce(masks)") == 1
        assert decoder.index("rect_alpha = np.maximum.reduce(masks)") < decoder.index(
            "for aug_frame in aug_frames:"
        )
        return tx.subprocess.CompletedProcess(
            command, 0, stdout='{"frames": 7, "fps": 30.0}\n', stderr=""
        )

    monkeypatch.setattr(tx, "cosmos_transfer_repo", lambda: tmp_path)
    monkeypatch.setattr(tx.subprocess, "run", fake_run)
    result = tx.preserve_source_chroma(
        {"video_path": str(generated), "video_bytes": generated.stat().st_size},
        source_video=str(source),
        regions_json="[[0.1,0.1,0.9,0.9]]",
    )
    assert Path(result["video_path"]).read_bytes() == b"corrected-video"
    assert result["protected_chroma"] == {
        "mode": "source-chroma",
        "method": "feathered-per-pixel-source-chroma",
        "region_count": 1,
        "feather_pixels": 12,
        "luma_max_delta": 32,
        "frame_count": 7,
    }


def test_preserve_source_chroma_accepts_frame_aligned_sam2_masks(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.mp4"
    generated = tmp_path / "generated.mp4"
    masks = tmp_path / "masks"
    masks.mkdir()
    source.write_bytes(b"source")
    generated.write_bytes(b"generated")
    (masks / "mask-000000.png").write_bytes(b"mask")

    def fake_run(command, **kwargs):  # noqa: ANN001
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"corrected-video")
            return tx.subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        assert str(masks) in command
        return tx.subprocess.CompletedProcess(
            command, 0, stdout='{"frames": 1, "fps": 30.0}\n', stderr=""
        )

    monkeypatch.setattr(tx, "cosmos_transfer_repo", lambda: tmp_path)
    monkeypatch.setattr(tx.subprocess, "run", fake_run)
    result = tx.preserve_source_chroma(
        {"video_path": str(generated), "video_bytes": generated.stat().st_size},
        source_video=str(source),
        masks_dir=str(masks),
        segmentation={"engine": "meta-sam2-upstream", "manifest_uri": "s3://x"},
    )

    assert result["protected_chroma"]["method"].startswith("sam2-mask-")
    assert result["protected_chroma"]["region_count"] == 0
    assert result["protected_chroma"]["segmentation"]["engine"] == "meta-sam2-upstream"


def test_preserve_source_chroma_decodes_encoded_video_and_keeps_generated_luma(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for the encoded-video integration test")
    source = tmp_path / "source.mp4"
    generated = tmp_path / "generated.mp4"
    for path, color in ((source, "0x3c78b4"), (generated, "0x28140a")):
        subprocess.run(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:size=64x48:rate=6:duration=1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
        )
    monkeypatch.setattr(tx, "_venv_python", lambda _repo: Path(sys.executable))
    result = tx.preserve_source_chroma(
        {"video_path": str(generated), "video_bytes": generated.stat().st_size},
        source_video=str(source),
        regions_json="[[0,0,1,1]]",
        feather_pixels=1,
    )

    def first_rgb(path: Path) -> np.ndarray:
        raw = subprocess.check_output(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ]
        )
        return np.frombuffer(raw, dtype=np.uint8).reshape(48, 64, 3).astype(np.float32)

    def ycbcr_means(frame: np.ndarray) -> tuple[float, float, float]:
        red, green, blue = frame[..., 0], frame[..., 1], frame[..., 2]
        y = 0.299 * red + 0.587 * green + 0.114 * blue
        cb = 128.0 - 0.168736 * red - 0.331264 * green + 0.5 * blue
        cr = 128.0 + 0.5 * red - 0.418688 * green - 0.081312 * blue
        return float(y.mean()), float(cb.mean()), float(cr.mean())

    source_y, source_cb, source_cr = ycbcr_means(first_rgb(source))
    generated_y, generated_cb, generated_cr = ycbcr_means(first_rgb(generated))
    restored_y, restored_cb, restored_cr = ycbcr_means(
        first_rgb(Path(result["video_path"]))
    )
    assert abs(source_cb - generated_cb) > 10
    assert abs(source_cr - generated_cr) > 10
    assert abs(restored_cb - source_cb) < 5
    assert abs(restored_cr - source_cr) < 5
    assert abs(restored_y - generated_y) > 40
    assert 20 < abs(restored_y - source_y) < 37


def test_frame_aligned_masks_protect_only_the_selected_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for the encoded-video integration test")
    from PIL import Image

    source = tmp_path / "source.mp4"
    generated = tmp_path / "generated.mp4"
    masks = tmp_path / "masks"
    masks.mkdir()
    for path, color in ((source, "0x3c78b4"), (generated, "0x28140a")):
        subprocess.run(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:size=64x48:rate=6:duration=1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
        )
    for index in range(6):
        mask = Image.new("L", (64, 48), 0)
        mask.paste(255, (0, 0, 32, 48))
        mask.save(masks / f"mask-{index:06d}.png")

    monkeypatch.setattr(tx, "_venv_python", lambda _repo: Path(sys.executable))
    result = tx.preserve_source_chroma(
        {"video_path": str(generated), "video_bytes": generated.stat().st_size},
        source_video=str(source),
        masks_dir=str(masks),
        segmentation={"engine": "meta-sam2-upstream"},
        feather_pixels=8,
    )

    def first_rgb(path: Path) -> np.ndarray:
        raw = subprocess.check_output(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ]
        )
        return np.frombuffer(raw, dtype=np.uint8).reshape(48, 64, 3).astype(np.float32)

    source_frame = first_rgb(source)
    generated_frame = first_rgb(generated)
    restored = first_rgb(Path(result["video_path"]))
    protected_error = np.abs(restored[:, :28] - source_frame[:, :28]).mean()
    protected_baseline_error = np.abs(
        generated_frame[:, :28] - source_frame[:, :28]
    ).mean()
    unprotected_error = np.abs(restored[:, 36:] - generated_frame[:, 36:]).mean()
    assert protected_error < protected_baseline_error * 0.55
    assert unprotected_error < 5.0
    assert result["protected_chroma"]["frame_count"] == 6


def test_load_refinement_validates_numeric_settings(tmp_path: Path) -> None:
    from npa.workflows import data_factory_stages as dfs

    policy = tmp_path / "refinement.json"
    committed = dfs.prepare_refinement(
        str(tmp_path / "grade"),
        str(policy),
        base_control_weight=1.0,
        base_guidance=2,
    )
    loaded = cosmos2._load_refinement(str(policy))
    assert committed["attempt"] == loaded["attempt"] == 0
    assert loaded["settings"] == {"control_weight": 1.0, "guidance": 2}


def test_refinement_prompt_emphasizes_only_exact_failed_attributes() -> None:
    combo = {
        "prompt": "Preserve the scene and change its appearance.",
        "lighting": "soft diffuse daylight",
        "background": "solid gray backdrop",
        "surface_finish": "matte",
    }

    adapted = cosmos2._apply_refinement_prompt(
        combo,
        {
            "failed_attributes": [
                "lighting",
                "background",
                "not-a-configured-variable",
            ]
        },
    )

    assert "lighting is unambiguously soft diffuse daylight" in adapted["prompt"]
    assert "background is unambiguously solid gray backdrop" in adapted["prompt"]
    assert "not-a-configured-variable" not in adapted["prompt"]
    assert combo == {
        "prompt": "Preserve the scene and change its appearance.",
        "lighting": "soft diffuse daylight",
        "background": "solid gray backdrop",
        "surface_finish": "matte",
    }


@pytest.mark.parametrize(
    "settings",
    [
        {"control_weight": 1.1, "guidance": 2},
        {"control_weight": 0.8, "guidance": 2.5},
    ],
)
def test_load_refinement_rejects_values_cosmos_cannot_load(
    tmp_path: Path, settings: dict[str, float]
) -> None:
    from npa.workflows import data_factory_stages as dfs

    policy = tmp_path / "refinement.json"
    pointer = dfs.prepare_refinement(
        str(tmp_path / "grade"),
        str(policy),
        base_control_weight=1.0,
        base_guidance=2,
    )
    immutable = dict(pointer)
    immutable.pop("policy_sha256")
    immutable["settings"] = settings
    digest = dfs._payload_sha256(immutable)
    Path(str(immutable["history_uri"])).write_bytes(
        dfs._canonical_json_bytes(immutable)
    )
    Path(str(immutable["commit_uri"])).write_bytes(
        dfs._canonical_json_bytes(
            {
                "schema": "npa.data_factory.refinement.commit.v1",
                "attempt": immutable["attempt"],
                "history_uri": immutable["history_uri"],
                "policy_sha256": digest,
            }
        )
    )
    policy.write_bytes(
        dfs._canonical_json_bytes({**immutable, "policy_sha256": digest})
    )
    with pytest.raises(typer.BadParameter):
        cosmos2._load_refinement(str(policy))


def test_load_refinement_rejects_an_uncommitted_policy(tmp_path: Path) -> None:
    policy = tmp_path / "refinement.json"
    policy.write_text(
        json.dumps(
            {
                "schema": "npa.data_factory.refinement.v1",
                "attempt": 0,
                "settings": {"control_weight": 1.0, "guidance": 3},
            }
        )
    )

    with pytest.raises(typer.BadParameter, match="policy digest"):
        cosmos2._load_refinement(str(policy))


def test_run_cosmos_transfer_conditions_on_input(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    (repo / "examples").mkdir(parents=True)
    _fake_env(monkeypatch, repo)
    clip = tmp_path / "myinput.mp4"
    clip.write_bytes(b"x" * 1000)

    res = tx.run_cosmos_transfer(
        run_id="r1", input_video=str(clip), prompt="foggy morning", control="edge"
    )
    assert res["input_conditioned"] is True
    assert res["input_video"] == str(clip)
    assert res["control"] == "edge"
    assert Path(res["video_path"]).exists()
    # A conditioned spec was written that points at the input clip.
    assert res["spec"]
    # The synthesized controlnet spec is ephemeral (removed after inference to
    # avoid accumulating in the repo dir); its content is returned for inspection.
    assert not (repo / res["spec"]).exists()
    spec = res["spec_json"]
    assert spec["video_path"] == str(clip.resolve())
    assert "edge" in spec
    assert spec["prompt"] == "foggy morning"


def test_run_cosmos_transfer_requires_input_or_explicit_spec(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    (repo / "examples").mkdir(parents=True)
    _fake_env(monkeypatch, repo)
    monkeypatch.delenv("COSMOS_TRANSFER_SPEC", raising=False)
    monkeypatch.delenv("COSMOS_TRANSFER_PROMPT", raising=False)

    with pytest.raises(ValueError, match="no upstream media is bundled"):
        tx.run_cosmos_transfer(run_id="r2")
    assert not list(repo.glob("_npa_input_spec_*.json"))


def test_run_cosmos_transfer_refuses_missing_token_before_env_or_download(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(tx, "cosmos_transfer_repo", lambda: repo)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    called = False

    def fail_if_called(_repo: Path) -> Path:
        nonlocal called
        called = True
        raise AssertionError("environment/download path must not run")

    monkeypatch.setattr(tx, "ensure_env", fail_if_called)
    with pytest.raises(RuntimeError, match="no model download was attempted"):
        tx.run_cosmos_transfer(input_video=str(tmp_path / "missing.mp4"))
    assert called is False


def test_run_cosmos_transfer_accepts_small_guardrailed_video(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    (repo / "examples").mkdir(parents=True)
    monkeypatch.setattr(tx, "cosmos_transfer_repo", lambda: repo)
    monkeypatch.setattr(tx, "ensure_env", lambda _repo: Path("/usr/bin/python3"))
    monkeypatch.setattr(tx, "prepare_guardrail_nltk_data", lambda **_kwargs: 0)
    monkeypatch.setenv("HF_TOKEN", "unit-test-placeholder")

    def fake_run(cmd, *_args, **kwargs):
        out = cmd[cmd.index("-o") + 1]
        outdir = Path(kwargs["cwd"]) / out
        outdir.mkdir(parents=True)
        (outdir / "small.mp4").write_bytes(b"x" * 8_932)

    monkeypatch.setattr(tx.subprocess, "run", fake_run)

    result = tx.run_cosmos_transfer(run_id="small", spec="assets/custom.json")

    assert result["video_bytes"] == 8_932
    assert result["video_path"].endswith("small.mp4")


def test_run_cosmos_transfer_content_guardrail_opt_out_is_explicit(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    (repo / "examples").mkdir(parents=True)
    monkeypatch.setattr(tx, "cosmos_transfer_repo", lambda: repo)
    monkeypatch.setattr(tx, "ensure_env", lambda _repo: Path("/usr/bin/python3"))
    monkeypatch.setattr(tx, "prepare_guardrail_nltk_data", lambda **_kwargs: 0)
    monkeypatch.setenv("HF_TOKEN", "unit-test-placeholder")
    seen: list[list[str]] = []

    def fake_run(cmd, *_args, **kwargs):
        seen.append(cmd)
        outdir = Path(kwargs["cwd"]) / cmd[cmd.index("-o") + 1]
        outdir.mkdir(parents=True)
        (outdir / "result.mp4").write_bytes(b"x" * 8_932)

    monkeypatch.setattr(tx.subprocess, "run", fake_run)

    guarded = tx.run_cosmos_transfer(run_id="guarded", spec="assets/custom.json")
    assert "--disable-guardrails" not in seen[-1]
    assert guarded["content_guardrails_enabled"] is True

    monkeypatch.setenv(tx.DISABLE_CONTENT_GUARDRAILS_ENV, "1")
    opted_out = tx.run_cosmos_transfer(run_id="opted-out", spec="assets/custom.json")
    assert "--disable-guardrails" in seen[-1]
    assert opted_out["content_guardrails_enabled"] is False


def test_publish_marks_real_gpu_mode_and_conditioning(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "out.mp4"
    video.write_bytes(b"x" * 200_000)
    monkeypatch.setattr(tx, "extract_frames", lambda vp, dest, max_frames=8: [])

    recorded: dict[str, str] = {}

    class FakeStorage:
        def upload_file(self, local: str, uri: str) -> str:
            if uri.endswith("manifest.json"):
                recorded["manifest"] = Path(local).read_text()
            elif uri.endswith("metadata.json"):
                recorded["metadata"] = Path(local).read_text()
            return uri

    manifest = tx.publish_transfer_to_s3(
        {
            "video_path": str(video),
            "video_bytes": 200_000,
            "spec": "_npa_input_spec_r1.json",
            "input_conditioned": True,
            "input_video": "/tmp/robot_input.mp4",
            "conditioning_clip_uri": "s3://bkt/run1/input/conditioning.mp4",
            "control": "edge",
            "content_guardrails_enabled": False,
            "protected_chroma": {"mode": "source-chroma", "region_count": 2},
            "refinement": {"attempt": 1},
            "effective_control_weight": 1.5,
            "effective_guidance": 2.25,
        },
        "s3://bkt/run1/cosmos_augmented/",
        run_id="run1",
        variables={"weather": "rainy"},
        storage_client=FakeStorage(),
    )
    assert manifest["mode"] == "cosmos_transfer2.5_gpu"
    assert manifest["input_conditioned"] is True
    assert manifest["conditioned_input"] == "robot_input.mp4"
    assert manifest["conditioning_clip_uri"] == "s3://bkt/run1/input/conditioning.mp4"
    assert manifest["control"] == "edge"
    assert manifest["content_guardrails_enabled"] is False
    meta = json.loads(recorded["metadata"])
    assert meta["mode"] == "cosmos_transfer2.5_gpu"
    assert meta["input_conditioned"] is True
    assert meta["conditioned_input"] == "robot_input.mp4"
    assert meta["content_guardrails_enabled"] is False
    assert meta["conditioning_clip_uri"] == "s3://bkt/run1/input/conditioning.mp4"
    assert meta["protected_chroma"]["mode"] == "source-chroma"
    assert meta["refinement"]["attempt"] == 1
    assert meta["effective_control_weight"] == 1.5
    assert meta["effective_guidance"] == 2.25


def test_multi_variant_publish_writes_one_clip_per_combo(
    tmp_path: Path, monkeypatch
) -> None:
    """publish_transfer_clip (per combo) + write_run_manifest (once) must emit one
    clip dir per sampled scenario and a run manifest that records the fan-out."""
    monkeypatch.setattr(tx, "extract_frames", lambda vp, dest, max_frames=8: [])

    uploaded: list[str] = []

    class FakeStorage:
        def upload_file(self, local: str, uri: str) -> str:
            uploaded.append(uri)
            return uri

    storage = FakeStorage()
    combos = [
        {"cloth_color": "blue", "prompt": "a blue cloth, bright daylight"},
        {"cloth_color": "red", "prompt": "a red cloth, dim evening light"},
        {"cloth_color": "white", "prompt": "a white cloth, cool overhead light"},
    ]
    clips = []
    for i, combo in enumerate(combos):
        video = tmp_path / f"out{i}.mp4"
        video.write_bytes(b"x" * 200_000)
        clips.append(
            tx.publish_transfer_clip(
                {
                    "video_path": str(video),
                    "video_bytes": 200_000,
                    "spec": f"spec{i}",
                    "input_conditioned": True,
                    "input_video": "/tmp/robot.mp4",
                    "control": "edge",
                },
                "s3://bkt/run1/cosmos_augmented/",
                run_id="run1",
                clip_name=f"aug-run1-{i}",
                variables=combo,
                storage_client=storage,
            )
        )
    manifest = tx.write_run_manifest(
        clips, "s3://bkt/run1/cosmos_augmented/", run_id="run1", storage_client=storage
    )

    assert manifest["variant_count"] == 3
    assert manifest["multiply_mode"] == "multi-variant"
    assert manifest["clips"] == ["aug-run1-0", "aug-run1-1", "aug-run1-2"]
    assert len(manifest["variants"]) == 3
    assert manifest["variants"][1]["prompt"] == "a red cloth, dim evening light"
    # One metadata.json + one augmented_video.mp4 per clip, plus one run manifest.
    assert sum(u.endswith("metadata.json") for u in uploaded) == 3
    assert sum(u.endswith("augmented_video.mp4") for u in uploaded) == 3
    assert sum(u.endswith("cosmos_augmented/manifest.json") for u in uploaded) == 1


def test_single_variant_manifest_reports_single_mode(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(tx, "extract_frames", lambda vp, dest, max_frames=8: [])
    video = tmp_path / "out.mp4"
    video.write_bytes(b"x" * 200_000)

    class FakeStorage:
        def upload_file(self, local: str, uri: str) -> str:
            return uri

    manifest = tx.publish_transfer_to_s3(
        {"video_path": str(video), "video_bytes": 200_000, "spec": "s"},
        "s3://bkt/run1/cosmos_augmented/",
        run_id="run1",
        variables={"cloth_color": "blue"},
        storage_client=FakeStorage(),
    )
    assert manifest["variant_count"] == 1
    assert manifest["multiply_mode"] == "single-variant"
    assert manifest["clips"] == ["aug-run1"]


def test_materialize_input_clip_local_path(tmp_path: Path) -> None:
    from npa.cli.workbench.cosmos2 import _materialize_input_clip

    clip = tmp_path / "local.mp4"
    clip.write_bytes(b"x" * 10)
    assert _materialize_input_clip(str(clip)) == str(clip)
    assert _materialize_input_clip("") == ""
    assert _materialize_input_clip(str(tmp_path / "missing.mp4")) == ""


def test_materialize_input_clip_empty_s3_prefix_is_no_video(monkeypatch) -> None:
    from npa.clients.storage import StorageClient

    materialized_dirs: list[Path] = []

    class EmptyPrefixStorage:
        def download_directory(self, _src: str, local_dir: str) -> str:
            materialized_dirs.append(Path(local_dir))
            assert materialized_dirs[-1].is_dir()
            return local_dir

    monkeypatch.setattr(StorageClient, "from_environment", lambda: EmptyPrefixStorage())

    assert cosmos2._materialize_input_clip("s3://test-bucket/empty/") == ""
    assert materialized_dirs and not materialized_dirs[0].exists()


def test_materialize_paidf_prefix_prefers_prepared_conditioning_clip(
    monkeypatch,
) -> None:
    from npa.clients.storage import StorageClient

    class PreparedPrefixStorage:
        def download_directory(self, _src: str, local_dir: str) -> str:
            root = Path(local_dir)
            (root / "source.mp4").write_bytes(b"source")
            (root / "conditioning.mp4").write_bytes(b"conditioning")
            return local_dir

    monkeypatch.setattr(
        StorageClient, "from_environment", lambda: PreparedPrefixStorage()
    )

    clip = cosmos2._materialize_input_clip("s3://test-bucket/prepared/")

    assert Path(clip).name == "conditioning.mp4"
    assert Path(clip).read_bytes() == b"conditioning"


def test_materialize_paidf_frames_as_conditioning_clip(
    tmp_path: Path, monkeypatch
) -> None:
    from npa.clients.storage import StorageClient

    materialized_dirs: list[Path] = []
    ffmpeg_commands: list[list[str]] = []

    class FramePrefixStorage:
        def download_directory(self, _src: str, local_dir: str) -> str:
            root = Path(local_dir)
            materialized_dirs.append(root)
            (root / "frame_0000.png").write_bytes(b"png")
            nested = root / "nested"
            nested.mkdir()
            (nested / "frame_0001.jpg").write_bytes(b"jpg")
            (root / "ignore.txt").write_text("not an image", encoding="utf-8")
            return local_dir

    def fake_ffmpeg(command: list[str], *, check: bool) -> None:
        assert check is True
        ffmpeg_commands.append(command)
        Path(command[-1]).write_bytes(b"mp4")

    monkeypatch.setattr(StorageClient, "from_environment", lambda: FramePrefixStorage())
    # The helper imports subprocess locally, so patch the shared module object.
    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_ffmpeg)
    clip = cosmos2._materialize_input_clip(
        "s3://test-bucket/seeded/", allow_frame_sequence=True
    )

    assert clip.endswith("npa-paidf-conditioning.mp4")
    assert Path(clip).read_bytes() == b"mp4"
    assert materialized_dirs[0].exists()  # retained for inference
    assert ffmpeg_commands and ffmpeg_commands[0][0] == "ffmpeg"
    assert ffmpeg_commands[0][ffmpeg_commands[0].index("-frames:v") + 1] == "93"
    concat = Path(ffmpeg_commands[0][ffmpeg_commands[0].index("-i") + 1]).read_text()
    assert "frame-00000.png" in concat
    assert "frame-00001.jpg" in concat
    assert "ignore.txt" not in concat


def test_generated_conditioning_clip_is_persisted_for_evaluator(
    tmp_path: Path, monkeypatch
) -> None:
    from npa.clients.storage import StorageClient

    clip = tmp_path / "npa-paidf-conditioning.mp4"
    clip.write_bytes(b"conditioning")
    uploads: list[tuple[str, str]] = []

    class Storage:
        def upload_file(self, local: str, uri: str) -> str:
            uploads.append((local, uri))
            return uri

    monkeypatch.setattr(StorageClient, "from_environment", lambda: Storage())

    uri = cosmos2._persist_generated_conditioning_clip(
        str(clip), "s3://bucket/physical-ai-data-factory/run/input/"
    )

    assert uri == "s3://bucket/physical-ai-data-factory/run/input/conditioning.mp4"
    assert uploads == [(str(clip), uri)]
    assert (
        cosmos2._persist_generated_conditioning_clip(
            str(tmp_path / "user-video.mp4"), "s3://bucket/run/input/"
        )
        == ""
    )


def test_prepared_conditioning_clip_resolves_to_canonical_uri_without_reupload(
    tmp_path: Path, monkeypatch
) -> None:
    from npa.clients.storage import StorageClient

    clip = tmp_path / "conditioning.mp4"
    clip.write_bytes(b"prepared")
    monkeypatch.setattr(
        StorageClient,
        "from_environment",
        lambda: (_ for _ in ()).throw(AssertionError("must not upload")),
    )

    assert (
        cosmos2._persist_generated_conditioning_clip(
            str(clip), "s3://bucket/physical-ai-data-factory/run/input/"
        )
        == "s3://bucket/physical-ai-data-factory/run/input/conditioning.mp4"
    )


def test_materialize_standalone_does_not_convert_frame_prefix(monkeypatch) -> None:
    from npa.clients.storage import StorageClient

    materialized_dirs: list[Path] = []

    class FramePrefixStorage:
        def download_directory(self, _src: str, local_dir: str) -> str:
            root = Path(local_dir)
            materialized_dirs.append(root)
            (root / "frame.png").write_bytes(b"png")
            return local_dir

    monkeypatch.setattr(StorageClient, "from_environment", lambda: FramePrefixStorage())

    assert cosmos2._materialize_input_clip("s3://test-bucket/seeded/") == ""
    assert materialized_dirs and not materialized_dirs[0].exists()


@pytest.mark.parametrize("stage", ["authentication", "listing", "download"])
def test_materialize_input_clip_propagates_storage_failures(
    monkeypatch, stage: str
) -> None:
    from npa.clients.storage import StorageClient

    failure = PermissionError(f"{stage} failed: credential=must-not-surface")

    class FailingStorage:
        def download_directory(self, _src: str, _local_dir: str) -> str:
            raise failure

        def download_path(self, _src: str, _local_path: str) -> str:
            raise failure

    def from_environment():
        if stage == "authentication":
            raise failure
        return FailingStorage()

    monkeypatch.setattr(StorageClient, "from_environment", from_environment)
    source = (
        "s3://test-bucket/input.mp4"
        if stage == "download"
        else "s3://test-bucket/input/"
    )

    with pytest.raises(PermissionError) as caught:
        cosmos2._materialize_input_clip(source)
    assert caught.value is failure


def test_cli_materialization_error_is_sanitized_and_preserves_cause(
    monkeypatch,
) -> None:
    secret = "do-not-print-this-token"
    failure = PermissionError(f"access denied: token={secret}")
    source = f"s3://test-bucket/input/?token={secret}"

    def fail_materialize(_src: str) -> str:
        raise failure

    monkeypatch.setattr(cosmos2, "_materialize_input_clip", fail_materialize)

    with pytest.raises(cosmos2.typer.BadParameter) as caught:
        cosmos2._materialize_conditioning_input(source)
    assert isinstance(caught.value, cosmos2.typer.BadParameter)
    assert caught.value.__cause__ is failure
    message = str(caught.value)
    assert "could not inspect or download" in message
    assert "credentials" in message
    assert secret not in message


def test_transfer_cli_rejects_conditioning_without_input_video(monkeypatch) -> None:
    input_uri = "s3://test-bucket/run/input/"
    materialized: list[str] = []
    inference_called = False

    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: True)

    def fake_materialize(src: str) -> str:
        materialized.append(src)
        return ""

    def fail_if_inferred(**_kwargs) -> dict:
        nonlocal inference_called
        inference_called = True
        raise AssertionError("inference must not run without an input video")

    monkeypatch.setattr(cosmos2, "_materialize_input_clip", fake_materialize)
    monkeypatch.setattr(tx, "run_cosmos_transfer", fail_if_inferred)

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            input_uri,
            "--output-uri",
            "s3://test-bucket/run/augmented/",
            "--condition-on-input",
            "--execute",
        ],
    )

    assert result.exit_code != 0
    assert "no supported video" in result.output
    assert input_uri not in result.output
    assert materialized == [input_uri]
    assert inference_called is False


def test_transfer_cli_reports_storage_failure_without_secret(monkeypatch) -> None:
    secret = "do-not-print-this-token"
    input_uri = f"s3://test-bucket/run/input/?token={secret}"
    failure = PermissionError(f"access denied: credential={secret}")
    inference_called = False

    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: True)

    def fail_materialize(_src: str) -> str:
        raise failure

    def fail_if_inferred(**_kwargs) -> dict:
        nonlocal inference_called
        inference_called = True
        raise AssertionError("inference must not run after an input-storage failure")

    monkeypatch.setattr(cosmos2, "_materialize_input_clip", fail_materialize)
    monkeypatch.setattr(tx, "run_cosmos_transfer", fail_if_inferred)

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            input_uri,
            "--output-uri",
            "s3://test-bucket/run/augmented/",
            "--condition-on-input",
            "--execute",
        ],
    )

    assert result.exit_code != 0
    assert "could not inspect or download" in result.output
    assert "no supported video" not in result.output
    assert secret not in result.output
    assert inference_called is False


def test_augmentation_manifest_read_failure_is_sanitized_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "private-object-name"

    def fail_read(_uri: str) -> dict:
        raise PermissionError(f"denied object={secret}")

    monkeypatch.setattr(
        "npa.workflows.data_factory_stages._download_json", fail_read
    )

    with pytest.raises(typer.BadParameter, match="augmentation manifest") as exc:
        cosmos2._all_augmentations("s3://redacted/configs/")

    assert secret not in str(exc.value)


def test_augmentation_manifest_requires_real_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "npa.workflows.data_factory_stages._download_json",
        lambda _uri: {"augmentations": []},
    )

    with pytest.raises(typer.BadParameter, match="no augmentation variants"):
        cosmos2._all_augmentations("s3://redacted/configs/")


def test_paidf_transfer_invokes_optional_sam2_once_and_reuses_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npa.cli.workbench import cosmos2
    from npa.workbench.cosmos import sam2_masks

    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    generated = tmp_path / "generated.mp4"
    generated.write_bytes(b"generated")
    calls: dict[str, object] = {"generate": 0, "protected": []}

    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: True)
    monkeypatch.setattr(
        cosmos2, "_materialize_conditioning_input", lambda *_args, **_kwargs: str(source)
    )
    monkeypatch.setattr(
        cosmos2,
        "_all_augmentations",
        lambda _uri: [{"prompt": "generic-a"}, {"prompt": "generic-b"}],
    )
    monkeypatch.setattr(
        cosmos2,
        "_persist_generated_conditioning_clip",
        lambda *_args, **_kwargs: "s3://b/input.mp4",
    )

    def fake_generate(_video, output_dir, *, config, run_id):  # noqa: ANN001
        calls["generate"] = int(calls["generate"]) + 1
        assert run_id == "run"
        masks = Path(output_dir) / "masks"
        masks.mkdir()
        (masks / "mask-000000.png").write_bytes(b"mask")
        manifest = Path(output_dir) / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        return sam2_masks.Sam2MaskResult({}, manifest, masks)

    monkeypatch.setattr(sam2_masks, "generate_sam2_video_masks", fake_generate)
    monkeypatch.setattr(
        sam2_masks,
        "load_published_sam2_masks",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        lambda: object(),
    )
    monkeypatch.setattr(
        sam2_masks,
        "publish_sam2_masks",
        lambda *_args, **_kwargs: {
            "engine": sam2_masks.SAM2_ENGINE,
            "component": "Meta Segment Anything Model 2",
            "component_version": "1.0",
            "component_source": "https://github.com/facebookresearch/sam2",
            "component_revision": sam2_masks.SAM2_SOURCE_REVISION,
            "license": {"spdx": "Apache-2.0"},
            "config": {"mode": "sam2-auto"},
            "frame_count": 1,
            "object_count": 1,
            "mask_coverage": {"mean": 0.25},
            "runtime": {"device": "cuda", "seconds": 1.0},
            "manifest_uri": "s3://b/run/segmentation/manifest.json",
            "masks_uri": "s3://b/run/segmentation/masks/",
        },
    )
    monkeypatch.setattr(
        tx,
        "run_cosmos_transfer",
        lambda **_kwargs: {
            "video_path": str(generated),
            "video_bytes": generated.stat().st_size,
            "spec": "conditioned.json",
        },
    )

    def fake_protect(result, **kwargs):  # noqa: ANN001
        calls["protected"].append(kwargs)
        return result

    monkeypatch.setattr(tx, "preserve_source_chroma", fake_protect)
    monkeypatch.setattr(
        tx,
        "publish_transfer_clip",
        lambda *_args, **_kwargs: {"name": "aug-run-0", "frame_count": 1},
    )
    monkeypatch.setattr(
        tx,
        "write_run_manifest",
        lambda *_args, **_kwargs: {
            "augmented_video_uri": "s3://b/run/augment/video.mp4",
            "augmented_videos": ["s3://b/run/augment/video.mp4"],
            "frame_count": 1,
            "variant_count": 1,
            "multiply_mode": "single-variant",
            "variant_parallelism": 1,
            "clips": ["aug-run-0"],
            "conditioning_clip_uri": "s3://b/input.mp4",
            "control_spec": "conditioned.json",
            "control": "edge",
            "control_weight": 1.0,
            "control_prompt": "",
            "mask_prompt": "",
            "control_uris": {},
        },
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            "s3://b/run/input/",
            "--output-uri",
            "s3://b/run/augment/",
            "--configs-uri",
            "s3://b/run/configs/",
            "--run-id",
            "run",
            "--execute",
            "--condition-on-input",
            "--segmentation-mode",
            "sam2-auto",
            "--segmentation-uri",
            "s3://b/run/segmentation/",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert calls["generate"] == 1
    assert len(calls["protected"]) == 2
    assert calls["protected"][0]["masks_dir"]
    assert payload["segmentation"]["engine"] == sam2_masks.SAM2_ENGINE
    assert "prompt" not in payload
    assert "manifest_uri" not in json.dumps(payload)


def test_paidf_transfer_rejects_invalid_or_conflicting_sam2_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npa.cli.workbench import cosmos2

    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: True)
    monkeypatch.setattr(
        cosmos2, "_materialize_conditioning_input", lambda *_args, **_kwargs: str(source)
    )

    invalid = runner.invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            "s3://b/input/",
            "--output-uri",
            "s3://b/output/",
            "--configs-uri",
            "s3://b/configs/",
            "--run-id",
            "run",
            "--execute",
            "--condition-on-input",
            "--segmentation-mode",
            "sam2-auto",
            "--segmentation-uri",
            "s3://b/segmentation/",
            "--sam2-points-per-side",
            "2",
        ],
    )
    assert invalid.exit_code != 0
    assert "points_per_side" in invalid.output

    conflict = runner.invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            "s3://b/input/",
            "--output-uri",
            "s3://b/output/",
            "--configs-uri",
            "s3://b/configs/",
            "--run-id",
            "run",
            "--execute",
            "--condition-on-input",
            "--segmentation-mode",
            "sam2-auto",
            "--segmentation-uri",
            "s3://b/segmentation/",
            "--protected-regions-json",
            "[[0.1,0.1,0.2,0.2]]",
        ],
    )
    assert conflict.exit_code != 0
    assert "mutually" in conflict.output and "exclusive" in conflict.output


def test_detect_gpu_count_from_cuda_visible_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    assert cosmos2._detect_gpu_count() == 4
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # Falls back to nvidia-smi (absent in CI) -> at least 1.
    assert cosmos2._detect_gpu_count() >= 1


def test_variant_parallelism_env_override_and_cap(monkeypatch) -> None:
    monkeypatch.setenv("NPA_COSMOS_VARIANT_PARALLELISM", "4")
    # Capped at the number of variants so we never spawn idle workers.
    assert cosmos2._variant_parallelism(2) == 2
    assert cosmos2._variant_parallelism(8) == 4
    monkeypatch.delenv("NPA_COSMOS_VARIANT_PARALLELISM", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    assert cosmos2._variant_parallelism(6) == 4
    assert cosmos2._variant_parallelism(1) == 1


def test_candidate_inference_seed_is_optional_and_validated() -> None:
    assert cosmos2._inference_seed({}) is None
    assert cosmos2._inference_seed({"inference_seed": 123}) == 123
    with pytest.raises(typer.BadParameter, match="inference_seed"):
        cosmos2._inference_seed({"inference_seed": "not-a-number"})


def test_run_cosmos_transfer_pins_gpu_and_unique_spec(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    (repo / "examples").mkdir(parents=True)
    # A default spec the prompt override copies from.
    spec_path = repo / "spec.json"
    spec_path.write_text(
        json.dumps({"prompt": "orig", "video_path": "x"}), encoding="utf-8"
    )
    _fake_env(monkeypatch, repo)

    seen_env: dict[str, str] = {}

    def fake_run(cmd, *args, **kwargs):  # noqa: ANN001
        cwd = Path(kwargs["cwd"])
        out = cmd[cmd.index("-o") + 1]
        outdir = cwd / out
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "result.mp4").write_bytes(b"y" * 200_001)
        seen_env["CUDA_VISIBLE_DEVICES"] = kwargs["env"].get("CUDA_VISIBLE_DEVICES", "")
        seen_env["NLTK_DATA"] = kwargs["env"].get("NLTK_DATA", "")
        return None

    monkeypatch.setattr(tx.subprocess, "run", fake_run)

    res = tx.run_cosmos_transfer(
        run_id="run1-v2",
        spec="spec.json",
        prompt="a red cloth",
        cuda_visible_devices="2",
        variant_tag="run1-v2",
        hf_home=str(tmp_path / "model-cache"),
    )
    assert seen_env["CUDA_VISIBLE_DEVICES"] == "2"
    assert seen_env["NLTK_DATA"] == str(
        tmp_path
        / "model-cache"
        / tx.GUARDRAIL_NLTK_MATERIALIZED_DIR
        / tx.GUARDRAIL_REVISION
    )
    assert "/hub/" not in seen_env["NLTK_DATA"]
    # Variant-tagged patched spec keeps concurrent siblings from clobbering.
    assert "run1-v2" in res["spec"]
    assert Path(res["video_path"]).exists()


def test_write_run_manifest_records_variant_parallelism(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(tx, "extract_frames", lambda vp, dest, max_frames=8: [])

    class FakeStorage:
        def upload_file(self, local: str, uri: str) -> str:
            return uri

    storage = FakeStorage()
    clips = []
    for i in range(4):
        video = tmp_path / f"out{i}.mp4"
        video.write_bytes(b"x" * 200_000)
        clips.append(
            tx.publish_transfer_clip(
                {"video_path": str(video), "video_bytes": 200_000, "spec": f"spec{i}"},
                "s3://bkt/run1/cosmos_augmented/",
                run_id="run1",
                clip_name=f"aug-run1-{i}",
                variables={"prompt": f"scene {i}"},
                storage_client=storage,
            )
        )
    manifest = tx.write_run_manifest(
        clips,
        "s3://bkt/run1/cosmos_augmented/",
        run_id="run1",
        storage_client=storage,
        variant_parallelism=4,
    )
    assert manifest["variant_count"] == 4
    assert manifest["variant_parallelism"] == 4
    assert manifest["multiply_mode"] == "multi-variant"


def test_generic_publisher_writes_flat_frames_and_durable_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "out.mp4"
    video.write_bytes(b"video" * 50_000)

    def fake_extract(_video: str, dest: Path, *, max_frames: int = 8) -> list[Path]:
        dest.mkdir(parents=True, exist_ok=True)
        frames = [dest / f"frame-{index:05d}.png" for index in range(2)]
        for frame in frames:
            frame.write_bytes(b"png")
        return frames

    monkeypatch.setattr(tx, "extract_frames", fake_extract)
    uploads: dict[str, bytes] = {}

    class FakeStorage:
        def upload_file(self, local: str, uri: str) -> str:
            uploads[uri] = Path(local).read_bytes()
            return uri

    manifest = tx.publish_transfer_to_s3(
        {
            "video_path": str(video),
            "video_bytes": video.stat().st_size,
            "spec": "conditioned.json",
            "input_conditioned": True,
            "input_video": "/tmp/input.mp4",
            "control": "edge",
        },
        "s3://bucket/run/augment/",
        run_id="run",
        frames_output_uri="s3://bucket/run/augment/",
        require_frames=True,
        storage_client=FakeStorage(),
    )

    assert manifest["schema"] == tx.TRANSFER_MANIFEST_SCHEMA
    assert manifest["augmented_frames_uri"] == "s3://bucket/run/augment/"
    assert [frame["uri"] for frame in manifest["frames"]] == [
        "s3://bucket/run/augment/frame-00000.png",
        "s3://bucket/run/augment/frame-00001.png",
    ]
    assert tx.transfer_manifest_uri_for("s3://bucket/run/augment/") in uploads
    assert all(uri in uploads for uri in (frame["uri"] for frame in manifest["frames"]))


def test_generic_publisher_fails_if_frame_extraction_produces_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "out.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(tx, "extract_frames", lambda *_args, **_kwargs: [])

    class FakeStorage:
        def upload_file(self, _local: str, uri: str) -> str:
            return uri

    with pytest.raises(RuntimeError, match="no frames could be extracted"):
        tx.publish_transfer_to_s3(
            {"video_path": str(video), "video_bytes": 5, "spec": "spec.json"},
            "s3://bucket/run/augment/",
            frames_output_uri="s3://bucket/run/augment/",
            require_frames=True,
            storage_client=FakeStorage(),
        )


def test_frame_extraction_preserves_pyav_subprocess_failure(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "malformed.mp4"
    video.write_bytes(b"not-a-video")
    cause = tx.subprocess.CalledProcessError(
        1,
        ["python", "-c", "decode"],
        stderr="av.error.InvalidDataError: invalid data found when processing input",
    )

    def fail_decode(*_args, **_kwargs):
        raise cause

    monkeypatch.setattr(tx.subprocess, "run", fail_decode)

    with pytest.raises(
        tx.FrameExtractionError, match="InvalidDataError: invalid data"
    ) as raised:
        tx.extract_frames(str(video), tmp_path / "frames")

    assert raised.value.__cause__ is cause


def test_frame_extraction_subprocess_failure_raises_public_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "transfer-repo"
    _write_extractor_python(
        repo, "printf '%s\\n' 'av.error.InvalidDataError: broken stream' >&2; exit 23"
    )
    monkeypatch.setenv("COSMOS_TRANSFER_REPO", str(repo))

    with pytest.raises(tx.FrameExtractionError, match="exit code 23.*broken stream"):
        tx.extract_frames(str(tmp_path / "broken.mp4"), tmp_path / "frames")


def test_frame_extraction_os_failure_raises_public_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "missing-runtime"
    repo.mkdir()
    monkeypatch.setenv("COSMOS_TRANSFER_REPO", str(repo))

    with pytest.raises(
        tx.FrameExtractionError, match="could not start frame extraction"
    ):
        tx.extract_frames(str(tmp_path / "input.mp4"), tmp_path / "frames")


def test_successful_zero_frame_decode_is_not_an_extraction_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "transfer-repo"
    _write_extractor_python(repo, "exit 0")
    monkeypatch.setenv("COSMOS_TRANSFER_REPO", str(repo))

    assert tx.extract_frames(str(tmp_path / "empty.mp4"), tmp_path / "frames") == []


def test_publish_transfer_clip_requires_frames_before_any_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "transfer-repo"
    _write_extractor_python(repo, "exit 0")
    monkeypatch.setenv("COSMOS_TRANSFER_REPO", str(repo))
    video = tmp_path / "empty.mp4"
    video.write_bytes(b"video-with-no-decodable-frames")
    uploads: list[str] = []

    class RecordingStorage:
        def upload_file(self, _local: str, uri: str) -> str:
            uploads.append(uri)
            return uri

    with pytest.raises(RuntimeError, match="no frames could be extracted"):
        tx.publish_transfer_clip(
            {"video_path": str(video), "video_bytes": video.stat().st_size},
            "s3://bucket/run/augment/",
            require_frames=True,
            storage_client=RecordingStorage(),
        )

    assert uploads == []


def test_conditioned_execute_fails_closed_when_input_video_is_missing(
    monkeypatch,
) -> None:
    from typer.testing import CliRunner

    from npa.cli.main import app
    from npa.cli.workbench import cosmos2

    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: True)
    monkeypatch.setattr(cosmos2, "_materialize_input_clip", lambda _uri: "")
    monkeypatch.setattr(
        tx,
        "run_cosmos_transfer",
        lambda **_kwargs: pytest.fail(
            "inference must not run without conditioned input"
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            "s3://bucket/input/",
            "--output-uri",
            "s3://bucket/augment/",
            "--execute",
            "--condition-on-input",
        ],
    )

    assert result.exit_code != 0
    assert "no supported video" in result.output


def test_execute_fails_closed_when_vendor_runtime_is_unavailable(monkeypatch) -> None:
    from typer.testing import CliRunner

    from npa.cli.main import app

    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: False)
    monkeypatch.setattr(
        tx,
        "reference_augment_frames",
        lambda *_args, **_kwargs: pytest.fail(
            "--execute must not use the reference fallback"
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            "s3://bucket/input/",
            "--output-uri",
            "s3://bucket/augment/",
            "--execute",
            "--condition-on-input",
        ],
    )

    assert result.exit_code != 0
    assert "needs the cosmos-transfer2.5 runtime" in result.output


def test_conditioned_execute_uses_input_and_shared_generic_publisher(
    monkeypatch,
) -> None:
    from typer.testing import CliRunner

    from npa.cli.main import app
    from npa.cli.workbench import cosmos2

    seen: dict[str, object] = {}
    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: True)
    monkeypatch.setattr(
        cosmos2, "_materialize_input_clip", lambda _uri: "/tmp/input.mp4"
    )

    def fake_run(**kwargs):
        seen["input_video"] = kwargs.get("input_video")
        return {
            "video_path": "/tmp/generated.mp4",
            "video_bytes": 1234,
            "spec": "conditioned.json",
            "input_conditioned": True,
            "input_video": "/tmp/input.mp4",
            "control": "edge",
        }

    def fake_publish(transfer, output_uri, **kwargs):
        seen["published_transfer"] = transfer
        seen["publish_output_uri"] = output_uri
        seen["publish_kwargs"] = kwargs
        return {
            "augmented_video_uri": "s3://bucket/augment/aug-run/augmented_video.mp4",
            "augmented_frames_uri": "s3://bucket/augment/",
            "frame_count": 8,
        }

    monkeypatch.setattr(tx, "run_cosmos_transfer", fake_run)
    monkeypatch.setattr(tx, "publish_transfer_to_s3", fake_publish)

    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            "s3://bucket/input/",
            "--output-uri",
            "s3://bucket/augment/",
            "--run-id",
            "run",
            "--execute",
            "--condition-on-input",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert seen["input_video"] == "/tmp/input.mp4"
    assert seen["publish_output_uri"] == "s3://bucket/augment/"
    publish_kwargs = seen["publish_kwargs"]
    assert publish_kwargs["frames_output_uri"] == "s3://bucket/augment/"
    assert publish_kwargs["require_frames"] is True
    assert payload["input_conditioned"] is True
    assert payload["augmented_frames_uri"] == "s3://bucket/augment/"
    assert payload["manifest_uri"] == "s3://bucket/augment/manifest.json"


def test_sim2real_engine_real_manifest_uses_gpu_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.clients.storage import StorageClient
    from npa.workflows import sim2real_envgen
    from npa.workflows.sim2real import engine

    uploads: dict[str, bytes] = {}

    class FakeStorage:
        def upload_file(self, local: str, uri: str) -> str:
            uploads[uri] = Path(local).read_bytes()
            return uri

    monkeypatch.setattr(
        StorageClient,
        "from_environment",
        staticmethod(FakeStorage),
    )
    monkeypatch.setattr(
        engine,
        "_run_real_cosmos_transfer",
        lambda *_args: {
            "augmented_video_uri": "s3://bucket/run/video/augmented.mp4",
            "frame_count": 1,
            "frames": [
                {
                    "frame_id": "frame-00000",
                    "uri": "s3://bucket/run/frames/frame-00000.png",
                }
            ],
            "video_bytes": 1234,
            "spec": "spec.json",
        },
    )

    result = engine.run_cosmos2_transfer_component_from_s3(
        input_uri="s3://bucket/input/",
        output_uri="s3://bucket/run/result.json",
        augmented_frames_uri="s3://bucket/run/frames/",
        run_id="mode-test",
    )

    durable = json.loads(uploads["s3://bucket/run/manifest.json"])
    assert result["manifest"]["mode"] == durable["mode"] == "cosmos_transfer2.5_gpu"
    assert durable["frames"] == [
        {
            "frame_id": "frame-00000",
            "uri": "s3://bucket/run/frames/frame-00000.png",
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(uploads["s3://bucket/run/manifest.json"])
    assert sim2real_envgen.frame_uris_from_transfer_manifest(str(manifest_path)) == (
        "s3://bucket/run/frames/frame-00000.png",
    )


def test_sim2real_real_transfer_without_exact_frames_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.clients.storage import StorageClient
    from npa.workflows.sim2real import engine

    class FakeStorage:
        def upload_file(self, _local: str, uri: str) -> str:
            return uri

    monkeypatch.setattr(
        StorageClient,
        "from_environment",
        staticmethod(FakeStorage),
    )
    monkeypatch.setattr(
        engine,
        "_run_real_cosmos_transfer",
        lambda *_args: {
            "augmented_video_uri": "s3://bucket/run/video/augmented.mp4",
            "frame_count": 1,
            "video_bytes": 1234,
            "spec": "spec.json",
        },
    )

    with pytest.raises(engine.Sim2RealLoopError, match="non-empty exact frames"):
        engine.run_cosmos2_transfer_component_from_s3(
            input_uri="s3://bucket/input/",
            output_uri="s3://bucket/run/result.json",
            augmented_frames_uri="s3://bucket/run/frames/",
            run_id="missing-lineage",
        )


def test_sim2real_engine_real_frame_index_uses_gpu_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npa.workbench.cosmos import fixture as cosmos_fixture
    from npa.workflows.sim2real import engine

    video = tmp_path / "output.mp4"
    video.write_bytes(b"video")
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"frame")
    uploads: dict[str, bytes] = {}

    class FakeStorage:
        def upload_file(self, local: str, uri: str) -> str:
            uploads[uri] = Path(local).read_bytes()
            return uri

    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: True)
    monkeypatch.setenv("NPA_SIM2REAL_TRANSFER_SPEC", "unit-test-spec.json")
    monkeypatch.setattr(
        cosmos_fixture,
        "generate_fixture",
        lambda *_args, **_kwargs: pytest.fail(
            "an explicit transfer spec must bypass procedural fixture generation"
        ),
    )
    monkeypatch.setattr(
        tx,
        "run_cosmos_transfer",
        lambda **_kwargs: {
            "video_path": str(video),
            "video_bytes": video.stat().st_size,
            "spec": "spec.json",
        },
    )
    monkeypatch.setattr(
        tx,
        "extract_frames",
        lambda *_args, **_kwargs: [frame],
    )

    result = engine._run_real_cosmos_transfer(
        FakeStorage(),
        "s3://bucket/input/",
        "s3://bucket/run/",
        "s3://bucket/run/frames/",
        "mode-test",
    )

    assert result is not None
    index = json.loads(uploads["s3://bucket/run/frames/index.json"])
    assert index["mode"] == "cosmos_transfer2.5_gpu"
    assert result["frames"] == index["frames"]


def test_sim2real_engine_frame_extraction_failure_uses_descriptor_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from npa.workflows.sim2real import engine

    video = tmp_path / "output.mp4"
    video.write_bytes(b"video")
    uploads: list[str] = []

    class RecordingStorage:
        def upload_file(self, _local: str, uri: str) -> str:
            uploads.append(uri)
            return uri

    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: True)
    monkeypatch.setenv("NPA_SIM2REAL_TRANSFER_SPEC", "unit-test-spec.json")
    monkeypatch.setattr(
        tx,
        "run_cosmos_transfer",
        lambda **_kwargs: {
            "video_path": str(video),
            "video_bytes": video.stat().st_size,
            "spec": "spec.json",
        },
    )

    def fail_extraction(*_args, **_kwargs):
        raise tx.FrameExtractionError("decoder subprocess failed")

    monkeypatch.setattr(tx, "extract_frames", fail_extraction)

    result = engine._run_real_cosmos_transfer(
        RecordingStorage(),
        "s3://bucket/input/",
        "s3://bucket/run/",
        "s3://bucket/run/frames/",
        "fallback-test",
    )

    assert result is None
    assert uploads == []
    assert "frame_extraction_failed_fallback" in capsys.readouterr().err


def test_sim2real_engine_does_not_swallow_unrelated_extraction_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npa.workflows.sim2real import engine

    video = tmp_path / "output.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: True)
    monkeypatch.setenv("NPA_SIM2REAL_TRANSFER_SPEC", "unit-test-spec.json")
    monkeypatch.setattr(
        tx,
        "run_cosmos_transfer",
        lambda **_kwargs: {
            "video_path": str(video),
            "video_bytes": video.stat().st_size,
            "spec": "spec.json",
        },
    )

    def programmer_error(*_args, **_kwargs):
        raise AssertionError("unrelated bug")

    monkeypatch.setattr(tx, "extract_frames", programmer_error)

    with pytest.raises(AssertionError, match="unrelated bug"):
        engine._run_real_cosmos_transfer(
            object(),
            "s3://bucket/input/",
            "s3://bucket/run/",
            "s3://bucket/run/frames/",
            "error-test",
        )
