"""Unit tests for Cosmos Curator model resolution and the runtime weight fetch.

No weights are downloaded here: the registry is a fixture shaped like upstream's
``all_models.json`` and the downloader is injected, so these cover which models a
capability asks for, where they land, and the credential contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from npa.workbench.cosmos_curate import models as mod
from npa.workbench.cosmos_curate.upstream import CosmosCurateError

# Shaped exactly like the entries in upstream's cosmos_curator/configs/all_models.json.
REGISTRY = {
    "transnetv2": {
        "model_id": "Sn4kehead/TransNetV2",
        "version": "db6ceabeb692ec71ecc6beb6d00db67ad1412d7f",
        "filelist": None,
    },
    "internvideo2_mm": {
        "model_id": "OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
        "version": "4362e1f88a992e7edbfd7696f7f78b7f79426dfd",
        "filelist": ["InternVideo2-stage2_1b-224p-f4.pt"],
    },
    "bert": {
        "model_id": "google-bert/bert-large-uncased",
        "version": "6da4b6a26a1877e173fca3225479512db81a5e5b",
        "filelist": ["config.json", "model.safetensors"],
    },
    "qwen2.5_vl": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "version": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "filelist": None,
    },
    "clip_vit": {
        "model_id": "openai/clip-vit-large-patch14",
        "version": "32bd64288804d66eefd0ccbe215aa642df71cc41",
        "filelist": ["config.json", "model.safetensors"],
    },
    "aesthetic_scorer": {
        "model_id": "ttj/sac-logos-ava1-l14-linearMSE",
        "version": "1e77fa05081323d99725fc40a9bf9f88180490e7",
        "filelist": ["model.safetensors"],
    },
    "cosmos_embed1_336p": {
        "model_id": "nvidia/Cosmos-Embed1-336p",
        "version": "5d8309dd5c9ec8f856b16d589693004c907c9a57",
        "filelist": None,
    },
    "t5_xxl": {
        "model_id": "google-t5/t5-11b",
        "version": "90f37703b3334dfe9d2b8b0f1e1e1e1e1e1e1e1e",
        "filelist": None,
    },
}


@pytest.fixture()
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A directory shaped enough like an upstream checkout to hold the registry."""

    root = tmp_path / "cosmos-curate"
    (root / "cosmos_curator" / "pipelines").mkdir(parents=True)
    registry_path = root / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(REGISTRY), encoding="utf-8")
    monkeypatch.setenv("NPA_COSMOS_CURATE_SRC", str(root))
    return root


def test_weights_dir_defaults_to_upstreams_container_cache() -> None:
    assert str(mod.weights_dir(environ={})) == "/config/models"


def test_weights_dir_is_overridable(tmp_path: Path) -> None:
    override = tmp_path / "weights"
    assert mod.weights_dir(environ={mod.WEIGHTS_DIR_ENV: str(override)}) == override


@pytest.mark.parametrize("name", mod.HF_TOKEN_ENVS)
def test_hf_token_is_read_from_any_of_its_usual_names(name: str) -> None:
    assert mod.hf_token(environ={name: "tok"}) == "tok"


def test_hf_token_is_empty_when_unset() -> None:
    assert mod.hf_token(environ={}) == ""


def test_default_set_covers_the_split_annotate_pipeline(checkout: Path) -> None:
    """Upstream's `video-pipeline split` defaults to TransNetV2 + InternVideo2 + Qwen."""

    keys = [spec.key for spec in mod.resolve_models([])]
    assert keys == ["transnetv2", "internvideo2_mm", "bert", "qwen2.5_vl"]


def test_resolve_models_carries_upstreams_pinned_revision(checkout: Path) -> None:
    (spec,) = mod.resolve_models(["split-transnetv2"])
    assert spec.model_id == "Sn4kehead/TransNetV2"
    assert spec.revision == REGISTRY["transnetv2"]["version"]
    assert spec.files == ()


def test_resolve_models_carries_upstreams_file_list(checkout: Path) -> None:
    specs = {spec.key: spec for spec in mod.resolve_models(["filter-aesthetic"])}
    assert set(specs) == {"clip_vit", "aesthetic_scorer"}
    assert specs["aesthetic_scorer"].files == ("model.safetensors",)


def test_resolve_models_accepts_a_raw_model_key(checkout: Path) -> None:
    (spec,) = mod.resolve_models(["cosmos_embed1_336p"])
    assert spec.model_id == "nvidia/Cosmos-Embed1-336p"


def test_resolve_models_deduplicates_overlapping_sets(checkout: Path) -> None:
    keys = [spec.key for spec in mod.resolve_models(["embed-internvideo2", "bert", "split-annotate"])]
    assert len(keys) == len(set(keys))
    assert "bert" in keys


def test_resolve_models_rejects_an_unknown_name(checkout: Path) -> None:
    with pytest.raises(CosmosCurateError, match="unknown model or set"):
        mod.resolve_models(["not-a-model"])


def test_resolve_models_needs_a_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NPA_COSMOS_CURATE_SRC", str(tmp_path / "absent"))
    with pytest.raises(CosmosCurateError, match="no Cosmos Curator checkout"):
        mod.resolve_models([])


def test_model_status_reports_missing_then_present(checkout: Path, tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    environ = {"NPA_COSMOS_CURATE_SRC": str(checkout), mod.WEIGHTS_DIR_ENV: str(weights)}
    specs = mod.resolve_models(["filter-aesthetic"], environ=environ)

    missing = {status.key: status for status in mod.model_status(specs, environ=environ)}
    assert not any(status.present for status in missing.values())
    assert missing["clip_vit"].local_dir.endswith("openai/clip-vit-large-patch14")

    # A partial download does not count as present: the registry lists two files.
    partial = weights / "openai/clip-vit-large-patch14"
    partial.mkdir(parents=True)
    (partial / "config.json").write_text("{}", encoding="utf-8")
    assert not mod.model_status(specs, environ=environ)[0].present

    (partial / "model.safetensors").write_bytes(b"weights")
    status = {entry.key: entry for entry in mod.model_status(specs, environ=environ)}["clip_vit"]
    assert status.present
    assert status.file_count == 2
    assert status.bytes > 0


def test_model_status_needs_more_than_a_file_when_there_is_no_file_list(
    checkout: Path, tmp_path: Path
) -> None:
    """Without a registry file list, only a completion stamp can vouch for a cache.

    Treating any file as proof accepts a download that was killed part-way: the next
    run skips it and the missing shard surfaces later as a load error inside a stage.
    """

    weights = tmp_path / "weights"
    environ = {"NPA_COSMOS_CURATE_SRC": str(checkout), mod.WEIGHTS_DIR_ENV: str(weights)}
    specs = mod.resolve_models(["split-transnetv2"], environ=environ)
    local = weights / "Sn4kehead/TransNetV2"
    local.mkdir(parents=True)
    (local / "transnetv2-pytorch-weights.pth").write_bytes(b"weights")

    assert not mod.model_status(specs, environ=environ)[0].present

    mod.write_completion_stamp(local, specs[0])
    assert mod.model_status(specs, environ=environ)[0].present


def test_fetch_models_requires_a_hugging_face_token(checkout: Path, tmp_path: Path) -> None:
    environ = {"NPA_COSMOS_CURATE_SRC": str(checkout), mod.WEIGHTS_DIR_ENV: str(tmp_path / "w")}
    with pytest.raises(CosmosCurateError, match="no Hugging Face token"):
        mod.fetch_models(["split-transnetv2"], environ=environ)


def test_fetch_models_gives_upstream_the_config_it_reads_its_token_from(
    checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream loads huggingface.api_key from a config file, not from the env."""

    import yaml as yaml_mod

    weights = tmp_path / "weights"
    seen: dict[str, Any] = {}

    def fake_download(model_id: str, revision: str | None, files: list[str] | None) -> None:
        import sys

        config_mod = sys.modules["cosmos_curator.core.utils.config.config"]
        path = Path(config_mod.CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE)
        seen["path"] = path
        seen["config"] = yaml_mod.safe_load(path.read_text(encoding="utf-8"))
        seen["mode"] = path.stat().st_mode & 0o777
        local = weights / model_id
        local.mkdir(parents=True, exist_ok=True)
        (local / (files or ["weights.bin"])[0]).write_bytes(b"weights")

    _install_fake_upstream(monkeypatch, weights, fake_download)
    monkeypatch.setenv("HF_TOKEN", "hf-secret-token")
    monkeypatch.setenv(mod.WEIGHTS_DIR_ENV, str(weights))

    result = mod.fetch_models(["split-transnetv2"])
    assert result.fetched == ["transnetv2"]
    assert seen["config"] == {"huggingface": {"api_key": "hf-secret-token"}}
    # Written into a private temporary file, not a persistent credential location.
    assert seen["mode"] == 0o600
    assert not seen["path"].exists(), "the temporary token file must not outlive the fetch"

    import sys

    restored = sys.modules["cosmos_curator.core.utils.config.config"]
    assert str(restored.CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE) == (
        "/cosmos_curator/config/cosmos_curator.yaml"
    )


def test_fetch_models_calls_upstreams_downloader_with_upstreams_pins(
    checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "weights"
    calls: list[tuple[str, str | None, list[str] | None]] = []

    def fake_download(model_id: str, revision: str | None, files: list[str] | None) -> None:
        calls.append((model_id, revision, files))
        local = weights / model_id
        local.mkdir(parents=True, exist_ok=True)
        for name in files or ["weights.bin"]:
            (local / name).write_bytes(b"weights")

    _install_fake_upstream(monkeypatch, weights, fake_download)
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setenv(mod.WEIGHTS_DIR_ENV, str(weights))

    result = mod.fetch_models(["filter-aesthetic"])
    assert result.status == "completed"
    assert sorted(result.fetched) == ["aesthetic_scorer", "clip_vit"]
    assert not result.failed
    assert all(status.present for status in result.models)
    # Upstream's pinned revision and file list are what gets requested.
    by_model = {model_id: (revision, files) for model_id, revision, files in calls}
    assert by_model["ttj/sac-logos-ava1-l14-linearMSE"] == (
        REGISTRY["aesthetic_scorer"]["version"],
        ["model.safetensors"],
    )
    assert by_model["openai/clip-vit-large-patch14"][0] == REGISTRY["clip_vit"]["version"]


def test_fetch_models_skips_what_is_already_complete(
    checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "weights"
    local = weights / "Sn4kehead/TransNetV2"
    local.mkdir(parents=True)
    (local / "transnetv2-pytorch-weights.pth").write_bytes(b"weights")
    calls: list[str] = []

    _install_fake_upstream(
        monkeypatch, weights, lambda model_id, revision, files: calls.append(model_id)
    )
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setenv(mod.WEIGHTS_DIR_ENV, str(weights))

    # A finished download is one that left a stamp behind.
    environ = {"NPA_COSMOS_CURATE_SRC": str(checkout), mod.WEIGHTS_DIR_ENV: str(weights)}
    mod.write_completion_stamp(local, mod.resolve_models(["split-transnetv2"], environ=environ)[0])

    result = mod.fetch_models(["split-transnetv2"])
    assert result.already_present == ["transnetv2"]
    assert not calls

    forced = mod.fetch_models(["split-transnetv2"], force=True)
    assert forced.fetched == ["transnetv2"]
    assert calls == ["Sn4kehead/TransNetV2"]


def test_fetch_models_records_a_failure_and_keeps_going(
    checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "weights"

    def flaky(model_id: str, revision: str | None, files: list[str] | None) -> None:
        if "clip" in model_id:
            raise RuntimeError("403 gated repo")
        local = weights / model_id
        local.mkdir(parents=True, exist_ok=True)
        (local / (files or ["weights.bin"])[0]).write_bytes(b"weights")

    _install_fake_upstream(monkeypatch, weights, flaky)
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setenv(mod.WEIGHTS_DIR_ENV, str(weights))

    result = mod.fetch_models(["filter-aesthetic"])
    assert result.status == "partial"
    assert result.fetched == ["aesthetic_scorer"]
    assert "clip_vit" in result.failed
    assert "403" in result.failed["clip_vit"]


def test_describe_models_reports_credentials_and_set_membership(
    checkout: Path, tmp_path: Path
) -> None:
    environ = {
        "NPA_COSMOS_CURATE_SRC": str(checkout),
        mod.WEIGHTS_DIR_ENV: str(tmp_path / "w"),
        "HF_TOKEN": "tok",
    }
    payload = mod.describe_models(environ=environ)
    assert payload["hf_token_present"] is True
    assert payload["ngc_key_present"] is False
    assert payload["cpu_stages_need_no_weights"] is True
    assert payload["registry_size"] == len(REGISTRY)
    assert set(payload["sets"]) == set(mod.MODEL_SETS)
    assert payload["sets"]["caption-qwen"]["keys"] == ["qwen2.5_vl"]
    assert payload["sets"]["caption-qwen"]["models"][0]["present"] is False


def test_describe_models_reports_a_missing_checkout_instead_of_raising(
    tmp_path: Path,
) -> None:
    payload = mod.describe_models(environ={"NPA_COSMOS_CURATE_SRC": str(tmp_path / "absent")})
    assert "error" in payload
    assert payload["sets"] == {}


def _install_fake_upstream(
    monkeypatch: pytest.MonkeyPatch,
    weights: Path,
    download: Any,
) -> None:
    """Stand in for the upstream modules :func:`fetch_models` reaches for."""

    import sys
    import types

    monkeypatch.setattr(mod, "ensure_upstream_importable", lambda **_: None)

    # `from cosmos_curator.core.utils import environment` resolves through the
    # parent packages, so each level has to exist in sys.modules.
    packages = {}
    for name in (
        "cosmos_curator",
        "cosmos_curator.core",
        "cosmos_curator.core.utils",
        "cosmos_curator.core.utils.model",
    ):
        package = types.ModuleType(name)
        package.__path__ = []  # type: ignore[attr-defined]
        packages[name] = package
        monkeypatch.setitem(sys.modules, name, package)

    model_utils = types.ModuleType("cosmos_curator.core.utils.model.model_utils")
    model_utils.download_model_weights_from_huggingface_to_workspace = download  # type: ignore[attr-defined]
    environment = types.ModuleType("cosmos_curator.core.utils.environment")
    environment.CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR = Path("/config/models")  # type: ignore[attr-defined]
    # Upstream reads its HF token from this config path, not from the environment.
    config_pkg = types.ModuleType("cosmos_curator.core.utils.config")
    config_pkg.__path__ = []  # type: ignore[attr-defined]
    config_mod = types.ModuleType("cosmos_curator.core.utils.config.config")
    config_mod.CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE = Path(  # type: ignore[attr-defined]
        "/cosmos_curator/config/cosmos_curator.yaml"
    )
    config_pkg.config = config_mod  # type: ignore[attr-defined]

    for name, module in (
        ("cosmos_curator.core.utils.model.model_utils", model_utils),
        ("cosmos_curator.core.utils.environment", environment),
        ("cosmos_curator.core.utils.config", config_pkg),
        ("cosmos_curator.core.utils.config.config", config_mod),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    setattr(packages["cosmos_curator.core.utils"], "environment", environment)
    setattr(packages["cosmos_curator.core.utils"], "config", config_pkg)
    setattr(packages["cosmos_curator.core.utils.model"], "model_utils", model_utils)
