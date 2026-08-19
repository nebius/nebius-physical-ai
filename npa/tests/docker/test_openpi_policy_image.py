from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
IMAGE = ROOT / "docker" / "workbench" / "openpi-policy"
DOCKERFILE = IMAGE / "Dockerfile"


def test_openpi_policy_image_is_pinned_and_runtime_fetch_only() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "15a9616a00943ada6c20a0f158e3adb39df2ccac" in text
    assert "793488b5a55bb87200db90a61fd0af51922b686d94e1da4f4c587ab119b37d74" in text
    assert text.count("FROM nvidia/cuda:") == 2
    assert text.count("@sha256:") >= 2
    assert "openpi_checkpoint_cache.py" in text
    assert "openpi_policy_server.py" in text
    assert 'npa.version="pi05-polaris-runtime-cache-20260819-r9"' in text
    assert "OPENPI_DATA_HOME=/opt/npa-model-cache/openpi/openpi-data" in text
    assert "NPA_OPENPI_ACCEPT_GEMMA_TERMS=" not in text
    assert "download.maybe_download" not in text
    assert "COPY /opt/npa-model-cache" not in text
    assert 'test -z "$(find /opt/npa-model-cache/openpi -mindepth 1' in text
    assert "USER 1000" in text
    assert "EXPOSE 8000" in text
    assert "IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg" in text
    assert "imageio_ffmpeg/binaries/ffmpeg*' -delete" in text
    assert "deepdiff==8.6.2" in text
    assert "wandb/bin/wandb-core' -delete" in text


def test_openpi_policy_packaging_contract_is_public_service() -> None:
    contract = yaml.safe_load(
        (ROOT / "docker" / "workbench" / "packaging-contract.yaml").read_text()
    )
    image = contract["images"]["openpi-policy"]
    assert image == {
        "dockerfile": "openpi-policy/Dockerfile",
        "tier": "service",
        "ports": [8000],
        "redistribution": "public",
        "notes": image["notes"],
    }
    assert "runtime" in image["notes"].lower()


def test_openpi_policy_redistribution_record_names_excluded_payloads() -> None:
    record = (IMAGE / "REDISTRIBUTION.md").read_text(encoding="utf-8").lower()
    notices = (IMAGE / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8").lower()
    for prohibited in ("checkpoint", "credential", "isaac", "omniverse", "antioch"):
        assert prohibited in record
    assert "runtime cache only" in notices
    assert "gcs object-generation manifest" in notices
