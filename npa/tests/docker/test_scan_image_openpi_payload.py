from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "scan_image_openpi_payload.py"


def _scanner():
    spec = importlib.util.spec_from_file_location("scan_image_openpi_payload", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_openpi_payload_classifier_fails_on_cache_and_orbax_shards() -> None:
    scanner = _scanner()
    assert "cache" in scanner.classify_path(
        "opt/npa-model-cache/openpi/gcs/revision/checkpoint/assets/norm.json"
    )
    assert "Orbax" in scanner.classify_path(
        "workspace/model/params/ocdbt.process_0/d/abc"
    )
    assert scanner.classify_path("opt/byof/src/openpi/models/gemma.py") is None
    assert scanner.classify_path("opt/npa-model-cache/openpi/") is None


def test_openpi_history_classifier_fails_on_baked_acceptance_or_token() -> None:
    scanner = _scanner()
    assert "acceptance" in scanner.classify_history(
        "ENV NPA_OPENPI_ACCEPT_GEMMA_TERMS=YES"
    )
    assert "credential" in scanner.classify_history("ARG HF_TOKEN=not-a-real-value")
    assert scanner.classify_history("COPY openpi_checkpoint_cache.py /opt/") is None
