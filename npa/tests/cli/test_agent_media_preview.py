"""Hard contract tests for agent MP4/image artifact preview.

These markers have regressed when the live agent was bootstrapped from an older
revision or when template edits dropped the authenticated blob preview path.
Keep this file focused and fail-loud.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from npa.cli.agent import AGENT_MEDIA_PREVIEW_CONTRACT, AGENT_UI_VERSION
from npa.workflows.artifacts import (
    _IMAGE_EXTENSIONS,
    _TEXT_EXTENSIONS,
    _VIDEO_EXTENSIONS,
    artifact_media_type,
    render_hint_for_object,
)

AGENT_MODULE = Path(__file__).resolve().parents[2] / "src" / "npa" / "cli" / "agent.py"
AGENT_CONTRACTS_MODULE = AGENT_MODULE.with_name("agent_contracts.py")
AGENT_UI = AGENT_MODULE.with_name("agent_ui.html")
ARTIFACT_CONTENT_MODULE = AGENT_MODULE.with_name("agent_artifact_content.py")
ARTIFACTS_MODULE = (
    Path(__file__).resolve().parents[2] / "src" / "npa" / "workflows" / "artifacts.py"
)


def test_agent_media_preview_contract_present_in_source() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8") + AGENT_UI.read_text(encoding="utf-8")
    contract_source = (
        source
        + AGENT_CONTRACTS_MODULE.read_text(encoding="utf-8")
        + ARTIFACT_CONTENT_MODULE.read_text(encoding="utf-8")
    )
    assert f'AGENT_UI_VERSION = "{AGENT_UI_VERSION}"' in source
    for marker in AGENT_MEDIA_PREVIEW_CONTRACT:
        assert marker in contract_source, (
            f"missing media-preview contract marker: {marker!r}"
        )
    # Anti-patterns that previously broke MP4 playback under basic auth.
    assert '`<video controls src="${{previewUrl}}">`' not in source
    assert 'host.innerHTML = `<video controls src="${{previewUrl}}"></video>`' not in source
    assert 'host.innerHTML = `<img alt="artifact image" src="${{previewUrl}}" />`' not in source


def test_artifact_media_type_covers_inline_render_extensions() -> None:
    assert artifact_media_type("heldout-success.mp4") == "video/mp4"
    assert artifact_media_type("clip.webm") == "video/webm"
    assert artifact_media_type("clip.mov") == "video/quicktime"
    assert artifact_media_type("frame.png") == "image/png"
    assert artifact_media_type("frame.jpg") == "image/jpeg"
    assert artifact_media_type("frame.jpeg") == "image/jpeg"
    assert artifact_media_type("report.json") == "application/json"
    assert artifact_media_type("orchestrator.log").startswith("text/plain")
    assert artifact_media_type("unknown.fooz") == "application/octet-stream"
    assert artifact_media_type("path/with/dirs/video.MP4") == "video/mp4"


@pytest.mark.parametrize("ext", sorted(_VIDEO_EXTENSIONS | _IMAGE_EXTENSIONS | _TEXT_EXTENSIONS | {".json"}))
def test_artifact_media_type_aligned_with_render_hint(ext: str) -> None:
    key = f"run/artifacts/object{ext}"
    render = render_hint_for_object(key=key)
    media = artifact_media_type(key)
    if render == "video":
        assert media.startswith("video/")
    elif render == "image":
        assert media.startswith("image/")
    elif render == "json":
        assert media == "application/json"
    elif render == "text":
        assert media.startswith("text/") or media.startswith("application/x-ndjson")


def test_embedded_artifacts_source_includes_media_type_helper() -> None:
    from npa.cli.agent import _embedded_agent_artifacts_source

    embedded = _embedded_agent_artifacts_source()
    assert "def artifact_media_type(" in embedded
    assert '"video/mp4"' in embedded
    # Ensure the live backend can call the helper by name.
    assert "artifact_media_type" in ARTIFACTS_MODULE.read_text(encoding="utf-8")


def test_bootstrap_embeds_artifacts_module_with_media_type() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8") + ARTIFACT_CONTENT_MODULE.read_text(
        encoding="utf-8"
    )
    assert "_AGENT_ARTIFACTS_EMBED" in source
    assert "content_type = artifact_media_type(str(artifact.key))" in source
    assert '"X-Content-Type-Options": "nosniff"' in source
    assert "def _artifact_media_type(" not in source
    # load-run must echo the applied snapshot (not re-enter status) to avoid
    # racing concurrent UI artifact loads from returning the wrong run.
    assert "def _sim_viz_load_response(" in source
    assert '"sim_viz": _sim_viz_load_response(' in source
