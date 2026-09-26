"""Execute the partner labeling graph against operator-selected real services."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from uuid import uuid4

import pytest

from npa.clients.storage import StorageClient
from npa.orchestration.npa_workflow import build_plan, load_spec, run_workflow
from npa.orchestration.npa_workflow.submit import merge_config_overrides
from npa.workbench.encord.storage import ConditionalArtifactStore

ROOT = Path(__file__).resolve().parents[3]


def _live_spec():
    config = os.environ.get("NPA_E2E_ENCORD_LABEL_CONFIG", "")
    if not config:
        pytest.skip(
            "Set NPA_E2E_ENCORD_LABEL_CONFIG to explicit Encord labeling targets"
        )
    overrides = json.loads(Path(config).read_text())
    required = {
        "bucket",
        "prefix",
        "encord_media_uri",
        "encord_integration",
        "encord_label_plan_uri",
    }
    if not isinstance(overrides, dict) or not required <= overrides.keys():
        raise ValueError("Explicit storage and label-plan targets are required")
    spec = load_spec(ROOT / "workflows/partners/encord/encord-labeling-demo.yaml")
    return merge_config_overrides(spec, overrides)


def _check_videos(storage, demo, evidence):
    import av

    assert demo["status"] == "completed"
    assert demo["label_source"] == "encord_project_export"
    assert demo["videos"]
    for index, item in enumerate(demo["videos"]):
        target = evidence / f"annotated-{index + 1}.mp4"
        target.touch(mode=0o600)
        storage.download_file(item["uri"], str(target))
        assert hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]
        assert target.stat().st_size == item["bytes"]
        assert item["box_annotations"] > 0 and item["tracks"] > 0
        with av.open(str(target)) as container:
            assert sum(1 for _ in container.decode(video=0)) == item["frames"] > 0


def test_unconfigured_labeling_test_skips_without_remote_calls(monkeypatch):
    monkeypatch.delenv("NPA_E2E_ENCORD_LABEL_CONFIG", raising=False)
    with pytest.raises(pytest.skip.Exception):
        _live_spec()


@pytest.mark.e2e
def test_live_labeling_workflow_saves_exports_and_renders_real_annotations(
    tmp_path, monkeypatch
):
    spec = _live_spec()
    run_id = "encord-label-test-" + uuid4().hex
    root = Path(os.environ.get("NPA_E2E_ENCORD_EVIDENCE_DIR", str(tmp_path)))
    evidence = root / run_id
    evidence.mkdir(parents=True, mode=0o700)
    monkeypatch.setenv(
        "PATH",
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
    )
    try:
        plan = build_plan(spec, run_id=run_id)
        assert [step.state for step in plan.steps] == [
            "push",
            "import_labels",
            "pull",
            "verify",
            "render_labels",
        ]
        result = run_workflow(spec, run_id=run_id, execute=True, require_inputs=True)
        (evidence / "workflow.json").write_text(json.dumps(result, indent=2))
        storage = StorageClient.from_environment()
        demo = ConditionalArtifactStore(storage).read_json(
            plan.steps[-1].outputs[0]["uri"]
        )
        (evidence / "demo.json").write_text(json.dumps(demo, indent=2))
        _check_videos(storage, demo, evidence)
    except Exception as exc:
        (evidence / "failure.txt").write_text(traceback.format_exc())
        pytest.fail(
            f"Live Encord labeling failed ({type(exc).__name__}); inspect private evidence",
            pytrace=False,
        )
    finally:
        for artifact in evidence.iterdir():
            artifact.chmod(0o600)
