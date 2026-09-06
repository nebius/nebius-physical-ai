"""Real complete structural augmentation and publication-only CLI recovery.

Requires NPA_INTEGRATION_E2E=1, the existing Nano serving endpoint/token,
NPA_COSMOS3_AUGMENT_INPUT_URI, NPA_COSMOS3_AUGMENT_OUTPUT_URI,
NPA_COSMOS3_AUGMENT_PROMPT and an owner-only NPA_COSMOS3_VIDEO_EVIDENCE_DIR
outside the checkout. No mocked or fixture model output qualifies here.
"""

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.sdk.workbench.cosmos3 import nano_video_augment

pytestmark = pytest.mark.e2e


@pytest.mark.timeout(0)
def test_real_complete_augmentation_and_recover(monkeypatch):
    required = ("NPA_COSMOS3_VIDEO_ENDPOINT", "NPA_COSMOS3_VIDEO_TOKEN",
                "NPA_COSMOS3_VIDEO_EVIDENCE_DIR", "NPA_COSMOS3_AUGMENT_INPUT_URI",
                "NPA_COSMOS3_AUGMENT_OUTPUT_URI", "NPA_COSMOS3_AUGMENT_PROMPT")
    if any(not os.environ.get(name) for name in required):
        pytest.skip("requires explicit live augmentation source, destination and serving configuration")
    evidence = Path(os.environ[required[2]])
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    monkeypatch.setenv("NPA_COSMOS3_VIDEO_RECOVERY_DIR", str(evidence))
    result = nano_video_augment(
        input_path=os.environ[required[3]], output_path=os.environ[required[4]],
        prompt=os.environ[required[5]],
        seed=int(os.environ.get("NPA_COSMOS3_AUGMENT_SEED", "0")),
        negative_prompt=os.environ.get("NPA_COSMOS3_AUGMENT_NEGATIVE_PROMPT", ""),
        system_prompt=os.environ.get("NPA_COSMOS3_AUGMENT_SYSTEM_PROMPT", ""),
        guidance_scale=float(os.environ.get("NPA_COSMOS3_AUGMENT_GUIDANCE_SCALE", "3.0")),
        flow_shift=float(os.environ.get("NPA_COSMOS3_AUGMENT_FLOW_SHIFT", "10.0")),
        control_guidance=float(os.environ.get("NPA_COSMOS3_AUGMENT_CONTROL_GUIDANCE", "1.5")),
        edge_threshold=os.environ.get("NPA_COSMOS3_AUGMENT_EDGE_THRESHOLD", "medium"),
        num_inference_steps=int(os.environ.get("NPA_COSMOS3_AUGMENT_STEPS", "35")),
        chunk_frames=int(os.environ.get("NPA_COSMOS3_AUGMENT_CHUNK_FRAMES", "121")),
        max_sequence_length=int(os.environ.get("NPA_COSMOS3_AUGMENT_MAX_SEQUENCE_LENGTH", "4096")),
    )
    assert result["status"] == "succeeded"
    assert result["publication_verified"] is True and result["technical_validation_passed"] is True
    assert result["source_sha256"] != result["output_sha256"]
    assert result["video"]["decoded_frames"] == 720
    assert result["video"]["duration_seconds"] == 30
    # Remove serving authorization: verified local recovery must only contact S3.
    monkeypatch.delenv("NPA_COSMOS3_VIDEO_TOKEN")
    cli = CliRunner().invoke(app, ["workbench", "cosmos3", "nano-video-augment-recover",
                                 "--output-path", os.environ[required[4]]])
    for name, content in (("recover-stdout.json", cli.stdout), ("recover-stderr.txt", cli.stderr)):
        path = evidence / name
        path.write_text(content)
        path.chmod(0o600)
    assert cli.exit_code == 0, "Inspect retained private recovery diagnostics"
    recovered = json.loads(cli.stdout)
    assert recovered["request_id"] == result["request_id"]
    assert recovered["output_sha256"] == result["output_sha256"]
    assert recovered["publication_verified"] is True
