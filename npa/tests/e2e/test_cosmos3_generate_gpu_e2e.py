"""Live GPU e2e for the containerized Cosmos 3 generate path.

Runs the checked-in ``cosmos3-generate.yaml`` through raw ``sky launch`` in the
``npa-cosmos3`` image and asserts the run produced a real, non-blank image and
published it to S3. This is the capability the golden eval registers but cannot
prove without a GPU: the container ships no weights, so a passing run also proves
the runtime checkpoint download worked under the operator's own Hugging Face
credentials.

Opt-in, like the other Cosmos3 e2e: needs ``NPA_INTEGRATION_E2E=1``,
``NPA_COSMOS3_E2E=1``, an HF token, a pushed image in ``NPA_COSMOS3_E2E_IMAGE``,
and an ``s3://`` prefix to publish into.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]

ROOT = Path(__file__).resolve().parents[3]
YAML_PATH = (
    ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot" / "cosmos3-generate.yaml"
)
DEFAULT_PROMPT = (
    "a robot arm sorting colored blocks on a white workbench in a bright robotics lab"
)


def test_cosmos3_generate_publishes_a_real_image(tmp_path: Path) -> None:
    image = _require_runtime()
    sky_bin = _sky_bin()
    run_id = (
        "cosmos3-gen-"
        + time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        + "-"
        + uuid.uuid4().hex[:6]
    )
    output_uri = _output_uri(run_id)
    yaml_path = _render_template(tmp_path, image=image)

    cmd = [
        sky_bin,
        "launch",
        "--yes",
        "--cluster",
        run_id,
        "--infra",
        os.environ.get("NPA_COSMOS3_E2E_INFRA", "kubernetes"),
        "--gpus",
        os.environ.get("NPA_COSMOS3_E2E_GPU", "H100:1"),
        "--env",
        f"NPA_RUN_ID={run_id}",
        "--env",
        f"NPA_COSMOS3_OUTPUT_URI={output_uri}",
        "--env",
        f"NPA_COSMOS3_PROMPT={os.environ.get('NPA_COSMOS3_PROMPT', DEFAULT_PROMPT)}",
        "--env",
        "HF_TOKEN",
        "--env",
        "AWS_ACCESS_KEY_ID",
        "--env",
        "AWS_SECRET_ACCESS_KEY",
        "--env",
        "AWS_ENDPOINT_URL",
        str(yaml_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(os.environ.get("NPA_COSMOS3_E2E_TIMEOUT_SECONDS", "7200")),
            check=False,
        )
        (tmp_path / "sky.log").write_text(result.stdout, encoding="utf-8")
        assert result.returncode == 0, (
            f"sky launch failed (rc={result.returncode}); tail:\n"
            + "\n".join(result.stdout.splitlines()[-40:])
        )
        assert '"status": "executed"' in result.stdout, (
            "run did not report an executed generation; tail:\n"
            + "\n".join(result.stdout.splitlines()[-40:])
        )
    finally:
        subprocess.run(
            [sky_bin, "down", "--yes", run_id],
            text=True,
            capture_output=True,
            timeout=int(os.environ.get("NPA_COSMOS3_E2E_TEARDOWN_SECONDS", "1200")),
            check=False,
        )

    manifest, image_bytes = _fetch_published(output_uri)

    assert manifest["status"] == "executed"
    assert manifest["output_kind"] == "image"
    # The image bakes no weights, so a real run must have authenticated to HF.
    assert manifest["weights_baked"] is False
    assert manifest["hf_auth"] == "configured"
    # Guardrails stay on unless explicitly disabled.
    assert manifest["guardrails"] is True

    _assert_real_image(image_bytes)


def _require_runtime() -> str:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("NPA_INTEGRATION_E2E not set")
    if os.environ.get("NPA_COSMOS3_E2E") != "1":
        pytest.skip("NPA_COSMOS3_E2E not set")
    hf_env = os.environ.get("NPA_COSMOS3_HF_TOKEN_ENV", "HF_TOKEN")
    if not os.environ.get(hf_env):
        pytest.skip(f"{hf_env} is not set; the image bakes no weights")
    image = os.environ.get("NPA_COSMOS3_E2E_IMAGE", "").strip()
    if not image:
        pytest.skip("NPA_COSMOS3_E2E_IMAGE (pushed npa-cosmos3 ref) is not set")
    if not os.environ.get("NPA_COSMOS3_E2E_OUTPUT_PREFIX", "").strip():
        pytest.skip("NPA_COSMOS3_E2E_OUTPUT_PREFIX (s3:// prefix) is not set")
    return image


def _sky_bin() -> str:
    sky_bin = os.environ.get(
        "NPA_SKYPILOT_BIN", "/home/ubuntu/.npa/skypilot-venv/bin/sky"
    )
    if not Path(sky_bin).exists():
        pytest.skip(f"SkyPilot binary not found: {sky_bin}")
    return sky_bin


def _output_uri(run_id: str) -> str:
    prefix = os.environ["NPA_COSMOS3_E2E_OUTPUT_PREFIX"].rstrip("/")
    return f"{prefix}/{run_id}/"


def _render_template(tmp_path: Path, *, image: str) -> Path:
    """Substitute only ``${NPA_COSMOS3_IMAGE}``.

    SkyPilot does not expand variables inside ``resources.image_id``, so the
    submitter renders it — the same step the operator guide documents. Only that
    one token is replaced so the ``run:`` script's bash parameter expansions stay
    intact.
    """

    text = YAML_PATH.read_text(encoding="utf-8")
    rendered = text.replace("${NPA_COSMOS3_IMAGE}", image)
    assert "${NPA_COSMOS3_IMAGE}" not in rendered
    out = tmp_path / "cosmos3-generate.rendered.yaml"
    out.write_text(rendered, encoding="utf-8")
    return out


def _fetch_published(output_uri: str) -> tuple[dict, bytes]:
    import urllib.parse

    import boto3

    parsed = urllib.parse.urlparse(output_uri)
    bucket, prefix = parsed.netloc, parsed.path.lstrip("/")
    s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)
    keys = [
        obj["Key"]
        for obj in s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
    ]
    manifest_key = next((k for k in keys if k.endswith("generate.json")), "")
    assert manifest_key, f"no generate.json published under {output_uri}: {keys}"
    manifest = json.loads(
        s3.get_object(Bucket=bucket, Key=manifest_key)["Body"].read().decode("utf-8")
    )
    artifact_key = next(
        (k for k in keys if k.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))), ""
    )
    assert artifact_key, f"no image artifact published under {output_uri}: {keys}"
    return manifest, s3.get_object(Bucket=bucket, Key=artifact_key)["Body"].read()


def _assert_real_image(payload: bytes) -> None:
    """A decodable image with actual variation, not a blank or placeholder frame."""

    import io

    from PIL import Image

    with Image.open(io.BytesIO(payload)) as img:
        img.load()
        width, height = img.size
        rgb = img.convert("RGB")
    assert width >= 256 and height >= 256, f"suspicious dimensions: {width}x{height}"
    extrema = rgb.getextrema()
    assert any(lo != hi for lo, hi in extrema), f"flat/blank image: extrema={extrema}"
