"""Cosmos 3 generate golden eval — a REAL capability test.

Runs an actual text2image generation through the same
``npa.workbench.cosmos.generate.run_cosmos3_generate`` path the CLI, SDK, and
``cosmos3-generate`` workflow use, then asserts the artifact is a decodable image
with real dimensions. This exercises the container's job (Cosmos 3 world/image
generation), not a CUDA/import probe.

GPU-gated and heavy: no weights ship in the image, so the gated Cosmos3 checkpoint
downloads on first use under the operator's own Hugging Face license acceptance
(HF_TOKEN). Import-safe on the default interpreter — PIL is imported lazily inside
``main``, and torch only ever loads inside the framework's own venv.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_PROMPT = (
    "A medium shot of a robotic arm on a clean white workbench in a bright "
    "robotics lab, sorting small colored blocks."
)


class _WiringResult:
    def __init__(self, name: str, ok: bool, detail: str = "") -> None:
        self.name = name
        self.ok = ok
        self.detail = detail


def check_generate_wiring() -> _WiringResult:
    """Infra-free precondition check that the generate path is wired end to end.

    The GPU capability run lives in ``main``; this helper is the fast, import-only
    sanity check used by the standard unit suite. It proves the CLI command, the
    shared runner, and the guardrails-on-by-default plan all agree without a GPU.
    """

    try:
        from npa.cli.workbench import cosmos3 as cosmos3_cli
        from npa.workbench.cosmos.generate import generate_plan, run_cosmos3_generate  # noqa: F401
    except Exception as exc:  # pragma: no cover - import failure path
        return _WiringResult("cosmos3 generate wiring", False, str(exc))
    if not hasattr(cosmos3_cli, "generate_cmd"):
        return _WiringResult(
            "cosmos3 generate wiring", False, "missing cosmos3 generate command"
        )
    plan = generate_plan(prompt="a robot arm on a workbench", output_dir="/tmp/npa-cosmos3-wiring")
    if plan.get("guardrails") is not True:
        return _WiringResult(
            "cosmos3 generate wiring", False, "guardrails are not on by default"
        )
    if "--no-guardrails" in plan.get("argv", []):
        return _WiringResult(
            "cosmos3 generate wiring", False, "--no-guardrails present by default"
        )
    return _WiringResult(
        "cosmos3 generate wiring",
        True,
        "npa.workbench.cosmos.generate.run_cosmos3_generate",
    )


def main() -> int:
    from npa.workbench.cosmos.generate import (
        Cosmos3GenerateError,
        cosmos3_generate_available,
        run_cosmos3_generate,
    )

    if not cosmos3_generate_available():
        print("[FAIL] cosmos-framework inference runtime not present in this image")
        return 1

    output_dir = os.environ.get(
        "NPA_COSMOS3_OUTPUT_DIR", "/tmp/npa-cosmos3-generate"
    )
    try:
        result = run_cosmos3_generate(
            mode="text2image",
            prompt=os.environ.get("NPA_COSMOS3_PROMPT", DEFAULT_PROMPT),
            output_dir=str(Path(output_dir) / "golden-eval"),
            name="golden-eval",
            checkpoint=os.environ.get("NPA_COSMOS3_CHECKPOINT", ""),
            seed=0,
        )
    except Cosmos3GenerateError as exc:
        print(f"[FAIL] cosmos3 generation: {exc}")
        return 1

    artifact = Path(str(result.get("output_path", "")))
    if result.get("output_kind") != "image":
        print(f"[FAIL] expected an image artifact, got: {result.get('output_kind')!r}")
        return 1
    if not artifact.is_file() or artifact.stat().st_size <= 0:
        print(f"[FAIL] missing or empty artifact: {artifact}")
        return 1

    from PIL import Image

    with Image.open(artifact) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        print(f"[FAIL] artifact has invalid dimensions: {width}x{height}")
        return 1
    if result.get("guardrails") is not True:
        print("[FAIL] guardrails were not enabled for the golden eval")
        return 1

    print(
        f"[PASS] cosmos3 generated {artifact} ({artifact.stat().st_size} bytes, "
        f"{width}x{height}) checkpoint={result.get('checkpoint')} "
        f"guardrails={result.get('guardrails')} weights_baked={result.get('weights_baked')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
