"""Run metric RGB-D reconstruction and portable Isaac scene assembly for field captures."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

from npa.workflows.field_failure.native_artifacts import (
    _archive,
    _bundle,
    _identity,
    _protocol,
    _recipe,
    _upload,
    _upload_json,
)


def reconstruct(request: dict) -> dict:
    """Reconstruct measured capture bundles and retain their native geometry evidence.

    Args:
        request: Sealed field-failure reconstruction request.
    Returns:
        Reconstruction record with derived scene bundles and actual quality reports.
    Raises:
        ValueError: Capture integrity, calibration, quality or protocol fails.
        ImportError: Metric reconstruction, Open3D or OpenUSD is unavailable.
        OSError: Native input or output cannot be read or published.
    """
    with tempfile.TemporaryDirectory(prefix="npa-field-reconstruction-") as temporary:
        root = Path(temporary)
        protocol = _protocol(request, root)
        scenes, reports = [], []
        for index, capture in enumerate(request["captures"]):
            scene, report = _reconstruct_capture(
                request, protocol, capture, root, index
            )
            scenes.append(scene)
            reports.append(report)
        evidence = _upload_json(
            reports,
            root / "reconstruction.json",
            request["output_prefix"] + "native-reconstruction.json",
        )
        return {
            **_identity(request, "npa.field-failure.reconstruction.v1"),
            "scenes": scenes,
            "evidence": evidence,
        }


def _reconstruct_capture(request, protocol, capture, root, index):
    from npa.workbench.nurec.navigation_reconstruction import reconstruct_capture
    from npa.workbench.nurec.navigation_scene import prepare_scene

    source = _bundle(capture["asset"], root, f"capture-{index}")
    recipe = _recipe(source, protocol)
    surface, scene = root / f"surface-{index}", root / f"scene-{index}"
    reconstruction = reconstruct_capture(str(source), str(surface))
    assembly = prepare_scene(str(surface), str(scene))
    prepared = root / f"navigation-{index}"
    prepared.mkdir()
    shutil.copyfile(scene / "scene.usdz", prepared / "scene.usdz")
    from npa.workflows.field_failure.artifacts import _digest

    recipe.update(
        scene_file="scene.usdz",
        scene_sha256=_digest((prepared / "scene.usdz").read_bytes()),
    )
    (prepared / "recipe.json").write_text(json.dumps(recipe, allow_nan=False))
    (prepared / "reconstruction.json").write_text(
        json.dumps(reconstruction, allow_nan=False)
    )
    (prepared / "assembly.json").write_text(json.dumps(assembly, allow_nan=False))
    archive = root / f"navigation-{index}.tar"
    _archive(prepared, archive)
    asset = _upload(archive, request["output_prefix"] + f"scene-{index}.tar")
    record = {key: capture[key] for key in ("scenario_id", "group_id")}
    record.update(capture_sha256=capture["asset"]["sha256"], asset=asset)
    return record, {
        "scenario_id": capture["scenario_id"],
        "reconstruction": reconstruction,
        "assembly": assembly,
    }
