"""Retain baseline route coverage by composing measured scenes without changing geometry."""

import copy
import json
from pathlib import Path

from npa.workflows.navigation.artifacts import file_sha256, write_json
from npa.workflows.navigation.reference_geometry import validate_scene_frame


def compose_scene(office, warehouse, output, *, translation=(50.0, 0.0, 0.0)):
    """Package separated office and warehouse geometry in one metric static scene.

    Args:
        office: Reconstructed metric Z-up collision USDZ.
        warehouse: Unmodified baseline warehouse collision USDZ.
        output: Fresh destination USDZ.
        translation: Office translation in metres; no rotation or scaling occurs.
    Returns:
        Source hashes, translation and composite scene identity.
    Raises:
        ValueError: Inputs have invalid frames, overlap, or cannot be composed.
        OSError: Required files cannot be read or written.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils

    output = Path(output)
    layer = output.with_suffix(".usda")
    stage = Usd.Stage.CreateNew(str(layer))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    for name, source, offset in (
        ("Warehouse", warehouse, (0.0, 0.0, 0.0)),
        ("Office", office, translation),
    ):
        validate_scene_frame(str(source))
        root = UsdGeom.Xform.Define(stage, "/World/" + name)
        root.GetPrim().GetReferences().AddReference(str(Path(source).resolve()))
        root.AddTranslateOp().Set(Gf.Vec3d(*offset))
    _require_separation(stage)
    stage.GetRootLayer().TransferContent(stage.Flatten())
    stage.GetRootLayer().Save()
    if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(layer)), str(output)):
        raise ValueError("could not package the replay collision scene")
    layer.unlink()
    return {
        "warehouse_sha256": file_sha256(Path(warehouse)),
        "office_sha256": file_sha256(Path(office)),
        "office_translation_m": list(translation),
        "scene_sha256": file_sha256(output),
        "geometry_rescaled": False,
        "added_support_geometry": False,
    }


def _require_separation(stage):
    from pxr import Usd, UsdGeom

    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    regions = [
        bounds.ComputeWorldBound(
            stage.GetPrimAtPath("/World/" + name)
        ).ComputeAlignedRange()
        for name in ("Warehouse", "Office")
    ]
    if any(region.IsEmpty() for region in regions):
        raise ValueError("replay scene has empty geometry")
    first, second = regions
    # Rays have a 12-metre horizon. Disjoint bounds alone would let one region
    # change the other's observations, despite having no physical overlap.
    if not any(
        first.GetMax()[axis] + 12 < second.GetMin()[axis]
        or second.GetMax()[axis] + 12 < first.GetMin()[axis]
        for axis in (0, 1)
    ):
        raise ValueError("replay regions must be separated beyond the ray horizon")


def translated_cases(cases, translation):
    """Translate reset poses and goals together while preserving route geometry.

    Args:
        cases: Explicit reset case dictionaries.
        translation: Metric XYZ scene translation.
    Returns:
        Independent translated cases with unchanged headings and identifiers.
    Raises:
        ValueError: Translation does not contain exactly three finite values.
    """
    import math

    if len(translation) != 3 or not all(math.isfinite(v) for v in translation):
        raise ValueError("scene translation requires three finite coordinates")
    result = copy.deepcopy(cases)
    for case in result:
        case["position_m"] = [
            v + d for v, d in zip(case["position_m"], translation, strict=True)
        ]
        case["goal_m"] = [
            v + d for v, d in zip(case["goal_m"], translation[:2], strict=True)
        ]
    return result


def apply_replay(source, prepared, recipe):
    """Compose a hash-bound baseline replay scene after actual capture reconstruction.

    Args:
        source: Sealed capture bundle that may contain replay.json.
        prepared: Native navigation output with reconstructed scene.usdz.
        recipe: Mutable office training recipe from the sealed capture.
    Returns:
        Optional composition evidence; None for existing single-scene inputs.
    Raises:
        ValueError: Replay metadata, baseline geometry or reset identities differ.
        OSError: Input files or output publication fail.
    """
    replay_path = source / "replay.json"
    if not replay_path.exists():
        return None
    replay = json.loads(replay_path.read_text())
    _validate_replay(source, replay)
    scene = prepared / "scene.usdz"
    office = prepared / "office.usdz"
    scene.rename(office)
    evidence = compose_scene(
        office,
        source / "replay.usdz",
        scene,
        translation=replay["office_translation_m"],
    )
    _verify_frozen_geometry(scene, replay, evidence)
    office_cases = translated_cases(
        recipe["train_cases"], replay["office_translation_m"]
    )
    recipe["train_cases"] = _training_order(office_cases, replay)
    recipe["eval_cases"] = replay["development_cases"]
    _verify_frozen_routes(recipe, replay)
    recipe["probe"] = replay["probe"]
    recipe["scene_sha256"] = evidence["scene_sha256"]
    evidence.update(
        warehouse_routes=len(replay["train_cases"]),
        office_routes=len(office_cases),
        scope="known office and warehouse layouts; new routes evaluated separately",
    )
    write_json(prepared / "replay-composition.json", evidence)
    return evidence


def _training_order(office_cases, replay):
    regions = (office_cases, replay["train_cases"])
    if replay["schema"] == "npa.navigation.replay.v1":
        regions = tuple(reversed(regions))
    return [case for pair in zip(*regions, strict=True) for case in pair]


def _validate_replay(source, replay):
    expected = {
        "schema",
        "scene_sha256",
        "office_translation_m",
        "train_cases",
        "development_cases",
        "probe",
    }
    if replay.get("schema") == "npa.navigation.replay.v2":
        expected.update({"frozen_geometry", "frozen_routes"})
    if set(replay) != expected or replay["schema"] not in {
        "npa.navigation.replay.v1",
        "npa.navigation.replay.v2",
    }:
        raise ValueError("invalid native replay contract")
    scene = source / "replay.usdz"
    if scene.is_symlink() or file_sha256(scene) != replay["scene_sha256"]:
        raise ValueError("baseline replay scene hash differs")


def _verify_frozen_geometry(scene, replay, evidence):
    if "frozen_geometry" not in replay:
        return
    from npa.workflows.navigation.reference_identity import collision_identity

    expected = replay["frozen_geometry"]
    actual = collision_identity(scene)
    if (
        actual != expected["collision_identity"]
        or evidence["warehouse_sha256"] != expected["warehouse_sha256"]
        or evidence["office_translation_m"] != expected["office_translation_m"]
    ):
        raise ValueError(
            "reconstructed collision geometry differs from the pre-learning freeze"
        )
    evidence.update(
        collision_identity=actual,
        frozen_geometry_matched=True,
        pre_learning_scene_sha256=expected["scene_sha256"],
    )


def _verify_frozen_routes(recipe, replay):
    if "frozen_routes" not in replay:
        return
    import hashlib
    from npa.workflows.navigation.contract import Case

    for cohort, key in (("training", "train_cases"), ("development", "eval_cases")):
        rows = [Case.model_validate(row).model_dump() for row in recipe[key]]
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        if digest != replay["frozen_routes"][cohort]:
            raise ValueError("replay routes differ from the pre-learning freeze")
