"""Reject incomplete fanout evidence and nondefault generation in starter audits."""

from io import BytesIO
import json
from pathlib import Path

import pytest


@pytest.fixture
def starter_audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from tests.e2e.test_paidf_cosmos3_starter_live import _assert_starter_batch

    return _assert_starter_batch


def _documents(count, attempt=0):
    variants = [
        {
            "clip": f"variant-{index:04d}",
            "seed": 17 + attempt * 1000 + index,
            "guidance": 5.0 - attempt * 0.5,
            "steps": 24 + attempt * 4,
        }
        for index in range(count)
    ]
    return {
        "cosmos_augmented/manifest.json": {
            "variants": variants,
            "variant_count": count,
            "variant_parallelism": 1,
            "motion_preservation": {"source_weight": 0},
            "attempt": attempt,
            "model": "Cosmos3-Nano",
            "structural_control": "edge",
        },
        "configs/manifest.json": {
            "n_augmentations": count,
            "augmentations": [{} for _ in variants],
            "augmentation_seed": "30",
            "appearance_profile_source": "default",
        },
        "grade/cosmos_evaluator.json": {
            "threshold": 0.2,
            "attribute_threshold": 0.25,
            "clips": [{"clip_id": variant["clip"]} for variant in variants],
        },
        "reports/final.json": {
            "variant_count": count,
            "annotated_variant_count": count,
        },
    }


class ArtifactClient:
    def __init__(self, documents):
        self.documents = documents

    def get_object(self, *, Bucket, Key):
        return {"Body": BytesIO(json.dumps(self.documents[Key]).encode())}


@pytest.mark.parametrize("count", [2, 12])
@pytest.mark.parametrize("attempt", [0, 1])
def test_starter_audit_accepts_complete_default_and_refined_batches(
    starter_audit, count, attempt
):
    starter_audit(
        ArtifactClient(_documents(count, attempt)), "example-bucket", "", count
    )


@pytest.mark.parametrize(
    "artifact,field",
    [
        ("cosmos_augmented/manifest.json", "variants"),
        ("configs/manifest.json", "augmentations"),
        ("grade/cosmos_evaluator.json", "clips"),
    ],
)
def test_starter_audit_rejects_missing_or_duplicate_variants(
    starter_audit, artifact, field
):
    documents = _documents(12)
    documents[artifact][field][-1] = documents[artifact][field][0]
    if field == "augmentations":
        documents[artifact][field].pop()
    with pytest.raises(AssertionError):
        starter_audit(ArtifactClient(documents), "example-bucket", "", 12)


@pytest.mark.parametrize("field", ["variant_count", "annotated_variant_count"])
def test_starter_audit_rejects_partial_finalization(starter_audit, field):
    documents = _documents(12)
    documents["reports/final.json"][field] = 2
    with pytest.raises(AssertionError):
        starter_audit(ArtifactClient(documents), "example-bucket", "", 12)


def test_starter_audit_rejects_recipe_tuning(starter_audit):
    documents = _documents(12)
    documents["cosmos_augmented/manifest.json"]["variants"][0]["steps"] = 35
    with pytest.raises(AssertionError):
        starter_audit(ArtifactClient(documents), "example-bucket", "", 12)


RECIPE_PROFILES = [
    {
        "lighting": f"lighting {index}",
        "background": f"tabletop {index}",
        "color_grade": f"palette {index}",
        "surface_finish": f"finish {index}",
    }
    for index in range(12)
]
RECIPE = {
    "variant_count": "12",
    "transfer_first_chunk_conditional_frames": "0",
    "appearance_profiles_json": json.dumps(RECIPE_PROFILES),
}


def _recipe_documents():
    documents = _documents(12)
    configs = documents["configs/manifest.json"]
    configs["appearance_profile_source"] = "custom"
    configs["appearance_profiles"] = json.loads(json.dumps(RECIPE_PROFILES))
    configs["augmentations"] = [dict(profile) for profile in RECIPE_PROFILES]
    for variant in documents["cosmos_augmented/manifest.json"]["variants"]:
        published = f"{variant['clip']}/_native/batch/video"
        variant["augmented_video_uri"] = (
            f"s3://example-bucket/{published}/augmented_video.mp4"
        )
        documents[f"{published}/transfer.json"] = {
            "first_chunk_conditional_frames": 0,
            "text_guardrail_passed": True,
            "video_guardrail_passed": True,
            "edge_threshold": "medium",
            "normalize_cfg": False,
            "output_fps": 24.0,
        }
    return documents


def _audit_recipe(starter_audit, documents):
    starter_audit(ArtifactClient(documents), "example-bucket", "", 12, RECIPE)


def _first_receipt(documents):
    return next(
        value for key, value in documents.items() if key.endswith("transfer.json")
    )


def test_starter_audit_accepts_matching_recipe_run(starter_audit):
    _audit_recipe(starter_audit, _recipe_documents())


def test_recipe_audit_rejects_default_profile_run(starter_audit):
    with pytest.raises(AssertionError):
        _audit_recipe(starter_audit, _documents(12))


def test_recipe_audit_rejects_changed_or_duplicate_profiles(starter_audit):
    changed = _recipe_documents()
    changed["configs/manifest.json"]["appearance_profiles"][0]["background"] = "other"
    with pytest.raises(AssertionError):
        _audit_recipe(starter_audit, changed)
    duplicated = _recipe_documents()
    augmentations = duplicated["configs/manifest.json"]["augmentations"]
    augmentations[-1] = dict(augmentations[0])
    with pytest.raises(AssertionError):
        _audit_recipe(starter_audit, duplicated)


@pytest.mark.parametrize(
    "field,value",
    [
        ("first_chunk_conditional_frames", 1),
        ("text_guardrail_passed", False),
        ("video_guardrail_passed", False),
        ("edge_threshold", "low"),
        ("normalize_cfg", True),
        ("output_fps", 30.0),
        ("rgb_conditioning", {"weight": 0.5}),
    ],
)
def test_recipe_audit_rejects_anchor_or_guardrail_receipt(starter_audit, field, value):
    documents = _recipe_documents()
    _first_receipt(documents)[field] = value
    with pytest.raises(AssertionError):
        _audit_recipe(starter_audit, documents)
