"""Exercise configurable appearance sampling and source-grounded prompt delivery."""

from __future__ import annotations

import json

import pytest

from npa.workflows import data_factory_stages as stages
from npa.workflows.data_factory_appearance import parse_appearance_profiles
from npa.workflows.paidf_cosmos3 import _caption_text


PROFILES = [
    {"lighting": lighting, "background": "gray work surface",
     "color_grade": "neutral balanced color palette",
     "surface_finish": "matte low-gloss work surface"}
    for lighting in ("soft warm room lighting", "soft cool room lighting")
]


@pytest.mark.parametrize("value", [None, [], "null", "{}", "[]", "[", '["warm"]',
                                   '[{"lighting": "warm"}]'])
def test_profiles_reject_invalid_contract(value):
    with pytest.raises(ValueError, match="appearance"):
        parse_appearance_profiles(value)


@pytest.mark.parametrize("invalid", ["", "   ", 1, True, None, ["warm"]])
def test_profiles_reject_invalid_attribute_values(invalid):
    with pytest.raises(ValueError, match="non-empty strings"):
        parse_appearance_profiles(json.dumps([{**PROFILES[0], "lighting": invalid}]))


def test_profiles_reject_unknown_fields_and_normalized_duplicates():
    with pytest.raises(ValueError, match="exactly"):
        parse_appearance_profiles(json.dumps([{**PROFILES[0], "prompt": "replace scene"}]))
    duplicate = {key: " " + value + " " for key, value in PROFILES[0].items()}
    with pytest.raises(ValueError, match="duplicate"):
        parse_appearance_profiles(json.dumps([PROFILES[0], duplicate]))


def test_custom_profiles_drive_prompts_options_and_reproducible_sampling(tmp_path):
    manifests = [stages.generate_configs(
        str(tmp_path / f"{run}.json"), n_augmentations=5, seed=run,
        augmentation_seed="fixed", appearance_profiles_json=json.dumps(PROFILES),
        augment_subject="precision manipulation",
    ) for run in ("first", "second")]
    first = manifests[0]
    assert first["augmentations"] == manifests[1]["augmentations"]
    assert first["appearance_profile_source"] == "custom"
    assert first["appearance_profiles"] == PROFILES
    assert len({combo["inference_seed"] for combo in first["augmentations"]}) == 5
    assert {combo["lighting"] for combo in first["augmentations"][:2]} == {
        profile["lighting"] for profile in PROFILES
    }
    for combo in first["augmentations"]:
        for key in stages.APPEARANCE_VARIABLES:
            assert combo[key] in first["variables"][key]
            assert combo[key] in combo["prompt"]
        assert "precision manipulation" in combo["prompt"]
        assert "insertion openings" in combo["prompt"]
    saved = json.loads((tmp_path / "first.json").read_text())
    assert saved["appearance_profiles"] == first["appearance_profiles"]
    assert saved["augmentations"] == first["augmentations"]
    assert saved["variables"] == first["variables"]


def test_default_sampling_remains_available(tmp_path):
    assert parse_appearance_profiles(" ") is None
    manifest = stages.generate_configs(str(tmp_path / "default.json"), seed="baseline")
    assert manifest["appearance_profile_source"] == "default"
    assert manifest["variables"] == stages.APPEARANCE_VARIABLES
    assert all({key: combo[key] for key in stages.APPEARANCE_VARIABLES}
               in stages.APPEARANCE_PROFILES for combo in manifest["augmentations"])


def test_custom_profiles_reject_anchor_before_read_or_write(monkeypatch, tmp_path):
    def unexpected(*args):
        raise AssertionError("must validate before artifact access")

    monkeypatch.setattr(stages, "_derive_quality_anchor", unexpected)
    with pytest.raises(ValueError, match="quality anchor"):
        stages.generate_configs(str(tmp_path / "unused.json"),
                                quality_anchor_uri="s3://example-bucket/anchor.json",
                                appearance_profiles_json=json.dumps(PROFILES))
    assert not (tmp_path / "unused.json").exists()


@pytest.mark.parametrize("count", [1, 2, 4, 8, 16])
def test_caption_sampling_includes_action_outcome_in_order(count):
    selected = _caption_text({"captions": [{"caption": f"frame-{i}"} for i in range(count)]}).split()
    assert selected[0] == "frame-0"
    assert selected[-1] == f"frame-{count - 1}"
    assert len(selected) == min(4, count)
    assert selected == sorted(selected, key=lambda item: int(item.split("-")[1]))
