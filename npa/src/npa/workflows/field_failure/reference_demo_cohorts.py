"""Freeze balanced adaptation and retention cohorts before baseline learning."""

import json

from npa.workflows.navigation.artifacts import write_json
from npa.workflows.navigation.reference_identity import collision_identity
from npa.workflows.navigation.reference_replay import compose_scene, translated_cases

TRANSLATION = [50.0, 0.0, 0.0]


def freeze_balanced_cohorts(baseline, scan_cases, office):
    """Bind measured geometry and disjoint balanced evaluation without seeing outcomes.

    Args:
        baseline: Prepared warehouse baseline directory.
        scan_cases: Measured office training and held-out route inventory.
        office: Verified measured office scene USDZ.
    Returns:
        Scene identities, route hashes and per-region case/seed mappings.
    Raises:
        ValueError: Populations, metric geometry or route separation are invalid.
        OSError: Required source bytes cannot be read or written.
    """
    from npa.workflows.field_failure.reference_demo_inputs import cohort_manifest

    recipe = json.loads((baseline / "recipe.json").read_text())
    count = recipe["num_envs"]
    if count % 2 or any(
        len(scan_cases[key]) != count for key in ("train_cases", "eval_cases")
    ):
        raise ValueError("balanced reference requires equal even route populations")
    regions = _region_cohorts(baseline, recipe, scan_cases)
    combined = {name: _interleave(rows) for name, rows in regions.items()}
    manifest = cohort_manifest(combined)
    geometry = compose_scene(
        office, baseline / "scene.usdz", baseline / "composite.usdz"
    )
    geometry["collision_identity"] = collision_identity(baseline / "composite.usdz")
    manifest.update(
        regions={name: _region_manifest(rows) for name, rows in regions.items()},
        geometry=geometry,
        scope="new routes in both known public layouts; not new-site transfer",
        final_used_for_selection=False,
        cohort_seed=json.loads((baseline / "cohorts.json").read_text())["cohort_seed"],
    )
    write_json(baseline / "development-cases.json", combined["development"])
    write_json(baseline / "final-cases.json", combined["final"])
    write_json(baseline / "cohorts.json", manifest)
    return manifest


def _region_cohorts(baseline, recipe, scan):
    half = recipe["num_envs"] // 2
    office = translated_cases(scan["eval_cases"], TRANSLATION)
    warehouse_final = json.loads((baseline / "final-cases.json").read_text())
    return {
        "training": {
            "office": translated_cases(scan["train_cases"], TRANSLATION),
            "warehouse": recipe["train_cases"],
        },
        "development": {
            "office": office[:half],
            "warehouse": recipe["eval_cases"][:half],
        },
        "final": {"office": office[half:], "warehouse": warehouse_final[:half]},
    }


def _interleave(regions):
    # Robot zero previews adaptation in the office; metrics still cover both regions.
    return [
        row
        for pair in zip(regions["office"], regions["warehouse"], strict=True)
        for row in pair
    ]


def _region_manifest(regions):
    from npa.workflows.field_failure.reference_demo_inputs import cohort_manifest

    result = cohort_manifest(regions)["cohorts"]
    for name, rows in regions.items():
        result[name].update(
            case_ids=[row["id"] for row in rows], seeds=[row["seed"] for row in rows]
        )
    return result
