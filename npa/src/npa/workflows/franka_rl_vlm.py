"""Audit paired Franka rollouts with blinded Token Factory judgments and simulator comparisons."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from npa.clients.token_factory import TokenFactoryClient
from npa.workbench.vlm_eval.temporal import RUBRIC, RUBRIC_VERSION, judge_manipulation
from npa.workflows.lerobot_transfer_data import file_sha256, write_json
from npa.workflows.franka_rl_embodiments import validate_capture_embodiment


def _capture_contract(evaluated: Path) -> tuple[dict, dict, dict]:
    evaluation = json.loads((evaluated / "evaluation.json").read_text())
    metadata = json.loads((evaluated / "trajectories/meta.json").read_text())
    recipe = evaluation["recipe"]
    validate_capture_embodiment(metadata, recipe)
    protocol = recipe["visual_eval"]
    if (
        protocol["rubric_version"] != RUBRIC_VERSION
        or protocol["rubric_sha256"] != hashlib.sha256(RUBRIC.encode()).hexdigest()
    ):
        raise ValueError("VLM rubric differs from the sealed experiment")
    expected = {
        (arm, condition, index)
        for arm in protocol["arms"]
        for condition in recipe["conditions"]
        for index in range(recipe["capture_episodes"])
    }
    rows = metadata["episode_results"]
    actual = {(row["arm"], row["condition"], row["capture_index"]) for row in rows}
    if (
        len(actual) != len(rows)
        or actual != expected
        or metadata["num_episodes"] != len(rows)
    ):
        raise ValueError("VLM capture grid has missing, extra, or duplicate episodes")
    if (
        not metadata["genuine_simulator_pixels"]
        or metadata["source_split"] != "capture"
    ):
        raise ValueError(
            "VLM evaluation requires independently captured simulator pixels"
        )
    if metadata["run_id"] != recipe["run_id"] or metadata["assets"] != recipe["assets"]:
        raise ValueError("VLM capture run or simulation asset identity differs")
    for row in rows:
        expected_hash = (
            evaluation["selection"]["selected_checkpoint_sha256"]
            if row["arm"] == "trained"
            else (evaluation["training"]["checkpoints"]["initial.pt"])
        )
        if row["checkpoint_sha256"] != expected_hash:
            raise ValueError(
                "VLM capture does not use the claimed initial or selected checkpoint"
            )
        if row["reset_seed"] != recipe["capture_seed"] + row["capture_index"]:
            raise ValueError("VLM capture uses an unsealed reset seed")
        _check_physics(row, recipe)
    for index in range(recipe["capture_episodes"]):
        if (
            len(
                {
                    row["initial_state_sha256"]
                    for row in rows
                    if row["capture_index"] == index
                }
            )
            != 1
        ):
            raise ValueError("VLM capture resets are not paired")
    return evaluation, metadata, recipe


def _check_physics(row: dict, recipe: dict) -> None:
    if "simulation_validity" in recipe:
        actual = row["applied_physics"].get("simulation_validity", {})
        if (
            actual.get("verified") is not True
            or actual.get("contract") != recipe["simulation_validity"]
            or actual.get("checked_batches", 0) <= 0
        ):
            raise ValueError(
                "Capture lacks required measured simulation validity evidence"
            )
    if (
        "embodiment" in recipe
        and row["applied_physics"].get("embodiment", {}).get("profile")
        != recipe["embodiment"]
    ):
        raise ValueError("Capture physics belongs to a different robot embodiment")
    if (
        "physics_capacity" in recipe
        and row["applied_physics"].get("physics_capacity") != recipe["physics_capacity"]
    ):
        raise ValueError(
            "Franka capture physics capacity differs from the sealed experiment"
        )
    condition = recipe["conditions"][row["condition"]]
    expected = {
        "mass_kg": recipe["assets"]["nominal_mass_kg"] * condition["mass_scale"],
        "static_friction": condition["friction"],
        "dynamic_friction": condition["friction"],
    }
    for name, value in expected.items():
        actual = row["applied_physics"][name]
        if not np.allclose([actual["min"], actual["max"]], value, rtol=1e-5, atol=1e-6):
            raise ValueError("Franka capture physics differs from its named condition")


def _judge_episode(
    evaluated: Path,
    output: Path,
    index: int,
    metadata: dict,
    recipe: dict,
    client,
    previous: Path | None = None,
) -> dict:
    trajectory = evaluated / "trajectories" / f"episode_{index:06d}"
    row = metadata["episode_results"][index]
    pixels = np.load(trajectory / "rgb.npy", mmap_mode="r", allow_pickle=False)
    geometry = np.load(trajectory / "object_metrics.npy", allow_pickle=False)
    if (
        len(pixels) != row["length"]
        or geometry.shape != (len(pixels), 3)
        or not np.isfinite(geometry).all()
    ):
        raise ValueError("VLM frames and synchronized physical measurements disagree")
    target = recipe["assets"]["description"]
    task = f"Grasp the {target} initially on the table, lift it, and hold it steady. Leave the tray and other parts alone."
    result = judge_manipulation(
        rgb_path=trajectory / "rgb.npy",
        output=output / f"episode-{index:06d}",
        task=task,
        fps=metadata["fps"],
        frame_count=recipe["visual_eval"]["frame_count"],
        model=recipe["visual_eval"]["model"],
        client=client,
        previous=previous,
    )
    sampled = [frame["index"] for frame in result["frames"]]
    high = geometry[sampled, 2] > recipe["minimum_object_height_m"]
    return {
        "episode_index": index,
        **row,
        "visual": result,
        "reference_valid": (
            "simulation_validity" in recipe
            and row.get("applied_physics", {})
            .get("simulation_validity", {})
            .get("verified")
            is True
        ),
        "reference_lifted": bool(np.count_nonzero(high) >= 2),
        "reference_sample_indices": sampled,
        "rgb_sha256": file_sha256(trajectory / "rgb.npy"),
        "geometry_sha256": file_sha256(trajectory / "object_metrics.npy"),
    }


def summarize_visual(rows: list[dict], recipe: dict) -> dict:
    """Compare visual judgments with physical lift evidence without overriding strict success.

    Args:
        rows: Completed per-episode hosted judgments and synchronized simulator references.
        recipe: Sealed thresholds and condition identities.
    Returns:
        Confusion counts, per-arm visual outcomes, and explicit audit and visual quality gates.
    Raises:
        ValueError: The audit contains no episodes.
    """
    if not rows:
        raise ValueError("Cannot summarize an empty Franka visual audit")
    _validate_judgment_grid(rows, recipe)
    reference_valid = [row for row in rows if row.get("reference_valid", False)]
    confusion = _reference_confusion(reference_valid)
    positive = sum(row["reference_lifted"] for row in reference_valid)
    negative = len(reference_valid) - positive
    sensitivity = confusion["positive_yes"] / positive if positive else None
    specificity = confusion["negative_no"] / negative if negative else None
    balanced = (sensitivity + specificity) / 2 if positive and negative else None
    rates = _visual_rates(rows, recipe)
    invalid = sum(row["visual"]["verdict"] is None for row in rows)
    valid_physics = len(reference_valid) == len(rows)
    audit_passed = (
        valid_physics
        and not invalid
        and balanced is not None
        and balanced >= recipe["visual_eval"]["minimum_lift_agreement"]
    )
    visual_passed = valid_physics and all(
        row["held_at_end_rate"] >= recipe["minimum_success"]
        and row["scene_disturbed_count"] == 0
        and row["scene_uncertain_count"] == 0
        and row["invalid_count"] == 0
        for row in rates["trained"].values()
    )
    return {
        "episodes": len(rows),
        "invalid_responses": invalid,
        "lift_confusion": dict(confusion),
        "lift_sensitivity": sensitivity,
        "reference_valid_episodes": len(reference_valid),
        "invalid_or_unverified_physics_episodes": len(rows) - len(reference_valid),
        "lift_specificity": specificity,
        "lift_balanced_accuracy": balanced,
        "outcomes": rates,
        "visual_audit_passed": audit_passed,
        "visual_task_passed": visual_passed,
        "quality_role": "Additional gate; cannot override failed simulator success",
        "calibrated_on_independent_human_labels": False,
        "limitations": [
            "Sampled monocular frames cannot verify exact speed, force, goal distance, or continuous stability.",
            "Agreement uses object height in the same supplied frames as an elevation proxy; it does not verify grasp or strict lift-and-hold success.",
            "Capture episodes are independent of held-out test episodes; their denominators differ.",
        ],
    }


def _reference_confusion(rows):
    confusion = Counter()
    for row in rows:
        verdict = row["visual"]["verdict"]
        prediction = verdict["lifted"]["verdict"] if verdict is not None else "invalid"
        reference = "positive" if row["reference_lifted"] else "negative"
        confusion[f"{reference}_{prediction}"] += 1
    return confusion


def _validate_judgment_grid(rows: list[dict], recipe: dict) -> None:
    expected = {
        (arm, condition, index)
        for arm in recipe["visual_eval"]["arms"]
        for condition in recipe["conditions"]
        for index in range(recipe["capture_episodes"])
    }
    actual = {(row["arm"], row["condition"], row["capture_index"]) for row in rows}
    if actual != expected or len(actual) != len(rows):
        raise ValueError(
            "Franka visual judgments have missing, extra, or duplicate episode coverage"
        )
    for row in rows:
        if "reference_valid" in row and type(row["reference_valid"]) is not bool:
            raise ValueError("Capture physical reference validity must be boolean")
        visual = row["visual"]
        status = visual.get("status", "valid")
        if (
            status not in {"valid", "invalid_response"}
            or (status == "invalid_response") != (visual["verdict"] is None)
            or (status == "invalid_response" and not visual.get("validation_error"))
        ):
            raise ValueError("Franka visual judgment validation status is inconsistent")
        if (
            visual["model"] != recipe["visual_eval"]["model"]
            or visual["backend"] != "token_factory"
            or visual["rubric_sha256"] != recipe["visual_eval"]["rubric_sha256"]
            or type(row["reference_lifted"]) is not bool
        ):
            raise ValueError(
                "Franka visual judgment model, rubric, or physical reference differs"
            )


def _visual_rates(rows: list[dict], recipe: dict) -> dict:
    rates = {}
    for arm in recipe["visual_eval"]["arms"]:
        rates[arm] = {}
        for condition in recipe["conditions"]:
            selected = [
                row
                for row in rows
                if row["arm"] == arm and row["condition"] == condition
            ]
            verdicts = [
                row["visual"]["verdict"]
                for row in selected
                if row["visual"]["verdict"] is not None
            ]
            rates[arm][condition] = {
                "episodes": len(selected),
                "invalid_count": len(selected) - len(verdicts),
                "lifted_rate": sum(v["lifted"]["verdict"] == "yes" for v in verdicts)
                / len(selected),
                "held_at_end_rate": sum(
                    v["held_at_end"]["verdict"] == "yes" for v in verdicts
                )
                / len(selected),
                "scene_disturbed_count": sum(
                    v["scene_disturbed"]["verdict"] == "yes" for v in verdicts
                ),
                "scene_uncertain_count": sum(
                    v["scene_disturbed"]["verdict"] == "uncertain" for v in verdicts
                ),
                "uncertain_count": sum(
                    any(
                        v[key]["verdict"] == "uncertain"
                        for key in ("lifted", "held_at_end", "scene_disturbed")
                    )
                    for v in verdicts
                ),
                "failure_modes": dict(
                    Counter(tag for v in verdicts for tag in v["failure_modes"])
                ),
            }
    return rates


def _prior_responses(
    previous: Path | None, evaluated: Path, metadata: dict
) -> dict[int, Path]:
    if previous is None:
        return {}
    contract = json.loads((previous / "capture-contract.json").read_text())
    if contract != {
        "evaluation_sha256": file_sha256(evaluated / "evaluation.json"),
        "capture_sha256": file_sha256(evaluated / "trajectories/meta.json"),
    }:
        raise ValueError(
            "Prior VLM judgments belong to a different evaluation or capture"
        )
    recorded = {}
    for episode in previous.glob("episode-*"):
        index = int(episode.name.removeprefix("episode-"))
        if (
            not 0 <= index < metadata["num_episodes"]
            or index in recorded
            or not (episode / "response.json").is_file()
        ):
            raise ValueError(
                "Prior VLM judgments contain incomplete or duplicate responses"
            )
        recorded[index] = episode
    return recorded


def evaluate_captures(
    evaluated: Path, output: Path, *, previous: Path | None = None
) -> None:
    """Run the actual hosted judge for every sealed capture and publish a paired audit.

    Args:
        evaluated: Verified evaluation stage with policy captures and physical telemetry.
        output: New visual evaluation artifact directory.
        previous: Optional checksum-verified interrupted audit with a matching capture contract.
    Returns:
        None.
    Raises:
        ValueError: Capture lineage, model availability, or prior evidence is invalid.
        TokenFactoryError: Hosted model access or inference fails.
        OSError: Input or output files cannot be accessed.
    """
    _, metadata, recipe = _capture_contract(evaluated)
    recorded = _prior_responses(previous, evaluated, metadata)
    client = TokenFactoryClient()
    if recipe["visual_eval"]["model"] not in client.list_models():
        raise ValueError(
            "Sealed VLM model is unavailable to the Token Factory credential"
        )
    output.mkdir(parents=True)
    write_json(
        output / "capture-contract.json",
        {
            "evaluation_sha256": file_sha256(evaluated / "evaluation.json"),
            "capture_sha256": file_sha256(evaluated / "trajectories/meta.json"),
        },
    )
    rows = []
    order = np.random.default_rng(recipe["capture_seed"]).permutation(
        metadata["num_episodes"]
    )
    for index in order.tolist():
        rows.append(
            _judge_episode(
                evaluated, output, index, metadata, recipe, client, recorded.get(index)
            )
        )
        print(f"Token Factory judged capture {len(rows)}/{len(order)}", flush=True)
    rows.sort(key=lambda row: row["episode_index"])
    report = {
        "schema": "npa.franka-rl.visual-evaluation.v1",
        "recipe": recipe,
        "evaluation_sha256": file_sha256(evaluated / "evaluation.json"),
        "capture_sha256": file_sha256(evaluated / "trajectories/meta.json"),
        "summary": summarize_visual(rows, recipe),
        "episodes": rows,
    }
    write_json(output / "visual-evaluation.json", report)
