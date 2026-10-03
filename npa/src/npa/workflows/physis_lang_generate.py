"""Run paired Physis-Lang prompt ablations through the pinned native Wan 2.1 pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re

from npa.solutions.video_generation import generate_video, load_video_pipeline
from npa.workflows.physis_lang_artifacts import file_hash, write_json
from npa.workflows.physis_lang_contract import ARMS, SOLUTION, arm_prompts


def expected_grid(recipe: dict) -> set[tuple[str, int, str]]:
    """Validate the frozen recipe and enumerate its exact comparison grid.

    Args:
        recipe: Prepared experiment.
    Returns:
        Unique scenario, seed and arm identities.
    Raises:
        ValueError: Recipe identities or coverage are invalid.
    """
    if (
        recipe.get("schema") != "npa.physis-lang.recipe.v1"
        or recipe.get("solution") != SOLUTION
    ):
        raise ValueError("Unsupported Physis recipe")
    if recipe.get("arms") != list(ARMS):
        raise ValueError("Recipe must retain all three comparison arms")
    cases, seeds = recipe.get("cases", []), recipe.get("seeds", [])
    names = [case["id"] for case in cases]
    if (
        not names
        or len(names) != len(set(names))
        or not seeds
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("Recipe contains empty or duplicate scenarios/seeds")
    if any(
        not isinstance(name, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
        for name in names
    ):
        raise ValueError("Scenario identifiers must be safe artifact names")
    if any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
        raise ValueError("Recipe seeds are invalid")
    return {(name, seed, arm) for name in names for seed in seeds for arm in ARMS}


def _check_tokens(pipeline, prompt: str, negative: str) -> None:
    for text in (prompt, negative):
        tokens = pipeline.tokenizer(text, truncation=False, add_special_tokens=True)[
            "input_ids"
        ]
        if len(tokens) > 512:
            raise ValueError(
                "Physical conditioning exceeds Wan's 512-token encoder context"
            )


def _generate_one(case, arm, seed, pipeline, output: Path) -> dict:
    prompt, negative = arm_prompts(case, arm)
    _check_tokens(pipeline, prompt, negative)
    name = f"{case['id']}-seed-{seed}-{arm}"
    result = generate_video(
        SOLUTION,
        prompt,
        seed,
        output / name,
        negative_prompt=negative,
        pipeline=pipeline,
    )
    result.update(
        {"case_id": case["id"], "arm": arm, "relative_video": f"{name}/video.mp4"}
    )
    write_json(output / name / "generation.json", result)
    print(
        f"Generated {name}: {result['observed']['frame_count']} decoded frames",
        flush=True,
    )
    return result


def generate(prepared: Path, output: Path, seed: int) -> None:
    """Generate every arm/scenario for one seed with one resident GPU model.

    Args:
        prepared: Verified recipe directory.
        output: Empty local generation directory.
        seed: One seed declared in the frozen recipe.
    Returns:
        None.
    Raises:
        ValueError: Recipe, token lengths or selected seed are invalid.
        RuntimeError: Native GPU execution or media validation fails.
    """
    recipe = json.loads((prepared / "recipe.json").read_text())
    expected_grid(recipe)
    if seed not in recipe["seeds"]:
        raise ValueError("Generation seed was not sealed in the recipe")
    pipeline = load_video_pipeline(SOLUTION)
    requests = [(case, arm) for case in recipe["cases"] for arm in ARMS]
    random.Random(seed).shuffle(requests)
    results = [
        _generate_one(case, arm, seed, pipeline, output) for case, arm in requests
    ]
    write_json(output / "recipe.json", recipe)
    write_json(
        output / "generation.json",
        {
            "schema": "npa.physis-lang.generation.v1",
            "status": "completed",
            "recipe_sha256": file_hash(prepared / "recipe.json"),
            "seed": seed,
            "videos": results,
            "generation_order": [r["relative_video"] for r in results],
        },
    )


def main() -> None:
    """Execute a local GPU shard in the reviewed model runtime.

    Args:
        None; arguments come from the command line.
    Returns:
        None.
    Raises:
        ValueError: Invalid recipe or command arguments.
        RuntimeError: Model execution fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    generate(args.input_path, args.output_path, args.seed)


if __name__ == "__main__":
    main()
