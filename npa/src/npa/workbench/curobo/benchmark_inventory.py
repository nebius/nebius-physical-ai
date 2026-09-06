"""Compact identity inventory independently derived from pinned benchmark YAML.

Source: https://github.com/fishbotics/robometrics/tree/81e3d1d605de84100d8ab880b43096aba221a48b/robometrics/content/dataset
Counts and zero-based exclusions come from each YAML mapping's list length and
``collision_buffer_ik < 0`` values, not planner output. Dataset payloads are not
copied here. MotionBenchMaker is BSD-3-Clause and MPiNets is MIT; see the pinned
repository's Licenses file and the cuRobo image redistribution record.
"""

DATASET_FILES = {
    "motion_benchmaker": (
        "mb_set.yaml",
        "5165ad4fb55f93c63dbbdda6f25a14f3ceb24ee3305e27ad317f38f1e0d6ed0d",
    ),
    "mpinets": (
        "mpinets_set.yaml",
        "9189186d83e51600a2c768aa7657933aa052150501852771994757052b797bbf",
    ),
}

# group: (population, invalid zero-based indices)
BENCHMARK_GROUPS = {
    "motion_benchmaker": {
        "bookshelf_small_panda": (100, ()),
        "bookshelf_tall_panda": (100, ()),
        "bookshelf_thin_panda": (100, ()),
        "box_panda": (100, ()),
        "box_panda_flipped": (100, ()),
        "cage_panda": (100, ()),
        "table_pick_panda": (100, ()),
        "table_under_pick_panda": (100, (74,)),
    },
    "mpinets": {
        "cubby_neutral_goal": (75, ()),
        "cubby_neutral_start": (75, ()),
        "cubby_task_oriented": (150, ()),
        "dresser_neutral_goal": (150, ()),
        "dresser_neutral_start": (150, ()),
        "dresser_task_oriented": (300, (39, 248)),
        "merged_cubby_neutral_goal": (75, ()),
        "merged_cubby_neutral_start": (75, ()),
        "merged_cubby_task_oriented": (150, ()),
        "tabletop_neutral_goal": (150, ()),
        "tabletop_neutral_start": (150, (6, 125)),
        "tabletop_task_oriented": (300, (43, 66, 225, 257, 268)),
    },
}


def benchmark_identities(modes: list[str]) -> dict[tuple[str, str, str], bool]:
    """Map every exact requested identity to whether upstream excludes it."""
    return {
        (mode, dataset, f"{group}/{index}"): index in invalid
        for mode in modes
        for dataset, groups in BENCHMARK_GROUPS.items()
        for group, (count, invalid) in groups.items()
        for index in range(count)
    }
