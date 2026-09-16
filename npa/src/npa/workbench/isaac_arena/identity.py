"""Immutable upstream identities and the supported Arena execution inventory."""

ISAAC_ARENA_VERSION = "0.3.0"
ISAAC_ARENA_REVISION = "ed0fd12be862078be316c73eb7cf423ba9b1c5cd"
ISAAC_ARENA_ARCHIVE_SHA256 = (
    "4e62ddbd7edc40fb47e62a0d5ba523eebc129482b4ce083612f17c79f1fc40a8"
)
LIGHTWHEEL_SDK_VERSION = "1.0.3"
ISAAC_ARENA_ROOT = "/opt/isaac-arena"
SOURCE_IDENTITY_SCHEMA = "npa.workbench.isaac_arena.source_identity.v1"
ARTIFACT_SCHEMA = "npa.workbench.isaac_arena.evaluation.v1"
CAPABILITIES_SCHEMA = "npa.workbench.isaac_arena.capabilities.v1"
SUPPORTED_POLICIES = frozenset({"zero_action", "replay", "rsl_rl"})
REGISTERED_ENVIRONMENTS = frozenset(
    {
        "cube_goal_pose",
        "dexsuite_lift",
        "droid_table_multi_object_placement",
        "franka_put_and_close_door",
        "galileo_g1_locomanip_pick_and_place",
        "galileo_pick_and_place",
        "gr1_open_microwave",
        "gr1_table_multi_object_no_collision",
        "gr1_turn_stand_mixer_knob",
        "kitchen_pick_and_place",
        "lift_object",
        "peg_insert",
        "pick_and_place_maple_table",
        "press_button",
        "put_item_in_fridge_and_close_door",
        "gear_mesh",
        "tabletop_place_upright",
        "tabletop_sort_cubes",
    }
)
UNSUPPORTED_ENVIRONMENTS = frozenset(
    {"droid_table_multi_object_placement", "gr1_table_multi_object_no_collision"}
)
SUPPORTED_ENVIRONMENTS = REGISTERED_ENVIRONMENTS - UNSUPPORTED_ENVIRONMENTS
