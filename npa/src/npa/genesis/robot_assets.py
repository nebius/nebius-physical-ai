"""RobotSpec schema, presets, asset download, and provenance for BYO robots.

This module describes the *robot embodiment* of a pick-and-place scene
independently of the simulator, mirroring the object-side ``scene_assets``
module. It is deliberately free of ``torch`` / ``genesis`` / ``isaaclab``
imports at module level so it can be unit-tested without a GPU.

A RobotSpec selects a ``robot_source``:

- ``stock_franka`` (default): the Franka Emika Panda MJCF shipped inside the
  Genesis package (``xml/franka_emika_panda/panda.xml``). No download. This
  reproduces today's hardcoded robot byte-for-byte.
- ``byo_urdf`` / ``byo_mjcf`` / ``byo_usd``: a customer arm fetched from an
  S3/URI as an *articulated* description (URDF + meshes, MJCF, or USD). The
  file is downloaded, hashed, and loaded; provenance records
  ``{robot_source, robot_uri, local_path, sha256, loaded}``. A BYO robot that
  fails to load raises ``RobotSpecError`` — there is **no silent fallback to
  Franka**.
- ``genesis_builtin``: an articulated robot description shipped inside the
  Genesis ``assets`` tree (e.g. a built-in UR/xarm URDF), resolved by path.

The robot must be an articulated description (URDF/MJCF/USD). A plain visual
mesh (``.obj`` / ``.glb`` / ``.stl`` / ``.ply``) is only valid for the
manipulated *object*, never the robot; supplying one as the robot raises a
clear ``RobotSpecError``.

Control/morphology config (``ee_link``, ``finger_links``, ``dof_count``,
``kp`` / ``kv`` gains, force ranges, ``home_qpos``) lets the env operate on a
configured embodiment instead of Franka constants. Presets are provided for
Franka and for UR/Flexiv-class arms so a customer can pick a preset or supply
a full URDF plus minimal config.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from npa.genesis.scene_assets import download_asset, sha256_file

ROBOT_SPEC_SCHEMA = "npa.sim2real.robot_spec.v1"

ROBOT_SOURCE_STOCK_FRANKA = "stock_franka"
ROBOT_SOURCE_BYO_URDF = "byo_urdf"
ROBOT_SOURCE_BYO_MJCF = "byo_mjcf"
ROBOT_SOURCE_BYO_USD = "byo_usd"
ROBOT_SOURCE_GENESIS_BUILTIN = "genesis_builtin"
ROBOT_SOURCES = (
    ROBOT_SOURCE_STOCK_FRANKA,
    ROBOT_SOURCE_BYO_URDF,
    ROBOT_SOURCE_BYO_MJCF,
    ROBOT_SOURCE_BYO_USD,
    ROBOT_SOURCE_GENESIS_BUILTIN,
)

# BYO sources that require a downloadable, articulated robot description.
ROBOT_SOURCES_BYO = (
    ROBOT_SOURCE_BYO_URDF,
    ROBOT_SOURCE_BYO_MJCF,
    ROBOT_SOURCE_BYO_USD,
)

# Articulated description suffixes accepted per source. A robot MUST be one of
# these — a plain visual mesh (.obj/.glb/.stl/.ply/.dae) is rejected loudly.
URDF_SUFFIXES = (".urdf",)
MJCF_SUFFIXES = (".xml", ".mjcf")
USD_SUFFIXES = (".usd", ".usda", ".usdc")
ARTICULATED_SUFFIXES = URDF_SUFFIXES + MJCF_SUFFIXES + USD_SUFFIXES
# Visual-mesh suffixes that are valid for OBJECTS but never for the ROBOT.
VISUAL_MESH_SUFFIXES = (".obj", ".glb", ".gltf", ".ply", ".stl", ".dae")

_SOURCE_SUFFIXES = {
    ROBOT_SOURCE_BYO_URDF: URDF_SUFFIXES,
    ROBOT_SOURCE_BYO_MJCF: MJCF_SUFFIXES,
    ROBOT_SOURCE_BYO_USD: USD_SUFFIXES,
}

# Stock Franka MJCF shipped with Genesis (the path the env hardcodes today).
STOCK_FRANKA_MJCF = "xml/franka_emika_panda/panda.xml"

# Franka Panda home joint configuration (ready pose) — matches FRANKA_HOME in
# env_pick_place.py so the default path is byte-for-byte identical.
FRANKA_HOME = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04)


class RobotSpecError(RuntimeError):
    """Raised when a RobotSpec is malformed or a robot asset cannot load."""


@dataclass
class RobotSpec:
    """A parsed, simulator-agnostic robot embodiment description."""

    robot_source: str = ROBOT_SOURCE_STOCK_FRANKA
    name: str = "franka_panda"
    # Source references
    robot_uri: str = ""  # byo_* source (s3:// or local path)
    builtin_path: str = ""  # genesis_builtin path relative to <genesis>/assets
    # Morphology / control
    ee_link: str = "hand"  # end-effector link name (IK + ee_pos)
    base_link: str = "panda_link0"  # articulation root link (ee_frame source frame)
    finger_links: tuple[str, ...] = ("left_finger", "right_finger")
    n_arm_joints: int = 7
    n_gripper_joints: int = 2
    joint_names: tuple[str, ...] = ()
    kp: tuple[float, ...] = (4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100)
    kv: tuple[float, ...] = (450, 450, 350, 350, 200, 200, 200, 10, 10)
    force_lower: tuple[float, ...] = (-87, -87, -87, -87, -12, -12, -12, -100, -100)
    force_upper: tuple[float, ...] = (87, 87, 87, 87, 12, 12, 12, 100, 100)
    home_qpos: tuple[float, ...] = FRANKA_HOME
    gripper_open: float = 0.04
    gripper_close: float = 0.0
    # Isaac-side hint: the task whose robot this preset corresponds to, used to
    # pick a sane default action/observation contract. Optional metadata only.
    isaac_robot_hint: str = ""
    # Resolved fields (populated by resolve_robot_asset)
    local_path: str = ""
    sha256: str = ""
    loaded: bool = False

    @property
    def dof_count(self) -> int:
        return self.n_arm_joints + self.n_gripper_joints

    @property
    def has_gripper(self) -> bool:
        return self.n_gripper_joints > 0 and bool(self.finger_links)

    def is_stock_franka(self) -> bool:
        return self.robot_source == ROBOT_SOURCE_STOCK_FRANKA

    def is_byo(self) -> bool:
        return self.robot_source in ROBOT_SOURCES_BYO

    def requires_download(self) -> bool:
        return self.robot_source in ROBOT_SOURCES_BYO

    def provenance(self) -> dict[str, Any]:
        """Robot provenance record (mirrors the object asset provenance)."""

        return {
            "schema": "npa.sim2real.robot_provenance.v1",
            "name": self.name,
            "robot_source": self.robot_source,
            "robot_uri": self.robot_uri,
            "builtin_path": self.builtin_path,
            "local_path": self.local_path,
            "sha256": self.sha256,
            "loaded": bool(self.loaded),
            "ee_link": self.ee_link,
            "dof_count": self.dof_count,
            "n_arm_joints": self.n_arm_joints,
            "n_gripper_joints": self.n_gripper_joints,
            # No silent fallback to Franka is ever permitted for a BYO robot.
            "robot_fallback_used": False,
        }

    def validate(self) -> None:
        """Validate source/articulation invariants. Raises RobotSpecError."""

        if self.robot_source not in ROBOT_SOURCES:
            raise RobotSpecError(
                f"robot_source must be one of {ROBOT_SOURCES}, "
                f"got {self.robot_source!r}"
            )
        if self.n_arm_joints <= 0:
            raise RobotSpecError("n_arm_joints must be positive")
        if self.n_gripper_joints < 0:
            raise RobotSpecError("n_gripper_joints must be non-negative")
        for gain, label in ((self.kp, "kp"), (self.kv, "kv")):
            if len(gain) != self.dof_count:
                raise RobotSpecError(
                    f"{label} must have dof_count ({self.dof_count}) entries, "
                    f"got {len(gain)}"
                )
        if len(self.force_lower) != self.dof_count or len(self.force_upper) != self.dof_count:
            raise RobotSpecError(
                f"force_lower/force_upper must have dof_count ({self.dof_count}) "
                "entries"
            )
        if len(self.home_qpos) != self.dof_count:
            raise RobotSpecError(
                f"home_qpos must have dof_count ({self.dof_count}) entries, "
                f"got {len(self.home_qpos)}"
            )
        if not self.ee_link:
            raise RobotSpecError("ee_link must be a non-empty link name")
        if self.is_byo():
            if not self.robot_uri:
                raise RobotSpecError(
                    f"robot_source={self.robot_source} requires a non-empty robot_uri"
                )
            _validate_articulated_uri(self.robot_uri, self.robot_source)
        elif self.robot_source == ROBOT_SOURCE_GENESIS_BUILTIN and not self.builtin_path:
            raise RobotSpecError("genesis_builtin robot requires builtin_path")


def _suffix_of(uri: str) -> str:
    return Path(str(uri).split("?", 1)[0].rstrip("/")).suffix.lower()


def _infer_robot_source_from_uri(uri: str) -> str | None:
    """Map a robot asset URI suffix to the matching BYO robot_source."""

    suffix = _suffix_of(uri)
    if suffix in URDF_SUFFIXES:
        return ROBOT_SOURCE_BYO_URDF
    if suffix in MJCF_SUFFIXES:
        return ROBOT_SOURCE_BYO_MJCF
    if suffix in USD_SUFFIXES:
        return ROBOT_SOURCE_BYO_USD
    return None


def _validate_articulated_uri(uri: str, robot_source: str) -> None:
    """Reject a non-articulated file given as the robot (clear error)."""

    suffix = _suffix_of(uri)
    if suffix in VISUAL_MESH_SUFFIXES:
        raise RobotSpecError(
            f"robot asset {uri!r} is a visual mesh ({suffix}); the ROBOT requires "
            "an articulated URDF/MJCF/USD description (UR/Flexiv publish these). "
            "A plain mesh/OBJ is only valid for a manipulated OBJECT, not the robot."
        )
    expected = _SOURCE_SUFFIXES.get(robot_source, ARTICULATED_SUFFIXES)
    if suffix and suffix not in expected:
        raise RobotSpecError(
            f"robot_source={robot_source} expects one of {expected} for robot_uri, "
            f"got {suffix!r} ({uri!r})"
        )
    if not suffix:
        raise RobotSpecError(
            f"robot_uri {uri!r} has no file extension; the robot must be an "
            f"articulated description ({ARTICULATED_SUFFIXES})"
        )


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #

# Franka Panda — the current default. Values mirror env_pick_place.py exactly.
_FRANKA_PRESET = RobotSpec(
    robot_source=ROBOT_SOURCE_STOCK_FRANKA,
    name="franka_panda",
    ee_link="hand",
    finger_links=("left_finger", "right_finger"),
    n_arm_joints=7,
    n_gripper_joints=2,
    kp=(4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100),
    kv=(450, 450, 350, 350, 200, 200, 200, 10, 10),
    force_lower=(-87, -87, -87, -87, -12, -12, -12, -100, -100),
    force_upper=(87, 87, 87, 87, 12, 12, 12, 100, 100),
    home_qpos=FRANKA_HOME,
    gripper_open=0.04,
    gripper_close=0.0,
    isaac_robot_hint="franka",
)

# Universal Robots UR5e — 6-DOF arm, flange end-effector link ``tool0``. UR
# arms ship without a gripper in the base ur_description URDF, so there are no
# finger links by default (gripper control is a BYO follow-up). Gains are
# conservative position-control defaults sized to the 6 arm joints.
_UR5E_PRESET = RobotSpec(
    robot_source=ROBOT_SOURCE_BYO_URDF,
    name="ur5e",
    ee_link="tool0",
    base_link="base_link",
    finger_links=(),
    n_arm_joints=6,
    n_gripper_joints=0,
    joint_names=(
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ),
    kp=(3000, 3000, 2000, 1500, 1000, 1000),
    kv=(300, 300, 200, 150, 100, 100),
    force_lower=(-150, -150, -150, -28, -28, -28),
    force_upper=(150, 150, 150, 28, 28, 28),
    home_qpos=(0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0),
    isaac_robot_hint="ur",
)

# Universal Robots UR10e — 6-DOF, larger payload/torques than the UR5e.
_UR10E_PRESET = replace(
    _UR5E_PRESET,
    name="ur10e",
    kp=(4500, 4500, 3000, 2000, 1500, 1500),
    kv=(450, 450, 300, 200, 150, 150),
    force_lower=(-330, -330, -150, -56, -56, -56),
    force_upper=(330, 330, 150, 56, 56, 56),
)

# Flexiv Rizon — 7-DOF serial arm, tool flange link ``flange``. Like the UR
# arms, the published URDF has no integrated gripper, so finger control is a
# BYO follow-up. Gains sized to the 7 arm joints.
_FLEXIV_RIZON_PRESET = RobotSpec(
    robot_source=ROBOT_SOURCE_BYO_URDF,
    name="flexiv_rizon",
    ee_link="flange",
    base_link="base_link",
    finger_links=(),
    n_arm_joints=7,
    n_gripper_joints=0,
    joint_names=(
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ),
    kp=(3000, 3000, 2500, 2500, 1500, 1500, 1000),
    kv=(300, 300, 250, 250, 150, 150, 100),
    force_lower=(-123, -123, -64, -64, -39, -39, -39),
    force_upper=(123, 123, 64, 64, 39, 39, 39),
    home_qpos=(0.0, -0.698, 0.0, 1.571, 0.0, 0.698, 0.0),
    isaac_robot_hint="flexiv",
)

ROBOT_PRESETS: dict[str, RobotSpec] = {
    "franka": _FRANKA_PRESET,
    "stock_franka": _FRANKA_PRESET,
    "ur5e": _UR5E_PRESET,
    "ur10e": _UR10E_PRESET,
    "flexiv": _FLEXIV_RIZON_PRESET,
    "flexiv_rizon": _FLEXIV_RIZON_PRESET,
    "rizon": _FLEXIV_RIZON_PRESET,
}


def default_franka_robot_spec() -> RobotSpec:
    """Return the stock Franka RobotSpec (today's hardcoded embodiment)."""

    return replace(_FRANKA_PRESET)


def robot_spec_from_preset(name: str) -> RobotSpec:
    """Return a copy of a named preset (franka/ur5e/ur10e/flexiv)."""

    key = str(name or "").strip().lower()
    if key not in ROBOT_PRESETS:
        raise RobotSpecError(
            f"unknown robot preset {name!r}; known presets: "
            f"{sorted(set(ROBOT_PRESETS))}"
        )
    return replace(ROBOT_PRESETS[key])


def _coerce_float_tuple(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise RobotSpecError(f"{label} must be a list of numbers, got {value!r}")
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise RobotSpecError(f"{label} must be numeric: {value!r}") from exc


def _coerce_str_tuple(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)):
        raise RobotSpecError(f"{label} must be a list of strings, got {value!r}")
    return tuple(str(v) for v in value)


def parse_robot_spec(doc: dict[str, Any]) -> RobotSpec:
    """Parse and validate a RobotSpec JSON document into a RobotSpec.

    A ``preset`` key (franka/ur5e/ur10e/flexiv) seeds the defaults; any other
    keys override the preset, so a customer can supply only a URDF uri plus a
    couple of overrides on top of, say, the ``ur5e`` preset.
    """

    if not isinstance(doc, dict):
        raise RobotSpecError(f"RobotSpec must be a JSON object, got {type(doc)!r}")

    preset_name = str(doc.get("preset") or "").strip().lower()
    spec = robot_spec_from_preset(preset_name) if preset_name else RobotSpec()

    explicit_source = doc.get("robot_source") is not None
    if explicit_source:
        spec.robot_source = str(doc["robot_source"]).strip()
    if doc.get("name"):
        spec.name = str(doc["name"]).strip()
    if doc.get("robot_uri") is not None:
        spec.robot_uri = str(doc["robot_uri"]).strip()
    if not explicit_source and spec.robot_uri:
        inferred = _infer_robot_source_from_uri(spec.robot_uri)
        if inferred is not None:
            spec.robot_source = inferred
    if doc.get("builtin_path") is not None:
        spec.builtin_path = str(doc["builtin_path"]).strip()
    if doc.get("ee_link"):
        spec.ee_link = str(doc["ee_link"]).strip()
    if "finger_links" in doc:
        spec.finger_links = _coerce_str_tuple(doc["finger_links"], "finger_links")
    if doc.get("joint_names") is not None:
        spec.joint_names = _coerce_str_tuple(doc["joint_names"], "joint_names")
    if doc.get("n_arm_joints") is not None:
        spec.n_arm_joints = int(doc["n_arm_joints"])
    if doc.get("n_gripper_joints") is not None:
        spec.n_gripper_joints = int(doc["n_gripper_joints"])
    if doc.get("kp") is not None:
        spec.kp = _coerce_float_tuple(doc["kp"], "kp")
    if doc.get("kv") is not None:
        spec.kv = _coerce_float_tuple(doc["kv"], "kv")
    if doc.get("force_lower") is not None:
        spec.force_lower = _coerce_float_tuple(doc["force_lower"], "force_lower")
    if doc.get("force_upper") is not None:
        spec.force_upper = _coerce_float_tuple(doc["force_upper"], "force_upper")
    if doc.get("home_qpos") is not None:
        spec.home_qpos = _coerce_float_tuple(doc["home_qpos"], "home_qpos")
    if doc.get("gripper_open") is not None:
        spec.gripper_open = float(doc["gripper_open"])
    if doc.get("gripper_close") is not None:
        spec.gripper_close = float(doc["gripper_close"])
    if doc.get("isaac_robot_hint"):
        spec.isaac_robot_hint = str(doc["isaac_robot_hint"]).strip()

    spec.validate()
    return spec


def adapt_robot_spec_for_sim_backend(spec: RobotSpec, sim_backend: str) -> RobotSpec:
    """Prefer URDF over MJCF when the held-out backend is Isaac Lab."""

    backend = str(sim_backend or "").strip().lower()
    if backend != "isaac" or not spec.robot_uri:
        return spec
    if spec.robot_source != ROBOT_SOURCE_BYO_MJCF:
        return spec
    suffix = _suffix_of(spec.robot_uri)
    if suffix not in MJCF_SUFFIXES:
        return spec
    stem = str(spec.robot_uri).rsplit(".", 1)[0]
    return replace(spec, robot_uri=f"{stem}.urdf", robot_source=ROBOT_SOURCE_BYO_URDF)


def robot_spec_from_inputs(
    *,
    robot_spec_uri: str = "",
    robot_source: str = "",
    robot_preset: str = "",
) -> RobotSpec | None:
    """Build a RobotSpec from loose CLI/env inputs (no download).

    Resolution order:
    - ``robot_spec_uri``: handled by the caller (downloads + parses JSON).
    - ``robot_preset``: a named preset (ur5e/ur10e/flexiv/franka).
    - ``robot_source``: a bare source (e.g. byo_urdf) — caller must also supply
      a uri via the spec; used mainly for the stock_franka default.

    Returns ``None`` when nothing robot-related is requested (default Franka).
    """

    preset = str(robot_preset or "").strip().lower()
    source = str(robot_source or "").strip().lower()
    if preset:
        return robot_spec_from_preset(preset)
    if source and source != ROBOT_SOURCE_STOCK_FRANKA:
        spec = RobotSpec(robot_source=source)
        return spec
    if source == ROBOT_SOURCE_STOCK_FRANKA:
        return default_franka_robot_spec()
    return None


def resolve_robot_asset(
    spec: RobotSpec,
    *,
    dest_dir: str | Path,
    client: Any = None,
    endpoint_url: str = "",
    downloader: Any = None,
    builtin_resolver: Any = None,
) -> RobotSpec:
    """Download/resolve the robot description, computing local_path + sha256.

    Mutates and returns ``spec``. BYO robots are downloaded and hashed;
    genesis_builtin robots are resolved under the Genesis assets tree;
    stock_franka resolves to the built-in MJCF path with no download. A robot
    that cannot be resolved raises ``RobotSpecError`` — there is no silent
    fallback to Franka.
    """

    spec.validate()
    downloader = downloader or download_asset
    dest_dir = Path(dest_dir)

    if spec.robot_source == ROBOT_SOURCE_STOCK_FRANKA:
        # Resolved lazily inside the GPU image (built-in MJCF). Record the
        # built-in path as provenance; no download / sha256.
        spec.local_path = ""
        spec.builtin_path = spec.builtin_path or STOCK_FRANKA_MJCF
        return spec

    if spec.is_byo():
        dest_dir.mkdir(parents=True, exist_ok=True)
        local = downloader(
            spec.robot_uri,
            dest_dir / _safe_name(spec.name),
            client=client,
            endpoint_url=endpoint_url,
        )
        # Re-validate the *downloaded* file is articulated (defense in depth).
        _validate_articulated_uri(str(local), spec.robot_source)
        spec.local_path = str(local)
        spec.sha256 = sha256_file(local)
        return spec

    # genesis_builtin
    builtin_resolver = builtin_resolver or _resolve_genesis_builtin_robot
    local = builtin_resolver(spec.builtin_path)
    local_path = Path(local)
    if not local_path.is_file() or local_path.stat().st_size == 0:
        raise RobotSpecError(
            f"genesis_builtin robot not resolved to a file: {spec.builtin_path}"
        )
    spec.local_path = str(local_path)
    spec.sha256 = sha256_file(local_path)
    return spec


def _resolve_genesis_builtin_robot(builtin_path: str) -> Path:
    """Resolve a builtin robot path relative to the installed Genesis package."""

    builtin_path = str(builtin_path or "").strip().lstrip("/")
    if not builtin_path:
        raise RobotSpecError("genesis_builtin robot requires a builtin_path")
    import genesis as gs  # noqa: PLC0415 - lazy, GPU image only

    assets_root = Path(gs.__file__).resolve().parent / "assets"
    candidate = assets_root / builtin_path
    if not candidate.is_file():
        raise RobotSpecError(
            f"genesis_builtin robot not found under {assets_root}: {builtin_path}"
        )
    return candidate


def _safe_name(value: str) -> str:
    chars = [c if c.isalnum() or c in ("-", "_", ".") else "-" for c in str(value)]
    return "".join(chars) or "robot"
