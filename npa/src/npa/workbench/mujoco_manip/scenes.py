"""Programmatic MJCF scene builders for contact-rich manipulation tasks.

Every scene is generated as an XML string: no downloads, no asset servers, so
environments load on machines without network access.  Each scene models a
position-controlled wrist (three slide joints plus a yaw hinge) carrying a tool
over a static fixture.  Contacts between the tool and the fixture are resolved
by MuJoCo's real contact solver; the scripted policies in
:mod:`npa.workbench.mujoco_manip.policies` drive the wrist through the
actuated joints.

Shared conventions:

* The wrist body carries joints ``wx`` / ``wy`` / ``wz`` (slides) and
  ``wyaw`` (hinge).  A 4-D action in ``[-1, 1]`` maps to a Cartesian target
  delta per control step.
* A ``tip`` site marks the tool tip; success criteria are computed from live
  simulation state, never from the policy's intentions.
* ``TASK_SPECS`` holds the geometry constants the XML builders, environments,
  and scripted policies share, so a number cannot drift between them.
"""

from __future__ import annotations

PEG_INSERTION = "peg_insertion"
SCREW_DRIVING = "screw_driving"
REACH = "reach"

TASKS: tuple[str, ...] = (PEG_INSERTION, SCREW_DRIVING, REACH)


def _f(spec: dict[str, object], key: str) -> float:
    value = spec[key]
    assert isinstance(value, float), f"TASK_SPECS[{key}] must be a float"
    return value


def _xyz(spec: dict[str, object], key: str) -> tuple[float, float, float]:
    value = spec[key]
    assert (
        isinstance(value, tuple) and len(value) == 3
    ), f"TASK_SPECS[{key}] must be an (x, y, z) tuple"
    return (float(value[0]), float(value[1]), float(value[2]))

#: Per-task geometry shared by the scene builders, envs, and scripted policies.
TASK_SPECS: dict[str, dict[str, object]] = {
    PEG_INSERTION: {
        # Plate: 0.10 x 0.10 x 0.02, top surface at z=0.10, square hole
        # with half-width 0.012 (peg radius 0.008 -> 4 mm clearance/side).
        "plate_top_z": 0.10,
        "plate_thickness": 0.02,
        "hole_half": 0.012,
        "peg_radius": 0.008,
        "peg_length": 0.12,
        # The peg is rigidly attached under the wrist; tip = wrist_z - 0.18.
        "tip_offset_z": -0.18,
        "hole_center": (0.0, 0.0),
        # Success: tip 3 cm below the plate top, within 11 mm of hole center.
        "insert_depth": 0.03,
        "radial_tol": 0.011,
        "wrist_start": (0.0, 0.0, 0.30),
        # Waypoints are tool-tip targets the expert policy follows.
        "waypoints": ((0.0, 0.0, 0.17), (0.0, 0.0, 0.105), (0.0, 0.0, 0.055)),
        "goal": (0.0, 0.0, 0.10),
    },
    SCREW_DRIVING: {
        # Socket block: 0.12 x 0.12 x 0.04, top surface at z=0.12, square
        # socket with half-width 0.013 (screw radius 0.010).
        "socket_top_z": 0.12,
        "socket_half": 0.013,
        "screw_radius": 0.010,
        "screw_length": 0.14,
        # The screw hangs from the wrist on two passive hinge joints
        # (tx, ty): contact torques tilt it for real, and the tilt is part of
        # the success criterion.  Tip = wrist_z - 0.18 at zero tilt.
        "tip_offset_z": -0.18,
        "tilt_limit": 0.35,
        # Success: tip 3.5 cm below the socket top, within 11 mm of the
        # socket axis, with tilt under 0.10 rad.
        "drive_depth": 0.035,
        "radial_tol": 0.011,
        "tilt_tol": 0.10,
        "wrist_start": (0.0, 0.0, 0.32),
        "waypoints": ((0.0, 0.0, 0.21), (0.0, 0.0, 0.125), (0.0, 0.0, 0.075)),
        "yaw_rate": 0.6,
        "goal": (0.0, 0.0, 0.12),
    },
    REACH: {
        # Smoke task: move the probe tip to a target sphere.  No contacts.
        "target": (-0.12, 0.10, 0.14),
        "target_radius": 0.015,
        "tip_offset_z": -0.18,
        "reach_tol": 0.02,
        "wrist_start": (0.12, -0.06, 0.30),
        "waypoints": ((-0.12, 0.10, 0.14),),
        "goal": (-0.12, 0.10, 0.14),
    },
}

_WRIST_JOINTS = """
      <joint name="wx" type="slide" axis="1 0 0" range="-0.3 0.3" damping="4"/>
      <joint name="wy" type="slide" axis="0 1 0" range="-0.3 0.3" damping="4"/>
      <joint name="wz" type="slide" axis="0 0 1" range="-0.28 0.06" damping="6"/>
      <joint name="wyaw" type="hinge" axis="0 0 1" damping="0.5"/>
"""

_WRIST_ACTUATORS = """
  <actuator>
    <position name="ax" joint="wx" kp="400" ctrlrange="-0.3 0.3"/>
    <position name="ay" joint="wy" kp="400" ctrlrange="-0.3 0.3"/>
    <position name="az" joint="wz" kp="600" ctrlrange="-0.28 0.06"/>
    <position name="ayaw" joint="wyaw" kp="30"/>
  </actuator>
"""

_SCENE_HEAD = """<mujoco model="{name}">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <default>
    <geom contype="1" conaffinity="1" condim="3" friction="0.8 0.1 0.1"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="0.6 0.6 0.05" rgba="0.85 0.85 0.85 1"/>
"""


def peg_insertion_xml() -> str:
    """MJCF for the peg-in-hole insertion task."""
    spec = TASK_SPECS[PEG_INSERTION]
    plate_top = _f(spec, "plate_top_z")
    hole = _f(spec, "hole_half")
    plate_z = plate_top - _f(spec, "plate_thickness") / 2
    outer = 0.05
    strip = outer - hole
    strip_half = strip / 2
    mid = hole + strip / 2
    wx, wy, wz = _xyz(spec, "wrist_start")
    return (
        _SCENE_HEAD.format(name=PEG_INSERTION)
        + f"""    <body name="plate" pos="0 0 {plate_z:.3f}">
      <geom name="plate_n" type="box" size="{outer:.3f} {strip_half:.3f} 0.01" pos="0 {mid:.3f} 0" rgba="0.4 0.5 0.7 1"/>
      <geom name="plate_s" type="box" size="{outer:.3f} {strip_half:.3f} 0.01" pos="0 {-mid:.3f} 0" rgba="0.4 0.5 0.7 1"/>
      <geom name="plate_e" type="box" size="{strip_half:.3f} {hole:.3f} 0.01" pos="{mid:.3f} 0 0" rgba="0.4 0.5 0.7 1"/>
      <geom name="plate_w" type="box" size="{strip_half:.3f} {hole:.3f} 0.01" pos="{-mid:.3f} 0 0" rgba="0.4 0.5 0.7 1"/>
    </body>
    <body name="wrist" pos="{wx:.3f} {wy:.3f} {wz:.3f}">
{_WRIST_JOINTS}      <geom name="wrist_geom" type="box" size="0.02 0.02 0.015" rgba="0.2 0.2 0.2 1"/>
      <body name="tool" pos="0 0 -0.10">
        <geom name="peg" type="cylinder" size="0.008 0.06" pos="0 0 -0.02" rgba="0.8 0.3 0.2 1"/>
        <site name="tip" pos="0 0 -0.08"/>
      </body>
    </body>
  </worldbody>
{_WRIST_ACTUATORS}</mujoco>
"""
    )


def screw_driving_xml() -> str:
    """MJCF for the screw-driving task (compliant tool holder)."""
    spec = TASK_SPECS[SCREW_DRIVING]
    top = _f(spec, "socket_top_z")
    hole = _f(spec, "socket_half")
    block_z = top - 0.02
    outer = 0.06
    strip = outer - hole
    strip_half = strip / 2
    mid = hole + strip / 2
    wx, wy, wz = _xyz(spec, "wrist_start")
    lim_deg = _f(spec, "tilt_limit") * 57.29578
    return (
        _SCENE_HEAD.format(name=SCREW_DRIVING)
        + f"""    <body name="socket_block" pos="0 0 {block_z:.3f}">
      <geom name="block_n" type="box" size="{outer:.3f} {strip_half:.3f} 0.02" pos="0 {mid:.3f} 0" rgba="0.35 0.45 0.6 1"/>
      <geom name="block_s" type="box" size="{outer:.3f} {strip_half:.3f} 0.02" pos="0 {-mid:.3f} 0" rgba="0.35 0.45 0.6 1"/>
      <geom name="block_e" type="box" size="{strip_half:.3f} {hole:.3f} 0.02" pos="{mid:.3f} 0 0" rgba="0.35 0.45 0.6 1"/>
      <geom name="block_w" type="box" size="{strip_half:.3f} {hole:.3f} 0.02" pos="{-mid:.3f} 0 0" rgba="0.35 0.45 0.6 1"/>
    </body>
    <body name="wrist" pos="{wx:.3f} {wy:.3f} {wz:.3f}">
{_WRIST_JOINTS}      <geom name="wrist_geom" type="box" size="0.02 0.02 0.015" rgba="0.2 0.2 0.2 1"/>
      <body name="holder" pos="0 0 -0.08">
        <body name="tool" pos="0 0 0">
          <joint name="tx" type="hinge" axis="1 0 0" range="{-lim_deg:.1f} {lim_deg:.1f}" damping="0.6"/>
          <joint name="ty" type="hinge" axis="0 1 0" range="{-lim_deg:.1f} {lim_deg:.1f}" damping="0.6"/>
          <geom name="screw" type="cylinder" size="0.010 0.07" pos="0 0 -0.03" rgba="0.75 0.6 0.2 1"/>
          <site name="tip" pos="0 0 -0.10"/>
        </body>
      </body>
    </body>
  </worldbody>
{_WRIST_ACTUATORS}</mujoco>
"""
    )


def reach_xml() -> str:
    """MJCF for the reach smoke task (no contacts)."""
    spec = TASK_SPECS[REACH]
    tx, ty, tz = _xyz(spec, "target")
    tr = _f(spec, "target_radius")
    wx, wy, wz = _xyz(spec, "wrist_start")
    return (
        _SCENE_HEAD.format(name=REACH)
        + f"""    <geom name="target" type="sphere" size="{tr:.3f}" pos="{tx:.3f} {ty:.3f} {tz:.3f}" contype="0" conaffinity="0" rgba="0.2 0.7 0.3 0.8"/>
    <body name="wrist" pos="{wx:.3f} {wy:.3f} {wz:.3f}">
{_WRIST_JOINTS}      <geom name="wrist_geom" type="box" size="0.02 0.02 0.015" rgba="0.2 0.2 0.2 1"/>
      <body name="tool" pos="0 0 -0.10">
        <geom name="probe" type="sphere" size="0.008" pos="0 0 -0.08" rgba="0.8 0.3 0.2 1"/>
        <site name="tip" pos="0 0 -0.08"/>
      </body>
    </body>
  </worldbody>
{_WRIST_ACTUATORS}</mujoco>
"""
    )


_SCENE_BUILDERS = {
    PEG_INSERTION: peg_insertion_xml,
    SCREW_DRIVING: screw_driving_xml,
    REACH: reach_xml,
}


def build_scene(task: str) -> str:
    """Return the MJCF XML string for *task* (raises KeyError listing TASKS)."""
    try:
        return _SCENE_BUILDERS[task]()
    except KeyError:
        raise KeyError(f"unknown mujoco_manip task {task!r}; known tasks: {TASKS}")
