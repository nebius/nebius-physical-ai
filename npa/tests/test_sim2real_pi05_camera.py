from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workflows.sim2real import pi05_isaac
from npa.workflows.sim2real.pi05_closed_loop import _new_episode_record
from npa.workflows.sim2real.pi05_contract import Pi05ContractError
from npa.workflows.sim2real.pi05_isaac import (
    DESTINATION_LABEL,
    DESTINATION_RADIUS_M,
    _bind_rgb_clock,
    _camera_config,
    _camera_pose_sample,
    _camera_time,
    _contact_config,
    _destination_pixel_count,
    _finger_bounds,
    _finger_grasp_reference_geometry,
    _finger_in_frustum,
    _finger_pixel_counts,
    _hand_origin_control_receipt,
    _multiply_xyzw,
    _measured_proximal_clearance,
    _object_asset_geometry,
    _object_grasp_geometry,
    _observation_state,
    _phase_converged,
    _robot_plant_receipt,
    _launch_camera_app,
    _live_gravity_receipt,
    _axis_angle_error_xyzw,
    _expert_waypoints,
    _expert_command,
    _quaternion_angle_xyzw,
    _relative_camera_quaternion,
    _require_isaac_camera_runtime,
    _ros_look_at_xyzw,
    _ros_to_opengl_xyzw,
    _rotate_vector,
    _sim_time,
    _sensor_clock_reference,
    _semantic_pixel_count,
    _verify_scene_semantics,
    _settle_expert_reset,
    _synchronize_wrist_before_capture,
    _target_pixel_count,
    _untransformed_aligned_bounds,
    _validate_capture_ages,
    _validated_kit_args,
    _wrist_mount_receipt,
    _world_pose_matrix,
    _xyzw_to_wxyz,
    configure_destination_marker,
    configure_finger_semantics,
    configure_hand_origin_frames,
    configure_object_reset,
    configure_robot_plant,
    configure_support_surface,
)


class _Tensor:
    def __init__(self, value: object) -> None:
        self.value = np.asarray(value)

    def __getitem__(self, item: object) -> _Tensor:
        return _Tensor(self.value[item])

    def detach(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.value

    def item(self) -> object:
        return self.value.item()


class _Proxy:
    def __init__(self, value: object) -> None:
        self.torch = _Tensor(value)


class _FabricCameraView:
    def __init__(
        self,
        world: object,
        local: object,
        parent: str = "env_0",
        local_quaternion: object = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        if parent == "env_0":
            path = "/World/envs/env_0/Pi05WristCamera"
        else:
            path = f"/World/envs/env_0/Robot/{parent}/Pi05WristCamera"
        self.prim_paths = [path]
        self.world = np.asarray(world, dtype=np.float32)
        self.local = np.asarray(local, dtype=np.float32)
        self.local_quaternion = np.asarray(local_quaternion, dtype=np.float32)
        self.world_quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
        self._use_fabric = True
        self._fabric_initialized = True
        self._fabric_usd_sync_done = True

    def get_world_poses(self) -> tuple[_Proxy, _Proxy]:
        return _Proxy([self.world]), _Proxy([self.world_quaternion])

    def get_local_poses(self) -> tuple[_Proxy, _Proxy]:
        return _Proxy([self.local]), _Proxy([self.local_quaternion])


class _PinnedCamera:
    def __init__(self, cached: object, view: _FabricCameraView) -> None:
        self._view = view
        rgb = np.indices((224, 224, 3)).sum(axis=0).astype(np.uint8)[None]
        self.data = SimpleNamespace(
            pos_w=_Proxy([cached]),
            quat_w_ros=_Proxy([[0.0, 0.0, 0.0, 1.0]]),
            output={"rgb": _Tensor(rgb)},
        )
        self._timestamp = _Tensor([0.0])
        self._timestamp_last_update = _Tensor([0.0])
        self._is_outdated = _Tensor([0.0])
        self.frame = _Proxy([0])
        self._is_initialized = True
        self._renderer = SimpleNamespace()
        self._render_data = SimpleNamespace()
        self.set_calls = 0
        self.update_calls = 0
        self.render_pumps = 0
        self.renderer_position = np.asarray(cached, dtype=np.float32)
        self.renderer_quaternion = np.asarray([0.0, 0.0, 0.0, 1.0])

    def _npa_renderer_usd_pose(self) -> tuple[object, object]:
        return self.renderer_position, self.renderer_quaternion

    def _npa_publish_renderer_pose(self, position: object, quaternion: object) -> None:
        self.renderer_position = np.asarray(position, dtype=np.float32)
        self.renderer_quaternion = np.asarray(quaternion, dtype=np.float32)

    def _npa_render_after_pose(self) -> None:
        self.render_pumps += 1

    def set_world_poses(
        self, positions: object, orientations: object, *, convention: str
    ) -> None:
        assert convention == "ros"
        self.set_calls += 1
        self._view.world = positions.detach().cpu().numpy()[0]
        self._quaternion = orientations.detach().cpu().numpy()[0]
        self._view.world_quaternion = _ros_to_opengl_xyzw(self._quaternion)

    def _update_poses(self) -> None:
        self.data.pos_w = _Proxy([self._view.world])
        quaternion = getattr(self, "_quaternion", [0.0, 0.0, 0.0, 1.0])
        self.data.quat_w_ros = _Proxy([quaternion])

    def update(self, dt: float, *, force_recompute: bool) -> None:
        assert dt == 0.0
        assert force_recompute is True
        self.update_calls += 1
        self._update_poses()
        self.data.output["rgb"].value = np.roll(
            self.data.output["rgb"].value, self.update_calls, axis=2
        )
        self.frame = _Proxy([int(self.frame.torch.value[0]) + 1])


class _Offset:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


class _CameraCfg:
    OffsetCfg = _Offset

    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


class _AssetCfg(_CameraCfg):
    InitialStateCfg = _CameraCfg


class _Prim:
    def __init__(self, path: str) -> None:
        self.path = path

    def GetName(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    def GetPath(self) -> str:
        return self.path

    def IsValid(self) -> bool:
        return True


class _LazyCamera:
    def __init__(self, clock: object) -> None:
        self._timestamp_last_update = clock
        self.data_accesses = 0

    @property
    def data(self) -> object:
        self.data_accesses += 1
        return object()


def test_gen3_camera_config_requests_live_pose_and_integer_semantics(
    monkeypatch,
) -> None:
    sensors = types.ModuleType("isaaclab.sensors")
    sensors.TiledCameraCfg = _CameraCfg
    sim = types.ModuleType("isaaclab.sim")
    sim.PinholeCameraCfg = _CameraCfg
    isaaclab = types.ModuleType("isaaclab")
    renderers = types.ModuleType("isaaclab_physx.renderers")
    renderers.IsaacRtxRendererCfg = _CameraCfg
    isaaclab.sensors = sensors
    isaaclab.sim = sim
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", sensors)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim)
    monkeypatch.setitem(sys.modules, "isaaclab_physx.renderers", renderers)
    cfg = SimpleNamespace(scene=SimpleNamespace())

    _camera_config(cfg)

    for camera in (cfg.scene.pi05_exterior, cfg.scene.pi05_wrist):
        assert camera.update_latest_camera_pose is True
        assert camera.renderer_cfg.colorize_semantic_segmentation is False
        assert camera.renderer_cfg.semantic_filter == ["class"]
        assert camera.data_types == ["rgb", "semantic_segmentation"]
    assert cfg.scene.pi05_wrist.offset.rot == pi05_isaac.WRIST_MOUNT_QUATERNION_XYZW
    assert cfg.scene.pi05_wrist.offset.convention == "ros"
    assert cfg.scene.pi05_wrist.prim_path == "{ENV_REGEX_NS}/Pi05WristCamera"
    assert cfg.scene.pi05_exterior.offset.convention == "ros"
    assert cfg.scene.pi05_wrist.spawn.focal_length == 18.0


def test_ros_camera_axes_aim_at_declared_geometry() -> None:
    position = np.asarray(pi05_isaac.EXTERIOR_CAMERA_POSITION_M)
    target = np.asarray(pi05_isaac.EXTERIOR_CAMERA_TARGET_M)
    quaternion = _ros_look_at_xyzw(position, target)
    optical_axis = _rotate_vector(_xyzw_to_wxyz(quaternion), [0.0, 0.0, 1.0])
    expected = (target - position) / np.linalg.norm(target - position)
    assert optical_axis == pytest.approx(expected)
    wrist_axis = _rotate_vector(
        _xyzw_to_wxyz(pi05_isaac.WRIST_MOUNT_QUATERNION_XYZW), [0, 0, 1]
    )
    wrist_target = np.subtract(
        pi05_isaac.WRIST_MOUNT_TARGET_LOCAL_M,
        pi05_isaac.WRIST_MOUNT_TRANSLATION_M,
    )
    assert wrist_axis == pytest.approx(wrist_target / np.linalg.norm(wrist_target))


def test_exterior_oblique_geometry_contains_source_and_destination() -> None:
    position = np.asarray(pi05_isaac.EXTERIOR_CAMERA_POSITION_M)
    quaternion = _ros_look_at_xyzw(position, pi05_isaac.EXTERIOR_CAMERA_TARGET_M)
    rotation = np.column_stack(
        [_rotate_vector(_xyzw_to_wxyz(quaternion), axis) for axis in np.eye(3)]
    )
    focal_pixels = 18.0 / 20.955 * 224.0
    pixels = []
    for point in ([0.42, -0.25, 0.024], [0.55, -0.08, 0.024], [0.5, 0.18, 0.024]):
        camera = rotation.T @ (np.asarray(point) - position)
        assert camera[2] > 0.03
        pixels.append(112.0 + focal_pixels * camera[:2] / camera[2])
    assert np.allclose(
        pixels,
        [[115.409, 80.768], [100.372, 103.640], [120.189, 140.292]],
        atol=0.002,
    )


def test_side_mount_keeps_default_cube_faces_beyond_near_clip() -> None:
    camera = np.asarray(pi05_isaac.WRIST_MOUNT_TRANSLATION_M)
    target = np.asarray(pi05_isaac.WRIST_MOUNT_TARGET_LOCAL_M)
    center_depth = float(np.linalg.norm(target - camera))
    assert center_depth - 0.072 / 2 > 0.03


def test_object_reset_maps_declared_world_ranges_from_pinned_origin() -> None:
    cfg = SimpleNamespace(
        scene=SimpleNamespace(
            object=SimpleNamespace(init_state=SimpleNamespace(pos=[0.5, 0, 0.055]))
        ),
        events=SimpleNamespace(
            reset_object_position=SimpleNamespace(params={"pose_range": {}})
        ),
    )

    configure_object_reset(cfg)

    pose = cfg.events.reset_object_position.params["pose_range"]
    assert pose["x"] == pytest.approx((-0.08, 0.05))
    assert pose["y"] == pytest.approx((-0.25, -0.08))
    assert pose["z"] == (0.0, 0.0)
    assert tuple(0.5 + value for value in pose["x"]) == pytest.approx((0.42, 0.55))


def test_renderer_world_matrix_preserves_pose_in_usd_row_convention() -> None:
    matrix = _world_pose_matrix([0.4, -0.1, 0.6], [0.0, 0.0, 0.0, 1.0])
    assert matrix.shape == (4, 4)
    assert matrix[3, :3] == pytest.approx([0.4, -0.1, 0.6])
    assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1.0)


def test_destination_marker_is_rendered_semantic_and_noncolliding(monkeypatch) -> None:
    assets = types.ModuleType("isaaclab.assets")
    assets.AssetBaseCfg = _AssetCfg
    sim = types.ModuleType("isaaclab.sim")
    sim.CylinderCfg = _CameraCfg
    sim.PreviewSurfaceCfg = _CameraCfg
    isaaclab = types.ModuleType("isaaclab")
    isaaclab.assets = assets
    isaaclab.sim = sim
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.assets", assets)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim)
    cfg = SimpleNamespace(scene=SimpleNamespace())

    configure_destination_marker(cfg, (0.5, 0.18, 0.025))

    marker = cfg.scene.pi05_destination
    assert marker.init_state.pos == (0.5, 0.18, 0.0005)
    assert marker.spawn.radius == DESTINATION_RADIUS_M
    assert marker.spawn.collision_props is None
    assert marker.spawn.semantic_tags == [("class", DESTINATION_LABEL)]
    assert marker.spawn.visual_material.diffuse_color == (0.0, 0.85, 0.85)


def test_scene_semantics_use_authored_config_not_runtime_view_cfg(monkeypatch) -> None:
    paths = {
        "/World/envs/env_0/Object": _Prim("/World/envs/env_0/Object"),
        "/World/envs/env_0/Pi05DestinationMarker": _Prim(
            "/World/envs/env_0/Pi05DestinationMarker"
        ),
    }
    sim = types.ModuleType("isaaclab.sim")
    sim.find_first_matching_prim = lambda path: paths[path]
    semantics = types.ModuleType("isaaclab.sim.utils.semantics")
    semantics.get_labels = lambda prim: {
        "class": [
            "pi05_target_object" if prim.GetName() == "Object" else DESTINATION_LABEL
        ]
    }
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils.semantics", semantics)
    monkeypatch.setattr(
        pi05_isaac,
        "_install_finger_semantics",
        lambda _native: {"left": "/left", "right": "/right"},
    )
    native = SimpleNamespace(
        scene={"object": object(), "pi05_destination": object()},
        cfg=SimpleNamespace(
            scene=SimpleNamespace(
                object=SimpleNamespace(prim_path="/World/envs/env_0/Object"),
                pi05_destination=SimpleNamespace(
                    prim_path="/World/envs/env_0/Pi05DestinationMarker"
                ),
            )
        ),
    )

    receipt = _verify_scene_semantics(native)

    assert receipt["pi05_target_object"] == "/World/envs/env_0/Object"
    assert receipt[DESTINATION_LABEL].endswith("/Pi05DestinationMarker")


def test_camera_launcher_consumes_writable_portable_state(
    monkeypatch, tmp_path
) -> None:
    state = tmp_path / "kit-state"
    state.mkdir()
    monkeypatch.setenv("NPA_ISAAC_STATE_DIR", str(state))
    args = f"--portable-root {state / 'kit'} --/structuredLog/enable=false"
    monkeypatch.setenv("NPA_ISAAC_KIT_ARGS", args)
    calls = []
    app_module = types.ModuleType("isaaclab.app")

    class AppLauncher:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.app = object()

    app_module.AppLauncher = AppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)

    _launch_camera_app()

    assert calls == [{"headless": True, "enable_cameras": True, "kit_args": args}]


def test_camera_launcher_rejects_readonly_cache_as_portable_root(
    monkeypatch, tmp_path
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("NPA_ISAAC_STATE_DIR", str(state))
    monkeypatch.setenv("NPA_ISAAC_KIT_ARGS", "--portable-root /opt/isaac-cache/kit")
    with pytest.raises(RuntimeError, match="inside NPA_ISAAC_STATE_DIR"):
        _launch_camera_app()


@pytest.mark.parametrize(
    "arguments",
    (
        "--portable-root {state}/../../opt/isaac-cache",
        "--portable-root {state}/kit --portable-root={second}",
        "--portable-root={state}/kit --portable-root {second}",
        "--portable-root {state}",
    ),
)
def test_kit_state_rejects_escape_duplicate_and_root_alias(
    monkeypatch, tmp_path, arguments
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    escaped = arguments.format(state=state, second=tmp_path / "second")
    monkeypatch.setenv("NPA_ISAAC_STATE_DIR", str(state))
    monkeypatch.setenv("NPA_ISAAC_KIT_ARGS", escaped)
    with pytest.raises(RuntimeError, match="portable-root|portable root"):
        _validated_kit_args()


def test_kit_state_accepts_single_equal_form_inside_resolved_root(
    monkeypatch, tmp_path
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    value = f"--portable-root={state / 'kit'} --/structuredLog/enable=false"
    monkeypatch.setenv("NPA_ISAAC_STATE_DIR", str(state))
    monkeypatch.setenv("NPA_ISAAC_KIT_ARGS", value)
    assert _validated_kit_args() == value


def test_support_surface_is_one_exact_contact_reporting_body(monkeypatch) -> None:
    assets = types.ModuleType("isaaclab.assets")
    assets.AssetBaseCfg = _AssetCfg
    sim = types.ModuleType("isaaclab.sim")
    for name in (
        "CuboidCfg",
        "RigidBodyPropertiesCfg",
        "CollisionPropertiesCfg",
        "PreviewSurfaceCfg",
    ):
        setattr(sim, name, _CameraCfg)
    isaaclab = types.ModuleType("isaaclab")
    isaaclab.assets, isaaclab.sim = assets, sim
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.assets", assets)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim)
    cfg = SimpleNamespace(scene=SimpleNamespace())

    configure_support_surface(cfg)

    assert cfg.scene.table.prim_path == "{ENV_REGEX_NS}/Table"
    assert cfg.scene.table.spawn.size == (1.0, 0.8, 0.05)
    assert cfg.scene.table.spawn.rigid_props.kinematic_enabled is True
    assert cfg.scene.table.spawn.activate_contact_sensors is True


def test_finger_labels_are_authored_by_spawn_before_physics(monkeypatch) -> None:
    root = _Prim("/World/envs/env_0/Robot")
    children = [
        root,
        _Prim(f"{root.path}/panda_leftfinger"),
        _Prim(f"{root.path}/panda_rightfinger"),
    ]
    applied: list[tuple[str, list[str], str]] = []
    semantics = types.ModuleType("isaaclab.sim.utils.semantics")

    def add_labels(prim, labels, instance_name) -> None:
        applied.append((prim.path, labels, instance_name))

    semantics.add_labels = add_labels
    usd = SimpleNamespace(PrimRange=lambda _root: children)
    pxr = types.ModuleType("pxr")
    pxr.Usd = usd
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils.semantics", semantics)
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    calls: list[str] = []

    def original(prim_path, _cfg, **_kwargs):
        calls.append(prim_path)
        return root

    spawn = SimpleNamespace(func=original)
    cfg = SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(spawn=spawn)))
    configure_finger_semantics(cfg)
    assert applied == []

    returned = cfg.scene.robot.spawn.func("/World/envs/env_0/Robot", spawn)

    assert returned is root
    assert calls == ["/World/envs/env_0/Robot"]
    assert applied == [
        (children[1].path, ["pi05_left_finger"], "class"),
        (children[2].path, ["pi05_right_finger"], "class"),
    ]


def test_contact_filters_name_exact_single_bodies(monkeypatch) -> None:
    sensors = types.ModuleType("isaaclab.sensors")
    sensors.ContactSensorCfg = _CameraCfg
    isaaclab = types.ModuleType("isaaclab")
    isaaclab.sensors = sensors
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", sensors)

    def spawn() -> SimpleNamespace:
        return SimpleNamespace(activate_contact_sensors=False)

    cfg = SimpleNamespace(
        scene=SimpleNamespace(
            robot=SimpleNamespace(spawn=spawn()),
            object=SimpleNamespace(spawn=spawn()),
            table=SimpleNamespace(spawn=spawn()),
        )
    )

    _contact_config(cfg)

    assert cfg.scene.object_support_contact.filter_prim_paths_expr == [
        "{ENV_REGEX_NS}/Table"
    ]
    assert cfg.scene.object_support_contact.prim_path == "{ENV_REGEX_NS}/Object"
    assert all(
        asset.spawn.activate_contact_sensors
        for asset in (cfg.scene.robot, cfg.scene.object, cfg.scene.table)
    )


def test_inherited_sensor_and_ik_tool_offsets_are_zeroed_to_hand_origin() -> None:
    sensor_offset = SimpleNamespace(pos=(0.0, 0.0, 0.1034), rot=(0, 0, 0, 1))
    ik_offset = SimpleNamespace(pos=(0.0, 0.0, 0.107), rot=(0, 0, 0, 1))
    cfg = SimpleNamespace(
        scene=SimpleNamespace(
            ee_frame=SimpleNamespace(
                target_frames=[
                    SimpleNamespace(
                        prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                        offset=sensor_offset,
                    )
                ]
            )
        ),
        actions=SimpleNamespace(
            arm_action=SimpleNamespace(body_name="panda_hand", body_offset=ik_offset)
        ),
    )

    configure_hand_origin_frames(cfg, require_ik=True)

    assert sensor_offset.pos == (0.0, 0.0, 0.0)
    assert ik_offset.pos == (0.0, 0.0, 0.0)
    assert sensor_offset.rot == ik_offset.rot == (0.0, 0.0, 0.0, 1.0)


def test_expert_and_policy_select_the_same_upstream_high_pd_plant(monkeypatch) -> None:
    selected = SimpleNamespace(prim_path="")
    high_pd = SimpleNamespace(
        replace=lambda *, prim_path: SimpleNamespace(
            prim_path=prim_path, selected=selected
        )
    )
    franka = types.ModuleType("isaaclab_assets.robots.franka")
    franka.FRANKA_PANDA_HIGH_PD_CFG = high_pd
    monkeypatch.setitem(
        sys.modules, "isaaclab_assets", types.ModuleType("isaaclab_assets")
    )
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_assets.robots",
        types.ModuleType("isaaclab_assets.robots"),
    )
    monkeypatch.setitem(sys.modules, "isaaclab_assets.robots.franka", franka)
    cfg = SimpleNamespace(
        scene=SimpleNamespace(robot=SimpleNamespace(prim_path="/Robot"))
    )

    configure_robot_plant(cfg)

    assert cfg.scene.robot.prim_path == "/Robot"
    assert cfg.scene.robot.selected is selected


def test_runtime_robot_plant_receipt_reads_actual_actuator_tensors() -> None:
    import torch

    actuator = SimpleNamespace(
        joint_names=["panda_joint1", "panda_joint2"],
        stiffness=torch.tensor([[400.0, 400.0]]),
        damping=torch.tensor([[80.0, 80.0]]),
    )
    robot = SimpleNamespace(
        cfg=SimpleNamespace(
            spawn=SimpleNamespace(rigid_props=SimpleNamespace(disable_gravity=True))
        ),
        actuators={"panda_shoulder": actuator},
        num_instances=1,
        num_bodies=9,
        root_view=SimpleNamespace(
            get_disable_gravities=lambda: torch.ones((1, 9), dtype=torch.uint8)
        ),
    )

    receipt = _robot_plant_receipt(SimpleNamespace(scene={"robot": robot}))

    assert receipt["configured_disable_robot_gravity"] is True
    assert receipt["observed_disable_gravity"]["all_bodies_disabled"] is True
    assert receipt["observed_disable_gravity"]["shape"] == [1, 9]
    assert receipt["observed_disable_gravity"]["converter"] == "torch"
    assert receipt["actuators"]["panda_shoulder"]["stiffness"] == [400.0]
    assert receipt["actuators"]["panda_shoulder"]["damping"] == [80.0]


def test_runtime_robot_plant_rejects_unapplied_gravity_config() -> None:
    import torch

    actuator = SimpleNamespace(
        joint_names=["panda_joint1"],
        stiffness=torch.tensor([[400.0]]),
        damping=torch.tensor([[80.0]]),
    )
    robot = SimpleNamespace(
        cfg=SimpleNamespace(
            spawn=SimpleNamespace(rigid_props=SimpleNamespace(disable_gravity=True))
        ),
        actuators={"panda_shoulder": actuator},
        num_instances=1,
        num_bodies=1,
        root_view=SimpleNamespace(
            get_disable_gravities=lambda: torch.zeros((1, 1), dtype=torch.uint8)
        ),
    )

    with pytest.raises(RuntimeError, match="gravity disabling is not active"):
        _robot_plant_receipt(SimpleNamespace(scene={"robot": robot}))


def test_runtime_robot_plant_converts_raw_warp_gravity_flags(monkeypatch) -> None:
    import torch

    raw = object()
    warp = types.ModuleType("warp")
    warp.to_torch = lambda value: torch.ones((1, 2)) if value is raw else None
    monkeypatch.setitem(sys.modules, "warp", warp)
    robot = _plant_robot_for_gravity(SimpleNamespace(get_disable_gravities=lambda: raw))

    receipt = _robot_plant_receipt(SimpleNamespace(scene={"robot": robot}))

    gravity = receipt["observed_disable_gravity"]
    assert gravity["converter"] == "warp.to_torch"
    assert gravity["return_type"] == "builtins.object"


def _plant_robot_for_gravity(view: SimpleNamespace) -> SimpleNamespace:
    import torch

    actuator = SimpleNamespace(
        joint_names=["panda_joint1"],
        stiffness=torch.tensor([[400.0]]),
        damping=torch.tensor([[80.0]]),
    )
    return SimpleNamespace(
        cfg=SimpleNamespace(
            spawn=SimpleNamespace(rigid_props=SimpleNamespace(disable_gravity=True))
        ),
        actuators={"panda_shoulder": actuator},
        num_instances=1,
        num_bodies=2,
        root_view=view,
    )


@pytest.mark.parametrize(
    ("robot", "message"),
    [
        (SimpleNamespace(root_view=None), "lacks its PhysX articulation root view"),
        (SimpleNamespace(root_view=SimpleNamespace()), "lacks get_disable_gravities"),
    ],
)
def test_live_gravity_receipt_distinguishes_missing_api(robot, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        _live_gravity_receipt(robot)


def test_live_gravity_receipt_rejects_shape_and_nonbinary_values() -> None:
    import torch

    wrong_shape = SimpleNamespace(
        root_view=SimpleNamespace(get_disable_gravities=lambda: torch.ones((1, 1))),
        num_instances=1,
        num_bodies=2,
    )
    with pytest.raises(RuntimeError, match="shape is invalid"):
        _live_gravity_receipt(wrong_shape)
    nonbinary = SimpleNamespace(
        root_view=SimpleNamespace(
            get_disable_gravities=lambda: torch.tensor([[1.0, 2.0]])
        ),
        num_instances=1,
        num_bodies=2,
    )
    with pytest.raises(RuntimeError, match="flags are invalid"):
        _live_gravity_receipt(nonbinary)


def test_live_rotated_hand_sensor_and_ik_control_point_receipt() -> None:
    import torch

    position = [0.39, -0.02, 0.46]
    quaternion = [0.0, 0.0, 2**-0.5, 2**-0.5]
    offset = SimpleNamespace(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0))
    target = SimpleNamespace(prim_path="/World/Robot/panda_hand", offset=offset)
    term = SimpleNamespace(
        cfg=SimpleNamespace(body_name="panda_hand", body_offset=offset)
    )
    native = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                body_names=["panda_hand"],
                data=SimpleNamespace(
                    body_pos_w=torch.tensor([[position]], dtype=torch.float64),
                    body_quat_w=torch.tensor([[quaternion]], dtype=torch.float64),
                ),
            ),
            "ee_frame": SimpleNamespace(
                data=SimpleNamespace(
                    target_pos_w=torch.tensor([[position]], dtype=torch.float64),
                    target_quat_w=torch.tensor([[quaternion]], dtype=torch.float64),
                )
            ),
        },
        cfg=SimpleNamespace(
            scene=SimpleNamespace(ee_frame=SimpleNamespace(target_frames=[target]))
        ),
        action_manager=SimpleNamespace(get_term=lambda _name: term),
    )

    receipt = _hand_origin_control_receipt(native, require_ik=True)

    assert receipt["controlled_frame"] == "panda_hand_origin"
    assert receipt["controlled_point_world_m"] == pytest.approx(position)
    assert receipt["sensor_position_m"] == pytest.approx(position)
    assert receipt["live_hand_quaternion_xyzw"] == pytest.approx(quaternion)
    assert receipt["ik_body_offset"]["position_m"] == pytest.approx([0.0] * 3)


def test_camera_contract_is_bound_to_exact_gen3_runtime(monkeypatch) -> None:
    monkeypatch.setattr("importlib.metadata.version", lambda _name: "3.0.0b2.post1")
    _require_isaac_camera_runtime()
    monkeypatch.setattr("importlib.metadata.version", lambda _name: "2.2.0")
    with pytest.raises(RuntimeError, match="requires Isaac Lab 3.0.0b2.post1"):
        _require_isaac_camera_runtime()


def test_camera_time_refreshes_data_then_reads_raw_warp_clock(monkeypatch) -> None:
    raw_warp = object()
    warp = types.ModuleType("warp")
    warp.to_torch = lambda value: _Tensor([0.2]) if value is raw_warp else None
    monkeypatch.setitem(sys.modules, "warp", warp)
    camera = _LazyCamera(raw_warp)
    native = SimpleNamespace(scene={"camera": camera})

    assert _camera_time(native, "camera") == pytest.approx(0.2)
    assert camera.data_accesses == 1


@pytest.mark.parametrize("clock", [_Tensor([0.3]), _Proxy([0.3])])
def test_camera_time_supports_torch_and_proxy_clock_boundaries(clock) -> None:
    camera = _LazyCamera(clock)
    assert _camera_time(
        SimpleNamespace(scene={"camera": camera}), "camera"
    ) == pytest.approx(0.3)


def test_episode_clock_restarts_with_each_sensor_reset_epoch() -> None:
    native = SimpleNamespace(
        common_step_counter=900,
        episode_length_buf=_Tensor([3]),
        step_dt=1.0 / 15.0,
    )
    assert _sim_time(native) == pytest.approx(0.2)

    native.common_step_counter = 901
    native.episode_length_buf = _Tensor([0])
    assert _sim_time(native) == 0.0
    native.episode_length_buf = _Tensor([1])
    assert _sim_time(native) == pytest.approx(1.0 / 15.0)


def test_capture_clock_reproduces_long_float32_accumulation_and_reset() -> None:
    native = SimpleNamespace(
        episode_length_buf=_Tensor([600]),
        cfg=SimpleNamespace(decimation=4),
        physics_dt=1.0 / 60.0,
    )
    raw = np.float32(0.0)
    for _ in range(600 * 4):
        raw = np.float32(raw + np.float32(1.0 / 60.0))
    reference = _sensor_clock_reference(native)
    assert reference["expected_sensor_time_s"] == float(raw)
    assert abs(float(raw) - 40.0) > 1.0e-6
    _validate_capture_ages((float(raw), float(raw)), reference, 1.0 / 15.0)

    with pytest.raises(RuntimeError, match="future"):
        _validate_capture_ages((float(raw) + 0.01, float(raw)), reference, 1 / 15)
    with pytest.raises(RuntimeError, match="stale"):
        _validate_capture_ages((float(raw) - 0.1, float(raw)), reference, 1 / 15)

    native.episode_length_buf = _Tensor([0])
    reset = _sensor_clock_reference(native)
    assert reset["expected_sensor_time_s"] == 0.0
    assert reset["reset_epoch"] == 1


def test_rgb_receipt_preserves_raw_age_and_binds_sensor_age() -> None:
    receipt = {
        "capture_time_s": float(np.float32(2.0 / 15.0)),
        "observation_time_s": 2.0 / 15.0,
        "age_s": 2.0 / 15.0 - float(np.float32(2.0 / 15.0)),
    }
    clock = {
        "expected_sensor_time_s": receipt["capture_time_s"],
        "physics_update_count": 8,
        "physics_period_s": float(np.float32(1.0 / 60.0)),
        "reset_epoch": 0,
    }

    _bind_rgb_clock(receipt, clock)

    assert receipt["sensor_causal_age_s"] == 0.0
    assert receipt["age_s"] == 0.0
    assert receipt["raw_observation_age_s"] != 0.0
    assert receipt["representation_offset_s"] == -receipt["raw_observation_age_s"]
    assert receipt["sensor_clock"] == clock


def _semantic_native(segmentation: np.ndarray) -> SimpleNamespace:
    data = SimpleNamespace(
        info={
            "semantic_segmentation": {
                "idToLabels": {
                    "(0, 0, 0, 0)": {"class": "BACKGROUND"},
                    "(7, 0, 0, 0)": {"class": "pi05_target_object"},
                }
            }
        },
        output={"semantic_segmentation": _Proxy(segmentation)},
    )
    return SimpleNamespace(
        scene={"camera": SimpleNamespace(data=data)},
        _npa_pi05_verified_semantic_paths={
            "pi05_target_object": "/World/envs/env_0/Object",
            DESTINATION_LABEL: "/World/envs/env_0/Pi05DestinationMarker",
            "pi05_left_finger": "/World/envs/env_0/Robot/panda_leftfinger",
            "pi05_right_finger": "/World/envs/env_0/Robot/panda_rightfinger",
        },
    )


def test_semantic_visibility_rejects_rgba_and_counts_packed_metadata_pixels() -> None:
    rgba = np.zeros((1, 224, 224, 4), dtype=np.uint8)
    rgba[..., 3] = 7
    with pytest.raises(RuntimeError, match="int32"):
        _target_pixel_count(_semantic_native(rgba), "camera")

    labels = np.zeros((1, 224, 224, 1), dtype=np.int32)
    labels[0, 10, 20, 0] = 7
    labels[0, 11, 20, 0] = 7
    assert _target_pixel_count(_semantic_native(labels), "camera") == 2

    destination = _semantic_native(labels)
    destination.scene["camera"].data.info["semantic_segmentation"]["idToLabels"] = {
        "(0, 0, 0, 0)": {"class": "BACKGROUND"},
        "(7, 0, 0, 0)": {"class": DESTINATION_LABEL},
    }
    assert _destination_pixel_count(destination, "camera") == 2

    finger_data = _semantic_native(labels).scene["camera"].data
    finger_data.info["semantic_segmentation"]["idToLabels"] = {
        "(0, 0, 0, 0)": {"class": "BACKGROUND"},
        "(7, 0, 0, 0)": {"class": "pi05_left_finger"},
        "(8, 0, 0, 0)": {"class": "pi05_right_finger"},
    }
    labels[0, 12, 20, 0] = 8
    native = _semantic_native(labels)
    native.scene["camera"] = SimpleNamespace(data=finger_data)
    assert _finger_pixel_counts(native, "camera") == {"left": 2, "right": 1}


def test_known_but_out_of_view_semantic_class_counts_as_zero() -> None:
    labels = np.zeros((1, 224, 224, 1), dtype=np.int32)
    native = _semantic_native(labels)
    native.scene["camera"].data.info["semantic_segmentation"]["idToLabels"] = {
        "(0, 0, 0, 0)": {"class": "BACKGROUND"}
    }

    assert _target_pixel_count(native, "camera") == 0
    del native._npa_pi05_verified_semantic_paths
    with pytest.raises(RuntimeError, match="unverified semantic label"):
        _target_pixel_count(native, "camera")
    with pytest.raises(RuntimeError, match="unverified semantic label"):
        _semantic_pixel_count(native, "camera", "operator_unverified_label")


def test_semantic_visibility_decodes_signed_live_packed_rgba_code() -> None:
    rgba = (27, 186, 239, 255)
    packed = int.from_bytes(bytes(rgba), byteorder="little", signed=True)
    labels = np.zeros((1, 224, 224, 1), dtype=np.int32)
    labels[0, 10, 20, 0] = packed
    native = _semantic_native(labels)
    native.scene["camera"].data.info["semantic_segmentation"]["idToLabels"] = {
        "(0, 0, 0, 0)": {"class": "BACKGROUND"},
        str(rgba): {"class": "pi05_target_object"},
    }
    assert packed < 0
    assert _target_pixel_count(native, "camera") == 1


@pytest.mark.parametrize(
    ("metadata", "pixels", "message"),
    (
        ({"not-rgba": {"class": "pi05_target_object"}}, [0], "metadata key"),
        (
            {"(7, 0, 0, 0)": {"class": "pi05_target_object"}},
            [8],
            "unmapped codes",
        ),
        ({"(7, 0, 0, 0)": "pi05_target_object"}, [7], "class metadata"),
        (
            {
                "(0, 0, 0, 0)": {"class": "BACKGROUND"},
                "(7, 0, 0, 0)": {"class": "pi05_target_object"},
                "(8, 0, 0, 0)": {"class": "pi05_target_object"},
            },
            [7],
            "class mapping is ambiguous",
        ),
    ),
)
def test_semantic_visibility_rejects_malformed_or_unmapped_encoding(
    metadata: object, pixels: list[int], message: str
) -> None:
    labels = np.zeros((1, 224, 224, 1), dtype=np.int32)
    labels[0, 0, 0, 0] = pixels[0]
    native = _semantic_native(labels)
    native.scene["camera"].data.info["semantic_segmentation"]["idToLabels"] = metadata
    with pytest.raises(RuntimeError, match=message):
        _target_pixel_count(native, "camera")


def test_camera_pose_uses_proxy_arrays_and_preserves_xyzw_order() -> None:
    robot = SimpleNamespace(
        body_names=["panda_hand"],
        data=SimpleNamespace(
            body_pos_w=_Tensor([[[0.1, 0.2, 0.3]]]),
            body_quat_w=_Proxy([[[0.0, 0.0, 0.0, 1.0]]]),
        ),
    )
    wrist = SimpleNamespace(
        data=SimpleNamespace(
            pos_w=_Proxy([[0.2, 0.3, 0.4]]),
            quat_w_ros=_Proxy([[0.1, 0.2, 0.3, 0.9]]),
        ),
        _npa_renderer_usd_pose=lambda: ([0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.9]),
    )
    exterior = SimpleNamespace(
        data=SimpleNamespace(
            pos_w=_Proxy([[1.0, 2.0, 3.0]]),
            quat_w_world=_Proxy([[0.0, 0.0, 0.0, 1.0]]),
        )
    )
    pose = _camera_pose_sample(
        SimpleNamespace(
            scene={"robot": robot, "pi05_wrist": wrist, "pi05_exterior": exterior}
        )
    )
    assert pose["wrist_camera_quaternion_xyzw"] == [0.1, 0.2, 0.3, 0.9]
    assert _xyzw_to_wxyz(pose["wrist_camera_quaternion_xyzw"]).tolist() == [
        0.9,
        0.1,
        0.2,
        0.3,
    ]
    assert _relative_camera_quaternion(
        [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]
    ).tolist() == [1.0, 0.0, 0.0, 0.0]


def _mount_native(
    *,
    camera_position: object,
    view_position: object,
    parent: str = "env_0",
    local_quaternion: object = (1.0, 0.0, 0.0, 0.0),
) -> SimpleNamespace:
    robot = SimpleNamespace(
        body_names=["panda_hand"],
        data=SimpleNamespace(
            body_pos_w=_Proxy([[[0.4, 0.0, 0.4]]]),
            body_quat_w=_Proxy([[[0.0, 0.0, 0.0, 1.0]]]),
        ),
    )
    view = _FabricCameraView(
        view_position,
        [0.0, 0.0, 1.0],
        parent,
        local_quaternion,
    )
    camera = _PinnedCamera(camera_position, view)
    return SimpleNamespace(scene={"robot": robot, "pi05_wrist": camera})


def test_wrist_mount_repairs_static_fabric_world_pose_with_receipt() -> None:
    native = _mount_native(
        camera_position=[0.025, 0.0, 0.035],
        view_position=[0.025, 0.0, 0.035],
    )
    buffers = _new_episode_record()

    receipt = _wrist_mount_receipt(native, buffers)

    assert receipt["refresh_action"] == (
        "usd_and_fabric_world_pose_synchronized_from_live_hand_mount"
    )
    assert receipt["before"]["camera_data_position_m"] == [0.025, 0.0, 0.035]
    assert np.allclose(receipt["after"]["camera_data_position_m"], [0.49, 0.0, 0.415])
    assert np.allclose(receipt["after"]["renderer_usd_position_m"], [0.49, 0.0, 0.415])
    assert receipt["pose_sources"]["camera_live"] == ("FabricFrameView.get_world_poses")
    assert buffers["wrist_mount_receipts"] == [receipt]
    assert buffers["queries"] == []
    assert native.scene["pi05_wrist"].set_calls == 1
    assert native.scene["pi05_wrist"].update_calls == 1
    assert native.scene["pi05_wrist"].render_pumps == 1
    assert receipt["post_refresh_render"]["renderer_frame"] == 1
    assert len(receipt["post_refresh_render"]["rgb_sha256"]) == 64


def test_wrist_mount_synchronizes_stale_camera_data_and_renderer() -> None:
    native = _mount_native(
        camera_position=[0.025, 0.0, 0.035],
        view_position=[0.425, 0.0, 0.435],
    )

    receipt = _wrist_mount_receipt(native, {"wrist_mount_receipts": []})

    assert receipt["refresh_action"] == (
        "usd_and_fabric_world_pose_synchronized_from_live_hand_mount"
    )
    assert native.scene["pi05_wrist"].set_calls == 1
    assert native.scene["pi05_wrist"].update_calls == 1
    assert np.allclose(receipt["after"]["camera_data_position_m"], [0.49, 0.0, 0.415])


def test_wrist_mount_rejects_detached_prim_and_preserves_raw_pose() -> None:
    native = _mount_native(
        camera_position=[0.025, 0.0, 0.035],
        view_position=[0.025, 0.0, 0.035],
        parent="detached_mount",
    )
    buffers = {"wrist_mount_receipts": []}

    with pytest.raises(RuntimeError, match="world-pose bridge"):
        _wrist_mount_receipt(native, buffers)

    receipt = buffers["wrist_mount_receipts"][0]
    assert np.allclose(
        receipt["before"]["frame_view_world_position_m"], [0.025, 0.0, 0.035]
    )
    assert receipt["failure"].startswith("RuntimeError:")
    assert native.scene["pi05_wrist"].set_calls == 0


def test_verified_wrist_pose_tracks_later_live_hand_motion() -> None:
    native = _mount_native(
        camera_position=[0.025, 0.0, 0.035],
        view_position=[0.025, 0.0, 0.035],
    )
    _wrist_mount_receipt(native, {"wrist_mount_receipts": []})
    native.scene["robot"].data.body_pos_w = _Proxy([[[0.5, -0.1, 0.6]]])

    _synchronize_wrist_before_capture(native)

    assert np.allclose(
        native.scene["pi05_wrist"].data.pos_w.torch.value[0],
        [0.59, -0.1, 0.615],
    )
    assert native.scene["pi05_wrist"].set_calls == 2
    assert native.scene["pi05_wrist"].update_calls == 2


def test_wrist_mount_composes_offset_and_orientation_in_live_hand_frame() -> None:
    native = _mount_native(
        camera_position=[0.025, 0.0, 0.035],
        view_position=[0.025, 0.0, 0.035],
    )
    native.scene["robot"].data.body_quat_w = _Proxy([[[0.0, 0.0, 1.0, 0.0]]])

    receipt = _wrist_mount_receipt(native, {"wrist_mount_receipts": []})

    assert np.allclose(receipt["after"]["camera_data_position_m"], [0.31, 0.0, 0.415])
    assert np.allclose(
        receipt["after"]["camera_data_quaternion_xyzw"],
        _multiply_xyzw([0.0, 0.0, 1.0, 0.0], pi05_isaac.WRIST_MOUNT_QUATERNION_XYZW),
    )


def test_wrist_mount_composes_nonidentity_calibrated_rotation(monkeypatch) -> None:
    local_ros = np.asarray([0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)])
    hand = np.asarray([0.0, 0.0, 1.0, 0.0])
    monkeypatch.setattr(
        "npa.workflows.sim2real.pi05_isaac.WRIST_MOUNT_QUATERNION_XYZW",
        tuple(local_ros),
    )
    native = _mount_native(
        camera_position=[0.025, 0.0, 0.035],
        view_position=[0.025, 0.0, 0.035],
    )
    native.scene["robot"].data.body_quat_w = _Proxy([[hand]])

    receipt = _wrist_mount_receipt(native, {"wrist_mount_receipts": []})

    assert receipt["after"]["camera_data_quaternion_xyzw"] == pytest.approx(
        _multiply_xyzw(hand, local_ros)
    )
    assert receipt["after"]["renderer_usd_quaternion_xyzw"] == pytest.approx(
        _multiply_xyzw(hand, local_ros)
    )


@pytest.mark.parametrize(
    "quaternion",
    ([0.0, 0.0, 0.0, 0.0], [np.nan, 0.0, 0.0, 1.0]),
)
def test_camera_orientation_rejects_invalid_quaternion(quaternion: object) -> None:
    with pytest.raises(RuntimeError, match="actual camera quaternion is invalid"):
        _quaternion_angle_xyzw(quaternion, [0.0, 0.0, 0.0, 1.0])


def test_finger_asset_bounds_projection_is_in_frustum_not_visibility(
    monkeypatch,
) -> None:
    intrinsics = [[100.0, 0.0, 112.0], [0.0, 100.0, 112.0], [0.0, 0.0, 1.0]]
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            pos_w=_Proxy([[0.0, 0.0, 0.0]]),
            quat_w_ros=_Proxy([[0.0, 0.0, 0.0, 1.0]]),
            intrinsic_matrices=_Proxy([intrinsics]),
        )
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.pi05_isaac._finger_bounds",
        lambda _native: _projected_finger_bounds(),
    )
    result = _finger_in_frustum(SimpleNamespace(scene={"pi05_wrist": sensor}))
    assert all(item["in_frustum"] for item in result.values())
    assert all(
        item["evidence_scope"] == "asset_bounds_in_frustum_not_rendered_visibility"
        for item in result.values()
    )


def _projected_finger_bounds() -> dict:
    def bounds(x_values: tuple[float, float]) -> dict:
        return {
            "local_bounds_m": [[-0.01, -0.01, -0.1], [0.01, 0.01, 0.1]],
            "world_corners_m": np.asarray(
                [[x, y, z] for x in x_values for y in (-0.01, 0.01) for z in (0.8, 1.0)]
            ),
            "pose_source": "live_articulation_body_pose_xyzw",
        }

    return {"left": bounds((-0.03, -0.01)), "right": bounds((0.01, 0.03))}


def test_finger_geometry_excludes_the_authored_link_transform() -> None:
    class _Aligned:
        def GetMin(self) -> tuple[float, float, float]:
            return (-0.01, -0.02, 0.0)

        def GetMax(self) -> tuple[float, float, float]:
            return (0.01, 0.02, 0.1)

    class _Bound:
        def ComputeAlignedBox(self) -> _Aligned:
            return _Aligned()

    class _Cache:
        def ComputeLocalBound(self, _prim: object) -> object:
            raise AssertionError("link transform must not be included twice")

        def ComputeUntransformedBound(self, prim: object) -> _Bound:
            assert prim == "nonidentity-finger-link"
            return _Bound()

    bounds = _untransformed_aligned_bounds(_Cache(), "nonidentity-finger-link")
    assert bounds == ((-0.01, -0.02, 0.0), (0.01, 0.02, 0.1))


def test_custom_object_geometry_uses_composed_runtime_extent(monkeypatch) -> None:
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    cube = UsdGeom.Cube.Define(stage, "/Object")
    cube.GetSizeAttr().Set(0.073)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(0.45, -0.15, 0.03))
    xform.AddScaleOp().Set(Gf.Vec3d(0.8, 0.9, 1.1))
    sim = types.ModuleType("isaaclab.sim")
    sim.find_first_matching_prim = lambda _path: cube.GetPrim()
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim)
    native = SimpleNamespace(
        scene={"object": object()},
        cfg=SimpleNamespace(
            scene=SimpleNamespace(object=SimpleNamespace(prim_path="/Object"))
        ),
    )

    geometry = _object_asset_geometry(native, (0.8, 0.9, 1.1))

    assert geometry["source_untransformed_extent_m"] == pytest.approx([0.073] * 3)
    assert geometry["effective_local_extent_m"] == pytest.approx(
        [0.0584, 0.0657, 0.0803]
    )
    assert geometry["grasp_width_m"] == pytest.approx(0.0657)
    assert geometry["support_height_m"] == pytest.approx(0.0803)
    assert geometry["source_untransformed_center_m"] == pytest.approx([0.0] * 3)


def test_custom_object_geometry_rejects_offset_rigid_origin(monkeypatch) -> None:
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Object")
    cube = UsdGeom.Cube.Define(stage, "/Object/Cube")
    cube.GetSizeAttr().Set(0.04)
    UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.03))
    sim = types.ModuleType("isaaclab.sim")
    sim.find_first_matching_prim = lambda _path: root.GetPrim()
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim)
    native = SimpleNamespace(
        scene={"object": object()},
        cfg=SimpleNamespace(
            scene=SimpleNamespace(object=SimpleNamespace(prim_path="/Object"))
        ),
    )

    with pytest.raises(RuntimeError, match="centered on its rigid root"):
        _object_asset_geometry(native, (1.0, 1.0, 1.0))


def test_downward_orientation_and_measured_grasp_reference_waypoints() -> None:
    tilted = np.asarray([-0.912, 0.02, -0.406, 0.054])
    target = np.asarray([1.0, 0.0, 0.0, 0.0])
    error = _axis_angle_error_xyzw(tilted, target)
    assert np.linalg.norm(error) > 0.1
    angle = np.linalg.norm(error)
    delta = np.r_[error / angle * np.sin(angle / 2.0), np.cos(angle / 2.0)]
    reached = _multiply_xyzw(delta, tilted)
    assert abs(float(np.dot(reached, target))) == pytest.approx(1.0)
    assert np.linalg.norm(_axis_angle_error_xyzw(target, target)) < 1.0e-8

    grasp_reference = np.asarray([0.01, -0.02, 0.0854166])
    geometry = _object_grasp_geometry(
        {
            "hand_to_finger_link_centroid_local_m": grasp_reference.tolist(),
            "finger_link_hand_axis_interval_m": [0.058533, 0.1123],
        },
        0.072,
    )
    waypoints = _expert_waypoints(np.zeros(3), np.ones(3), geometry)
    contact = next(goal for phase, goal, _intent in waypoints if phase == "contact")
    downward = _xyzw_to_wxyz([1.0, 0.0, 0.0, 0.0])
    reconstructed_reference = contact + _rotate_vector(downward, grasp_reference)
    assert reconstructed_reference == pytest.approx(
        [0.0, 0.0, geometry["contact_reference_above_object_center_m"]]
    )
    assert contact == pytest.approx([-0.01, -0.02, 0.102533])


@pytest.mark.parametrize("height", [0.048, 0.060, 0.072])
def test_size_aware_contact_keeps_proximal_finger_clear_of_cube_top(height) -> None:
    geometry = _object_grasp_geometry(
        {
            "hand_to_finger_link_centroid_local_m": [0.0, 0.0, 0.0854166],
            "finger_link_hand_axis_interval_m": [0.058533, 0.1123],
        },
        height,
    )
    waypoints = _expert_waypoints(
        np.asarray([0.5, 0.0, height / 2]), np.zeros(3), geometry
    )
    contact = next(goal for phase, goal, _intent in waypoints if phase == "contact")

    assert contact[2] == pytest.approx(height + 0.058533 + 0.008)
    assert geometry["contact_hand_origin_above_object_center_m"] == pytest.approx(
        height / 2 + 0.058533 + 0.008
    )


def test_live_clearance_uses_downward_hand_pose_and_object_top(monkeypatch) -> None:
    import torch

    robot = SimpleNamespace(
        body_names=["panda_hand"],
        data=SimpleNamespace(
            body_pos_w=torch.tensor([[[0.5, 0.0, 0.14]]]),
            body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        ),
    )
    obj = SimpleNamespace(
        data=SimpleNamespace(root_pos_w=torch.tensor([[0.5, 0.0, 0.03]]))
    )
    native = SimpleNamespace(scene={"robot": robot, "object": obj})
    world = np.asarray([[0.5, 0.0, 0.08], [0.5, 0.0, 0.03]])
    local = np.asarray([[0.0, 0.0, 0.06], [0.0, 0.0, 0.11]])
    monkeypatch.setattr(
        pi05_isaac, "_finger_points_in_hand_frame", lambda _native: (world, local)
    )
    geometry = {
        "object_height_m": 0.06,
    }

    assert _measured_proximal_clearance(native, geometry) == pytest.approx(0.02)


def test_contact_convergence_requires_measured_proximal_clearance() -> None:
    assert not _phase_converged("contact", 0.0, 0.0, 0.007)
    assert _phase_converged("contact", 0.0, 0.0, 0.008)
    assert _phase_converged("transport", 0.0, 0.0, -0.02)


def test_collector_state_saturates_only_allowed_open_width_overshoot() -> None:
    state = _observation_state({"q": np.arange(7), "width": 0.085})

    assert state.dtype == np.float32
    assert state.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0]
    with pytest.raises(Pi05ContractError, match="measurement tolerance"):
        _observation_state({"q": np.arange(7), "width": 0.101})


def test_grasp_reference_recovers_selected_asset_link_centroid(monkeypatch) -> None:
    import torch

    hand = np.asarray([0.39, 0.0, 0.46])
    local = np.asarray([3.38e-7, 3.08e-9, 0.0854165984])
    robot = SimpleNamespace(
        body_names=["panda_hand"],
        data=SimpleNamespace(
            body_pos_w=torch.as_tensor(hand.reshape(1, 1, 3), dtype=torch.float64),
            body_quat_w=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]),
        ),
    )
    corners = np.asarray(
        [
            hand + [local[0] - 0.005, local[1] - 0.005, 0.058533],
            hand + [local[0] + 0.005, local[1] + 0.005, 0.1123001968],
        ]
    )
    bounds = {side: {"world_corners_m": corners} for side in ("left", "right")}
    monkeypatch.setattr(pi05_isaac, "_finger_bounds", lambda _native: bounds)

    receipt = _finger_grasp_reference_geometry(SimpleNamespace(scene={"robot": robot}))

    assert receipt["hand_to_finger_link_centroid_local_m"] == pytest.approx(local)
    assert receipt["finger_link_hand_axis_interval_m"] == pytest.approx(
        [0.058533, 0.1123001968]
    )
    assert "bilateral contact required" in receipt["contact_geometry_claim"]


def _expert_settle_fixture():
    import torch

    native = SimpleNamespace(
        device="cpu",
        step_dt=1.0 / 15.0,
        cfg=SimpleNamespace(decimation=4),
        episode_length_buf=torch.tensor([0]),
        action_manager=SimpleNamespace(
            active_terms=["arm_action", "gripper_action"], action_term_dim=[6, 1]
        ),
    )

    class Env:
        def step(self, action):
            assert action.shape == (1, 7) and float(action[0, 6]) == 1.0
            native.episode_length_buf += 1
            return None, None, torch.tensor([False]), torch.tensor([False]), {}

    return native, Env()


def test_expert_command_uses_all_six_scaled_relative_pose_channels() -> None:
    import torch

    data = SimpleNamespace(
        target_pos_w=torch.tensor([[[0.4, 0.0, 0.3]]]),
        target_quat_w=torch.tensor([[[-0.912, 0.02, -0.406, 0.054]]]),
    )
    context = {
        "native": SimpleNamespace(scene={"ee_frame": SimpleNamespace(data=data)}),
        "total_dim": 7,
        "arm_start": 0,
        "grip_start": 6,
        "expert_action_scale": np.asarray([0.5] * 6),
    }
    command, distance, orientation = _expert_command(
        context, np.asarray([0.45, 0.0, 0.25]), 0.0
    )

    assert distance > 0.0 and orientation > 0.1
    assert np.linalg.norm(command[0, 3:6].numpy()) > 0.1
    assert float(command[0, 6]) == 1.0
    assert context["orientation_control"]["target_quaternion_xyzw"] == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]


def test_reset_settle_advances_physics_without_admitting_training_row(
    monkeypatch,
) -> None:
    native, env = _expert_settle_fixture()
    samples = iter(
        [
            (np.asarray([0.45, -0.15, 0.02775]), 0.08),
            (np.asarray([0.45, -0.15, 0.024]), 0.0),
        ]
    )
    support = iter([(0.0, 0.0, 0.0), (0.0, 0.0, 0.6)])
    monkeypatch.setattr(pi05_isaac, "_object_kinematics", lambda _native: next(samples))
    monkeypatch.setattr(pi05_isaac, "_contact_forces", lambda _native: next(support))
    receipt = _settle_expert_reset(native, env)
    assert receipt["control_steps"] == 2
    assert receipt["physics_steps"] == 8
    assert [row["object_position_m"][2] for row in receipt["samples"]] == pytest.approx(
        [0.02775, 0.024]
    )
    assert receipt["object_support_force_n"] == pytest.approx(0.6)
    assert receipt["training_row"] is False


def test_finger_bounds_transform_local_geometry_with_live_body_pose(
    monkeypatch,
) -> None:
    robot = SimpleNamespace(
        body_names=["panda_leftfinger", "panda_rightfinger"],
        data=SimpleNamespace(
            body_pos_w=_Proxy([[[0.1, 0.2, 0.3], [-0.1, 0.2, 0.3]]]),
            body_quat_w=_Proxy([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]]),
        ),
    )
    local = {
        side: ([-0.01, -0.02, 0.0], [0.01, 0.02, 0.1]) for side in ("left", "right")
    }
    monkeypatch.setattr(
        "npa.workflows.sim2real.pi05_isaac._finger_local_bounds",
        lambda _native: local,
    )

    first = _finger_bounds(SimpleNamespace(scene={"robot": robot}))
    robot.data.body_pos_w = _Proxy([[[0.2, 0.2, 0.3], [-0.2, 0.2, 0.3]]])
    second = _finger_bounds(SimpleNamespace(scene={"robot": robot}))

    assert np.allclose(
        second["left"]["world_corners_m"] - first["left"]["world_corners_m"],
        [0.1, 0.0, 0.0],
    )
    assert np.allclose(
        second["right"]["world_corners_m"] - first["right"]["world_corners_m"],
        [-0.1, 0.0, 0.0],
    )
    assert all(
        value["pose_source"] == "live_articulation_body_pose_xyzw"
        for value in second.values()
    )
