from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workflows.sim2real.pi05_closed_loop import _new_episode_record
from npa.workflows.sim2real.pi05_isaac import (
    DESTINATION_LABEL,
    DESTINATION_RADIUS_M,
    _camera_config,
    _camera_pose_sample,
    _camera_time,
    _contact_config,
    _destination_pixel_count,
    _finger_bounds,
    _finger_in_frustum,
    _finger_pixel_counts,
    _multiply_xyzw,
    _quaternion_angle_xyzw,
    _relative_camera_quaternion,
    _require_isaac_camera_runtime,
    _ros_look_at_xyzw,
    _ros_to_opengl_xyzw,
    _rotate_vector,
    _sim_time,
    _sensor_clock_reference,
    _synchronize_wrist_before_capture,
    _target_pixel_count,
    _untransformed_aligned_bounds,
    _validate_capture_ages,
    _wrist_mount_receipt,
    _world_pose_matrix,
    _xyzw_to_wxyz,
    configure_destination_marker,
    configure_finger_semantics,
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
    assert cfg.scene.pi05_wrist.offset.rot == (0.0, 0.0, 0.0, 1.0)
    assert cfg.scene.pi05_wrist.offset.convention == "ros"
    assert cfg.scene.pi05_wrist.prim_path == "{ENV_REGEX_NS}/Pi05WristCamera"
    assert cfg.scene.pi05_exterior.offset.convention == "ros"
    assert cfg.scene.pi05_wrist.spawn.focal_length == 18.0


def test_ros_camera_axes_aim_at_declared_geometry() -> None:
    position = np.asarray([1.25, -1.25, 1.05])
    target = np.asarray([0.5, 0.0, 0.05])
    quaternion = _ros_look_at_xyzw(position, target)
    optical_axis = _rotate_vector(_xyzw_to_wxyz(quaternion), [0.0, 0.0, 1.0])
    expected = (target - position) / np.linalg.norm(target - position)
    assert optical_axis == pytest.approx(expected)
    wrist_axis = _rotate_vector(_xyzw_to_wxyz([0.0, 0.0, 0.0, 1.0]), [0, 0, 1])
    assert wrist_axis == pytest.approx([0.0, 0.0, 1.0])


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
    return SimpleNamespace(scene={"camera": SimpleNamespace(data=data)})


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
    native = SimpleNamespace(scene={"camera": SimpleNamespace(data=finger_data)})
    assert _finger_pixel_counts(native, "camera") == {"left": 2, "right": 1}


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
    assert np.allclose(receipt["after"]["camera_data_position_m"], [0.425, 0.0, 0.435])
    assert np.allclose(receipt["after"]["renderer_usd_position_m"], [0.425, 0.0, 0.435])
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
    assert np.allclose(receipt["after"]["camera_data_position_m"], [0.425, 0.0, 0.435])


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
        [0.525, -0.1, 0.635],
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

    assert np.allclose(receipt["after"]["camera_data_position_m"], [0.375, 0.0, 0.435])
    assert np.allclose(
        receipt["after"]["camera_data_quaternion_xyzw"], [0.0, 0.0, 1.0, 0.0]
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
            "pose_source": "live_articulation_body_pos_w_body_quat_w_xyzw",
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
        value["pose_source"] == "live_articulation_body_pos_w_body_quat_w_xyzw"
        for value in second.values()
    )
