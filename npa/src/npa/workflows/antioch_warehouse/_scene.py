"""Run the contact conveyor and scripted palletizer authored in the local Antioch demo."""

from __future__ import annotations

import json
from ._layout import _WarehouseLayout


class _WarehouseCell(_WarehouseLayout):
    """Own the native PhysX warehouse state and measured batch controller."""

    def mat(self, name, color, metal=0.0, rough=0.4, emission=0.0):
        Gf, Sdf, Shade = (self.Gf, self.Sdf, self.Shade)
        mat = Shade.Material.Define(self.stage, "/World/Looks/" + name)
        sh = Shade.Shader.Define(self.stage, str(mat.GetPath()) + "/Surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*(c * emission for c in color))
        )
        sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        self.materials[name] = mat
        return mat

    def bind(self, prim, material):
        self.Shade.MaterialBindingAPI.Apply(prim).Bind(self.materials[material])

    def group(self, path, pos=(0, 0, 0)):
        obj = self.Geom.Xform.Define(self.stage, path)
        op = obj.AddTranslateOp()
        op.Set(self.Gf.Vec3d(*pos))
        return (obj, op)

    def box(self, path, pos, size, material, collider=False, rotate=None):
        obj = self.Geom.Cube.Define(self.stage, path)
        obj.CreateSizeAttr(1)
        obj.AddTranslateOp().Set(self.Gf.Vec3d(*pos))
        if rotate:
            obj.AddRotateXYZOp().Set(self.Gf.Vec3f(*rotate))
        obj.AddScaleOp().Set(self.Gf.Vec3f(*size))
        self.bind(obj.GetPrim(), material)
        if collider:
            self.Physics.CollisionAPI.Apply(obj.GetPrim())
        return obj

    def cylinder(self, path, pos, radius, height, material, axis="Z"):
        obj = self.Geom.Cylinder.Define(self.stage, path)
        obj.CreateRadiusAttr(radius)
        obj.CreateHeightAttr(height)
        obj.CreateAxisAttr(axis)
        obj.AddTranslateOp().Set(self.Gf.Vec3d(*pos))
        self.bind(obj.GetPrim(), material)
        return obj

    def event(self, name, **details):
        event = {"sim_s": self.t, "event": name, **details}
        self.events.append(event)
        print(json.dumps(event), flush=True)

    def __init__(self, world, output_directory):
        self.world, self.stage = (world, world.stage)
        from pxr import Gf, UsdGeom, UsdShade, UsdPhysics, UsdLux, Sdf, PhysxSchema

        self.Gf, self.Geom, self.Shade, self.Physics = (
            Gf,
            UsdGeom,
            UsdShade,
            UsdPhysics,
        )
        self.Lux, self.Sdf, self.Px = (UsdLux, Sdf, PhysxSchema)
        self.materials = {}
        self.events, self.samples, self.packages = ([], [], [])
        self.frames = {}
        self.t, self.last_sample = (0.0, -1.0)
        self.output_directory = output_directory
        from isaacsim.storage.native import get_assets_root_path, path_join

        self.asset_root = get_assets_root_path()
        self.path_join = path_join
        self.asset_records = {}
        self.phase = "CONVEYING"
        self.phase_since = 0.0
        self.placed = []
        self.completed_batches = 0
        self.total_placements = 0
        self.active = None
        self.grip_position = [0.0, -2.0, 2.3]
        self.move = None
        self.attached = False
        self.max_grip_error = 0.0
        self.max_belt_dx = 0.0
        self.hold_speed = None
        self.episode = 0
        self.reset_at = None
        self.measured_settled = []
        self.last_board = None
        self.belt_speed = 0.45
        self.initial_x = [0.0, -0.9, -1.8, -2.7, -3.6, -4.5]

    def set_phase(self, phase):
        self.phase = phase
        self.phase_since = self.t
        self.event(phase, episode=self.episode, carton=self.active)

    def move_to(self, target, next_phase):
        import numpy as np

        origin = np.array(self.grip_position, float)
        goal = np.array(target, float)
        duration = max(0.4, 1.5 * float(np.linalg.norm(goal - origin)) / 0.9)
        self.move = (self.t, duration, origin, goal, next_phase)

    def vacuum(self, attached):
        from pxr import UsdPhysics

        path = "/World/Cell/VacuumAttachment"
        if attached:
            p, _ = self.packages[self.active].get_world_pose()
            joint = UsdPhysics.FixedJoint.Define(self.stage, path)
            joint.CreateBody0Rel().SetTargets(["/World/Cell/Gripper"])
            joint.CreateBody1Rel().SetTargets([f"/World/Cartons/Carton{self.active}"])
            joint.CreateLocalPos0Attr(self.Gf.Vec3f(0, 0, 0))
            joint.CreateLocalPos1Attr(self.Gf.Vec3f(0, 0, 0.25))
            joint.CreateLocalRot0Attr(self.Gf.Quatf(1))
            joint.CreateLocalRot1Attr(self.Gf.Quatf(1))
            self.event(
                "vacuum_attached", carton=self.active, base=[float(v) for v in p]
            )
        elif self.stage.GetPrimAtPath(path):
            self.stage.RemovePrim(path)
            self.event("vacuum_released", carton=self.active)
        self.attached = attached

    def target_slot(self):
        n = len(self.placed)
        return [2.7 + (-0.28 if n % 2 == 0 else 0.28), 0.65, 0.212 + n // 2 * 0.25]

    def animate(self):
        x, y, z = self.grip_position
        Gf = self.Gf
        self.gripper_op.Set(Gf.Vec3d(x, y, z))
        self.beam_op.Set(Gf.Vec3d(1.8, y, 3.65))
        self.trolley_op.Set(Gf.Vec3d(x, y, 3.8))
        self.shaft_ops[0].Set(Gf.Vec3d(x, y, (3.6 + z) / 2))
        self.shaft_ops[-1].Set(Gf.Vec3f(0.12, 0.12, max(0.05, 3.6 - z)))
        moving = self.phase == "CONVEYING"
        self.bind(
            self.stacklight.GetPrim(),
            "yellow" if self.phase in ("BATCH_COMPLETE", "DEMO_RESET") else "green",
        )
        for name, phase in [("Complete", "BATCH_COMPLETE"), ("Reset", "DEMO_RESET")]:
            face = self.Geom.Imageable(
                self.stage.GetPrimAtPath("/World/PrintedSigns/" + name + "/Face")
            )
            if self.phase == phase:
                face.MakeVisible()
            else:
                face.MakeInvisible()
        for i, op in enumerate(self.slats):
            if moving:
                op.Set(
                    Gf.Vec3d(-5 + (i * 0.158 + self.t * self.belt_speed) % 6, -2, 0.851)
                )

    def physics_tick(self, dt):
        import numpy as np

        self.t += dt
        for i, b in enumerate(self.packages):
            p, _ = b.get_world_pose()
            if i not in self.placed and i != self.active:
                self.max_belt_dx = max(
                    self.max_belt_dx, float(p[0]) - self.initial_x[i]
                )
        if self.move is not None:
            start, duration, origin, goal, next_phase = self.move
            u = min(1.0, (self.t - start) / duration)
            smooth = u * u * (3 - 2 * u)
            self.grip_position = list(origin + (goal - origin) * smooth)
            if u >= 1:
                self.move = None
                self.set_phase(next_phase)
        if self.attached:
            p, _ = self.packages[self.active].get_world_pose()
            error = float(
                np.linalg.norm(p + np.array([0, 0, 0.25]) - self.grip_position)
            )
            self.max_grip_error = max(self.max_grip_error, error)
        getattr(self, "_phase_" + self.phase.lower(), lambda: None)()
        if self.t - self.last_sample > 0.5:
            self.record_sample()

    def record_sample(self):
        self.last_sample = self.t
        sample = {
            "sim_s": self.t,
            "phase": self.phase,
            "episode": self.episode,
            "placed": len(self.placed),
            "gripper": self.grip_position.copy(),
            "cartons": [
                [float(value) for value in body.get_world_pose()[0]]
                for body in self.packages
            ],
        }
        if self.samples and self.samples[-1]["sim_s"] == self.t:
            self.samples[-1] = sample
        else:
            self.samples.append(sample)
        (self.output_directory / "progress.json").write_text(json.dumps(sample) + "\n")

    def _phase_conveying(self):
        self.belt_velocity.Set(self.Gf.Vec3f(self.belt_speed, 0, 0))
        candidates = [
            (i, b.get_world_pose()[0])
            for i, b in enumerate(self.packages)
            if i not in self.placed
        ]
        if candidates:
            i, p = max(candidates, key=lambda item: float(item[1][0]))
            if p[0] >= 0.35:
                self.active = i
                self.belt_velocity.Set(self.Gf.Vec3f(0, 0, 0))
                self.set_phase("ACCUMULATION_STOP")

    def _phase_accumulation_stop(self):
        import numpy as np

        if self.t - self.phase_since <= 0.8:
            return
        p, _ = self.packages[self.active].get_world_pose()
        speed = float(np.linalg.norm(self.packages[self.active].get_linear_velocity()))
        if speed < 0.04:
            self.hold_speed = speed
            self.pick = [float(v) for v in p]
            self.set_phase("APPROACH")
            self.move_to([p[0], p[1], 2.3], "LOWER")

    def _phase_lower(self):
        if self.move is not None:
            return
        self.move_to([self.pick[0], self.pick[1], self.pick[2] + 0.251], "ATTACH")

    def _phase_attach(self):
        if self.attached:
            return
        self.vacuum(True)
        self.set_phase("GRIP_DWELL")

    def _phase_grip_dwell(self):
        if self.t - self.phase_since <= 0.4:
            return
        self.set_phase("LIFT")
        self.move_to([self.pick[0], self.pick[1], 2.3], "TRANSFER")

    def _phase_transfer(self):
        if self.move is not None:
            return
        self.slot = self.target_slot()
        self.move_to([self.slot[0], self.slot[1], 2.3], "LOWER_TO_PALLET")

    def _phase_lower_to_pallet(self):
        if self.move is not None:
            return
        self.move_to([self.slot[0], self.slot[1], self.slot[2] + 0.255], "RELEASE")

    def _phase_release(self):
        self.vacuum(False)
        self.set_phase("SETTLING")

    def _phase_settling(self):
        import numpy as np

        if self.t - self.phase_since <= 0.8:
            return
        p, _ = self.packages[self.active].get_world_pose()
        error = float(np.linalg.norm(p - np.array(self.slot)))
        speed = float(np.linalg.norm(self.packages[self.active].get_linear_velocity()))
        self.measured_settled.append(
            {
                "carton": self.active,
                "episode": self.episode,
                "sim_s": self.t,
                "target": self.slot,
                "actual": [float(v) for v in p],
                "error_m": error,
                "speed_m_s": speed,
            }
        )
        self.placed.append(self.active)
        self.total_placements += 1
        self.event("carton_placed", carton=self.active, error_m=error, speed_m_s=speed)
        self.set_phase("RETRACT")
        self.move_to([self.slot[0], self.slot[1], 2.3], "NEXT_CARTON")

    def _phase_next_carton(self):
        self.active = None
        if len(self.placed) == len(self.packages):
            self.completed_batches += 1
            self.set_phase("BATCH_COMPLETE")
        else:
            self.set_phase("CONVEYING")
