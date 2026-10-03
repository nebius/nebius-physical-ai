"""Build the warehouse geometry with runtime-fetched NVIDIA asset references."""


class _WarehouseLayout:
    def asset(self, path, relative, position=(0, 0, 0), yaw=0):
        """Preserve native transforms; convert units on a separate parent."""
        from pxr import Usd, UsdGeom

        url = self.path_join(
            self.asset_root, "Isaac/Environments/Simple_Warehouse/" + relative
        )
        if relative not in self.asset_records:
            source = Usd.Stage.Open(url)
            if not source or not source.GetDefaultPrim():
                raise RuntimeError("Asset has no default prim: " + url)
            mpu = UsdGeom.GetStageMetersPerUnit(source)
            up = str(UsdGeom.GetStageUpAxis(source))
            if up != "Z":
                raise ValueError("Asset requires explicit axis conversion: " + url)
            bound = (
                UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
                .ComputeWorldBound(source.GetDefaultPrim())
                .ComputeAlignedRange()
            )
            self.asset_records[relative] = {
                "url": url,
                "meters_per_unit": mpu,
                "min": list(bound.GetMin()),
                "max": list(bound.GetMax()),
                "size_m": [float(v) * mpu for v in bound.GetSize()],
            }
        group, _ = self.group(path, position)
        if yaw:
            group.AddRotateZOp().Set(yaw)
        mpu = self.asset_records[relative]["meters_per_unit"]
        group.AddScaleOp().Set(self.Gf.Vec3f(mpu, mpu, mpu))
        prim = self.stage.DefinePrim(path + "/Asset", "Xform")
        if not prim.GetReferences().AddReference(url):
            raise RuntimeError("Reference failed: " + url)
        return group

    def sign(
        self,
        name,
        label,
        center,
        width=2.2,
        height=0.48,
        bg=(241, 242, 236),
        fg=(25, 31, 30),
    ):
        texture = self._sign_texture(name, label, width, height, bg, fg)
        path = "/World/PrintedSigns/" + name
        material = self._sign_material(path, texture)
        self._sign_mesh(path, width, height, center, material)

    def _build_materials(self):
        for args in [
            ("paint", (0.63, 0.66, 0.64), 0.2, 0.48, 0),
            ("frame", (0.12, 0.19, 0.22), 0.55, 0.4, 0),
            ("steel", (0.43, 0.46, 0.47), 0.82, 0.28, 0),
            ("rubber", (0.035, 0.038, 0.04), 0, 0.85, 0),
            ("yellow", (0.95, 0.62, 0.045), 0.1, 0.65, 0),
            ("black", (0.022, 0.025, 0.025), 0, 0.7, 0),
            ("white", (0.84, 0.86, 0.81), 0, 0.72, 0),
            ("orange", (0.85, 0.26, 0.035), 0.1, 0.58, 0),
            ("green", (0.035, 0.48, 0.08), 0.1, 0.25, 0.8),
            ("red", (0.6, 0.025, 0.02), 0.1, 0.25, 0.6),
            ("screen", (0.1, 0.19, 0.17), 0.1, 0.5, 0.1),
        ]:
            self.mat(*args)
        self.asset("/World/Building", "warehouse.usd")

    def _build_lighting(self):
        for p in self.stage.Traverse():
            if str(p.GetPath()).startswith("/World/Building"):
                if p.IsA(self.Lux.RectLight):
                    p.GetAttribute("inputs:intensity").Set(2200.0)
                if p.IsA(self.Lux.DistantLight):
                    p.GetAttribute("inputs:intensity").Set(150.0)
        for i, (x, y) in enumerate(((-4, -3), (2, -3), (-4, 3), (3, 4))):
            light = self.Lux.RectLight.Define(self.stage, f"/World/WorkLights/L{i}")
            light.CreateWidthAttr(1.8)
            light.CreateHeightAttr(0.6)
            light.CreateIntensityAttr(1800)
            light.CreateColorAttr(self.Gf.Vec3f(1, 0.96, 0.9))
            light.AddTranslateOp().Set(self.Gf.Vec3d(x, y, 6.8))
        fill = self.Lux.DomeLight.Define(self.stage, "/World/WorkLights/Ambient")
        fill.CreateIntensityAttr(65)
        fill.CreateColorAttr(self.Gf.Vec3f(0.92, 0.95, 1))
        self.box("/World/Support", (0, 0, -0.07), (23.5, 35, 0.14), "paint", True)

    def _build_belt(self):
        from pxr import UsdPhysics, PhysxSchema

        belt = self.box(
            "/World/Cell/Belt", (-2, -2, 0.815), (6, 0.76, 0.07), "rubber", True
        )
        body = UsdPhysics.RigidBodyAPI.Apply(belt.GetPrim())
        body.CreateKinematicEnabledAttr(True)
        surface = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(belt.GetPrim())
        surface.CreateSurfaceVelocityEnabledAttr(True)
        surface.CreateSurfaceVelocityLocalSpaceAttr(False)
        self.belt_velocity = surface.CreateSurfaceVelocityAttr(
            self.Gf.Vec3f(self.belt_speed, 0, 0)
        )
        contact = self.Shade.Material.Define(self.stage, "/World/Looks/BeltContact")
        props = UsdPhysics.MaterialAPI.Apply(contact.GetPrim())
        props.CreateStaticFrictionAttr(0.65)
        props.CreateDynamicFrictionAttr(0.5)
        props.CreateRestitutionAttr(0.02)
        self.Shade.MaterialBindingAPI.Apply(belt.GetPrim()).Bind(
            contact, self.Shade.Tokens.weakerThanDescendants, "physics"
        )

    def _build_conveyor_frame(self):
        for i, y in enumerate((-2.46, -1.54)):
            self.box(
                f"/World/Cell/Conveyor/Rail{i}",
                (-2, y, 0.86),
                (6.18, 0.12, 0.22),
                "steel",
                True,
            )
            for j, x in enumerate((-4.6, -2.5, -0.4, 0.8)):
                self.box(
                    f"/World/Cell/Conveyor/Leg{i}_{j}",
                    (x, y, 0.39),
                    (0.09, 0.09, 0.78),
                    "frame",
                )
                self.box(
                    f"/World/Cell/Conveyor/Foot{i}_{j}",
                    (x, y, 0.045),
                    (0.24, 0.18, 0.05),
                    "black",
                )
        self._build_conveyor_drive()

    def _build_seams(self):
        self.slats = []
        for i in range(38):
            obj = self.box(
                f"/World/Cell/Conveyor/Seam{i}",
                (-5 + i * 0.158, -2, 0.851),
                (0.011, 0.75, 0.002),
                "black",
            )
            self.slats.append(obj.GetOrderedXformOps()[0])

    def _build_cartons(self):
        from pxr import Usd, UsdPhysics, PhysxSchema
        from isaacsim.core.prims import SingleRigidPrim

        for i, x in enumerate(self.initial_x):
            path = f"/World/Cartons/Carton{i}"
            root, _ = self.group(path, (x, -2, 0.87))
            self.asset(path + "/Appearance", "Props/SM_CardBoxC_01.usd")
            for p in Usd.PrimRange(self.stage.GetPrimAtPath(path + "/Appearance")):
                if p.HasAPI(UsdPhysics.RigidBodyAPI):
                    p.RemoveAPI(UsdPhysics.RigidBodyAPI)
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    p.RemoveAPI(UsdPhysics.CollisionAPI)
                if p.HasAPI(UsdPhysics.MeshCollisionAPI):
                    p.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            collider = self.box(
                path + "/Collider", (0, 0, 0.125), (0.5, 0.5, 0.25), "paint", True
            )
            collider.CreateVisibilityAttr().Set(self.Geom.Tokens.invisible)
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
            UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(2.5)
            px = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
            px.CreateEnableCCDAttr(True)
            px.CreateSolverPositionIterationCountAttr(32)
            px.CreateSolverVelocityIterationCountAttr(8)
            self.packages.append(
                self.world.scene.add(
                    SingleRigidPrim(path, name=f"warehouse_carton_{i}", mass=2.5)
                )
            )

    def _build_pallet(self):
        self.asset("/World/Cell/Pallet", "Props/SM_PaletteA_01.usd", (2.7, 0.65, 0))
        deck = self.box(
            "/World/Cell/PalletDeck",
            (2.7, 0.65, 0.194),
            (1.2, 0.99, 0.034),
            "paint",
            True,
        )
        deck.CreateVisibilityAttr().Set(self.Geom.Tokens.invisible)

    def _build_gantry_frame(self):
        for i, (x, y) in enumerate(
            ((-0.65, -3.1), (4.25, -3.1), (-0.65, 2.35), (4.25, 2.35))
        ):
            self.box(
                f"/World/Cell/Gantry/Post{i}", (x, y, 1.8), (0.14, 0.14, 3.6), "paint"
            )
            self.box(
                f"/World/Cell/Gantry/Base{i}", (x, y, 0.04), (0.4, 0.4, 0.08), "steel"
            )
            for j, (dx, dy) in enumerate(
                ((-0.13, -0.13), (0.13, -0.13), (-0.13, 0.13), (0.13, 0.13))
            ):
                self.cylinder(
                    f"/World/Cell/Gantry/Bolt{i}_{j}",
                    (x + dx, y + dy, 0.09),
                    0.022,
                    0.018,
                    "steel",
                )
        for i, x in enumerate((-0.65, 4.25)):
            self.box(
                f"/World/Cell/Gantry/YRail{i}",
                (x, -0.375, 3.65),
                (0.18, 5.7, 0.23),
                "frame",
            )

    def _build_gantry_axes(self):
        beam = self.box(
            "/World/Cell/Gantry/MovingBeam", (1.8, -2, 3.65), (5.0, 0.23, 0.25), "steel"
        )
        self.beam_op = beam.GetOrderedXformOps()[0]
        trolley = self.box(
            "/World/Cell/Gantry/Trolley", (0, -2, 3.8), (0.42, 0.42, 0.21), "frame"
        )
        self.trolley_op = trolley.GetOrderedXformOps()[0]
        shaft = self.box(
            "/World/Cell/Gantry/VerticalSlide", (0, -2, 2.8), (0.12, 0.12, 1.8), "steel"
        )
        self.shaft_ops = shaft.GetOrderedXformOps()

    def _build_gripper(self):
        from pxr import UsdPhysics

        gripper, self.gripper_op = self.group("/World/Cell/Gripper", self.grip_position)
        UsdPhysics.RigidBodyAPI.Apply(gripper.GetPrim()).CreateKinematicEnabledAttr(
            True
        )
        self.box("/World/Cell/Gripper/Plate", (0, 0, 0.08), (0.38, 0.34, 0.07), "steel")
        for i, (x, y) in enumerate(
            ((-0.12, -0.1), (0.12, -0.1), (-0.12, 0.1), (0.12, 0.1))
        ):
            self.cylinder(
                f"/World/Cell/Gripper/Cup{i}", (x, y, 0.025), 0.045, 0.05, "rubber"
            )
        self.cylinder(
            "/World/Cell/Gripper/AirFitting", (0, 0, 0.16), 0.035, 0.12, "orange"
        )

    def _build_guards_front(self):
        for i, x in enumerate((-0.9, 0.5, 1.9, 3.3, 4.7)):
            self.box(
                f"/World/Cell/Fence/Post{i}",
                (x, -3.6, 1.05),
                (0.055, 0.055, 2.1),
                "yellow",
            )
        for i, z in enumerate((0.13, 1.99)):
            self.box(
                f"/World/Cell/Fence/Rail{i}",
                (1.9, -3.6, z),
                (5.65, 0.035, 0.035),
                "frame",
            )
        for i in range(81):
            self.box(
                f"/World/Cell/Fence/WireV{i}",
                (-0.9 + i * 0.07, -3.6, 1.04),
                (0.007, 0.012, 1.82),
                "frame",
            )
        for i in range(27):
            self.box(
                f"/World/Cell/Fence/WireH{i}",
                (1.9, -3.6, 0.13 + i * 0.07),
                (5.6, 0.012, 0.007),
                "frame",
            )

    def _build_guards_sides(self):
        for side, x, y0, y1 in [("Right", 4.7, -3.6, 2.7), ("Left", -0.9, -1.3, 2.7)]:
            center = (y0 + y1) / 2
            length = y1 - y0
            for j, y in enumerate((y0, center, y1)):
                self.box(
                    f"/World/Cell/Fence/{side}Post{j}",
                    (x, y, 1.05),
                    (0.055, 0.055, 2.1),
                    "yellow",
                    True,
                )
            for j in range(round(length / 0.1) + 1):
                self.box(
                    f"/World/Cell/Fence/{side}V{j}",
                    (x, y0 + j * 0.1, 1.04),
                    (0.012, 0.007, 1.82),
                    "frame",
                )
            for j in range(20):
                self.box(
                    f"/World/Cell/Fence/{side}H{j}",
                    (x, center, 0.13 + j * 0.098),
                    (0.012, length, 0.007),
                    "frame",
                )
            guard = self.box(
                f"/World/Cell/Fence/{side}Collision",
                (x, center, 1.04),
                (0.018, length, 1.82),
                "frame",
                True,
            )
            guard.CreateVisibilityAttr().Set(self.Geom.Tokens.invisible)

    def _build_guards_rear(self):
        for j, x in enumerate((-0.9, 0.5, 1.9, 3.3, 4.7)):
            self.box(
                f"/World/Cell/Fence/RearPost{j}",
                (x, 2.7, 1.05),
                (0.055, 0.055, 2.1),
                "yellow",
                True,
            )
        for j in range(57):
            self.box(
                f"/World/Cell/Fence/RearV{j}",
                (-0.9 + j * 0.1, 2.7, 1.04),
                (0.007, 0.012, 1.82),
                "frame",
            )
        for j in range(20):
            self.box(
                f"/World/Cell/Fence/RearH{j}",
                (1.9, 2.7, 0.13 + j * 0.098),
                (5.6, 0.012, 0.007),
                "frame",
            )
        guard = self.box(
            "/World/Cell/Fence/RearCollision",
            (1.9, 2.7, 1.04),
            (5.6, 0.018, 1.82),
            "frame",
            True,
        )
        guard.CreateVisibilityAttr().Set(self.Geom.Tokens.invisible)

    def _build_markings(self):
        for i, y in enumerate((-4.05, -5.65)):
            self.box(
                f"/World/Markings/Walkway{i}",
                (-0.5, y, 0.003),
                (16, 0.06, 0.005),
                "yellow",
            )
        for i in range(9):
            self.box(
                f"/World/Markings/Crossing{i}",
                (-6.6, -4.9 + i * 0.18, 0.004),
                (1.2, 0.085, 0.004),
                "white",
            )
        for i, x in enumerate((-0.9, 4.7)):
            self.box(
                f"/World/Markings/CellEdge{i}",
                (x, -0.5, 0.004),
                (0.06, 6.2, 0.005),
                "yellow",
            )

    def _build_signs(self):
        self.sign("CellID", "PACKING CELL 03", (1.8, -3.62, 2.15), 3.3, 0.4)
        self.sign(
            "Complete", "PALLET COMPLETE", (1.8, -3.63, 2.55), 2.6, 0.28, (240, 194, 70)
        )
        self.sign(
            "Reset",
            "DEMO EPISODE RESTART",
            (1.8, -3.63, 2.55),
            2.6,
            0.28,
            (240, 194, 70),
        )
        self.Geom.Imageable(
            self.stage.GetPrimAtPath("/World/PrintedSigns/Complete/Face")
        ).MakeInvisible()
        self.Geom.Imageable(
            self.stage.GetPrimAtPath("/World/PrintedSigns/Reset/Face")
        ).MakeInvisible()
        self._build_warning_signs()

    def _build_controls(self):
        self.box(
            "/World/Controls/Pedestal", (4.9, -3.7, 0.63), (0.15, 0.15, 1.26), "frame"
        )
        self.box(
            "/World/Controls/Cabinet", (4.9, -3.7, 1.37), (0.64, 0.24, 0.43), "paint"
        )
        self.box(
            "/World/Controls/Screen",
            (4.81, -3.831, 1.39),
            (0.33, 0.008, 0.22),
            "screen",
        )
        self.cylinder(
            "/World/Controls/EStop", (5.12, -3.85, 1.33), 0.04, 0.035, "red", "Y"
        )
        self.sign(
            "RunState",
            "RUNNING",
            (4.81, -3.838, 1.39),
            0.3,
            0.15,
            (27, 48, 41),
            (225, 239, 220),
        )
        self.cylinder(
            "/World/Controls/StackPole", (4.9, -3.7, 1.98), 0.023, 0.6, "steel"
        )
        self.stacklight = self.cylinder(
            "/World/Controls/Stacklight", (4.9, -3.7, 2.3), 0.065, 0.16, "green"
        )

    def _build_inventory(self):
        for row, y in enumerate((5.0, 9.0)):
            for i, x in enumerate((-2.2, 1.8, 5.8)):
                self.asset(
                    f"/World/Inventory/Frame{row}_{i}",
                    "Props/SM_RackFrame_03.usd",
                    (x, y, 0),
                )
            for bay, x in enumerate((-0.2, 3.8)):
                self._build_inventory_bay(row, bay, x, y)

    def _build_inventory_bay(self, row, bay, x, y):
        for level, z in enumerate((0.52, 1.63, 2.74)):
            self.asset(
                f"/World/Inventory/Shelf{row}_{bay}_{level}",
                "Props/SM_RackShelf_01.usd",
                (x, y, z),
            )
            for carton in range(5):
                self.asset(
                    f"/World/Inventory/Carton{row}_{bay}_{level}_{carton}",
                    "Props/SM_CardBoxA_01.usd",
                    (x - 1.45 + carton * 0.72, y, z + 0.03),
                )

    def _build_staging(self):
        for i, (x, y) in enumerate(((-6, 1.6), (-6, 3.4), (5.9, 1.8), (7.7, 1.8))):
            self.asset(
                f"/World/Staging/Pallet{i}", "Props/SM_PaletteA_01.usd", (x, y, 0)
            )
            for j in range(4):
                self.asset(
                    f"/World/Staging/Box{i}_{j}",
                    "Props/SM_CardBoxC_01.usd",
                    (x + (-0.28 if j % 2 == 0 else 0.28), y, 0.212 + j // 2 * 0.25),
                )
        self.asset(
            "/World/Staging/HandCart", "Props/SM_PushcartA_02.usd", (-6.8, 6.5, 0), 90
        )

    def _build_render_settings(self):
        import carb

        settings = carb.settings.get_settings()
        for k, v in {
            "/rtx/rendermode": "RayTracedLighting",
            "/rtx/post/tonemap/op": 4,
            "/rtx/post/tonemap/filmIso": 160.0,
            "/rtx/post/tonemap/whitepoint": 6500.0,
            "/rtx/post/tonemap/enabled": True,
            "/rtx/post/aa/op": 3,
        }.items():
            settings.set(k, v)

    def build(self):
        self._build_materials()
        self._build_lighting()
        self._build_belt()
        self._build_conveyor_frame()
        self._build_seams()
        self._build_cartons()
        self._build_pallet()
        self._build_gantry_frame()
        self._build_gantry_axes()
        self._build_gripper()
        self._build_guards_front()
        self._build_guards_sides()
        self._build_guards_rear()
        self._build_markings()
        self._build_signs()
        self._build_controls()
        self._build_inventory()
        self._build_staging()
        self._build_render_settings()

    def _sign_texture(self, name, label, width, height, bg, fg):
        from PIL import Image, ImageDraw, ImageFont

        out = self.output_directory / "signs"
        out.mkdir(parents=True, exist_ok=True)
        im = Image.new("RGB", (1200, max(120, round(1200 * height / width))), bg)
        draw = ImageDraw.Draw(im)
        font = ImageFont.load_default(size=int(im.height * 0.46))
        while draw.textbbox((0, 0), label, font=font)[2] > im.width * 0.91:
            font = ImageFont.load_default(size=font.size - 2)
        box = draw.textbbox((0, 0), label, font=font)
        draw.text(
            ((im.width - box[2]) / 2, (im.height - box[3] - box[1]) / 2),
            label,
            font=font,
            fill=fg,
        )
        texture = out / (name + ".png")
        im.save(texture)
        return texture

    def _sign_material(self, path, texture):
        Sdf, Shade = (self.Sdf, self.Shade)
        mat = Shade.Material.Define(self.stage, path + "/Material")
        shader = Shade.Shader.Define(self.stage, path + "/Material/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
        uv = Shade.Shader.Define(self.stage, path + "/Material/UV")
        uv.CreateIdAttr("UsdPrimvarReader_float2")
        uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex = Shade.Shader.Define(self.stage, path + "/Material/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            uv.ConnectableAPI(), "result"
        )
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    def _sign_mesh(self, path, width, height, center, mat):
        Sdf, Shade, Gf = (self.Sdf, self.Shade, self.Gf)
        mesh = self.Geom.Mesh.Define(self.stage, path + "/Face")
        mesh.CreatePointsAttr(
            [
                (-width / 2, 0, -height / 2),
                (width / 2, 0, -height / 2),
                (width / 2, 0, height / 2),
                (-width / 2, 0, height / 2),
            ]
        )
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        self.Geom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, self.Geom.Tokens.vertex
        ).Set([(0, 0), (1, 0), (1, 1), (0, 1)])
        mesh.AddTranslateOp().Set(Gf.Vec3d(*center))
        Shade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

    def _build_conveyor_drive(self):
        for i, x in enumerate((-5.04, 1.04)):
            self.cylinder(
                f"/World/Cell/Conveyor/EndRoller{i}",
                (x, -2, 0.8),
                0.115,
                0.95,
                "steel",
                "Y",
            )
        self.box(
            "/World/Cell/Conveyor/Stop",
            (1.02, -2, 0.99),
            (0.08, 0.75, 0.27),
            "steel",
            True,
        )
        self.box(
            "/World/Cell/Conveyor/Motor", (-4.4, -2.7, 0.68), (0.38, 0.4, 0.3), "frame"
        )
        self.cylinder(
            "/World/Cell/Conveyor/MotorEnd",
            (-4.4, -2.92, 0.68),
            0.13,
            0.08,
            "steel",
            "Y",
        )

    def _build_warning_signs(self):
        self.sign(
            "Guard",
            "AUTOMATIC EQUIPMENT - KEEP CLEAR",
            (2.1, -3.63, 1.1),
            2.5,
            0.35,
            (245, 201, 62),
        )
        self.sign(
            "Access", "SERVICE ACCESS", (3.9, 2.675, 1.55), 1.1, 0.22, (245, 201, 62)
        )
        self.sign("Receiving", "INBOUND CARTONS", (-3.3, -1.35, 1.45), 2.6, 0.34)
        self.sign(
            "Walkway",
            "PEDESTRIAN WALKWAY",
            (-5.9, 3.3, 2.45),
            2.7,
            0.38,
            (43, 93, 65),
            (242, 245, 235),
        )
