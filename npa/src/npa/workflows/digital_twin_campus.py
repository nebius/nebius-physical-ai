"""Build an authored industrial campus with instanced geometry and measured asset counts."""

from __future__ import annotations

from collections import Counter
import math


class _Campus:
    def __init__(self, bpy):
        self.bpy = bpy
        self.meshes = {}
        self.materials = _materials(bpy)

    def _object(self, name, mesh, position, size, role="detail"):
        obj = self.bpy.data.objects.new(name, mesh)
        self.bpy.context.scene.collection.objects.link(obj)
        obj.location, obj.scale = position, size
        obj["asset_role"] = role
        return obj

    def _box(self, name, position, size, material, role="detail"):
        key = ("box", material)
        if key not in self.meshes:
            vertices = [
                (x / 2, y / 2, z / 2)
                for x, y, z in (
                    (-1, -1, -1),
                    (1, -1, -1),
                    (1, 1, -1),
                    (-1, 1, -1),
                    (-1, -1, 1),
                    (1, -1, 1),
                    (1, 1, 1),
                    (-1, 1, 1),
                )
            ]
            faces = [
                (3, 2, 1, 0),
                (4, 5, 6, 7),
                (0, 1, 5, 4),
                (1, 2, 6, 5),
                (2, 3, 7, 6),
                (3, 0, 4, 7),
            ]
            mesh = self.bpy.data.meshes.new(f"Shared {material} box")
            mesh.from_pydata(vertices, [], faces)
            mesh.materials.append(self.materials[material])
            self.meshes[key] = mesh
        return self._object(name, self.meshes[key], position, size, role)

    def _cylinder(self, name, position, radius, depth, material, role="detail"):
        key = ("cylinder", material)
        if key not in self.meshes:
            self.bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=1, depth=1)
            template = self.bpy.context.object
            mesh = template.data
            mesh.materials.append(self.materials[material])
            for face in mesh.polygons:
                face.use_smooth = len(face.vertices) == 4
            self.meshes[key] = mesh
            self.bpy.data.objects.remove(template, do_unlink=True)
        return self._object(
            name, self.meshes[key], position, (radius, radius, depth), role
        )

    def _sphere(self, name, position, size, material):
        key = ("sphere", material)
        if key not in self.meshes:
            self.bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8)
            template = self.bpy.context.object
            mesh = template.data
            mesh.materials.append(self.materials[material])
            for face in mesh.polygons:
                face.use_smooth = True
            self.meshes[key] = mesh
            self.bpy.data.objects.remove(template, do_unlink=True)
        return self._object(name, self.meshes[key], position, size)

    def _beam(self, name, start, end, radius, material):
        from mathutils import Vector

        direction = Vector(end) - Vector(start)
        obj = self._cylinder(
            name, (Vector(start) + Vector(end)) / 2, radius, direction.length, material
        )
        obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
        return obj


def _materials(bpy):
    palette = {
        "ground": ((0.038, 0.059, 0.071), 0.0, 0.82),
        "concrete": ((0.26, 0.32, 0.35), 0.1, 0.65),
        "road": ((0.024, 0.037, 0.047), 0.1, 0.35),
        "white": ((0.75, 0.81, 0.81), 0.32, 0.3),
        "steel": ((0.16, 0.23, 0.29), 0.85, 0.24),
        "glass": ((0.026, 0.14, 0.20), 0.68, 0.13),
        "orange": ((0.94, 0.20, 0.032), 0.48, 0.28),
        "teal": ((0.012, 0.37, 0.39), 0.48, 0.25),
        "blue": ((0.025, 0.12, 0.36), 0.5, 0.3),
        "red": ((0.57, 0.045, 0.021), 0.4, 0.32),
        "yellow": ((0.96, 0.61, 0.09), 0.2, 0.38),
        "grass": ((0.038, 0.13, 0.075), 0.0, 0.94),
        "leaf": ((0.035, 0.20, 0.105), 0.0, 0.8),
        "solar": ((0.018, 0.061, 0.16), 0.72, 0.2),
    }
    result = {name: _material(bpy, name, *values) for name, values in palette.items()}
    for name, color in (
        ("cyan-light", (0.05, 0.8, 1)),
        ("warm-light", (1, 0.61, 0.22)),
    ):
        result[name] = _material(bpy, name, color, 0, 0.3)
        shader = result[name].node_tree.nodes["Principled BSDF"]
        shader.inputs["Emission Color"].default_value = (*color, 1)
        shader.inputs["Emission Strength"].default_value = 4
    return result


def _material(bpy, name, color, metallic, roughness):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = (*color, 1)
    shader = material.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    return material


def _site(campus):
    campus._box("Campus foundation", (0, 0, -1.8), (500, 360, 3.6), "concrete", "site")
    campus._box("Surrounding terrain", (0, 0, -4), (4000, 4000, 3), "ground", "terrain")
    for y in (-155, -75, 45, 155):
        campus._box("East-west campus road", (0, y, 0.04), (490, 15, 0.08), "road")
        for x in range(-240, 245, 12):
            campus._box("Lane divider", (x, y, 0.095), (5, 0.2, 0.025), "white")
    for x in (-225, -15, 220):
        campus._box("North-south campus road", (x, 0, 0.05), (14, 345, 0.1), "road")
        for y in range(-165, 166, 12):
            campus._box("Lane divider", (x, y, 0.12), (0.2, 5, 0.025), "white")
    for x in (-246, 246):
        campus._box("Perimeter curb", (x, 0, 0.4), (1, 350, 0.8), "white")
    for y in (-176, 176):
        campus._box("Perimeter curb", (0, y, 0.4), (492, 1, 0.8), "white")
    for x in range(-205, 211, 25):
        for y in (-144, 144):
            _tree(campus, x, y)
            _streetlight(campus, x + 8, y)


def _tree(campus, x, y):
    campus._box("Landscape planter", (x, y, 0.4), (7, 7, 0.8), "concrete")
    campus._box("Planting bed", (x, y, 0.85), (6.7, 6.7, 0.15), "grass")
    campus._cylinder("Tree trunk", (x, y, 3.2), 0.4, 5, "steel", "tree")
    campus._sphere("Tree canopy", (x, y, 7), (3.8, 3.1, 4.1), "leaf")
    campus._sphere("Tree canopy", (x + 1.6, y, 5.8), (2.7, 2.4, 2.7), "grass")


def _streetlight(campus, x, y):
    campus._cylinder("Road luminaire mast", (x, y, 5.5), 0.16, 11, "steel")
    campus._box("Road luminaire", (x, y, 11), (3.1, 0.8, 0.18), "warm-light")


def _factory(campus):
    campus._box(
        "Robotics hall floor",
        (-119, -13, 0.45),
        (172, 100, 0.9),
        "concrete",
        "building",
    )
    campus._box("Robotics hall back wall", (-119, 36, 10), (172, 1.2, 20), "white")
    campus._box("Robotics hall west wall", (-204, -13, 10), (1.2, 100, 20), "glass")
    for x in range(-199, -34, 16):
        for y in (-59, 33):
            campus._box("Hall structural column", (x, y, 10), (0.65, 0.65, 20), "steel")
        campus._beam("Open roof truss", (x, -59, 20), (x, 34, 20), 0.35, "white")
        campus._beam("Roof truss chord", (x, -59, 18), (x, 34, 18), 0.22, "steel")
        for y in range(-59, 30, 10):
            campus._beam("Truss diagonal", (x, y, 18), (x, y + 10, 20), 0.12, "steel")
    campus._box("Cutaway roof strip", (-196, -13, 20.3), (18, 98, 0.5), "white")
    for row in range(4):
        y = -49 + row * 22
        _production_line(campus, y)
    for x in range(-188, -37, 20):
        campus._box("Clerestory window", (x, 35.3, 14), (16, 0.2, 6), "glass")
    _sign(campus, "01  /  ROBOTICS", (-174, -62, 21), 3.4)


def _production_line(campus, y):
    campus._box("Automated production belt", (-112, y, 2.2), (143, 3.8, 0.6), "steel")
    campus._box("Conveyor rubber", (-112, y, 2.55), (142, 3.3, 0.08), "road")
    for x in range(-181, -40, 5):
        campus._box("Conveyor roller", (x, y, 2.65), (0.25, 3.5, 0.15), "white")
    for index, x in enumerate(range(-177, -38, 18)):
        _robot(campus, x, y + 6, -1 if index % 2 else 1)
        campus._box("Production pallet", (x, y, 2.9), (3.8, 2.7, 0.45), "teal")
        campus._box("Machined assembly", (x, y, 3.5), (2.8, 1.8, 0.8), "white")
        campus._box("Controller cabinet", (x + 5, y + 7, 1.7), (2, 1.1, 3.4), "white")
        campus._box(
            "Controller display", (x + 5, y + 6.4, 2.3), (1.4, 0.08, 0.8), "cyan-light"
        )
        _fence(campus, x - 6, y + 11, 13)
    for offset in (-3, 13):
        campus._box(
            "Safety walkway stripe",
            (-113, y + offset, 0.97),
            (148, 0.15, 0.025),
            "yellow",
        )


def _robot(campus, x, y, direction):
    campus._cylinder("Robot pedestal", (x, y, 1.5), 1.1, 2, "steel", "robot")
    points = [
        (x, y, 2.5),
        (x, y, 4),
        (x + direction, y, 7),
        (x + 2 * direction, y - 3.5, 8.5),
        (x + direction, y - 5.7, 6.4),
    ]
    for start, end in zip(points, points[1:]):
        campus._beam("Industrial robot link", start, end, 0.55, "orange")
        campus._sphere("Robot actuator housing", start, (0.75, 0.75, 0.75), "steel")
    end = points[-1]
    campus._box("Robot gripper", end, (1.5, 1, 0.45), "steel")
    for offset in (-0.55, 0.55):
        campus._box(
            "Robot gripper jaw",
            (end[0] + offset, end[1], end[2] - 0.5),
            (0.2, 0.7, 0.9),
            "white",
        )


def _fence(campus, x, y, length):
    for offset in (0, length / 2, length):
        campus._box(
            "Safety fence post", (x + offset, y, 2.3), (0.15, 0.15, 3.3), "yellow"
        )
    for height in (1, 2, 3.7):
        campus._box(
            "Safety fence rail",
            (x + length / 2, y, height),
            (length, 0.07, 0.07),
            "steel",
        )
    for offset in range(int(length)):
        campus._box(
            "Safety fence wire", (x + offset, y, 2.3), (0.025, 0.025, 3), "steel"
        )


def _warehouse(campus):
    campus._box(
        "Fulfillment hall floor", (79, -5, 0.6), (152, 80, 1.2), "concrete", "building"
    )
    campus._box("Warehouse rear wall", (79, 34, 14), (152, 1, 28), "white")
    campus._box("Warehouse east wall", (154, -5, 14), (1, 80, 28), "glass")
    for aisle in range(5):
        for bay in range(12):
            _rack_bay(campus, 10 + bay * 11, -31 + aisle * 12)
    for x in range(9, 151, 22):
        campus._box("Warehouse roof column", (x, 32, 14), (0.7, 0.7, 28), "steel")
        campus._beam("Warehouse roof girder", (x, -43, 28), (x, 34, 28), 0.32, "white")
    campus._box("Warehouse cutaway roof", (80, 25, 28.3), (151, 18, 0.6), "white")
    for index in range(9):
        _agv(campus, 12 + index * 15, -53, index)
    _sign(campus, "02  /  FULFILLMENT", (14, -47, 29), 3.4)


def _rack_bay(campus, x, y):
    for offset in (-4.5, 4.5):
        campus._box(
            "High bay rack upright", (x + offset, y, 11.8), (0.23, 3.8, 21), "blue"
        )
    for level in range(5):
        z = 2 + level * 4
        campus._box("Warehouse shelf", (x, y, z), (9.5, 4, 0.3), "orange")
        for offset in (-2.4, 2.4):
            campus._box(
                "Inventory unit",
                (x + offset, y, z + 1.6),
                (4, 3.6, 2.8),
                "teal" if level % 2 else "white",
                "inventory_unit",
            )
            campus._box(
                "Inventory label",
                (x + offset, y - 1.83, z + 1.8),
                (1, 0.025, 0.35),
                "yellow",
            )


def _agv(campus, x, y, index):
    campus._box("Autonomous carrier", (x, y, 0.95), (5, 3.3, 1.3), "teal", "carrier")
    campus._box("Carrier payload", (x, y, 2.5), (3.4, 2.7, 1.8), "white")
    campus._box("Carrier lightbar", (x - 2.5, y, 1.1), (0.08, 2.6, 0.2), "cyan-light")
    for offset in (-1.6, 1.6):
        campus._sphere(
            "Carrier wheel", (x + offset, y - 1.5, 0.6), (0.5, 0.22, 0.5), "road"
        )


def _tower(campus):
    campus._box(
        "Operations tower podium", (-68, 93, 4), (68, 68, 8), "white", "building"
    )
    campus._box("Operations tower glazing", (-68, 93, 44), (39, 43, 76), "glass")
    for floor in range(15):
        z = 9 + floor * 5
        campus._box("Tower horizontal mullion", (-68, 93, z), (40, 44, 0.5), "steel")
        for x in range(-85, -48, 6):
            campus._box(
                "Occupied tower window",
                (x, 71.45, z + 1.8),
                (3.6, 0.1, 2.3),
                "warm-light" if (floor + x) % 4 == 0 else "glass",
            )
    for x in (-88, -78, -68, -58, -48):
        campus._box("Tower facade fin", (x, 71, 44), (0.35, 0.9, 76), "white")
    campus._box("Operations tower crown", (-68, 93, 83), (42, 46, 3), "steel")
    campus._box("Tower crown accent", (-68, 69.9, 82), (39, 0.15, 0.5), "cyan-light")
    campus._beam("Tower antenna", (-68, 93, 84), (-68, 93, 99), 0.24, "white")
    campus._box("Enclosed pedestrian bridge", (-68, 54, 12), (8, 35, 5), "glass")
    for x in (-101, -35):
        for y in (63, 123):
            _tree(campus, x, y)
    _sign(campus, "NPA  /  INDUSTRIAL CAMPUS", (-96, 59, 9), 2.2)


def _container(campus, x, y, z, material):
    campus._box(
        "Intermodal container", (x, y, z), (12.2, 2.44, 2.59), material, "container"
    )
    for offset in range(24):
        campus._box(
            "Container corrugation",
            (x - 5.8 + offset * 0.5, y - 1.25, z),
            (0.055, 0.07, 2.5),
            "steel",
        )
    for offset in (-0.8, 0.8):
        campus._box(
            "Container locking bar",
            (x + 6.13, y + offset, z),
            (0.045, 0.08, 2.4),
            "white",
        )


def _freight(campus):
    campus._box("Freight apron", (11, -112, 0.18), (404, 59, 0.36), "concrete")
    colors = ("teal", "orange", "blue", "red", "white")
    for row in range(5):
        for column in range(11):
            for level in range(2 + (column + row) % 2):
                _container(
                    campus,
                    -155 + column * 16,
                    -129 + row * 5.4,
                    1.6 + level * 2.62,
                    colors[(row + column + level) % 5],
                )
    for x in (51, 126):
        _crane(campus, x, -112)
    for index in range(8):
        _truck(campus, -161 + index * 45, -158, colors[index % 5])
    _sign(campus, "03  /  INTERMODAL", (47, -142, 1.2), 3.1)


def _crane(campus, x, y):
    for offset in (-22, 22):
        campus._box(
            "Gantry crane leg",
            (x, y + offset, 17),
            (2.4, 2.4, 34),
            "orange",
            "crane_leg",
        )
        campus._box("Gantry crane bogie", (x, y + offset, 1), (12, 3.5, 2), "steel")
        campus._box("Crane guide rail", (x, y + offset, 0.4), (72, 0.4, 0.3), "steel")
    campus._box("Gantry crossbeam", (x, y, 34), (4, 55, 3.5), "orange", "crane")
    campus._box("Gantry crane trolley", (x, y - 4, 31), (5, 7, 3), "steel")
    for offset in (-2, 2):
        campus._beam(
            "Crane hoist cable",
            (x + offset, y - 4, 30),
            (x + offset, y - 4, 12),
            0.055,
            "steel",
        )
    campus._box("Container spreader", (x, y - 4, 12), (13, 3, 0.7), "yellow")
    _container(campus, x, y - 4, 10.3, "teal")


def _truck(campus, x, y, material):
    campus._box(
        "Electric freight tractor", (x + 8, y, 2.2), (4, 2.8, 3.8), "white", "truck"
    )
    campus._box("Truck windshield", (x + 10.03, y, 3), (0.07, 2.5, 1.1), "glass")
    campus._box("Freight trailer", (x, y, 2.9), (12, 2.8, 4.2), material)
    campus._box("Trailer underframe", (x, y, 0.9), (13, 2.5, 0.4), "steel")
    for offset in (-4, -2, 6, 9):
        for side in (-1.4, 1.4):
            campus._sphere(
                "Truck tire", (x + offset, y + side, 0.8), (0.75, 0.28, 0.75), "road"
            )


def _energy(campus):
    campus._box("Energy plant apron", (120, 99, 0.3), (173, 80, 0.6), "concrete")
    for x in (61, 83, 105):
        for y in (82, 112):
            _tank(campus, x, y)
    for index in range(6):
        x = 133 + index * 11
        campus._box("Battery module", (x, 81, 4), (7, 13, 8), "white", "battery")
        campus._box(
            "Battery status strip", (x, 74.4, 4.5), (5, 0.12, 0.35), "cyan-light"
        )
        for z in range(2, 7):
            campus._box("Battery cooling vent", (x, 74.45, z), (5, 0.1, 0.2), "steel")
    for y in (66, 131):
        for z in (3, 4.2, 5.4):
            campus._beam("Utility pipe main", (42, y, z), (194, y, z), 0.28, "steel")
        for x in range(44, 195, 15):
            campus._box("Pipe rack support", (x, y, 2.7), (0.35, 3.5, 5.4), "white")
    for x in (143, 182):
        _turbine(campus, x, 126)
    _sign(campus, "04  /  ENERGY SYSTEMS", (46, 58, 1), 2.7)


def _tank(campus, x, y):
    campus._cylinder("Process vessel", (x, y, 10), 8, 19, "white", "vessel")
    campus._sphere("Vessel dome", (x, y, 19.5), (8, 8, 2.4), "steel")
    campus._cylinder("Vessel plinth", (x, y, 0.8), 8.5, 1.2, "steel")
    for z in (3, 10, 17):
        campus._cylinder("Vessel reinforcing band", (x, y, z), 8.07, 0.22, "steel")
    for z in range(2, 22):
        campus._box("Vessel ladder rung", (x, y - 8.2, z), (1, 0.15, 0.12), "steel")
    for offset in (-0.6, 0.6):
        campus._box(
            "Vessel ladder upright",
            (x + offset, y - 8.2, 11.5),
            (0.09, 0.15, 21),
            "steel",
        )


def _turbine(campus, x, y):
    campus._cylinder("Wind turbine tower", (x, y, 32), 1.2, 64, "white", "turbine")
    campus._box("Turbine nacelle", (x, y, 64), (2.7, 5, 2.8), "white")
    campus._sphere("Rotor hub", (x, y - 3, 64), (1.3, 1.3, 1.3), "steel")
    for index in range(3):
        angle = index * math.tau / 3 + 0.2
        end = (x + 23 * math.sin(angle), y - 3, 64 + 23 * math.cos(angle))
        campus._beam("Wind turbine blade", (x, y - 3, 64), end, 0.65, "white")


def _solar(campus):
    campus._box("Solar garden", (-163, 97, 0.25), (88, 79, 0.5), "grass")
    for row in range(9):
        for column in range(12):
            x, y = -201 + column * 6.7, 62 + row * 8
            panel = campus._box(
                "Photovoltaic array",
                (x, y, 2.7),
                (5.8, 5.3, 0.15),
                "solar",
                "solar_array",
            )
            panel.rotation_euler.x = math.radians(22)
            campus._box("Solar array stand", (x, y, 1.25), (0.15, 2, 2.5), "steel")
            for offset in (-1.9, 0, 1.9):
                rib = campus._box(
                    "Solar cell separator",
                    (x + offset, y, 2.81),
                    (0.035, 5.25, 0.025),
                    "white",
                )
                rib.rotation_euler.x = math.radians(22)


def _sign(campus, text, position, size):
    curve = campus.bpy.data.curves.new("Architectural lettering", "FONT")
    curve.body, curve.size, curve.extrude = text, size, 0.025
    curve.materials.append(campus.materials["white"])
    obj = campus.bpy.data.objects.new(text, curve)
    campus.bpy.context.scene.collection.objects.link(obj)
    obj.location = position
    obj.rotation_euler.x = math.pi / 2


def _lighting(bpy):
    from mathutils import Vector

    scene = bpy.context.scene
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes["Background"]
    background.inputs[0].default_value = (0.19, 0.29, 0.44, 1)
    background.inputs[1].default_value = 0.42
    light = bpy.data.lights.new("Late afternoon sunlight", "SUN")
    light.energy, light.angle, light.color = 2.3, 0.08, (1, 0.76, 0.5)
    obj = bpy.data.objects.new(light.name, light)
    scene.collection.objects.link(obj)
    obj.rotation_euler = Vector((0.6, 0.3, -0.85)).to_track_quat("-Z", "Y").to_euler()
    light = bpy.data.lights.new("Open sky fill", "AREA")
    light.energy, light.shape, light.size = 1800000, "DISK", 450
    light.color = (0.45, 0.7, 1)
    obj = bpy.data.objects.new(light.name, light)
    scene.collection.objects.link(obj)
    obj.location = (-50, -30, 350)
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"


def _build(bpy):
    campus = _Campus(bpy)
    for stage in (_site, _factory, _warehouse, _tower, _freight, _energy, _solar):
        stage(campus)
    _lighting(bpy)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    counts = Counter(obj.get("asset_role", "detail") for obj in meshes)
    for mesh in bpy.data.meshes:
        mesh.calc_loop_triangles()
    return {
        "site_extent_m": [500, 360],
        "site_area_hectares": 18,
        "mesh_objects": len(meshes),
        "instanced_triangles": sum(len(obj.data.loop_triangles) for obj in meshes),
        "unique_meshes": len({obj.data.name for obj in meshes}),
        "assets": dict(sorted(counts.items())),
    }


def _camera(index, count):
    route = min(index * 4 // count, 3)
    first = math.ceil(route * count / 4)
    length = math.ceil((route + 1) * count / 4) - first
    fraction = (index - first) / max(length - 1, 1)
    if route == 0:
        angle = math.radians(-53 + 66 * fraction)
        return (
            "Campus aerial",
            (720 * math.cos(angle), 720 * math.sin(angle), 390),
            (0, 0, 10),
            46,
        )
    if route == 1:
        return (
            "Robotics hall",
            (-169 + 135 * fraction, -116, 69),
            (-119 + 40 * fraction, -8, 4),
            48,
        )
    if route == 2:
        return "Freight terminal", (128 + 140 * fraction, -245, 112), (38, -112, 12), 52
    return "Energy and operations", (270 - 110 * fraction, 240, 135), (75, 84, 22), 48
