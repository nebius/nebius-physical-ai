"""Fetch pinned Go1 geometry and a walking policy with their upstream notices."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from defusedxml.ElementTree import parse

from .acquisition import _download
from .api import MarbleError


def acquire_robot(root):
    """Materialize hash-pinned upstream assets without changing the world bundle.

    Args: Local collection directory.
    Returns: The converted robot description and exact upstream provenance.
    Raises: MarbleError when an upstream artifact fails its pinned hash.
    """
    entries = json.loads(Path(__file__).with_name("quadruped-assets.json").read_text())
    for entry in entries:
        path = root / entry["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{entry['repository']}/{entry['revision']}/{entry['path']}"
        payload = path.read_bytes() if path.exists() else _download(url)
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise MarbleError("Pinned quadruped asset hash mismatch")
        path.write_bytes(payload)
    source = root / "robot-assets/unitree_ros/robots/go1_description"
    _convert_meshes(source)
    description = _description(source, root / "go1.urdf")
    return description, entries


def _convert_meshes(source):
    import trimesh

    for path in sorted((source / "meshes").glob("*.dae")):
        output = path.with_suffix(".glb")
        if not output.exists():
            trimesh.load(path, force="scene").export(output)


def _description(source, destination):
    tree = parse(source / "urdf/go1.urdf", forbid_dtd=True)
    # The ROS-only dummy root has no inertia. Promote the actual inertial trunk
    # rather than letting Bullet treat a massless root as fixed to the world.
    for element in list(tree.getroot()):
        if (element.tag, element.get("name")) in {
            ("link", "base"),
            ("joint", "floating_base"),
        }:
            tree.getroot().remove(element)
    for mesh in tree.findall(".//mesh"):
        mesh.set(
            "filename",
            mesh.get("filename").replace(
                "package://go1_description/",
                str(source.relative_to(destination.parent)) + "/",
            ),
        )
    # ROS optical frames have no mass; Bullet otherwise invents a 1 kg body.
    for link in tree.findall("link"):
        if link.find("inertial") is None:
            inertia = ET.SubElement(link, "inertial")
            ET.SubElement(inertia, "mass", value="0")
            ET.SubElement(
                inertia, "inertia", ixx="0", iyy="0", izz="0", ixy="0", ixz="0", iyz="0"
            )
    tree.write(destination)
    _write_visuals(tree, destination.with_name("render-robot.json"))
    return destination


def _visual(link, visual):
    origin = visual.find("origin")
    geometry = visual.find("geometry")
    for kind in ("mesh", "sphere", "box", "cylinder"):
        shape = geometry.find(kind)
        if shape is not None:
            break
    else:
        raise MarbleError("Unsupported robot visual geometry")
    return {
        "link": link.get("name"),
        "position": origin.get("xyz", "0 0 0") if origin is not None else "0 0 0",
        "angles": origin.get("rpy", "0 0 0") if origin is not None else "0 0 0",
        "geometry": {"kind": kind, **shape.attrib},
    }


def _write_visuals(tree, destination):
    # Blender consumes plain visual records; its bundled Python needs no XML parser.
    visuals = []
    for link in tree.findall("link"):
        for visual in link.findall("visual"):
            visuals.append(_visual(link, visual))
    destination.write_text(json.dumps(visuals, allow_nan=False))


def policy_path(root):
    """Locate the verified upstream Go1 ONNX checkpoint.

    Args: Collection directory populated by acquire_robot.
    Returns: Local policy path.
    Raises: None.
    """
    return (
        root
        / "robot-assets/mujoco_playground/mujoco_playground/experimental/sim2sim/onnx/go1_policy.onnx"
    )
