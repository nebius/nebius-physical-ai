"""Stage only the pinned official converter/reader source and its license records.

The Bazel binary is a py_binary, not a native COLMAP executable. Its PYTHONPATH
contract includes both the trueprice repository root and its pycolmap directory.
Do not replace it with PyPI's unrelated modern COLMAP bindings.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import urllib.request


NCORE_METADATA = {
    "LICENSE",
    "NOTICE",
    "MODULE.bazel",
    "ncore/BUILD.bazel",
    "tools/data_converter/BUILD.bazel",
    "tools/data_converter/colmap/BUILD.bazel",
    "deps/pycolmap/pycolmap.BUILD",
    "deps/pycolmap/fix-python3-map.patch",
    "deps/pip/requirements_ncore.in",
    "deps/pip/requirements_colmap.in",
    "deps/pip/requirements_3_11.txt",
}
NCORE_TOOLS = {
    "tools/debug.py",
    "tools/data_converter/cli.py",
    "tools/data_converter/colmap/converter.py",
}


def selected(name: str, component: str) -> bool:
    path = PurePosixPath(name)
    if component == "ncore":
        return name in NCORE_METADATA | NCORE_TOOLS or (
            path.parts[0] == "ncore"
            and (path.suffix == ".py" or path.name == "py.typed")
            and not path.name.endswith("_test.py")
            and "tests" not in path.parts
        )
    return name == "LICENSE.txt" or (
        path.parts[0] == "pycolmap" and path.suffix == ".py"
    )


def patch_numpy_sentinel(path: Path) -> None:
    """Preserve trueprice's uint64 maximum sentinel under NumPy 1.x and 2.x."""
    original = "INVALID_POINT3D = np.uint64(-1)"
    source = path.read_text()
    if source.count(original) != 1:
        raise ValueError("trueprice sentinel source changed")
    path.write_text(
        source.replace(
            original,
            "INVALID_POINT3D = np.uint64(np.iinfo(np.uint64).max)  # NPA: NumPy 2 compatibility",
        )
    )


def patch_downsample_camera(path: Path) -> None:
    """Select the current camera, not the last image's camera, when downsampling."""
    original = (
        "                    cameras[ncore_camera_id] = ColmapCamera(\n"
        "                        camera_id=ncore_camera_id,\n"
        "                        colmap_camera=self.scene_manager.cameras[imdata[k].camera_id],\n"
        "                        image_path=parent_dir / image_root,\n"
    )
    source = path.read_text()
    if source.count(original) != 1:
        raise ValueError("NCore downsample camera source changed")
    path.write_text(
        source.replace(
            original,
            original.replace(
                "colmap_camera=self.scene_manager.cameras[imdata[k].camera_id],",
                "colmap_camera=camera,  # NPA: use the current downsample camera",
            ),
        )
    )


def stage(lock: dict, output: Path, archives: Path | None = None) -> None:
    for component in ("ncore", "pycolmap"):
        pin = lock[component]
        if archives is None:
            with urllib.request.urlopen(pin["url"]) as response:
                data = response.read()
        else:
            data = (archives / f"{component}.tar.gz").read_bytes()
        if hashlib.sha256(data).hexdigest() != pin["sha256"]:
            raise ValueError(f"{component}: archive SHA-256 mismatch")
        prefix = f"{component}-{pin['revision']}/"
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive:
                if member.isdir() and member.name == prefix.rstrip("/"):
                    continue
                if not member.name.startswith(prefix):
                    raise ValueError(f"{component}: unexpected archive prefix")
                relative = member.name[len(prefix) :]
                if not relative:
                    continue
                path = PurePosixPath(relative)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"{component}: unsafe archive path")
                if not selected(relative, component):
                    continue
                if not member.isfile():
                    raise ValueError(f"{component}: source must be a regular file")
                destination = output / component / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source:
                    destination.write_bytes(source.read())
                destination.chmod(0o644)
    patch = (output / "ncore" / lock["pycolmap"]["patch"]).resolve()
    subprocess.run(
        ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch)],
        cwd=output / "pycolmap",
        check=True,
    )
    patch_numpy_sentinel(output / "pycolmap" / "pycolmap" / "scene_manager.py")
    patch_downsample_camera(output / "ncore/tools/data_converter/colmap/converter.py")
    # Preserve the exact patched source identity; no .git database or test data.
    inventory = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    (output / "source-inventory.json").write_text(
        json.dumps({"sources": lock, "files": inventory}, indent=2, sort_keys=True)
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archives", type=Path)
    args = parser.parse_args()
    stage(json.loads(args.lock.read_text()), args.output, args.archives)
