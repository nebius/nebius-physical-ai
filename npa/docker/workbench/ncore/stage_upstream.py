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

try:
    from npa._public_https import download_public_https
except ModuleNotFoundError as error:
    if error.name != "npa":
        raise
    # Docker copies the same stdlib-only helper beside this bootstrap recipe.
    from _public_https import download_public_https


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
            and "sensors" not in path.parts
            and path.name != "test.py"
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


def patch_colmap_text_readers(path: Path) -> None:
    """Distinguish whitespace from EOF without replacing the real COLMAP reader.

    Hash the complete methods after NVIDIA's Python-3 map patch so every edit
    fails closed on drift, including changes outside the old sentinel loops.
    """
    expected = {
        "_load_cameras_txt": "9f8d3c2fb96ff0c51ad4080e980ecbfc345149bdb341270486ce095ce798fdab",
        "_load_images_txt": "3dde132bbf18620930e408e6f5f3332e0e54bdf1eb13e673836a07e7c7b77b1a",
        "_load_points3D_txt": "c332de6334bc69c360e7873a7dd4b9ecbba97350bb29392ab35273d7a638eb12",
    }
    source = path.read_text()
    for name, digest in expected.items():
        signature = f"    def {name}(self, input_file):\n"
        if source.count(signature) != 1:
            raise ValueError("trueprice text reader source changed")
        start = source.index(signature)
        end = source.find("\n    #-----", start)
        original = source[start:end]
        if end == -1 or hashlib.sha256(original.encode()).hexdigest() != digest:
            raise ValueError("trueprice text reader source changed")
        if name != "_load_images_txt":
            replacement = original.replace(
                "for line in iter(lambda: f.readline().strip(), ''):",
                "for line in f:\n                line = line.strip()",
            )
        else:
            replacement = """    def _load_images_txt(self, input_file):
        self.images = OrderedDict()

        with open(input_file, 'r') as f:
            for header in f:
                header = header.strip()
                if not header or header.startswith('#'):
                    continue
                data = header.split(maxsplit=9)
                if len(data) != 10:
                    raise ValueError('invalid COLMAP text image header')
                image_id = int(data[0])
                if image_id in self.images:
                    raise ValueError('duplicate COLMAP text image ID')
                image = Image(data[9], int(data[8]),
                              Quaternion(np.array(list(map(float, data[1:5])))),
                              np.array(list(map(float, data[5:8]))))

                # A blank observation line is valid; only actual EOF is incomplete.
                for observations in f:
                    if not observations.lstrip().startswith('#'):
                        break
                else:
                    raise ValueError('incomplete COLMAP text image record')
                data = observations.split()
                if len(data) % 3:
                    raise ValueError('invalid COLMAP text image observations')
                image.points2D = np.array(
                    [list(map(float, data[::3])), list(map(float, data[1::3]))]).T
                image.point3D_ids = np.array([
                    SceneManager.INVALID_POINT3D if int(value) == -1 else int(value)
                    for value in data[2::3]], dtype=np.uint64)

                self.images[image_id] = image
                self.name_to_image_id[image.name] = image_id
                self.last_image_id = max(self.last_image_id, image_id)
"""
        source = source[:start] + replacement + source[end:]
    path.write_text(source)


def patch_downsample_masks(path: Path) -> None:
    """Resize only verified original masks, to the actual per-frame image size."""
    original = """                if mask_path is not None:
                    mask_array = np.array(PILImage.open(str(mask_path)).convert("L"), dtype=np.uint8)
                    generic_data["mask"] = mask_array
                    masks_found += 1
"""
    replacement = """                if mask_path is not None:
                    with PILImage.open(str(image_path)) as frame_image:
                        target_size = frame_image.size
                    with PILImage.open(str(mask_path)) as mask_image:
                        mask_image = mask_image.convert("L")
                        if mask_image.size != target_size:
                            source_camera = colmap_camera.colmap_camera
                            source_size = (source_camera.width, source_camera.height)
                            if colmap_camera.downsample_factor == 1 or mask_image.size != source_size:
                                raise ValueError("mask dimensions differ from source image")
                            source_path = self.sequence_path / self.images_dir / image_name
                            with PILImage.open(str(source_path)) as source_image:
                                source_image.load()
                                if source_image.size != source_size:
                                    raise ValueError("original image dimensions differ from calibration")
                            mask_image = mask_image.resize(target_size, resample=PILImage.Resampling.NEAREST)
                        generic_data["mask"] = np.array(mask_image, dtype=np.uint8)
                    masks_found += 1
"""
    source = path.read_text()
    if source.count(original) != 1:
        raise ValueError("NCore downsample mask source changed")
    path.write_text(source.replace(original, replacement))


def stage(lock: dict, output: Path, archives: Path | None = None) -> None:
    for component in ("ncore", "pycolmap"):
        pin = lock[component]
        if archives is None:
            with io.BytesIO() as response:
                download_public_https(
                    pin["url"],
                    response,
                    allowed_hosts=frozenset({"codeload.github.com"}),
                )
                data = response.getvalue()
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
    patch_colmap_text_readers(output / "pycolmap" / "pycolmap" / "scene_manager.py")
    patch_downsample_camera(output / "ncore/tools/data_converter/colmap/converter.py")
    patch_downsample_masks(output / "ncore/tools/data_converter/colmap/converter.py")
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
