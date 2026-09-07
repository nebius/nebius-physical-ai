"""Offline checks of native sources, shared-library linkage and real codecs."""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys


ANNEX = Path("/opt/ncore/native-sources")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    import numpy as np
    import scipy
    from scipy import fft, linalg, ndimage
    from PIL import Image

    converter = "converter-venv" in sys.prefix
    expected = ("1.26.4", "1.15.2") if converter else ("2.5.3", "1.18.1")
    assert (np.__version__, scipy.__version__) == expected
    lock = json.loads((ANNEX / "native-source-lock.json").read_text())
    for component in lock["components"]:
        for item in component["artifacts"]:
            assert digest(ANNEX / item["filename"]) == item["sha256"], item["filename"]
    site = Path(np.__file__).parent.parent
    for name in ("numpy.libs", "scipy.libs", "av.libs"):
        assert not (site / name).exists(), f"opaque wheel runtime returned: {name}"
    array = np.array([[3.0, 1.0], [1.0, 2.0]])
    np.testing.assert_allclose(
        array @ linalg.solve(array, np.eye(2)), np.eye(2), atol=1e-14
    )
    np.testing.assert_allclose(fft.ifft(fft.fft(np.arange(8.0))).real, np.arange(8.0))
    assert ndimage.zoom(np.arange(16.0).reshape(4, 4), 2).shape == (8, 8)
    packages = ["numpy", "scipy"]
    codec_proof = {}
    if not converter:
        import av

        assert av.__version__ == "17.1.0"
        packages.append("av")
        major = av.library_versions["libavcodec"][0]
        codec = ctypes.CDLL(f"libavcodec.so.{major}")
        codec.avcodec_configuration.restype = ctypes.c_char_p
        codec.avcodec_license.restype = ctypes.c_char_p
        configuration = codec.avcodec_configuration().decode()
        for option in ("--disable-gpl", "--disable-nonfree", "--disable-autodetect"):
            assert option in configuration
        assert (
            "--enable-libx264" not in configuration
            and "--enable-libx265" not in configuration
        )
        assert (
            "libx264" not in av.codecs_available
            and "libx265" not in av.codecs_available
        )
        for fmt in ("PNG", "JPEG"):
            buffer = io.BytesIO()
            Image.new("RGB", (16, 12), (70, 100, 130)).save(buffer, format=fmt)
            buffer.seek(0)
            with av.open(buffer) as container:
                (frame,) = list(container.decode(video=0))
                assert frame.to_ndarray(format="rgb24").shape == (12, 16, 3)
        codec_proof = {
            "configuration": configuration,
            "license": codec.avcodec_license().decode(),
            "decoded_formats": ["PNG", "JPEG"],
        }
    system = {}
    debian_lock = json.loads((ANNEX / "debian-native-lock.json").read_text())
    for package in debian_lock["packages"]:
        version = subprocess.check_output(
            ["dpkg-query", "-W", "-f=${Version}", package["package"]], text=True
        )
        assert version == package["version"], package["package"]
        for name, expected_hash in package["libraries"].items():
            path = Path(name).resolve()
            assert digest(path) == expected_hash, name
            system[str(path)] = {
                "sha256": expected_hash,
                "package": package["package"],
                "version": version,
                "source": package["source"],
            }
    libraries = {}
    paths = [
        p
        for name in packages
        for p in (site / name).rglob("*")
        if p.is_file() and (p.name.endswith(".so") or ".so." in p.name)
    ]
    paths += list(Path("/opt/ncore/ffmpeg/lib").glob("*.so.*"))
    paths += list(Path("/opt/ncore/ffmpeg/bin").iterdir())
    built_libraries = set(p.resolve() for p in paths)
    for path in sorted(built_libraries):
        linked = subprocess.check_output(["ldd", str(path)], text=True)
        assert "not found" not in linked, linked
        assert "libx264" not in linked and "libx265" not in linked, linked
        assert not any(
            name in linked for name in ("numpy.libs", "scipy.libs", "av.libs")
        ), linked
        for line in linked.splitlines():
            fields = line.split()
            candidates = [field for field in fields if field.startswith("/")]
            for dependency in candidates:
                actual = Path(dependency).resolve()
                if actual in built_libraries:
                    continue
                assert str(actual) in system, f"unmapped native runtime: {actual}"
        libraries[str(path)] = {"sha256": digest(path), "ldd": linked}
    print(
        json.dumps(
            {
                "interpreter": sys.executable,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "codecs": codec_proof,
                "libraries": libraries,
                "debian_runtime": system,
                "validation": "offline-native-packaging-only",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
