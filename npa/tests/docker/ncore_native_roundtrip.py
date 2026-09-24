"""Run inside the NCore image: synthetic COLMAP -> native converter -> V4 reader.

This is CPU packaging evidence, not capture, NRE, model, or GPU acceptance.
"""

import json
from pathlib import Path
import struct
import subprocess
import tempfile

from PIL import Image

from npa.workbench.nurec import colmap


def fixture(root: Path, extension: str, binary: bool) -> None:
    sparse = root / "sparse/0"
    sparse.mkdir(parents=True)
    (root / "images").mkdir()
    for index in (1, 2):
        Image.new("RGB", (16, 12), (index * 70, 20, 50)).save(
            root / f"images/frame{index}.{extension}"
        )
    if binary:
        (sparse / "cameras.bin").write_bytes(
            struct.pack("<QIiQQ4d", 1, 1, 1, 16, 12, 13, 13, 8, 6)
        )
        with (sparse / "images.bin").open("wb") as stream:
            stream.write(struct.pack("<Q", 2))
            for index in (1, 2):
                stream.write(
                    struct.pack("<I4d3dI", index, 1, 0, 0, 0, 1 - index, 0, 0, 1)
                )
                stream.write(f"frame{index}.{extension}".encode() + b"\0")
                stream.write(struct.pack("<QddQ", 1, 0, 0, index))
        with (sparse / "points3D.bin").open("wb") as stream:
            stream.write(struct.pack("<Q", 2))
            for index in (1, 2):
                stream.write(
                    struct.pack(
                        "<Q3d3BdQII", index, index, 2, 3, 20, 30, 40, 0, 1, index, 0
                    )
                )
    else:
        (sparse / "cameras.txt").write_text("1 PINHOLE 16 12 13 13 8 6\n")
        (sparse / "images.txt").write_text(
            f"1 1 0 0 0 0 0 0 1 frame1.{extension}\n0 0 1\n"
            f"2 1 0 0 0 -1 0 0 1 frame2.{extension}\n0 0 2\n"
        )
        (sparse / "points3D.txt").write_text(
            "1 1 2 3 20 30 40 0 1 0\n2 2 2 3 20 30 40 0 2 0\n"
        )


def main() -> None:
    results = []
    with tempfile.TemporaryDirectory(prefix="ncore-native-roundtrip-") as temporary:
        for extension in ("png", "jpg"):
            for binary in (False, True):
                case = Path(temporary) / f"{extension}-{binary}"
                source = case / "capture"
                output = case / "output"
                fixture(source, extension, binary)
                request = colmap.ColmapConversionRequest(
                    input_path="s3://ncore-fixture/input/",
                    output_path="s3://ncore-fixture/output/",
                    include_downsampled_images=False,
                )
                expected = colmap.inspect_colmap_source(source, request)
                subprocess.run(
                    [
                        "/opt/ncore/bin/colmap-convert",
                        "--root-dir",
                        str(source),
                        "--output-dir",
                        str(output),
                        "colmap-v4",
                        "--no-include-downsampled-images",
                    ],
                    check=True,
                )
                counts = colmap.validate_ncore_sequence(
                    output / "capture/capture.json", expected
                )
                assert counts == {"cameras": 1, "images": 2, "poses": 2, "points": 2}
                results.append(
                    {
                        "format": extension,
                        "colmap": "binary" if binary else "text",
                        **counts,
                    }
                )
    print(
        json.dumps(
            {
                "validation": "UNPUBLISHED/UNACCEPTED synthetic CPU conversion",
                "roundtrips": results,
            }
        )
    )


if __name__ == "__main__":
    main()
