"""Exercise the locked Datasets folder-metadata security fix on real inputs.

The controls cover lexical traversal, absolute paths and URL schemes. This is
not a sandbox for untrusted datasets or a claim about symlink containment.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from importlib.metadata import version
from pathlib import Path

from datasets.exceptions import DatasetGenerationError
from datasets.packaged_modules.imagefolder.imagefolder import ImageFolder
from PIL import Image


def _load_case(root: Path, name: str, reference: str) -> tuple[int, int, int]:
    """Decode one image selected through actual imagefolder metadata."""
    folder = root / name
    folder.mkdir()
    (folder / "nested").mkdir()
    Image.new("RGB", (8, 8), (19, 73, 131)).save(folder / "nested" / "inside.png")
    (folder / "metadata.jsonl").write_text(
        json.dumps({"file_name": reference, "control": name}) + "\n",
        encoding="utf-8",
    )
    # Construct the installed, hash-locked local builder directly. No Hub
    # repository resolution or remote dataset script participates in this test.
    builder = ImageFolder(
        data_dir=str(folder),
        data_files={
            "train": [
                str(folder / "metadata.jsonl"),
                str(folder / "nested" / "inside.png"),
            ]
        },
        cache_dir=str(root / "cache" / name),
    )
    builder.download_and_prepare()
    dataset = builder.as_dataset(split="train")
    assert len(dataset) == 1, len(dataset)
    return tuple(dataset[0]["image"].convert("RGB").getpixel((0, 0)))


def main() -> None:
    """Require genuine positive decoding and all fixed rejection classes."""
    assert version("datasets") == "5.0.1"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        outside = root / "outside.png"
        Image.new("RGB", (8, 8), (211, 3, 7)).save(outside)
        positive = ("nested/inside.png", "./nested/inside.png")
        for index, reference in enumerate(positive):
            assert _load_case(root, f"positive-{index}", reference) == (19, 73, 131)
        negative = {
            "parent": "../outside.png",
            "nested-parent": "nested/../../outside.png",
            "absolute": str(outside),
            "file-scheme": "file://" + str(outside),
            "local-scheme": "local://" + str(outside),
        }
        rejected = []
        for name, reference in negative.items():
            try:
                _load_case(root, name, reference)
            except DatasetGenerationError as exc:
                cause = exc.__cause__
                assert isinstance(cause, ValueError), repr(cause)
                assert "Invalid metadata file_name" in str(cause), str(cause)
                rejected.append(name)
            else:
                raise AssertionError(f"Datasets accepted {name} metadata")
        control_hash = hashlib.sha256(outside.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "datasets": version("datasets"),
                "positive_cases": len(positive),
                "rejected_cases": rejected,
                "outside_control_sha256": control_hash,
                "symlink_containment_tested": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
