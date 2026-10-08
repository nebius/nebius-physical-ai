"""Validate and snapshot real pallet datasets before purchasing a Marble world."""

import hashlib
import io
import json
import tempfile
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from npa.cli.path_contract import validate_read_path
from npa.workbench.dataset.storage import read_bytes_uri

from .api import MarbleError, require_api_key
from .runtime import _paths, _publish


class PalletImage(BaseModel):
    """Describe a real image and all its pallet boxes in pixel coordinates.

    Args: Stable image ID, capture group, S3 URI, and XYXY boxes.
    Returns: A validated image record.
    Raises: ValueError for invalid fields or a non-S3 source.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(min_length=1)
    group: str = Field(min_length=1)
    uri: str
    boxes: list[tuple[float, float, float, float]]


class PalletCutout(BaseModel):
    """Identify an alpha-masked pallet derived from the training partition.

    Args: S3 PNG URI and the source training image ID.
    Returns: A validated cutout record.
    Raises: ValueError for invalid fields.
    """

    model_config = ConfigDict(extra="forbid")
    uri: str
    source_id: str = Field(min_length=1)


class PalletDataset(BaseModel):
    """Keep real training and held-out evaluation captures separate.

    Args: Real image partitions and cutouts sourced exclusively from training.
    Returns: A validated dataset manifest.
    Raises: ValueError for overlapping identities, capture groups, or cutout sources.
    """

    model_config = ConfigDict(extra="forbid")
    train: list[PalletImage] = Field(min_length=1)
    evaluation: list[PalletImage] = Field(min_length=1)
    cutouts: list[PalletCutout] = Field(min_length=1)

    @model_validator(mode="after")
    def _separate_partitions(self):
        records = self.train + self.evaluation
        if len({item.id for item in records}) != len(records):
            raise ValueError("Image IDs must be unique across partitions")
        if {r.group for r in self.train} & {r.group for r in self.evaluation}:
            raise ValueError("Train and evaluation capture groups must be disjoint")
        training_ids = {item.id for item in self.train if item.boxes}
        if any(item.source_id not in training_ids for item in self.cutouts):
            raise ValueError("Cutouts must reference positive training images only")
        for item in [*records, *self.cutouts]:
            validate_read_path(item.uri, tool="marble", allow_hf=False)
        return self


def _decode(payload):
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
        return image
    except (OSError, ValueError) as exc:
        raise MarbleError("Pallet input is not a decodable image") from exc


def _pixel_hash(image):
    rgb = image.convert("RGB")
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def _validate_boxes(image, boxes):
    for x1, y1, x2, y2 in boxes:
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise MarbleError("Pallet boxes must have positive area inside the image")


def _snapshot_partition(root, name, records):
    entries = []
    for index, record in enumerate(records):
        payload = read_bytes_uri(record.uri)
        image = _decode(payload).convert("RGB")
        _validate_boxes(image, record.boxes)
        filename = f"{name}/{index:06d}.png"
        (root / name).mkdir(exist_ok=True)
        image.save(root / filename)
        entries.append(
            {
                "id": record.id,
                "group": record.group,
                "file": filename,
                "boxes": record.boxes,
                "pixel_sha256": _pixel_hash(image),
                "source_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if not any(item["boxes"] for item in entries):
        raise MarbleError(f"The {name} partition must contain labeled pallets")
    return entries


def _snapshot_cutouts(root, records):
    entries = []
    (root / "cutouts").mkdir()
    for index, record in enumerate(records):
        image = _decode(read_bytes_uri(record.uri))
        if image.mode != "RGBA" or image.format != "PNG":
            raise MarbleError("Pallet cutouts must be RGBA images with transparency")
        alpha = image.getchannel("A")
        if alpha.getextrema()[0] != 0 or alpha.getextrema()[1] == 0:
            raise MarbleError("Pallet cutouts require visible and transparent pixels")
        filename = f"cutouts/{index:06d}.png"
        image.crop(alpha.getbbox()).save(root / filename)
        entries.append({"file": filename, "source_id": record.source_id})
    return entries


def _snapshot(root, dataset, run_id):
    train = _snapshot_partition(root, "train", dataset.train)
    evaluation = _snapshot_partition(root, "evaluation", dataset.evaluation)
    if {r["pixel_sha256"] for r in train} & {r["pixel_sha256"] for r in evaluation}:
        raise MarbleError("Train/evaluation leakage: identical decoded image pixels")
    return {
        "run_id": run_id,
        "schema_version": "npa.marble.pallet-data.v1",
        "train": train,
        "evaluation": evaluation,
        "cutouts": _snapshot_cutouts(root, dataset.cutouts),
    }


def pallet_preflight(request):
    """Require API auth and snapshot verified real data before world generation.

    Args: RunRequest with an input manifest URI and output snapshot prefix.
    Returns: Snapshot counts, without token or private source URI values.
    Raises: MarbleError or ValueError on missing auth, bad images, or data leakage.
    """
    require_api_key()
    _paths(request)
    dataset = PalletDataset.model_validate_json(read_bytes_uri(request.input_path))
    with tempfile.TemporaryDirectory(prefix="npa-pallet-data-") as directory:
        root = Path(directory)
        manifest = _snapshot(root, dataset, request.run_id)
        manifest["files"] = {
            str(p.relative_to(root)): {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()
            }
            for p in root.rglob("*.png")
        }
        (root / "dataset.json").write_text(json.dumps(manifest, allow_nan=False))
        _publish(root, request.output_path)
    return {
        "run_id": request.run_id,
        "train": len(dataset.train),
        "evaluation": len(dataset.evaluation),
        "cutouts": len(dataset.cutouts),
    }
