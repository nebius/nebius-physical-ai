from __future__ import annotations

import ast
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from npa.workflows import paidf_native as native


def _executor_fixture() -> bytes:
    return b'''from __future__ import annotations

import os
import multistorageclient as msc

_MEDIA_KINDS = frozenset({"image", "video", "control"})


class Executor:
    def _write(self, result, output_path: str) -> None:
        if result.media_bytes is not None:
            with msc.open(output_path, "wb") as f:
                f.write(result.media_bytes)
'''


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), "red").save(buffer, "PNG")
    return buffer.getvalue()


def _patched_encoder(source: bytes):
    tree = ast.parse(native._paidf_image_output_patch_bytes(source))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_media_bytes_for_output"
    )

    def decode(value, _mode):
        try:
            return np.asarray(Image.open(io.BytesIO(value.tobytes())).convert("RGB"))
        except Exception:  # noqa: BLE001 - emulate cv2's invalid-image sentinel
            return None

    def encode(_extension, value, _parameters):
        output = io.BytesIO()
        Image.fromarray(value).save(output, "JPEG", quality=95)
        return True, np.frombuffer(output.getvalue(), dtype=np.uint8)

    namespace = {
        "cv2": SimpleNamespace(
            IMREAD_COLOR=1,
            IMWRITE_JPEG_QUALITY=1,
            imdecode=decode,
            imencode=encode,
        ),
        "np": np,
        "os": __import__("os"),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<patch>", "exec"), namespace)
    return namespace["_media_bytes_for_output"]


def test_paidf_executor_patch_encodes_png_bytes_for_jpeg_output() -> None:
    encoder = _patched_encoder(_executor_fixture())
    png = _png_bytes()

    jpeg = encoder(png, "output.jpg")

    assert jpeg.startswith(b"\xff\xd8\xff")
    assert jpeg != png
    with Image.open(io.BytesIO(jpeg)) as decoded:
        assert decoded.format == "JPEG"
        assert decoded.size == (32, 24)
    assert encoder(png, "output.png") == png


@pytest.mark.parametrize("payload", [b"", b"not an image"])
def test_paidf_executor_patch_rejects_invalid_jpeg_output_bytes(payload: bytes) -> None:
    encoder = _patched_encoder(_executor_fixture())

    with pytest.raises(ValueError, match="undecodable image bytes"):
        encoder(payload, "output.jpg")


def test_paidf_executor_patch_manifest_binds_original_and_patched_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _executor_fixture()
    patched = native._paidf_image_output_patch_bytes(original)
    target = tmp_path / native._PAIDF_EXECUTOR_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(original)
    monkeypatch.setattr(native, "_PAIDF_EXECUTOR_SHA256", hashlib.sha256(original).hexdigest())
    monkeypatch.setattr(
        native, "_PAIDF_EXECUTOR_PATCHED_SHA256", hashlib.sha256(patched).hexdigest()
    )

    manifest = native._patch_paidf_image_output_contract(tmp_path)

    assert target.read_bytes() == patched
    assert manifest["schema"] == "npa.paidf.upstream-source-adaptation.v1"
    assert manifest["upstream_revision"] == native.PAIDF_AUGMENTATION_REVISION
    assert manifest["original_sha256"] == hashlib.sha256(original).hexdigest()
    assert manifest["patched_sha256"] == hashlib.sha256(patched).hexdigest()
    assert len(manifest["patch_sha256"]) == 64


def test_paidf_executor_patch_refuses_unreviewed_source(tmp_path: Path) -> None:
    target = tmp_path / native._PAIDF_EXECUTOR_PATH
    target.parent.mkdir(parents=True)
    target.write_text("unreviewed source", encoding="utf-8")

    with pytest.raises(native.PaidfNativeError, match="differ from the reviewed"):
        native._patch_paidf_image_output_contract(tmp_path)


@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_source_adaptation_identity_is_exact(mutation: str) -> None:
    adaptation = native._paidf_image_output_adaptation()
    if mutation == "missing":
        candidate = None
    else:
        candidate = {**adaptation, "patch_sha256": "f" * 64}

    with pytest.raises(native.PaidfNativeError, match="reviewed image MIME"):
        native._require_paidf_image_output_adaptation(candidate)

    native._require_paidf_image_output_adaptation(adaptation)
