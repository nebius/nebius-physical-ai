"""Exercise Gradio's image boundary against the installed Pillow 13 runtime."""

from __future__ import annotations

import importlib.metadata
import os
import tempfile
from pathlib import Path

from gradio.components import Image as GradioImage
from gradio.data_classes import FileData
from PIL import Image


def main() -> None:
    if importlib.metadata.version("gradio") != "6.16.0+npa.pillow13.1":
        raise RuntimeError(
            "compatibility probe requires the reviewed Gradio derivative"
        )
    if importlib.metadata.version("pillow") != "13.0.0":
        raise RuntimeError("compatibility probe requires pillow 13.0.0")

    with tempfile.TemporaryDirectory() as directory:
        os.environ["GRADIO_TEMP_DIR"] = directory
        expected = Image.new("RGBA", (3, 2), (17, 34, 51, 68))
        component = GradioImage(type="pil", image_mode="RGBA", format="png")
        source = Path(directory) / "source.png"
        expected.save(source)
        decoded = component.preprocess(
            FileData(path=str(source), orig_name=source.name, mime_type="image/png")
        )
        if not isinstance(decoded, Image.Image):
            raise RuntimeError("Gradio did not decode the image into Pillow")
        if decoded.mode != "RGBA" or decoded.size != expected.size:
            raise RuntimeError(
                "Gradio/Pillow preprocessing changed image shape or mode"
            )
        if decoded.tobytes() != expected.tobytes():
            raise RuntimeError("Gradio/Pillow preprocessing changed image pixels")
        encoded = component.postprocess(decoded)
        if encoded is None or not Path(encoded.path).is_file():
            raise RuntimeError("Gradio/Pillow postprocessing did not emit an image")
        with Image.open(encoded.path) as reopened:
            reopened.load()
            if reopened.convert("RGBA").tobytes() != expected.tobytes():
                raise RuntimeError("Gradio/Pillow roundtrip changed image pixels")


if __name__ == "__main__":
    main()
