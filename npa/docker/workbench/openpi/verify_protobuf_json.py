"""Exercise the patched protobuf parser and the real TFDS metadata boundary."""

from __future__ import annotations

from google.protobuf import any_pb2, json_format
import numpy as np
import tensorflow_datasets as tfds


def nested_any(depth: int) -> dict:
    result: dict = {}
    for _ in range(depth):
        result = {"@type": "type.googleapis.com/google.protobuf.Any", "value": result}
    return result


def main() -> None:
    shallow = nested_any(4)
    message = json_format.ParseDict(shallow, any_pb2.Any(), max_recursion_depth=5)
    assert json_format.MessageToDict(message) == shallow
    try:
        json_format.ParseDict(nested_any(8), any_pb2.Any(), max_recursion_depth=5)
    except json_format.ParseError as exc:
        assert "Max recursion depth is 5" in str(exc)
    else:
        raise RuntimeError("nested Any bypassed the configured recursion depth")

    features = tfds.features.FeaturesDict(
        {
            "image": tfds.features.Image(shape=(48, 48, 3)),
            "label": tfds.features.Scalar(np.int64),
        }
    )
    restored = tfds.features.FeatureConnector.from_json(features.to_json())
    assert restored["image"].shape == (48, 48, 3)
    assert restored["label"].np_dtype == np.int64
    try:
        tfds.features.FeatureConnector.from_json(
            {
                "type": "tensorflow_datasets.core.features.image_feature.Image",
                "proto_cls": "google.protobuf.Any",
                "content": nested_any(8),
            }
        )
    except KeyError:
        pass
    else:
        raise RuntimeError("TFDS accepted a proto type outside its feature allowlist")
    print("protobuf depth enforcement and real TFDS metadata round-trip passed")


if __name__ == "__main__":
    main()
