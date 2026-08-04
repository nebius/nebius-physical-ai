"""`get_image_features` does not always return a tensor.

Live, the BDD100K backfill stage failed inside the deployed LanceDB service with
`'BaseModelOutputWithPooling' object has no attribute 'norm'` (EVIDENCE §R41). Transformers has
returned a plain tensor in some versions and a model-output object in others, and nothing pinned
which one this code assumed — the tests only ever asserted the resulting vector's shape.
"""

from __future__ import annotations

from types import SimpleNamespace

from npa.workbench.lancedb.bdd100k_udfs import _image_embeddings


def test_a_plain_tensor_passes_through() -> None:
    tensor = object()

    assert _image_embeddings(tensor) is tensor


def test_image_embeds_is_preferred_on_a_model_output() -> None:
    embeds = object()
    result = SimpleNamespace(image_embeds=embeds, pooler_output=object())

    assert _image_embeddings(result) is embeds


def test_pooler_output_is_the_fallback() -> None:
    pooled = object()
    result = SimpleNamespace(image_embeds=None, pooler_output=pooled)

    assert _image_embeddings(result) is pooled


def test_last_hidden_state_is_the_last_resort() -> None:
    hidden = object()
    result = SimpleNamespace(last_hidden_state=hidden)

    assert _image_embeddings(result) is hidden
