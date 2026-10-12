"""Guard SeedVR2 golden-eval authenticated immutable image resolution."""

import pytest

from npa.orchestration.skypilot.registry_preflight import ImagePullCheck
from npa.smoke import serverless_runner


def test_seedvr2_golden_image_is_bound_to_pull_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = "registry.example/npa-seedvr2:candidate"
    digest = "sha256:" + "c" * 64
    monkeypatch.setattr(
        serverless_runner,
        "check_image_pulls_with_credentials",
        lambda images, **_kwargs: [
            ImagePullCheck(image=images[0], status="ok", digest=digest)
        ],
    )

    assert serverless_runner._digest_bound_seedvr2_image(image) == (
        "registry.example/npa-seedvr2@" + digest
    )


def test_seedvr2_golden_image_requires_pullable_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = "registry.example/npa-seedvr2:candidate"
    monkeypatch.setattr(
        serverless_runner,
        "check_image_pulls_with_credentials",
        lambda images, **_kwargs: [ImagePullCheck(image=images[0], status="forbidden")],
    )

    with pytest.raises(RuntimeError, match="not pullable by digest"):
        serverless_runner._digest_bound_seedvr2_image(image)
