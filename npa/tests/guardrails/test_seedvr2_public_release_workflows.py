"""Guard both SeedVR2 publication scans against the exact source and image."""

from pathlib import Path


PUBLISH = (
    Path(__file__).resolve().parents[3] / ".github/workflows/publish-public-images.yml"
)


def test_seedvr2_publication_scans_bind_source_and_image() -> None:
    text = PUBLISH.read_text(encoding="utf-8")
    seedvr2_scans = [
        index
        for index in range(len(text))
        if text.startswith("scan_image_seedvr2_payload.py", index)
    ]
    assert len(seedvr2_scans) == 2
    assert seedvr2_scans[0] < text.index(
        "Push only after every pre-publication gate passes"
    )
    assert text.index("Verify pushed bytes") < seedvr2_scans[1]
    for position in seedvr2_scans:
        invocation = text[position : text.index("\n          fi", position)]
        assert '--clean-root-source-sha "$DEVELOPMENT_SHA"' in invocation
    for position, image, archive in zip(
        seedvr2_scans, ("$IMAGE", "$exact"), ("${TOOL}.tar", "${TOOL}-pushed.tar")
    ):
        invocation = text[position : text.index("\n          fi", position)]
        assert (
            "--expected-image-id \"$(docker image inspect --format '{{.Id}}' \""
            + image
            + '")"'
        ) in invocation
        assert '--tarball "$RUNNER_TEMP/' + archive + '"' in invocation
