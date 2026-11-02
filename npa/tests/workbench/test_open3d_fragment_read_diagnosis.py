"""What the read-retry path tells an operator to do next.

Both failures below were reproduced against the shipped code before anything was
changed. A manifest that parsed and then failed schema validation was read three
times and reported as `after 3 attempts`, as though the transport had been at
fault. A fragment whose bytes were intact and wrong, and a fragment whose stream
kept ending early, produced the same sentence -- "the manifest and the stored scan
disagree, or the transport kept truncating" -- while the manifest recorded the byte
count that separates them:

    diverged store:  32 bytes read, 32 recorded, every attempt
    truncating link:  4, 9, 13 bytes read, 32 recorded

The actions differ. A diverged store has to be prepared again; a truncating link has
to be retried. A message that offers both leaves the reader to guess at exactly the
moment they have least information.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from npa.workbench.open3d import runtime
from npa.workbench.open3d.artifacts import Open3dError
from npa.workbench.open3d.schemas import Fragment

PREPARED = "s3://example-bucket/prepared"
SCAN = b"the bytes prepare saw when it wrote the manifest"


def _fragment(payload: bytes = SCAN, name: str = "a") -> Fragment:
    return Fragment(
        id=name,
        uri=f"s3://example-bucket/prepared/{name}.pcd",
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
    )


def _manifest(**overrides) -> bytes:
    document = {
        "fragments": [
            _fragment().model_dump(),
            _fragment(SCAN + b"-second", "b").model_dump(),
        ],
        "voxel_size": 0.05,
    }
    document.update(overrides)
    return json.dumps(document).encode()


def _serve(monkeypatch, responses: list[bytes | Exception]) -> list[str]:
    """Answer each read from `responses`, recording the attempts made."""

    reads: list[str] = []

    def read(uri: str) -> bytes:
        reads.append(uri)
        item = responses[min(len(reads) - 1, len(responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(runtime, "read_bytes_uri", read)
    return reads


def test_a_manifest_that_parsed_and_failed_validation_is_not_read_again(
    monkeypatch,
) -> None:
    """The reproduction: three reads of a document that cannot become valid.

    `json.loads` succeeding is this function's own evidence that the stream did not
    end early, so the next read returns the same wrong document. Spending two more
    requests to confirm that, and then blaming the attempt count, is the defect.
    """

    reads = _serve(monkeypatch, [json.dumps({"voxel_size": 0.05}).encode()])
    with pytest.raises(Open3dError) as raised:
        runtime._load_manifest(PREPARED)

    assert len(reads) == 1, (
        "a complete document of the wrong shape was read more than once; it will be "
        "the wrong shape every time"
    )
    message = str(raised.value)
    assert "parsed as JSON but is not a registration manifest" in message
    assert "not read differently on another attempt" in message
    assert "open3d prepare" in message, "the action is to write a manifest"
    assert "attempts" not in message.replace("another attempt", ""), (
        "an exhausted-attempts message describes a transport that was working here"
    )


def test_the_schema_error_itself_reaches_the_operator(monkeypatch) -> None:
    """Which field is wrong is the whole content of a schema rejection.

    The previous message recorded `attempt 1: ValidationError` and dropped the
    detail, so a reader learned that validation failed and nothing about why.
    """

    _serve(monkeypatch, [_manifest(voxel_size=-1.0)])
    with pytest.raises(Open3dError, match="voxel_size") as raised:
        runtime._load_manifest(PREPARED)
    assert "ValidationError" not in str(raised.value), (
        "the exception's name is not the information; its content is"
    )


def test_a_manifest_that_did_not_parse_is_still_retried(monkeypatch) -> None:
    """The behaviour the split must not break.

    Truncated JSON does not close, which is why a parse failure is worth retrying.
    A live run lost a stage to exactly this.
    """

    good = _manifest()
    responses: list[bytes | Exception] = [
        OSError("IncompleteRead(5848412 bytes read, 276537 more expected)"),
        good[: len(good) // 2],
        good,
    ]
    reads: list[str] = []

    def read(uri: str) -> bytes:
        reads.append(uri)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(runtime, "read_bytes_uri", read)
    manifest = runtime._load_manifest(PREPARED)
    assert [one.id for one in manifest.fragments] == ["a", "b"]
    assert len(reads) == 3, reads


def test_a_store_holding_different_intact_bytes_says_so(monkeypatch) -> None:
    """The reproduction: full-length reads with the wrong digest, three times.

    Nothing is wrong with the transport. The object is not the one prepare recorded,
    and the only action that helps is writing the manifest again.
    """

    fragment = _fragment()
    _serve(monkeypatch, [b"x" * fragment.bytes])
    with pytest.raises(Open3dError) as raised:
        runtime._read_fragment(fragment)

    message = str(raised.value)
    assert "intact and is not the one prepare recorded" in message, message
    assert "Retrying will not help" in message, message
    assert "truncat" not in message, (
        "a full-length read is not a truncation, and offering it as a possibility is "
        "what sent a reader looking at the network"
    )
    assert f"of {fragment.bytes} recorded" in message, (
        "the recorded length is the evidence for the diagnosis and belongs in it"
    )


def test_a_transport_that_keeps_truncating_says_so(monkeypatch) -> None:
    """The reproduction's other half, which produced the same sentence."""

    fragment = _fragment()
    lengths = iter([4, 9, 13])
    monkeypatch.setattr(runtime, "read_bytes_uri", lambda uri: SCAN[: next(lengths)])
    with pytest.raises(Open3dError) as raised:
        runtime._read_fragment(fragment)

    message = str(raised.value)
    assert "ended short of the recorded length" in message, message
    assert "retrying the stage is the action" in message, message
    assert "re-run `npa workbench open3d prepare`" not in message, (
        "re-preparing a prefix whose stored object is fine is wasted work"
    )


def test_the_two_diagnoses_are_not_the_same_sentence(monkeypatch) -> None:
    """The defect stated directly: one message for two causes and two actions."""

    fragment = _fragment()
    messages = []
    for responses in (
        [b"x" * fragment.bytes],
        [SCAN[:4], SCAN[:9], SCAN[:13]],
    ):
        served = iter(responses)
        monkeypatch.setattr(
            runtime,
            "read_bytes_uri",
            lambda uri, served=served, last=responses[-1]: next(served, last),
        )
        with pytest.raises(Open3dError) as raised:
            runtime._read_fragment(fragment)
        messages.append(str(raised.value))

    diverged, truncated = messages
    assert diverged != truncated
    # Not merely different text: different actions, each absent from the other.
    action = "re-run `npa workbench open3d prepare`"
    assert action in diverged and action not in truncated
    assert "retrying the stage" in truncated and "retrying the stage" not in diverged


def test_both_causes_in_one_run_are_reported_as_both(monkeypatch) -> None:
    """The case a two-way branch would get wrong by picking a side.

    An unreliable link in front of a diverged store produces short reads and
    full-length mismatches together. Reporting only the one that happened to come
    last would send the reader down one of two paths with no reason.
    """

    fragment = _fragment()
    responses = [SCAN[:4], b"x" * fragment.bytes, SCAN[:9]]
    served = iter(responses)
    monkeypatch.setattr(
        runtime, "read_bytes_uri", lambda uri: next(served, responses[-1])
    )
    with pytest.raises(Open3dError) as raised:
        runtime._read_fragment(fragment)

    message = str(raised.value)
    assert "2 read(s) ended early" in message, message
    assert "1 arrived at the recorded length" in message, message
    assert "open3d prepare` and, if it still fails, retry the stage" in message, message


def test_transport_errors_alone_are_not_reported_as_a_content_disagreement(
    monkeypatch,
) -> None:
    """No bytes arrived, so nothing was compared to the manifest."""

    monkeypatch.setattr(
        runtime,
        "read_bytes_uri",
        lambda uri: (_ for _ in ()).throw(OSError("connection reset")),
    )
    with pytest.raises(Open3dError) as raised:
        runtime._read_fragment(_fragment())

    message = str(raised.value)
    assert "no attempt returned any bytes to compare" in message, message
    assert "connection reset" in message, "the transport errors are the evidence"


def test_a_matching_read_is_returned_whenever_it_arrives(monkeypatch) -> None:
    """The point of retrying, kept under test at each position in the sequence."""

    fragment = _fragment()
    for position in range(runtime.FRAGMENT_READ_ATTEMPTS):
        responses: list[bytes] = [SCAN[:4]] * position + [SCAN]
        served = iter(responses)
        monkeypatch.setattr(runtime, "read_bytes_uri", lambda uri: next(served))
        assert runtime._read_fragment(fragment) == SCAN, position


def test_the_diagnosis_covers_every_combination_of_what_was_seen() -> None:
    """Four states, each with its own action, and none falling through to silence."""

    seen = {
        runtime._diagnose_fragment_read(short, complete)
        for short in (0, 2)
        for complete in (0, 3)
    }
    assert len(seen) == 4, seen
    assert all(one.strip() for one in seen)
