"""What the read-retry path tells an operator to do next.

Both failures below were reproduced against the shipped code before anything was
changed. A manifest that parsed and then failed schema validation was read three
times and reported as `after 3 attempts`, as though the transport had been at
fault. A fragment whose bytes were intact and wrong, and a fragment whose stream
kept ending early, produced the same sentence -- "the manifest and the stored scan
disagree, or the transport kept truncating" -- while the manifest recorded the byte
count that separates them:

    diverged store:   48 bytes read, 48 recorded, every attempt
    shorter object:   17, 17, 17 bytes read, 48 recorded
    truncating link:   4,  9, 13 bytes read, 48 recorded

The actions differ. A diverged store has to be prepared again; a truncating link has
to be retried. A message that offers both leaves the reader to guess at exactly the
moment they have least information.

The middle row is the correction to the first version of this module, which had only
the outer two and so could treat "short of the recorded length" as meaning truncation.
A store rewritten with a shorter object reads short on every attempt, and telling its
reader to retry sends them to repeat a request whose bytes are gone. Length alone does
not separate those two; whether the lengths repeat does.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from npa.workbench.open3d import runtime
from npa.workbench.open3d.artifacts import Open3dError
from npa.workbench.open3d.schemas import Fragment
from npa.workbench.storage_scope import StorageAuthorizationError

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
    assert "2 read(s) ended short of the recorded length" in message, message
    assert "1 arrived at the recorded length" in message, message
    # Both actions, in the order of what the evidence most supports. Asserted as
    # positions rather than one substring, so adding "against this prefix" to the
    # re-prepare instruction cannot break a test about ordering.
    assert message.index("open3d prepare`") < message.index("retry the stage"), message
    assert "if it still fails, retry the stage" in message, message


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
    """Every state the read lengths can be in, each with its own action.

    This test asserted four states when there are six, and counting is what hid the
    defect: it took `short` and `complete` as counts, so "two short reads" was one
    state whether both landed on the same byte or not. Those are different causes with
    opposite actions, and the four-state version could not express the difference.
    """

    recorded = 48
    states = {
        "nothing to compare": [],
        "stable short": [17, 17, 17],
        "varying short": [4, 9, 13],
        "one short": [9],
        "at the recorded length": [48, 48, 48],
        "longer than recorded": [60, 60],
        "short and not short": [9, 48, 48],
    }
    seen = {
        name: runtime._diagnose_fragment_read(lengths, recorded)
        for name, lengths in states.items()
    }

    assert len(set(seen.values())) == len(states), (
        "two states share a sentence, so a reader cannot tell them apart: "
        f"{sorted(seen)}"
    )
    assert all(one.strip() for one in seen.values())

    # The actions have to follow the cause, not just differ from each other.
    retry_is_pointless = (
        "stable short",
        "at the recorded length",
        "longer than recorded",
    )
    for name in retry_is_pointless:
        assert "Retrying will not help" in seen[name], (name, seen[name])
    assert "retrying the stage is the action" in seen["varying short"]
    assert "cannot say whether" in seen["one short"]


def test_a_store_rewritten_with_a_shorter_object_is_not_called_a_truncation(
    monkeypatch,
) -> None:
    """The diagnosis this path got wrong, reproduced before it was changed.

    A short read has two causes, not one. A stream that ended early is short, and so is
    a complete read of a shorter object that replaced the one `prepare` recorded. The
    shipped code took short to mean truncation and told the reader "the transport
    truncated the stream rather than the store holding different bytes; retrying the
    stage is the action" -- for a case where the bytes being asked for are gone and no
    number of retries can produce them.

    The table in this module's docstring assumed a diverged store keeps its recorded
    length. A store rewritten with a shorter object breaks that assumption while
    looking, by length alone, exactly like a truncating link.
    """

    fragment = _fragment()
    rewritten = b"a complete object, just not that one"
    assert len(rewritten) < fragment.bytes, (
        "the fixture has to read short to be the case"
    )
    monkeypatch.setattr(runtime, "read_bytes_uri", lambda uri: rewritten)

    with pytest.raises(Open3dError) as raised:
        runtime._read_fragment(fragment)

    message = str(raised.value)
    assert f"every read ended at the same {len(rewritten)} bytes" in message, message
    assert "complete read of a shorter object" in message, message
    assert "re-run `npa workbench open3d prepare`" in message, message
    assert "Retrying will not help" in message, message
    assert "retrying the stage is the action" not in message, (
        "this is the misdiagnosis: the recommended action can never succeed, because "
        "the object the manifest recorded is no longer in the store"
    )


def test_repetition_is_what_separates_the_two_short_read_causes(monkeypatch) -> None:
    """Same number of short reads, same shortfall, opposite diagnoses.

    Length alone cannot separate a truncated stream from a shorter object, so the
    discriminator has to be whether the reads agree. A stream that ends early stops
    wherever the connection broke; a complete read of a shorter object returns that
    object's length every time. This drives both through the real `_read_fragment` with
    the same count of short reads to show the decision is not the count.
    """

    fragment = _fragment()
    verdicts = {}
    for label, lengths in (("stable", [17, 17, 17]), ("varying", [15, 17, 19])):
        served = iter(lengths)
        monkeypatch.setattr(
            runtime,
            "read_bytes_uri",
            lambda uri, served=served: b"x" * next(served),
        )
        with pytest.raises(Open3dError) as raised:
            runtime._read_fragment(fragment)
        verdicts[label] = str(raised.value)

    assert "Retrying will not help" in verdicts["stable"]
    assert "retrying the stage is the action" in verdicts["varying"]
    assert "different lengths" in verdicts["varying"]
    # The lengths are the evidence for the diagnosis, so they belong in the sentence
    # that makes it. Asserted against the diagnosis alone: the full exception also
    # lists every attempt, so checking the whole message would pass on that list even
    # if the sentence showed nothing.
    sentence = runtime._diagnose_fragment_read([15, 17, 19], fragment.bytes)
    assert "different lengths" in sentence, sentence
    for length in (15, 17, 19, fragment.bytes):
        assert str(length) in sentence, (length, sentence)
    # Both sets are three short reads of comparable size, so neither the count nor the
    # magnitude can be what produced the difference.
    assert "re-run `npa workbench open3d prepare`" not in verdicts["varying"]


def test_one_short_read_is_reported_as_undecidable(monkeypatch) -> None:
    """A single length distinguishes nothing, so the message must not pick a side.

    With two transport errors and one short read there is no repetition to read. The
    honest answer names both causes, in the order of cost, rather than guessing.
    """

    fragment = _fragment()
    responses: list[bytes | Exception] = [
        OSError("connection reset"),
        SCAN[:9],
        OSError("connection reset"),
    ]
    served = iter(responses)

    def read(uri: str) -> bytes:
        value = next(served)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(runtime, "read_bytes_uri", read)
    with pytest.raises(Open3dError) as raised:
        runtime._read_fragment(fragment)

    message = str(raised.value)
    assert "a single length cannot say whether" in message, message
    assert "Retry the stage first" in message, message
    assert "re-run `npa workbench open3d prepare`" in message, message


def test_a_longer_object_is_not_described_as_the_recorded_length(monkeypatch) -> None:
    """`len(payload) < recorded` has an `else`, and it covered two different cases.

    A read longer than the recorded length was counted as having "arrived at the
    recorded length", which is false. The action is the same as for an equal-length
    divergence, but the evidence quoted back to the reader has to be the evidence.
    """

    fragment = _fragment()
    monkeypatch.setattr(
        runtime, "read_bytes_uri", lambda uri: b"x" * (fragment.bytes + 8)
    )
    with pytest.raises(Open3dError) as raised:
        runtime._read_fragment(fragment)

    message = str(raised.value)
    assert f"longer than the {fragment.bytes} bytes recorded" in message, message
    assert "arrived at the recorded length" not in message, message
    assert "Retrying will not help" in message, message


def test_a_refused_read_is_not_retried_and_keeps_its_own_cause(monkeypatch) -> None:
    """`except ValueError` swallowed a scope refusal and renamed it a missing manifest.

    `StorageAuthorizationError` subclasses `ValueError`, so a read refused for falling
    outside the active storage scope landed in the branch meant for JSON that did not
    parse. It was retried twice more and reported as "could not be read as a valid
    registration manifest after 3 attempts; run `npa workbench open3d prepare` against
    this prefix first" -- a remedy for a permission boundary, with the reason text
    dropped because that branch recorded only the exception class name.

    This is the module's own principle applied to its own code: a non-transient failure
    should not be retried. Reachable through `visualize`, which loads a manifest from a
    path read out of a published result rather than one that went through `_tool_paths`.
    """

    reads = []

    def refuse(uri: str) -> bytes:
        reads.append(uri)
        raise StorageAuthorizationError("uri outside the active scope")

    monkeypatch.setattr(runtime, "read_bytes_uri", refuse)
    with pytest.raises(StorageAuthorizationError) as raised:
        runtime._load_manifest(PREPARED)

    assert len(reads) == 1, (
        f"the refusal was read {len(reads)} times; retrying a scope refusal cannot "
        "change the answer"
    )
    assert "outside the active scope" in str(raised.value)
    # Still a ValueError, which is what the CLI renders for a clean exit rather than a
    # traceback, so surfacing it does not change the operator-facing contract.
    assert isinstance(raised.value, ValueError)


def test_a_manifest_that_did_not_decode_is_still_retried(monkeypatch) -> None:
    """Narrowing the retry branch must not stop retrying the case it exists for.

    A stream that ends mid-character raises `UnicodeDecodeError`, and one that ends
    mid-document raises `json.JSONDecodeError`. Both subclass `ValueError`, so catching
    `ValueError` covered them by accident; catching them by name has to cover them on
    purpose. The reason text is now carried too, which the old branch dropped.
    """

    document = _manifest()
    for truncated in (document[: len(document) // 2], b"\xff\xfe not utf-8 at all"):
        served = iter([truncated, truncated, document])
        monkeypatch.setattr(runtime, "read_bytes_uri", lambda uri: next(served))
        manifest = runtime._load_manifest(PREPARED)
        assert len(manifest.fragments) == 2, truncated[:8]

    # And when every attempt fails that way, the reason reaches the operator. This
    # branch used to record the exception class alone, unlike the transport branch
    # beside it, so "JSONDecodeError" was all a reader got.
    half = document[: len(document) // 2]
    monkeypatch.setattr(runtime, "read_bytes_uri", lambda uri: half)
    with pytest.raises(Open3dError) as raised:
        runtime._load_manifest(PREPARED)
    message = str(raised.value)
    assert "JSONDecodeError:" in message, message
    assert "Expecting" in message or "Unterminated" in message, (
        "the parse failure's own text is the content of that failure and is what says "
        "where the document stopped"
    )


def test_a_schema_rejection_keeps_what_the_earlier_attempts_did(monkeypatch) -> None:
    """The terminal cause was right and it threw away the triage detail beside it.

    A transport error on attempt 1 followed by a complete document of the wrong shape
    on attempt 2 reported only the schema rejection, dropping the recorded failure. An
    operator triaging a flaky store needs to see what each attempt did -- this module
    asserts exactly that principle elsewhere.
    """

    served = iter(
        [OSError("connection reset"), json.dumps({"voxel_size": 0.05}).encode()]
    )

    def read(uri: str) -> bytes:
        value = next(served)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(runtime, "read_bytes_uri", read)
    with pytest.raises(Open3dError) as raised:
        runtime._load_manifest(PREPARED)

    message = str(raised.value)
    assert "parsed as JSON but is not a registration manifest" in message, message
    assert "Earlier attempts:" in message, message
    assert "OSError: connection reset" in message, message
    # And a rejection on the first attempt has no earlier attempts to report.
    monkeypatch.setattr(
        runtime, "read_bytes_uri", lambda uri: json.dumps({"voxel_size": 0.05}).encode()
    )
    with pytest.raises(Open3dError) as clean:
        runtime._load_manifest(PREPARED)
    assert "Earlier attempts" not in str(clean.value)
