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
reader to retry sends them to repeat a request whose bytes are gone.

The second correction is that repetition does not separate those two either, which the
version before this one claimed. A transport that truncates at a fixed point -- a
buffer cap, a proxy cutoff, a repeatable stream failure -- returns the same prefix
every attempt. Against an unchanged 100-byte object, a deterministic 50-byte cutoff
produces [50, 50, 50]; so does a genuine 50-byte replacement. The second row and the
truncation row are therefore observationally identical, and the first repair replaced
one categorical misdiagnosis with its converse.

What these tests hold now is narrower: the message reports what was observed, names
every cause consistent with it, points at the store metadata that does discriminate,
and does not claim to have ruled anything out. `test_the_same_observation_under_both_
causes_is_not_told_apart` drives one stable-short observation through both labelled
causes and requires the two diagnoses to be the same sentence.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from npa.workbench.open3d import runtime
from npa.workbench.open3d.artifacts import Open3dError
from npa.workbench.open3d.schemas import Fragment
from npa.workbench.storage_scope import StorageAuthorizationError

#: Phrasings that claim a cause was identified. Every one of these was in the message
#: at some point, and each is unsupported by read lengths alone: the first repair said
#: stable lengths proved a shorter stored object, and root reproduced [50, 50, 50] from
#: a 100-byte object that never changed with a deterministic 50-byte cutoff. The
#: converse phrasings are here too, so this guard does not just forbid one direction.
_VERDICT_PHRASES = (
    "complete read of a shorter object rather than",
    "rather than a truncated stream",
    "rather than the store holding",
    "intact and is not the one prepare recorded",
    "Retrying will not help",
    "retrying the stage is the action",
    "so this is a",
    "the transport truncated the stream rather than",
    "rarely stop",
    "rarely land",
)

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


def test_a_full_length_read_names_both_divergence_and_corruption(monkeypatch) -> None:
    """The reproduction: full-length reads with the wrong digest, three times.

    This is the strongest case in the table and it still does not reach a verdict.
    Bytes of the recorded length that do not match the recorded digest are consistent
    with an object of the same length whose content differs, and with bytes corrupted
    in transit -- which `_read_fragment`'s own docstring already acknowledged while
    the message asserted the first.
    """

    fragment = _fragment()
    _serve(monkeypatch, [b"x" * fragment.bytes])
    with pytest.raises(Open3dError) as raised:
        runtime._read_fragment(fragment)

    message = str(raised.value)
    assert f"arrived at exactly {fragment.bytes} bytes" in message, message
    assert "content differs from what prepare digested" in message, message
    assert "corrupted in transit" in message, (
        "an in-transit corruption of the right length produces this observation too, "
        "and the function's own docstring says so"
    )
    assert "cannot settle which it is" in message, message
    assert f"against the {fragment.bytes} bytes" in message, (
        "the recorded length is the evidence for the observation and belongs in it"
    )


def test_varied_short_reads_do_not_exclude_a_changing_object(monkeypatch) -> None:
    """The reproduction's other half, which produced the same sentence.

    Varying lengths point at a stream ending wherever the connection broke, and that
    is said. What is not said is that the store is therefore fine: an object being
    rewritten between attempts produces varying short reads too.
    """

    fragment = _fragment()
    lengths = iter([4, 9, 13])
    monkeypatch.setattr(runtime, "read_bytes_uri", lambda uri: SCAN[: next(lengths)])
    with pytest.raises(Open3dError) as raised:
        runtime._read_fragment(fragment)

    message = str(raised.value)
    assert "ended short of the recorded 48 bytes at 3 different lengths" in message
    assert "ending wherever the connection broke" in message, message
    assert "being rewritten between attempts" in message, (
        "varying lengths do not rule out a changing object, and the previous version "
        "of this message said they did"
    )
    for phrase in _VERDICT_PHRASES:
        assert phrase not in message, (phrase, message)


def test_the_two_observations_are_not_the_same_sentence(monkeypatch) -> None:
    """The original defect stated directly: one message for two unlike observations.

    Full-length mismatches and varying short reads are different things to have seen
    and still read differently. What they now share is the discriminating step, which
    is correct: neither observation settles the cause, so both hand the operator the
    same comparison to make.
    """

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

    full_length, varied_short = messages
    assert full_length != varied_short
    # Different observations, each quoting its own evidence.
    assert f"arrived at exactly {fragment.bytes} bytes" in full_length
    assert "arrived at exactly" not in varied_short
    assert "at 3 different lengths" in varied_short
    assert "different lengths" not in full_length
    # And different candidate causes, because the observations differ.
    assert "corrupted in transit" in full_length
    assert "corrupted in transit" not in varied_short
    assert "ending wherever the connection broke" in varied_short
    assert "ending wherever the connection broke" not in full_length
    # The same discriminating step, because neither observation decides anything.
    for message in messages:
        assert "cannot settle which it is" in message
        assert "Compare what the store currently reports" in message


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
    assert f"2 read(s) ended short of the recorded {fragment.bytes} bytes" in message
    assert f"1 arrived at exactly {fragment.bytes} bytes" in message, message
    # Every cause either side of the mixed observation is consistent with, not a
    # choice between them. Three causes here: two for the short reads and one for the
    # full-length mismatch.
    for cause in (
        "a stream ending before the object did",
        "now shorter than the one prepare recorded",
        "content differs from what prepare digested",
        "corrupted in transit",
    ):
        assert cause in message, (cause, message)
    assert "cannot settle which it is" in message, message
    for phrase in _VERDICT_PHRASES:
        assert phrase not in message, (phrase, message)


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
    # This is the one state where an unqualified action is supportable -- nothing
    # arrived, so there is no content to disagree about. That makes it the state most
    # at risk of acquiring a claim about the store, which it cannot support either.
    # Asserted against the diagnosis alone: the exception's own prefix says "the bytes
    # the manifest recorded at prepare time", so a check against the whole message
    # would match that prefix and pass no matter what the diagnosis claimed.
    diagnosis = runtime._diagnose_fragment_read([], _fragment().bytes)
    assert diagnosis in message, "the diagnosis has to be the one in the exception"
    for claim in ("diverged", "shorter", "prepare recorded", "prepare saw"):
        assert claim not in diagnosis, (claim, diagnosis)
    assert "re-run `npa workbench open3d prepare`" not in diagnosis, (
        "no bytes were compared, so nothing here points at the manifest"
    )


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

    # Each state describes its own observation, and none of them reaches a verdict.
    assert "at the same 17 bytes every time" in seen["stable short"]
    assert "at 3 different lengths" in seen["varying short"]
    assert "there is not even a repeated length" in seen["one short"]
    assert "arrived at exactly 48 bytes" in seen["at the recorded length"]
    assert "longer than the 48 bytes recorded" in seen["longer than recorded"]
    assert "did not complete" in seen["nothing to compare"]

    for name, message in seen.items():
        if name == "nothing to compare":
            # Nothing arrived, so there is no content disagreement to be undecided
            # about; this is the one state where an unqualified action is supportable.
            continue
        assert "Consistent with" in message, (name, message)
        assert "cannot settle which it is" in message, (name, message)
        for phrase in _VERDICT_PHRASES:
            assert phrase not in message, (name, phrase, message)


def test_the_same_observation_under_both_causes_is_not_told_apart(monkeypatch) -> None:
    """Root's counterexample, as a test. Two causes, one observation, one sentence.

    Row A: the store has been rewritten with a 17-byte object. Every read is complete
    and returns 17 bytes.

    Row B: the store still holds exactly the 48 bytes `prepare` recorded and its
    digest is still correct, but the transport truncates at a fixed 17 bytes -- a
    buffer cap, a consistent proxy cutoff, a repeatable stream failure. Every read
    returns the same first 17 bytes.

    Row C: the control. A varying truncation, which is the only short case whose
    lengths carry any information at all, and even that does not exclude an object
    being rewritten between attempts.

    A and B are observationally identical: [17, 17, 17] of 48 recorded, digest never
    matching. The previous version of this code read that as proof of row A and told
    the operator the store no longer held what prepare recorded and that retrying
    would not help -- false for row B, where the store is intact and the transport is
    the whole problem. A and B must therefore produce the same sentence, and that
    sentence must not pick one.
    """

    fragment = _fragment()
    assert fragment.bytes == 48
    rewritten = b"a complete object, just not that one"[:17]

    def diagnose(serve) -> str:
        monkeypatch.setattr(runtime, "read_bytes_uri", serve)
        with pytest.raises(Open3dError) as raised:
            runtime._read_fragment(fragment)
        return str(raised.value)

    # Row A: a genuinely shorter stored object, read completely every time.
    assert len(rewritten) == 17
    shorter_object = diagnose(lambda uri: rewritten)

    # Row B: the recorded object, unchanged, behind a deterministic 17-byte cutoff.
    assert len(SCAN) == fragment.bytes
    assert hashlib.sha256(SCAN).hexdigest() == fragment.sha256, (
        "row B has to serve the object the manifest recorded, or it is not the case "
        "root reproduced: an unchanged store behind a truncating transport"
    )
    fixed_truncation = diagnose(lambda uri: SCAN[:17])

    # Row C: a varying truncation of the same unchanged object.
    varying = iter([15, 17, 19])
    varied_truncation = diagnose(lambda uri, it=varying: SCAN[: next(it)])

    def sentence(message: str) -> str:
        return message.split("fragment scan-a could not be read")[-1].split(
            "attempt 1:"
        )[0]

    assert sentence(shorter_object) == sentence(fixed_truncation), (
        "the two causes produce identical read lengths, so any difference in the "
        "diagnosis is an inference the observation does not support\n"
        f"A: {sentence(shorter_object)}\nB: {sentence(fixed_truncation)}"
    )
    assert sentence(varied_truncation) != sentence(fixed_truncation), (
        "stable and varying lengths are different observations and should read "
        "differently, even though neither is decisive"
    )

    for label, message in (
        ("shorter object", shorter_object),
        ("fixed truncation", fixed_truncation),
        ("varied truncation", varied_truncation),
    ):
        for phrase in _VERDICT_PHRASES:
            assert phrase not in message, (label, phrase, message)
        assert "Consistent with" in message, (label, message)
        assert "cannot settle which it is" in message, (label, message)
        assert "a stream ending before the object did" in message, (label, message)
        assert "now shorter than the one prepare recorded" in message, (label, message)


def test_the_message_points_at_the_check_that_does_discriminate(monkeypatch) -> None:
    """Naming both causes is only half an answer; the operator still has to choose.

    The discriminating evidence is not in the read at all -- it is what the store says
    it currently holds, against what the manifest says `prepare` saw. That comparison
    separates the two causes that the lengths cannot, and it is cheap.

    It also has to be the comparison and not the remedy. Re-preparing rewrites the
    manifest to match whatever is in the store now, so for row B above it would make
    the error disappear while leaving every later run reading truncated scans.
    """

    fragment = _fragment()
    monkeypatch.setattr(runtime, "read_bytes_uri", lambda uri: SCAN[:17])
    with pytest.raises(Open3dError) as raised:
        runtime._read_fragment(fragment)
    message = str(raised.value)

    assert "Compare what the store currently reports for this object" in message
    assert "checksum or version metadata" in message
    assert "If the store still reports what prepare saw" in message
    assert "the transport is the cause" in message
    assert "If the store reports something else" in message
    assert "re-run `npa workbench open3d prepare`" in message

    assert "Do not re-prepare on the strength of the read lengths alone" in message, (
        "re-preparing on the lengths alone rewrites the manifest around a truncating "
        "transport, which hides the fault instead of fixing it"
    )
    assert "hides a truncating transport rather than fixing it" in message, (
        "the caution has to say why, or the same sentence can be turned into its "
        "opposite -- 'which is the remedy' -- without any guard noticing"
    )
    assert message.index("Compare what the store") < message.index(
        "re-run `npa workbench open3d prepare`"
    ), "the comparison has to come before the action it selects"


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
    assert "there is not even a repeated length to describe" in message, message
    assert "a stream ending before the object did" in message, message
    assert "now shorter than the one prepare recorded" in message, message
    assert "cannot settle which it is" in message, message
    # No repetition was observed, so neither the fixed-cutoff nor the
    # rewritten-between-attempts qualifier is supportable here.
    assert "at the same byte on every attempt" not in message, message
    assert "being rewritten between attempts" not in message, message
    for phrase in _VERDICT_PHRASES:
        assert phrase not in message, (phrase, message)


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
    assert "arrived at exactly" not in message, message
    assert "ended short" not in message, message
    # A longer read is not a truncation, but it is not proof of a replaced object
    # either: a transport that re-sends or concatenates delivers more than the object.
    assert "has grown or been replaced since prepare" in message, message
    assert "re-send or concatenation" in message, message
    for phrase in _VERDICT_PHRASES:
        assert phrase not in message, (phrase, message)


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
