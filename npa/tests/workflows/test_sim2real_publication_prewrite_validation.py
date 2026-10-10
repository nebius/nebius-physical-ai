"""Known pre-transport failures are distinct from exact lost-response recovery."""

import hashlib

from botocore.exceptions import ConnectionClosedError, ParamValidationError
import pytest

from npa.clients.storage import StoragePreconditionFailed
from npa.workflows.sim2real import publication


@pytest.mark.parametrize("file_write", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        "validation",
        "precondition",
        "prewrite",
        "same-existing",
        "postwrite-lost-response",
    ],
)
def test_mutable_cas_preserves_operation_phase_and_current_version(
    tmp_path, file_write, failure
):
    payload = b"held-out-exact-output"
    path = tmp_path / "output"
    path.write_bytes(payload)
    uri = "s3://demo-bucket/run/reports/sim2real.rrd"

    class Client:
        current = None
        writes = 0

        def put_file_conditional(self, *_args, **kwargs):
            if failure == "validation":
                self.current = payload, "concurrent-writer-version"
                raise ParamValidationError(
                    report="synthetic SDK pre-transport rejection"
                )
            if failure == "precondition":
                self.current = payload, "concurrent-writer-version"
                raise StoragePreconditionFailed(
                    "synthetic actual precondition rejection"
                )
            if failure in {"prewrite", "same-existing"}:
                raise RuntimeError("synthetic failure without a completed write")
            assert kwargs["if_none_match"] is True
            self.current = payload, "this-writer-version"
            self.writes += 1
            raise ConnectionClosedError(endpoint_url="https://storage.example.test")

        put_bytes_conditional = put_file_conditional

        def read_bytes_with_etag(self, _uri):
            return self.current

    client = Client()
    prior = None
    if failure == "same-existing":
        client.current = payload, "prior-writer-version"
        prior = publication.RemoteObjectSnapshot(payload, "prior-writer-version")

    def mutate():
        if file_write:
            return publication._replace_mutable_file(
                client,
                path,
                uri,
                prior,
                digest=hashlib.sha256(payload).hexdigest(),
                size=len(payload),
            )
        return publication._replace_mutable_bytes(client, payload, uri, prior)

    expected = {
        "validation": ParamValidationError,
        "precondition": publication.PublicationConflict,
        "prewrite": RuntimeError,
        "same-existing": RuntimeError,
    }
    if failure == "postwrite-lost-response":
        assert mutate() == "this-writer-version"
        assert client.writes == 1 and client.current[0] == payload
    else:
        with pytest.raises(expected[failure]):
            mutate()
        assert client.writes == 0


@pytest.mark.parametrize("operation", ["write", "delete"])
@pytest.mark.parametrize("notes", ["native", "unavailable", "broken"])
@pytest.mark.parametrize(
    "secondary_type", [RuntimeError, KeyboardInterrupt, SystemExit]
)
def test_failed_publication_readback_preserves_primary_and_portable_notes(
    operation, notes, secondary_type
):
    primary = RuntimeError("original publication failure")
    if notes == "unavailable":
        primary.add_note = None
    elif notes == "broken":

        def broken_note(_note):
            raise KeyboardInterrupt("note attachment interrupted")

        primary.add_note = broken_note

    class Client:
        reads = 0
        mutations = 0

        def put_bytes_conditional(self, *_args, **_kwargs):
            self.mutations += 1
            raise primary

        delete_file_conditional = put_bytes_conditional

        def read_bytes_with_etag(self, _uri):
            self.reads += 1
            raise secondary_type("untrusted transport diagnostic")

    client = Client()
    uri = "s3://demo-bucket/run/reports/sim2real.rrd"
    with pytest.raises(RuntimeError) as caught:
        if operation == "write":
            publication._replace_mutable_bytes(client, b"new result", uri, None)
        else:
            publication._delete_mutable_object(client, uri, if_match="prior-version")

    assert caught.value is primary
    assert client.mutations == 1 and client.reads == 1
    assert str(primary) == "original publication failure"
    if notes == "broken":
        assert not getattr(primary, "__notes__", [])
    else:
        assert primary.__notes__ == [
            f"publication readback also failed: builtins.{secondary_type.__name__}"
        ]
