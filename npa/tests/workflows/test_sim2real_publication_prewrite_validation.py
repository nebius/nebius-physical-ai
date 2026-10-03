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
