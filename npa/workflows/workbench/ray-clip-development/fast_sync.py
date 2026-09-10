"""Qualify Ray Jobs working-directory source delivery with two revisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid


RAY_VERSION = "2.58.0"
RESULT_MARKER = "NPA_RAY_FAST_SYNC_RESULT "
PROBE_SOURCE = f'''"""Observe one editable module inside a real Ray task."""
import hashlib
import json
from pathlib import Path

import ray


@ray.remote
def observe_revision():
    import editable_value

    source = Path(editable_value.__file__).resolve()
    return {{
        "ray_version": ray.__version__,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "value": editable_value.VALUE,
    }}


ray.init()
try:
    result = ray.get(observe_revision.remote())
finally:
    ray.shutdown()
print({RESULT_MARKER!r} + json.dumps(result, sort_keys=True))
'''


def _parser() -> argparse.ArgumentParser:
    """Build the standalone qualification command parser.

    Args:
        None.
    Returns:
        Parser for the explicit Jobs address and private evidence directory.
    Raises:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Verify that Ray 2.58 working_dir delivers a local Python edit."
    )
    parser.add_argument("--address", required=True, help="Explicit private Ray Jobs API address.")
    parser.add_argument(
        "--evidence-dir",
        required=True,
        type=Path,
        help="Fresh owner-only directory outside Git for exact Job evidence.",
    )
    return parser


def _write_private(path: Path, content: str) -> None:
    """Create an owner-only evidence file.

    Args:
        path: New evidence path.
        content: Text to persist.
    Returns:
        None.
    Raises:
        FileExistsError: The evidence path already exists.
        OSError: The file cannot be created or written.
    """
    with open(path, "x", encoding="utf-8", opener=lambda name, flags: os.open(name, flags, 0o600)) as stream:
        stream.write(content)


def _prepare_evidence_directory(path: Path) -> Path:
    """Create and validate a fresh owner-only evidence directory.

    Args:
        path: Directory that must not already exist.
    Returns:
        The resolved evidence directory.
    Raises:
        FileExistsError: The requested directory already exists.
        OSError: The directory cannot be created or inspected.
        ValueError: Ownership or permissions are unsafe.
    """
    path.mkdir(mode=0o700, parents=False)
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    if path.is_symlink() or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise ValueError("evidence directory must be owned by the caller with mode 0700")
    return resolved


def _source_digest(source: Path) -> str:
    """Hash the editable module bytes.

    Args:
        source: Editable Python module.
    Returns:
        Lowercase SHA-256 digest.
    Raises:
        OSError: The source cannot be read.
    """
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _write_revision(source: Path, value: str) -> str:
    """Write one deterministic source revision and return its digest.

    Args:
        source: Editable module path.
        value: Observable value returned by the remote task.
    Returns:
        SHA-256 digest of the written bytes.
    Raises:
        OSError: The source cannot be written or read back.
    """
    source.write_text(f'VALUE = {value!r}\n', encoding="utf-8")
    return _source_digest(source)


def _parse_result(logs: str) -> dict[str, str]:
    """Extract exactly one structured result from Ray Job logs.

    Args:
        logs: Complete native Ray Job logs.
    Returns:
        Validated result fields from the remote task.
    Raises:
        ValueError: The marker count or result schema is invalid.
    """
    payloads = [line.split(RESULT_MARKER, 1)[1] for line in logs.splitlines() if RESULT_MARKER in line]
    if len(payloads) != 1:
        raise ValueError(f"expected one fast-sync result, found {len(payloads)}")
    result = json.loads(payloads[0])
    required = {"ray_version", "source_sha256", "value"}
    if set(result) != required or not all(isinstance(result[field], str) for field in required):
        raise ValueError("fast-sync result has an invalid schema")
    return result


def _wait_for_terminal(client: object, submission_id: str) -> object:
    """Wait for one exact Ray Job to become terminal.

    Args:
        client: Ray JobSubmissionClient-compatible object.
        submission_id: Exact owned submission identifier.
    Returns:
        Terminal Ray Job status.
    Raises:
        Exception: The Ray client cannot query status.
    """
    while True:
        status = client.get_job_status(submission_id)
        if status.is_terminal():
            return status
        time.sleep(0.5)


def _stop_nonterminal_jobs(client: object, submission_ids: list[str]) -> list[dict[str, str]]:
    """Stop only exact owned Jobs and record terminal cleanup status.

    Args:
        client: Ray JobSubmissionClient-compatible object.
        submission_ids: Exact IDs created by this invocation.
    Returns:
        One terminal cleanup record per submitted Job.
    Raises:
        Exception: A Job cannot be queried, stopped, or verified terminal.
    """
    cleanup = []
    for submission_id in submission_ids:
        status = client.get_job_status(submission_id)
        if not status.is_terminal():
            client.stop_job(submission_id)
            status = _wait_for_terminal(client, submission_id)
        cleanup.append({"submission_id": submission_id, "status": str(status)})
    return cleanup


def _submit_revision(client: object, source_root: Path, submission_id: str) -> dict[str, str]:
    """Submit one source revision and return its terminal native evidence.

    Args:
        client: Ray JobSubmissionClient-compatible object.
        source_root: Local directory uploaded by Ray.
        submission_id: Exact owned submission identifier.
    Returns:
        Complete logs plus terminal Job status.
    Raises:
        Exception: Native submission, status, or log retrieval fails.
    """
    client.submit_job(
        submission_id=submission_id,
        entrypoint="python sync_probe.py",
        runtime_env={"working_dir": str(source_root)},
    )
    status = _wait_for_terminal(client, submission_id)
    logs = client.get_job_logs(submission_id)
    return {"job_status": str(status), "logs": logs}


def _validate_revisions(results: list[dict[str, str]]) -> None:
    """Require two exact remote observations with different source and output.

    Args:
        results: Baseline and changed remote observations.
    Returns:
        None.
    Raises:
        ValueError: Results do not prove exact changed source delivery.
    """
    if len(results) != 2:
        raise ValueError("fast-sync qualification requires exactly two revisions")
    for result in results:
        if result["ray_version"] != RAY_VERSION or result["job_status"] != "SUCCEEDED":
            raise ValueError("both revisions must succeed on Ray 2.58.0")
        if result["source_sha256"] != result["expected_sha256"]:
            raise ValueError("remote task did not import the submitted source bytes")
    if results[0]["source_sha256"] == results[1]["source_sha256"]:
        raise ValueError("source digest did not change between revisions")
    if results[0]["value"] == results[1]["value"]:
        raise ValueError("source edit did not change the observable Ray task result")


def _exercise_revisions(
    client: object,
    evidence: Path,
    identifiers: list[str],
    owned: list[str],
) -> list[dict[str, str]]:
    """Submit baseline and changed source packages to one Ray endpoint.

    Args:
        client: Ray JobSubmissionClient-compatible object.
        evidence: Owner-only evidence directory.
        identifiers: Exact baseline and changed Job identifiers.
        owned: Mutable exact-ID ledger updated before each submission.
    Returns:
        Validated-shape observations for both revisions.
    Raises:
        RuntimeError: A native Job does not succeed.
        ValueError: A native result marker is malformed.
        OSError: Source or evidence bytes cannot be written.
        Exception: Native Ray submission or observation fails.
    """
    results = []
    with tempfile.TemporaryDirectory(prefix="npa-ray-fast-sync-") as temporary:
        source_root = Path(temporary)
        (source_root / "sync_probe.py").write_text(PROBE_SOURCE, encoding="utf-8")
        editable = source_root / "editable_value.py"
        revisions = zip(("baseline", "changed"), ("before", "after"), identifiers, strict=True)
        for name, value, submission_id in revisions:
            expected_digest = _write_revision(editable, value)
            owned.append(submission_id)
            observed = _submit_revision(client, source_root, submission_id)
            logs = observed.pop("logs")
            _write_private(evidence / f"{name}.log", logs)
            if observed["job_status"] != "SUCCEEDED":
                raise RuntimeError(f"Ray Job finished with status {observed['job_status']}")
            observed.update(_parse_result(logs), name=name, expected_sha256=expected_digest)
            results.append(observed)
    return results


def qualify(address: str, evidence_directory: Path, *, run_token: str | None = None) -> dict[str, object]:
    """Prove that a local edit changes a remote Ray task without an image build.

    Args:
        address: Explicit Ray Jobs API address.
        evidence_directory: Fresh owner-only directory outside the source tree.
        run_token: Optional public-safe identifier for deterministic tests.
    Returns:
        Sanitized two-revision qualification summary.
    Raises:
        ImportError: Ray 2.58 is unavailable in the client environment.
        RuntimeError: Ray has the wrong version or a Job fails.
        ValueError: Source delivery or evidence validation fails.
        OSError: Source or evidence files cannot be written.
    """
    import ray
    from ray.job_submission import JobSubmissionClient

    if ray.__version__ != RAY_VERSION:
        raise RuntimeError(f"fast-sync qualification requires Ray {RAY_VERSION}")
    evidence = _prepare_evidence_directory(evidence_directory)
    token = run_token or uuid.uuid4().hex
    identifiers = [f"npa-fast-sync-{token}-{name}" for name in ("baseline", "changed")]
    client = JobSubmissionClient(address)
    owned: list[str] = []
    try:
        results = _exercise_revisions(client, evidence, identifiers, owned)
    finally:
        cleanup = _stop_nonterminal_jobs(client, owned)
        _write_private(evidence / "cleanup.json", json.dumps(cleanup, indent=2, sort_keys=True))
    _validate_revisions(results)
    private_result = {"submission_ids": identifiers, "revisions": results, "cleanup": cleanup}
    _write_private(evidence / "result.json", json.dumps(private_result, indent=2, sort_keys=True))
    return {"status": "passed", "ray_version": RAY_VERSION, "revisions": results}


def main(argv: list[str] | None = None) -> int:
    """Run the two-revision fast-sync qualification.

    Args:
        argv: Optional command-line arguments.
    Returns:
        Zero after verified source delivery.
    Raises:
        Exception: Qualification or exact Job cleanup fails.
    """
    arguments = _parser().parse_args(argv)
    print(json.dumps(qualify(arguments.address, arguments.evidence_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
