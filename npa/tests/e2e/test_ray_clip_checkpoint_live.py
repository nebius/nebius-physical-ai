"""Native Ray Jobs stop/resume proof on an explicitly prepared two-GPU cluster.

The private config supplies address, ssh argv, remote_python argv, remote_root,
and evidence_dir. Infrastructure, model access and image preflight are external
prerequisites; this test owns only the unique Jobs and result paths it creates.
"""

import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import uuid

import pytest


pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.timeout(0)]


def _write_private(path, content):
    """Keep logs and receipts owner-only even with a permissive caller umask."""
    with open(path, "x", opener=lambda p, flags: os.open(p, flags, 0o600)) as stream:
        stream.write(content)


def _remote(config, code):
    """Execute inspection on the prepared driver using authenticated SSH."""
    argv = [*config["remote_python"], "-c", code]
    return subprocess.check_output([*config["ssh"], shlex.join(argv)])


def _download(config, source, destination):
    """Preserve actual driver files before infrastructure teardown."""
    payload = _remote(config, (
        "import sys,tarfile\n"
        "with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:\n"
        f"    archive.add({str(source)!r}, arcname='result')\n"
    ))
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        archive.extractall(destination, filter="data")
    for path in [destination, *destination.rglob("*")]:
        path.chmod(0o700 if path.is_dir() else 0o600)
    return destination / "result"


def _committed(directory):
    """Independently hash every committed Parquet file and retain provenance."""
    result = {}
    for marker in sorted(directory.glob("shards/*/commit.json")):
        receipt = json.loads(marker.read_text())
        data = marker.with_name("embeddings.parquet").read_bytes()
        assert hashlib.sha256(data).hexdigest() == receipt["parquet_sha256"]
        result[int(marker.parent.name)] = {"receipt": receipt, "marker_bytes": marker.read_bytes(), "data": data}
    return result


def _inference_events(log):
    """Read complete call measurements independently of the application report."""
    decoder = json.JSONDecoder()
    events = []
    for line in log.splitlines():
        if "RAY_CLIP_INFERENCE " in line:
            value, _ = decoder.raw_decode(line.split("RAY_CLIP_INFERENCE ", 1)[1])
            events.append(value)
    return events


def _verify_runtime(report, source_hashes):
    """Bind every model initialization and surviving GPU actor to shipped bytes."""
    fields = {"application.py": "application_sha256", "worker.py": "source_sha256",
              "validation.py": "validation_sha256", "npa_lancedb_bdd100k_udfs.py": "udf_sha256"}
    for filename, field in fields.items():
        assert report[field] == source_hashes[filename]
        assert all(actor[field] == source_hashes[filename] for actor in report["model_initializations"])
    assert len({(actor["node_id"], tuple(actor["gpu_ids"])) for actor in report["final_actors"]}) == 2
    assert all(actor["cuda"] and actor["gpu_ids"] for actor in report["final_actors"])


def _verify_vectors(directory, records):
    """Reopen Parquet and Lance and compare their real normalized vectors."""
    import lancedb
    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(directory / "embeddings.parquet")
    assert table["record_id"].to_pylist() == list(range(records))
    vectors = np.asarray(table["vector"].to_pylist())
    assert vectors.shape == (records, 512) and np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
    lance = lancedb.connect(str(directory / "lance")).open_table("embeddings").to_arrow().sort_by("record_id")
    assert lance["record_id"].to_pylist() == list(range(records))
    np.testing.assert_array_equal(lance["vector"].to_pylist(), table["vector"].to_pylist())


def _verify_measurements(commits, retained, report, logs):
    """Compare current calls with original committed producer measurements."""
    for index, before in retained.items():
        assert commits[index] == before
    expected = sorted(set(commits) - set(retained))
    assert report["inferred_shards"] == expected
    assert report["reused_checkpoint_shards"] == sorted(retained)
    assert sum(actor["inference_calls"] for actor in report["final_actors"]) == len(expected)
    expected_records = sorted(record for index in expected for record in commits[index]["receipt"]["identity"]["record_ids"])
    events = _inference_events(logs)
    assert sorted(record for event in events for record in event["record_ids"]) == expected_records
    by_records = {tuple(event["record_ids"]): event["inference"] for event in events}
    assert len(by_records) == len(events) == len(expected)
    current_actors = {actor["instance_id"] for actor in report["final_actors"]}
    for index in expected:
        receipt = commits[index]["receipt"]
        assert by_records[tuple(receipt["identity"]["record_ids"])] == receipt["inference"]
        assert receipt["inference"]["instance_id"] in current_actors
    retained_actors = {value["receipt"]["inference"]["instance_id"] for value in retained.values()}
    assert retained_actors.isdisjoint(actor["instance_id"] for actor in report["final_actors"])
    for field, indices in (("inference_actor_seconds_sum", expected),
                           ("retained_checkpoint_inference_actor_seconds_sum", retained)):
        assert report[field] == pytest.approx(sum(commits[i]["receipt"]["inference"]["inference_seconds"] for i in indices))


def _verify_result(directory, records, retained, logs, source_hashes):
    """Reopen vectors and compare independent call, checkpoint and actor counts."""
    commits = _committed(directory)
    report = json.loads((directory / "report.json").read_text())
    _verify_runtime(report, source_hashes)
    _verify_vectors(directory, records)
    _verify_measurements(commits, retained, report, logs)
    return report


def _configuration():
    """Require explicit private access to a preflighted native Jobs cluster."""
    if not (config_path := os.environ.get("NPA_RAY_CLIP_CHECKPOINT_LIVE_CONFIG")):
        pytest.skip("requires private preflighted native Ray Jobs configuration")
    private_config = Path(config_path)
    assert private_config.is_file() and not private_config.is_symlink()
    assert private_config.stat().st_uid == os.getuid() and private_config.stat().st_mode & 0o077 == 0
    return json.loads(private_config.read_text())


def _evidence_directory(config, token):
    """Isolate each invocation's immutable native evidence."""
    evidence_root = Path(config["evidence_dir"])
    evidence_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    assert not evidence_root.is_symlink() and evidence_root.stat().st_uid == os.getuid()
    assert evidence_root.stat().st_mode & 0o077 == 0
    evidence = evidence_root / token
    evidence.mkdir(mode=0o700)
    return evidence


def _package_source(tmp_path, evidence):
    """Ship one hashed application package to every native Job."""
    source = tmp_path / "source"
    source.mkdir()
    package = Path(__file__).parents[2]
    example = package / "workflows/workbench/ray-clip-development"
    for name in ("application.py", "worker.py", "validation.py"):
        shutil.copy2(example / name, source / name)
    shutil.copy2(package / "src/npa/workbench/lancedb/bdd100k_udfs.py", source / "npa_lancedb_bdd100k_udfs.py")
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    _write_private(evidence / "source.json", json.dumps(source_hashes))
    return source, source_hashes


class _NativeJobs:
    """Retain submissions and cleanup evidence across every live case."""

    def __init__(self, config, source, source_hashes, evidence, token):
        from ray.job_submission import JobSubmissionClient

        self.config, self.source, self.source_hashes = config, source, source_hashes
        self.evidence, self.token = evidence, token
        self.client = JobSubmissionClient(config["address"])
        self.root = Path(config["remote_root"]) / token
        assert self.root.is_absolute()
        self.jobs, self.outputs, self.preserved = [], {}, set()
        self.records, self.batch_size = 8192, 32

    def _submit(self, name, directory):
        """Use the upstream Jobs API with one unchanged source package."""
        identifier = f"clip-resume-{self.token}-{name}"
        self.jobs.append(identifier)
        self.outputs[identifier] = directory
        _write_private(self.evidence / f"{name}-submission.json", json.dumps({"job_id": identifier, "output_path": str(directory)}))
        argv = ["python", "application.py", "--actors", "2", "--records", str(self.records),
                "--batch-size", str(self.batch_size), "--output-path", str(directory)]
        self.client.submit_job(submission_id=identifier, entrypoint=shlex.join(argv),
                               runtime_env={"working_dir": str(self.source), "env_vars": {"RAY_DEDUP_LOGS": "0"}})
        return identifier

    def _finish(self, identifier, expected="SUCCEEDED"):
        """Retain native status and logs, including honest terminal failures."""
        while not self.client.get_job_status(identifier).is_terminal():
            time.sleep(1)
        details = self.client.get_job_info(identifier)
        log = self.client.get_job_logs(identifier)
        _write_private(self.evidence / f"{identifier}.log", log)
        _write_private(self.evidence / f"{identifier}.json", json.dumps(vars(details), default=str, indent=2))
        assert str(details.status) == expected
        return log

    def _download_result(self, name, identifier, directory):
        """Mark outputs preserved only after their actual download succeeds."""
        output = _download(self.config, directory, self.evidence / name)
        self.preserved.add(identifier)
        return output

    def _stop_for_cleanup(self, identifier, errors):
        """Establish terminal state before attempting a partial-output snapshot."""
        try:
            self.client.stop_job(identifier)
            while not self.client.get_job_status(identifier).is_terminal():
                time.sleep(1)
            return True
        except Exception as error:
            errors.append({"job_id": identifier, "operation": "stop", "error_type": type(error).__name__})
            return False

    def _status_for_cleanup(self, identifier, terminal, errors):
        """Preserve the final native status even after a failed stop attempt."""
        try:
            details = self.client.get_job_info(identifier)
            terminal = details.status.is_terminal()
            _write_private(self.evidence / f"{identifier}-final.json", json.dumps(vars(details), default=str, indent=2))
        except Exception as error:
            errors.append({"job_id": identifier, "operation": "status", "error_type": type(error).__name__})
        return terminal

    def _logs_for_cleanup(self, identifier, errors):
        """Preserve native failure logs independently of status-query success."""
        try:
            _write_private(self.evidence / f"{identifier}-final.log", self.client.get_job_logs(identifier))
        except Exception as error:
            errors.append({"job_id": identifier, "operation": "logs", "error_type": type(error).__name__})

    def _snapshot_for_cleanup(self, identifier, terminal, errors):
        """Recover partial output only when the corresponding Job is terminal."""
        if not terminal or identifier in self.preserved:
            return
        try:
            directory = self.outputs[identifier]
            exists = _remote(self.config, f"import pathlib,json; print(json.dumps(pathlib.Path({str(directory)!r}).is_dir()))")
            if json.loads(exists):
                _download(self.config, directory, self.evidence / f"{identifier}-cleanup-snapshot")
        except Exception as error:
            errors.append({"job_id": identifier, "operation": "snapshot", "error_type": type(error).__name__})

    def _cleanup(self, original_failure):
        """Attempt every cleanup boundary without masking an original test failure."""
        errors = []
        for identifier in self.jobs:
            terminal = self._stop_for_cleanup(identifier, errors)
            terminal = self._status_for_cleanup(identifier, terminal, errors)
            self._logs_for_cleanup(identifier, errors)
            self._snapshot_for_cleanup(identifier, terminal, errors)
        _write_private(self.evidence / "job-cleanup.json", json.dumps({"attempted": self.jobs, "errors": errors}))
        if errors and not original_failure:
            raise RuntimeError("Native Job cleanup failed; inspect private receipts")


def _resume_partial(jobs):
    """Stop a partial native run and validate its resumed result against saved bytes."""
    partial = jobs.root / "partial"
    interrupted = jobs._submit("interrupted", partial)
    while jobs.client.get_job_logs(interrupted).count("RAY_CLIP_CHECKPOINT ") < 2:
        assert not jobs.client.get_job_status(interrupted).is_terminal()
        time.sleep(0.25)
    assert jobs.client.stop_job(interrupted)
    jobs._finish(interrupted, "STOPPED")
    before = jobs._download_result("interrupted", interrupted, partial)
    committed = _committed(before)
    assert 1 < len(committed) < jobs.records // jobs.batch_size
    resumed = jobs._submit("resumed", partial)
    logs = jobs._finish(resumed)
    completed = jobs._download_result("resumed", resumed, partial)
    _verify_result(completed, jobs.records, committed, logs, jobs.source_hashes)
    return partial, _committed(completed)


def _prepare_boundary(config, partial, target, name, missing):
    """Copy factual checkpoints without replacing their identity or provenance."""
    code = (
        "import pathlib,shutil\n"
        f"source=pathlib.Path({str(partial)!r}); target=pathlib.Path({str(target)!r})\n"
        "target.mkdir(parents=True)\n"
        "shutil.copy2(source/'execution.json',target/'execution.json')\n"
        "shutil.copytree(source/'shards',target/'shards')\n"
    )
    if name == "corrupt":
        code += "(target/'shards/000004/embeddings.parquet').write_bytes(b'corrupt committed bytes')\n"
    elif name == "uncommitted":
        code += "(target/'shards/000003/commit.json').unlink()\n"
        code += "(target/'shards/000003/embeddings.parquet').write_bytes(b'partial uncommitted bytes')\n"
    else:
        code += f"for i in {missing!r}: shutil.rmtree(target/'shards'/f'{{i:06d}}')\n"
    _remote(config, code)


def _verify_boundary(jobs, partial, full, name, missing):
    """Prove each sparse, empty, uncommitted or corrupt checkpoint boundary."""
    target = jobs.root / name
    _prepare_boundary(jobs.config, partial, target, name, missing)
    identifier = jobs._submit(name, target)
    logs = jobs._finish(identifier, "FAILED" if name == "corrupt" else "SUCCEEDED")
    output = jobs._download_result(name, identifier, target)
    if name == "corrupt":
        assert "Checkpoint data hash mismatch" in logs
        assert _inference_events(logs) == []
        assert not (output / "report.json").exists()
        return
    retained = {index: entry for index, entry in full.items() if index not in missing}
    report = _verify_result(output, jobs.records, retained, logs, jobs.source_hashes)
    assert report["concurrency_observation"]["participants"] == min(2, len(missing))
    assert report["concurrent_actor_inference_observed"] == (len(missing) > 1)


def test_native_clip_stop_resume_sparse_and_invalid_checkpoints(tmp_path):
    """Stop after multiple commits, then prove full, sparse and zero-work resumes.

    Args:
        tmp_path: Pytest directory for the submitted source package.
    Returns:
        None.
    Raises:
        AssertionError: Native status, vectors, calls or checkpoint bytes differ.
        RuntimeError: Native Job cleanup fails without an earlier test failure.
        OSError: Private configuration, source or evidence cannot be accessed.
    """
    config = _configuration()
    token = uuid.uuid4().hex
    evidence = _evidence_directory(config, token)
    source, source_hashes = _package_source(tmp_path, evidence)
    jobs = _NativeJobs(config, source, source_hashes, evidence, token)
    try:
        partial, full = _resume_partial(jobs)
        cases = {"sparse": [1, 3], "single": [3], "zero": [], "uncommitted": [3], "corrupt": []}
        for name, missing in cases.items():
            _verify_boundary(jobs, partial, full, name, missing)
    finally:
        jobs._cleanup(original_failure=sys.exc_info()[0] is not None)
