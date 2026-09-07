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
    with path.open("x", opener=lambda p, flags: os.open(p, flags, 0o600)) as stream:
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


def _verify_result(directory, records, retained, logs, source_hashes):
    """Reopen vectors and compare independent call, checkpoint and actor counts."""
    import lancedb
    import numpy as np
    import pyarrow.parquet as pq

    commits = _committed(directory)
    report = json.loads((directory / "report.json").read_text())
    fields = {"application.py": "application_sha256", "worker.py": "source_sha256",
              "validation.py": "validation_sha256", "npa_lancedb_bdd100k_udfs.py": "udf_sha256"}
    for filename, field in fields.items():
        assert report[field] == source_hashes[filename]
        assert all(actor[field] == source_hashes[filename] for actor in report["model_initializations"])
    assert len({(actor["node_id"], tuple(actor["gpu_ids"])) for actor in report["final_actors"]}) == 2
    assert all(actor["cuda"] and actor["gpu_ids"] for actor in report["final_actors"])
    table = pq.read_table(directory / "embeddings.parquet")
    assert table["record_id"].to_pylist() == list(range(records))
    vectors = np.asarray(table["vector"].to_pylist())
    assert vectors.shape == (records, 512) and np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
    lance = lancedb.connect(str(directory / "lance")).open_table("embeddings").to_arrow().sort_by("record_id")
    assert lance["record_id"].to_pylist() == list(range(records))
    np.testing.assert_array_equal(lance["vector"].to_pylist(), table["vector"].to_pylist())
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
    return report


def test_native_clip_stop_resume_sparse_and_invalid_checkpoints(tmp_path):
    """Stop after multiple commits, then prove full, sparse and zero-work resumes."""
    if not (config_path := os.environ.get("NPA_RAY_CLIP_CHECKPOINT_LIVE_CONFIG")):
        pytest.skip("requires private preflighted native Ray Jobs configuration")
    from ray.job_submission import JobSubmissionClient

    private_config = Path(config_path)
    assert private_config.is_file() and not private_config.is_symlink()
    assert private_config.stat().st_uid == os.getuid() and private_config.stat().st_mode & 0o077 == 0
    config = json.loads(private_config.read_text())
    token = uuid.uuid4().hex
    evidence_root = Path(config["evidence_dir"])
    evidence_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    assert not evidence_root.is_symlink() and evidence_root.stat().st_uid == os.getuid()
    assert evidence_root.stat().st_mode & 0o077 == 0
    evidence = evidence_root / token
    evidence.mkdir(mode=0o700)
    source = tmp_path / "source"
    source.mkdir()
    package = Path(__file__).parents[2]
    example = package / "workflows/workbench/ray-clip-development"
    for name in ("application.py", "worker.py", "validation.py"):
        shutil.copy2(example / name, source / name)
    shutil.copy2(package / "src/npa/workbench/lancedb/bdd100k_udfs.py", source / "npa_lancedb_bdd100k_udfs.py")
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    _write_private(evidence / "source.json", json.dumps(source_hashes))
    client = JobSubmissionClient(config["address"])
    root = Path(config["remote_root"]) / token
    assert root.is_absolute()
    jobs = []
    records, batch_size = 8192, 32

    def submit(name, directory):
        """Use the upstream Jobs API with one unchanged source package."""
        identifier = f"clip-resume-{token}-{name}"
        jobs.append(identifier)
        _write_private(evidence / f"{name}-submission.json", json.dumps({"job_id": identifier, "output_path": str(directory)}))
        argv = ["python", "application.py", "--actors", "2", "--records", str(records),
                "--batch-size", str(batch_size), "--output-path", str(directory)]
        client.submit_job(submission_id=identifier, entrypoint=shlex.join(argv),
                          runtime_env={"working_dir": str(source), "env_vars": {"RAY_DEDUP_LOGS": "0"}})
        return identifier

    def finish(identifier, expected="SUCCEEDED"):
        """Retain native status and logs, including honest terminal failures."""
        while not client.get_job_status(identifier).is_terminal():
            time.sleep(1)
        details = client.get_job_info(identifier)
        log = client.get_job_logs(identifier)
        _write_private(evidence / f"{identifier}.log", log)
        _write_private(evidence / f"{identifier}.json", json.dumps(vars(details), default=str, indent=2))
        assert str(details.status) == expected
        return log

    try:
        partial = root / "partial"
        interrupted = submit("interrupted", partial)
        while client.get_job_logs(interrupted).count("RAY_CLIP_CHECKPOINT ") < 2:
            assert not client.get_job_status(interrupted).is_terminal()
            time.sleep(0.25)
        assert client.stop_job(interrupted)
        finish(interrupted, "STOPPED")
        before = _download(config, partial, evidence / "interrupted")
        committed = _committed(before)
        assert 1 < len(committed) < records // batch_size
        resumed = submit("resumed", partial)
        logs = finish(resumed)
        completed = _download(config, partial, evidence / "resumed")
        _verify_result(completed, records, committed, logs, source_hashes)

        # Each boundary starts from a separate copy of the completed factual
        # checkpoint tree. Never rewrite its identity, actor or timing receipts.
        full = _committed(completed)
        cases = {"sparse": [1, 3], "single": [3], "zero": [], "uncommitted": [3], "corrupt": []}
        for name, missing in cases.items():
            target = root / name
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
            identifier = submit(name, target)
            log = finish(identifier, "FAILED" if name == "corrupt" else "SUCCEEDED")
            output = _download(config, target, evidence / name)
            if name == "corrupt":
                assert "Checkpoint data hash mismatch" in log
                assert _inference_events(log) == []
                assert not (output / "report.json").exists()
            else:
                retained = {i: entry for i, entry in full.items() if i not in missing}
                report = _verify_result(output, records, retained, log, source_hashes)
                assert report["concurrency_observation"]["participants"] == min(2, len(missing))
                assert report["concurrent_actor_inference_observed"] == (len(missing) > 1)
    finally:
        original_failure = sys.exc_info()[0] is not None
        cleanup_errors = []
        for identifier in jobs:
            try:
                if not client.get_job_status(identifier).is_terminal():
                    client.stop_job(identifier)
                    finish(identifier, "STOPPED")
            except Exception as error:
                cleanup_errors.append({"job_id": identifier, "error_type": type(error).__name__})
        _write_private(evidence / "job-cleanup.json", json.dumps({"attempted": jobs, "errors": cleanup_errors}))
        if cleanup_errors and not original_failure:
            raise RuntimeError("Native Job cleanup failed; inspect private receipts")
