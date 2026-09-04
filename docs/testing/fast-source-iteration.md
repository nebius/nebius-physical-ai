# Reproduce a container source-change iteration

This is the measured CPU example for the [Ray and fast development audit](../architecture/ray-fast-development-audit.md).
It calls the real NPA dataset implementation to ingest, validate, curate and
query 100,000 synthetic sensor metadata records. The query uses the supported
manifest backend. It does not decode sensor payloads or invoke LanceDB,
FiftyOne, a GPU, Ray, or a remote scheduler.

Use an isolated checkout with an editable `npa/.venv` installed from
`npa[dev,adapter]`, Docker on the same host, and a clean
`npa/src/npa/workbench/dataset/curation.py`. Run from the repository root.
The experiment reads the committed module with `git show`, writes a private
copy, and changes its quality comparison from `<` to `<=`. A nested read-only
mount selects that copy inside the container. The checkout is never edited;
the runner removes the private override in `finally`. Keep the source unchanged
while comparing the runs so each version has a stable input tree.

## Prepare the existing image and dependency directory

The image below is an existing published release artifact, pinned by digest.
It supplies Python 3.11 and the control runtime, but its intentionally narrow
package set lacks Pydantic. Install that dependency **once**, inside the same
image/interpreter, into an isolated directory. The benchmark keeps those bytes
fixed across iterations; no image is built. For a different workload use its
compatible workbench image and dependency set. A successful source mount does
not validate a new native dependency or GPU ABI.

```bash
set -euo pipefail
umask 077
export NPA_DEV_ROOT="$(mktemp -d)"
export NPA_DEV_IMAGE='ghcr.io/nebius/nebius-physical-ai/npa-sim2real-control@sha256:87fe8530710eea43364a21ad76dbe4b4c2d60e4b49705824fcdb62dc7d185af7'
mkdir -p "$NPA_DEV_ROOT/deps" "$NPA_DEV_ROOT/output"
git diff --exit-code -- npa/src/npa/workbench/dataset/curation.py
git diff --cached --exit-code -- npa/src/npa/workbench/dataset/curation.py

docker pull "$NPA_DEV_IMAGE"
docker run --rm --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges \
  --mount "type=bind,src=$NPA_DEV_ROOT/deps,dst=/deps" \
  --entrypoint python3 "$NPA_DEV_IMAGE" \
  -m pip install --disable-pip-version-check --no-cache-dir \
  --target /deps pydantic==2.12.5 > "$NPA_DEV_ROOT/dependency-setup.log" 2>&1
```

Image pull and dependency installation are preparation, excluded from the
iteration timings. The dependency log records the resolved transitive versions;
archive it with the results. Promotion still needs the normal locked image build
and validation process. These containers receive no cloud credentials, no Docker
socket, no host home directory, and no network during workload execution. Source
and dependencies are read-only; only the output directory is writable.

## Write the workload and iteration runner

The workload verifies the imported file hash, all selected record IDs against an
independent predicate, the validation result, and child lineage. Artifact hashes
cover actual serialized files; they are distinct from NPA's manifest identity.

```bash
set -euo pipefail
cat > "$NPA_DEV_ROOT/workload.py" <<'PY_WORKLOAD'
import hashlib
import json
import sys
import time
from pathlib import Path

from npa.workbench.dataset import curation
from npa.workbench.dataset.ingestion import ingest_dataset
from npa.workbench.dataset.schemas import CurateRequest, IngestRequest, QueryRequest, ValidateRequest
from npa.workbench.dataset.validation import validate_manifest

started = time.perf_counter()
label, expected_hash = sys.argv[1:]
module = Path(curation.__file__)
module_hash = hashlib.sha256(module.read_bytes()).hexdigest()
assert module_hash == expected_hash, (module, module_hash, expected_hash)
root = Path('/output') / label
root.mkdir()
raw = [dict(record_id=f'frame-{i:06d}', modality='camera',
            uri=f'/synthetic/frame-{i:06d}.png',
            event='cut_in' if i % 2 == 0 else 'cruise',
            location='synthetic_west' if (i // 2) % 2 == 0 else 'synthetic_east',
            timestamp=f'{i / 30:.6f}', quality={'confidence': ((i // 4) % 10) / 10})
       for i in range(100000)]
input_path = root / 'raw.json'
input_path.write_text(json.dumps({'records': raw}))
ingested = ingest_dataset(IngestRequest(input_uri=str(input_path), output_uri=str(root / 'dataset'),
                                       dataset_id='synthetic-sensors', source='synthetic', version='v1'))
assert ingested.record_count == len(raw)
validated = validate_manifest(ValidateRequest(input_uri=ingested.manifest_uri,
                                             output_uri=str(root / 'validation'), completeness_min=0.7))
assert validated.passed and validated.record_count == len(raw)
curated = curation.curate_dataset(CurateRequest(input_uri=ingested.manifest_uri,
    output_uri=str(root / 'curated'), event='cut_in', location='synthetic_west',
    quality_metric='confidence', min_quality=0.5))
queried = curation.query_dataset(QueryRequest(input_uri=curated.manifest_uri, limit=len(raw)))
strict = label == 'changed'
expected = [r['record_id'] for r in raw if r['event'] == 'cut_in'
            and r['location'] == 'synthetic_west'
            and (r['quality']['confidence'] > 0.5 if strict else r['quality']['confidence'] >= 0.5)]
assert [r['record_id'] for r in queried.records] == expected
assert curated.record_count == len(expected) == (10000 if strict else 12500)
child = json.loads(Path(curated.manifest_uri).read_text())
assert child['lineage']['parent_dataset_id'] == 'synthetic-sensors'
assert child['lineage']['parent_version'] == 'v1'
artifacts = {str(p.relative_to(root)): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                                     'bytes': p.stat().st_size} for p in sorted(root.rglob('*.json'))}
result = dict(label=label, input_records=len(raw), selected_records=len(expected),
              module_path=str(module), module_sha256=module_hash,
              workload_seconds=round(time.perf_counter() - started, 6),
              validation_passed=validated.passed, query_backend=queried.backend,
              selected_ids_sha256=hashlib.sha256('\n'.join(expected).encode()).hexdigest(),
              artifacts=artifacts)
(root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
PY_WORKLOAD
```

The runner starts a fresh container for each source version, passes the expected
source hash, measures container wall time, and removes the private override even when a
workload assertion fails. Its result distinguishes wall time from the timed
workload, which includes synthetic input generation and artifact serialization.

```bash
set -euo pipefail
cat > "$NPA_DEV_ROOT/run.py" <<'PY_RUNNER'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

repo = Path.cwd()
root = Path(__file__).resolve().parent
image = 'ghcr.io/nebius/nebius-physical-ai/npa-sim2real-control@sha256:87fe8530710eea43364a21ad76dbe4b4c2d60e4b49705824fcdb62dc7d185af7'
source = repo / 'npa/src/npa/workbench/dataset/curation.py'
original = subprocess.check_output(['git', 'show', 'HEAD:npa/src/npa/workbench/dataset/curation.py'])
assert source.read_bytes() == original, 'Use a clean target source file'
override = root / 'curation-override.py'
needle = b'_record_metric(record, quality_metric) < min_quality:'
assert original.count(needle) == 1
changed = original.replace(needle, b'_record_metric(record, quality_metric) <= min_quality:')
results = []
try:
    for label, content in [('baseline', original), ('changed', changed), ('restored', original)]:
        override.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        cmd = ['docker', 'run', '--rm', '--network', 'none', '--read-only',
               '--user', f'{os.getuid()}:{os.getgid()}', '--cap-drop', 'ALL',
               '--security-opt', 'no-new-privileges',
               '--mount', f'type=bind,src={repo / "npa/src"},dst=/src,readonly',
               '--mount', f'type=bind,src={override},dst=/src/npa/workbench/dataset/curation.py,readonly',
               '--mount', f'type=bind,src={root / "deps"},dst=/deps,readonly',
               '--mount', f'type=bind,src={root / "output"},dst=/output',
               '--mount', f'type=bind,src={root / "workload.py"},dst=/workload.py,readonly',
               '--env', 'PYTHONPATH=/src:/deps', '--env', 'PYTHONDONTWRITEBYTECODE=1',
               '--entrypoint', 'python3', image, '/workload.py', label, expected]
        start = time.perf_counter()
        run = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.perf_counter() - start
        (root / f'{label}.log').write_text(run.stdout + run.stderr)
        results.append({'label': label, 'exit_code': run.returncode, 'container_wall_seconds': round(wall, 6)})
        if run.returncode:
            raise RuntimeError(f'{label} failed; see private log')
        results[-1].update(json.loads(run.stdout))
        print(json.dumps({k: v for k, v in results[-1].items() if k != 'artifacts'}), flush=True)
finally:
    override.unlink(missing_ok=True)
    assert source.read_bytes() == original, 'Host source changed during run'
    (root / 'workload-evidence.json').write_text(json.dumps({'image': image, 'runs': results,
        'host_source_unchanged': source.read_bytes() == original, 'container_builds': 0,
        'data': '100000 synthetic sensor metadata records; sensor bytes are not read',
        'execution': 'local Docker on operator VM; manifest backend; no GPU or remote scheduler'}, indent=2) + '\n')
PY_RUNNER
npa/.venv/bin/python "$NPA_DEV_ROOT/run.py"
git diff --exit-code -- npa/src/npa/workbench/dataset/curation.py
```

Expected counts are **12,500 → 10,000 → 12,500**, with identical baseline and
restored module/selected-ID hashes. Inspect `workload-evidence.json`, each
`output/*/result.json`, the dataset manifests, and validation reports beneath
`NPA_DEV_ROOT`. Containers remove themselves on completion. Keep the evidence
until reviewed, then remove only this experiment's temporary directory.

## Observed execution, 2026-09-04

| Source | Input records | Selected records | Workload seconds | Container wall seconds |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 100,000 | 12,500 | 5.011 | 5.653 |
| Changed comparison | 100,000 | 10,000 | 5.918 | 6.893 |
| Restored | 100,000 | 12,500 | 5.942 | 6.905 |

The release manifest was independently fetched anonymously from GHCR (HTTP
200), its SHA-256 matched the requested digest, and Docker pulled it successfully
before the measured runs. Runtime versions were Python 3.11.15, Pydantic 2.12.5,
pydantic-core 2.41.5, httpx 0.28.1, and boto3 1.43.62.

All three containers exited **0**, all validation reports passed, and the
host source stayed unchanged byte-for-byte. Each container imported
`/src/npa/workbench/dataset/curation.py`. Baseline/restored module SHA-256:
`77de310fc9af3fa415f001c2e622e69a7bbf5f155b631b2ba6a7a33e46d5362d`;
changed module SHA-256:
`870dd61f4d7ed842f224151f92c416d480852b8a2e010c74711595fb4236d3dd`.
The selected-ID digest also returned to its original value after restoration.

These are three observations on one operator VM with the image already cached,
not a throughput study or a measured speedup over rebuilding. There were **zero
container builds**. The experiment proves changed NPA Python executed against
unchanged container and dependency bytes; it does not measure S3 transfer,
SkyPilot scheduling, Ray performance, cold image pulls, or model loading.
