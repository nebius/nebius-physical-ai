# End-to-End (E2E) Testing

E2E tests exercise `npa` commands against real Nebius infrastructure such as
S3 buckets and, over time, live workbenches. They complement the mock-based
unit and integration tests, but they create real resources and require valid
operator credentials.

## Quick Start

```bash
cd ~/repos/nebius-physical-ai
source npa/.venv/bin/activate
NPA_INTEGRATION_E2E=1 pytest npa/tests/e2e/ -v
```

If the default project in `~/.npa/config.yaml` does not contain object-storage
credentials, set the e2e project explicitly:

```bash
NPA_E2E_PROJECT=eu-north1 NPA_INTEGRATION_E2E=1 pytest npa/tests/e2e/ -v
```

The harness also auto-selects `eu-north1` when that project has storage
credentials configured, matching the current operator demo environment.

Live GPU tests marked `gpu and e2e` run manually from the Nebius Dev VM, not
from GitHub Actions. See [Live GPU E2E On The Dev VM](live-e2e.md).

## What E2E Tests Cover

Current Python e2e tests cover:

- `npa demo stage` against real S3:
  artifact landing, sha256 metadata round-trip, idempotency, and
  `demo verify` agreement.
- E2E harness smoke coverage:
  real test bucket creation, empty listing, and teardown.
- Existing opt-in live checks for BYOVM and GR00T NGC status.

Known current result: the first real `npa demo stage` run surfaced a product
bug on Nebius S3 metadata casing. Nebius returns the uploaded `sha256` metadata
key as `Sha256`, while `demo stage` and `demo verify` currently perform a
case-sensitive lookup for `sha256`. The demo-stage e2e tests are expected to
fail until that lookup is normalized.

## Gating

E2E tests are skipped by default. They run only when
`NPA_INTEGRATION_E2E` is set:

```bash
# These commands skip general e2e tests:
pytest npa/tests/
pytest npa/tests/e2e/

# These commands run general e2e tests:
NPA_INTEGRATION_E2E=1 pytest npa/tests/e2e/
NPA_INTEGRATION_E2E=1 pytest npa/tests/ -m e2e
```

Some older live tests have their own additional opt-in variables, such as
`NPA_E2E_BYOVM_SELF_HEAL` or `NPA_TEST_GROOT_NGC_E2E`. Those tests still skip
unless their specific live target configuration is present.

## Required Credentials

E2E tests use the same project storage credentials as the CLI through
`npa.clients.project_credentials`. `~/.npa/credentials.yaml` is read by normal
credential resolution and must not be edited by tests.

The selected e2e project must have permission to:

- List buckets.
- Create buckets named `npa-e2e-test-*`.
- Upload, copy, list, head, and delete objects in those test buckets.
- Delete those test buckets after emptying them.

## Infrastructure Costs

The S3 e2e harness creates real Nebius S3 buckets. Each test bucket:

- Is named `npa-e2e-test-<purpose>-<timestamp>`.
- Is automatically emptied and deleted when the fixture exits.
- Is considered an orphan after 60 minutes.

The harness enforces:

- Maximum 3 concurrent `npa-e2e-test-*` buckets.
- Maximum 8 bucket creations per run.
- Per-bucket teardown with 3 attempts and backoff of 5, 15, and 45 seconds.

The per-run bucket creation counter lives at:

```text
/tmp/npa-e2e-run-bucket-counter.txt
```

Initialize it before a manual e2e run when you want budget enforcement from a
known clean counter:

```bash
echo "0" > /tmp/npa-e2e-run-bucket-counter.txt
```

## Cleanup

The fixture teardown is the primary cleanup path. If a process is killed and
leaves a bucket behind, list test buckets with:

```bash
python3 -c "
from npa.clients.project_credentials import s3_client_for_project

client = s3_client_for_project('eu-north1')
orphans = [
    b['Name']
    for b in client.list_buckets().get('Buckets', [])
    if b['Name'].startswith('npa-e2e-test-')
]
for name in orphans:
    print(name)
"
```

Delete one orphan bucket with:

```bash
python3 -c "
from npa.clients.project_credentials import s3_client_for_project

client = s3_client_for_project('eu-north1')
bucket = 'BUCKET_NAME'

paginator = client.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=bucket):
    objects = [{'Key': obj['Key']} for obj in page.get('Contents', [])]
    if objects:
        client.delete_objects(Bucket=bucket, Delete={'Objects': objects})

client.delete_bucket(Bucket=bucket)
print(f'Deleted {bucket}')
"
```

Change `eu-north1` to the project used by your e2e run, or set
`NPA_E2E_PROJECT` and use that same alias consistently.

## Adding New E2E Tests

1. Add tests under `npa/tests/e2e/test_<feature>_e2e.py`.
2. Mark general real-infrastructure tests with `@pytest.mark.e2e`.
3. Use shared fixtures from `npa/tests/e2e/conftest.py`:
   - `e2e_test_bucket` creates one bucket for one test.
   - `e2e_module_test_bucket` creates one bucket shared by a module.
   - `s3_helper` provides `head_object`, `get_sha256_metadata`,
     `list_objects`, `list_object_summaries`, and `count_objects`.
4. Keep every real-resource fixture responsible for its own teardown with
   `try/finally` or pytest `yield` fixtures.
5. Extend the shared harness only when the second use case needs it.

## Diagnosis

Run with verbose output and short tracebacks:

```bash
NPA_INTEGRATION_E2E=1 pytest npa/tests/e2e/ -v --tb=short
```

For full output capture:

```bash
NPA_INTEGRATION_E2E=1 pytest npa/tests/e2e/ -v 2>&1 | tee /tmp/e2e-run-output.log
```

Subprocess timeouts in the demo-stage e2e tests default to 300 seconds, with
`demo stage` allowed 600 seconds because it reads and stages the whole manifest.
