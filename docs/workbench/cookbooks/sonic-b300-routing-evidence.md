# SONIC B300 routing evidence workflow

`sonic-b300-routing-evidence.yaml` is a released, CPU-only workflow that executes
the installed SONIC accelerator resolver. It fails closed unless both `b300` and
`gpu-b300-sxm` resolve to `B300:1`, while retaining L40S, H100, and B200
comparison routes.

```bash
npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/sonic-b300-routing-evidence.yaml --json
npa workbench workflow plan-spec \
  npa/workflows/workbench/npa-workflows/sonic-b300-routing-evidence.yaml \
  --run-id sonic-b300-routing-preview --json
npa workbench workflow submit \
  npa/workflows/workbench/npa-workflows/sonic-b300-routing-evidence.yaml \
  --run-id <new-run-id> --var bucket=<configured-bucket> \
  --var tested_commit_sha=<git-sha>
```

The run-scoped prefix contains `manifest.json`, `test-report.json`, and
`reports/sonic-b300-routing.rrd`. The RRD includes an explicit blueprint, a
labelled target-to-accelerator view, a Markdown evidence table, and seven samples
over a six-second assertion timeline. Values are `-1` (failed), `0`
(unverified), and `1` (passed), so failures and unverified live stages remain
visible.

Routing resolution is not provider recognition or GPU execution. By default the
provider-recognition, scheduling/placement, workload-completion, checkpoint
verification, and cleanup assertions remain `unverified`. Operators may
populate those fields only from a separate real SONIC `train` or `finetune`
run, using sanitized accelerator/product text, immutable job/image/checkpoint
digests, nonzero checkpoint bytes, and a semantic-load result. Inconsistent
passed claims fail closed. The evidence workflow itself is not SONIC training
or policy execution; it visualizes sanitized evidence from that separate run.
