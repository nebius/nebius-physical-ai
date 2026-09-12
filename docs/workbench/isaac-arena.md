# Isaac Lab-Arena policy evaluation

NPA supports the official Isaac Lab-Arena 0.3.0 policy evaluator as a pinned,
artifact-producing workbench job. The integration invokes upstream
`policy_runner.py`; it does not substitute a synthetic environment or infer
success from imports.

Upstream describes 0.3.0 as alpha, with unstable and incomplete APIs, and says
not to use it in production. NPA therefore supports only the immutable
evaluation contract documented here. This is a production-quality packaging
and operations surface around an explicitly pre-release upstream—not a claim
that the broader upstream project is production-ready.

## What the image contains

The public `npa-isaac-arena` image adds the Apache-2.0 Arena source at commit
`ed0fd12be862078be316c73eb7cf423ba9b1c5cd` to the accepted payload-clean Isaac
Lab image. Its source archive is SHA-256 verified. Upstream tests, sample
checkpoints, demonstration data, and documentation media are removed.

The accepted release is
`npa-isaac-arena:0.3.0-isaaclab3-20260912`, manifest
`sha256:f07a7fd0f44e22ba3366437b0d0973869a0590919951d516150094220939416f`.
It was promoted without rebuilding from source revision
`22783a16abcd424df540b71e94600d705b317f9b` after complete payload/security,
SBOM, provenance, bootstrap, anonymous-pull, and two-platform workload gates.

The image contains no Isaac Sim/Lab/Omniverse Kit payload, weights, replay data,
operator checkpoints, credentials, results, or populated runtime cache. On
first use, `/isaac-sim/python.sh` fetches the pinned Isaac runtime from NVIDIA
under the operator's `ACCEPT_EULA` setting. The compatible NPA baseline is Isaac
Lab `3.0.0b2.post1` with Isaac Sim `6.0.1.0`.

## Supported policies

```bash
npa workbench isaac-arena evaluate \
  --output-path ./arena-results \
  --environment cube_goal_pose \
  --policy-type zero_action \
  --num-episodes 1
```

The CLI and Python SDK support the three policy adapters shipped by upstream:

- `zero_action`: no input; useful as a factual simulator and metric baseline.
- `replay`: `--input-path` resolves to an Isaac Lab episode HDF5 file.
- `rsl_rl`: `--input-path` resolves to `model*.pt` or a directory containing
  exactly one such checkpoint with sibling `params/agent.yaml`.

Inputs and outputs may be local or operator-owned S3 paths. The simulator
subprocess receives no cloud, model, or HTTP admission credentials. Each result
contains raw episode JSONL, upstream static HTML, the credential-isolated simulator
log, and `result.json` with aggregate metrics, GPU identity, byte sizes, and hashes.
`--record-video` additionally fails unless upstream writes an independently probed,
decodable MP4 with valid dimensions and positive duration.

The Python SDK is the same implementation:

```python
from npa.sdk.workbench.isaac_arena import evaluate

result = evaluate(
    output_path="./arena-results",
    environment="cube_goal_pose",
    policy_type="zero_action",
    num_episodes=1,
)
```

## Supported workflows

- `workflows/testing/isaac-arena-evaluation-b200.yaml` runs a sequential
  four-seed state-only evaluation suite on one B200. Each state completes and
  scores an independent episode. B200 has no RT cores, so this path makes no
  render claim.
- `workflows/testing/isaac-arena-evaluation-rtxpro.yaml` runs an independent
  RTX PRO 6000 evaluation and requires a viewport MP4.

Validate and plan before submission, substitute an operator-owned bucket, and
pass the standard S3 credentials through workflow secret handling. No model
credential is required for the zero-action baseline.

```bash
npa workbench health preflight --checks nebius,s3
npa workbench workflow validate-spec workflows/testing/isaac-arena-evaluation-b200.yaml
npa workbench workflow plan-spec workflows/testing/isaac-arena-evaluation-b200.yaml
```

On a cluster whose SkyPilot accelerator spelling is already verified, pin it
explicitly to keep submission identity stable. Use the corresponding RTX name
and YAML for the video workflow.

```bash
export NPA_WORKFLOW_GPU_ACCELERATOR='B200:1'
npa workbench workflow submit \
  workflows/testing/isaac-arena-evaluation-b200.yaml \
  --runtime --durable-s3 --max-wait-seconds 0 \
  --project <project-alias> --infra k8s/<context> \
  --var bucket=<operator-owned-bucket>
```

## Accepted workload evidence

The exact release digest completed independent one-episode evaluations on both
required GPU families. The B200 run measured capability `(10, 0)`, completed
1,050 scored steps, and retained five hash-verified task artifacts (86,082
bytes): one episode journal, three linked HTML pages, and the simulator log. It
made no video claim.

The RTX PRO 6000 run measured capability `(12, 0)`, completed the same 1,050
scored steps, and retained six hash-verified task artifacts (1,119,004 bytes).
Its required viewport artifact independently decoded as H.264, 1280×720, and
70.067 seconds. Both zero-action runs reported success rate 0.0; that is an
expected policy baseline result, while the evaluation capability and artifact
integrity gates passed.

The final supported B200 workflow then completed its exact four-state seed sweep.
All four episodes ran 1,050 scored steps at capability `(10, 0)` and retained
five independently hash-verified non-video artifacts each: 20 task artifacts /
344,330 bytes in total. Every live job was terminal or repeat-safely cancelled,
and an independent pod audit found no run-owned worker. Run-created controllers
were removed; the pre-existing shared controller, clusters, and operator storage
were retained.

For exact build, scan, qualification, and cleanup rules, use
`skills/tools/isaac-arena/SKILL.md`.
