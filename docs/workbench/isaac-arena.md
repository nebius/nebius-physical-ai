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
`--record-video` additionally fails unless upstream writes a non-empty MP4.

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

- `workflows/testing/isaac-arena-evaluation-b200.yaml` runs state-only
  completed-episode evaluation on one B200. B200 has no RT cores, so this path
  makes no render claim.
- `workflows/testing/isaac-arena-evaluation-rtxpro.yaml` runs an independent
  RTX PRO 6000 evaluation and requires a viewport MP4.

Validate and plan before submission, substitute an operator-owned bucket, and
pass the standard S3 credentials through workflow secret handling. No model
credential is required for the zero-action baseline.

```bash
npa workbench health preflight --checks nebius,storage
npa workbench workflow validate-spec workflows/testing/isaac-arena-evaluation-b200.yaml
npa workbench workflow plan-spec workflows/testing/isaac-arena-evaluation-b200.yaml
```

For exact build, scan, qualification, and cleanup rules, use
`skills/tools/isaac-arena/SKILL.md`.
