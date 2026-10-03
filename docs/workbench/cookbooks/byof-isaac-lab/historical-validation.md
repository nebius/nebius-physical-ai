# Historical Isaac Lab BYOF validation

[Current cookbook](README.md)

These W10 records used the generation 2 base in May 2026. They establish image
and entrypoint overrides for that tested runtime. The current example uses
Isaac Lab 3 beta / Isaac Sim 6 and `--visualizer none`; these older results do
not validate that replacement runtime. `--headless` below is historical.

## Published test image

W10 pushed
`ghcr.io/nebius/nebius-physical-ai/isaac-lab-byof-test:w10-byof-image-20260520T223706Z`.
The pushed manifest-list digest was
`sha256:c3e104601e31afaa833e3e73558ec9f0c6478f1dce59261fa45073a4d03518bf`;
the linux/amd64 platform digest was
`sha256:d9abdad36137a2f7cb38a6dbd85313ab0c5594582c248e174c9c4a13883d399c`.
Both differ from the vanilla base digest above.

## Image-only run

W10 validated this surface with:

```text
Run ID: w10-byof-image-only-20260520T232650Z
GPU: L40S
Output: s3://${NPA_S3_BUCKET}/checkpoints/isaac-lab-byof/w10-byof-image-only-20260520T232650Z/
Manifest train_script: /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py
```

## Image and command override run

W10 validated this surface with:

```text
Run ID: w10-byof-image-and-cmd-20260520T233113Z
GPU: L40S
Output: s3://${NPA_S3_BUCKET}/checkpoints/isaac-lab-byof/w10-byof-image-and-cmd-20260520T233113Z/
Manifest train_script: /opt/byof/custom_train.py
Sentinel: byof_sentinel.json
```

The validation sentinel included:

```json
{
  "byof": true,
  "script": "custom_train.py",
  "run_id": "w10-byof-image-and-cmd-20260520T233113Z",
  "task": "Isaac-Cartpole-v0",
  "num_envs": "64",
  "max_iterations": "1",
  "headless": true,
  "hydra_args": ["agent.save_interval=1"]
}
```

This proves that a non-default image and a non-default entrypoint can run while
still producing a normal Isaac Lab checkpoint.
