# Physical Reasoning Challenge (Token Factory + MiniMax-M3)

[Cookbooks](README.md)

> This cookbook uses the current hosted vision/reasoning default,
> `MiniMaxAI/MiniMax-M3`, through `https://api.tokenfactory.nebius.com/v1/`.
> The public `nvidia/Cosmos3-Super-Reasoner` route was retired under the
> [August 2026 notice](https://docs.tokenfactory.nebius.com/august-2026-deprecation-notice).
> A dedicated or self-hosted Cosmos3 Reasoner is a separate backend; do not
> substitute its name for a MiniMax request. Confirm model availability with
> `npa workbench token-factory models` before the challenge.

A zero-GPU challenge built on Nebius Token Factory's hosted
`MiniMaxAI/MiniMax-M3`. Teams walk a robot (or just a phone camera) to
different scenes around campus, capture images, and use the reasoner to extract
**scene understanding** and a **plan of action** for a task at that scene.

No GPUs, no servers to manage: inference is hosted, so a team needs only a
`NEBIUS_TOKEN_FACTORY_KEY` and the `npa` CLI.

## Why this works

`MiniMaxAI/MiniMax-M3` is the current hosted vision/reasoning backend for this
challenge. Given scene images plus a task, it returns a concrete, ordered plan
based on visible objects and spatial layout — the "what do I see, what should I
do" loop this challenge needs. It does not act, certify physical validity, or
establish access to a separate Cosmos3 Reasoner backend.

## Setup (once per team)

```bash
npa configure                         # enter the Token Factory API key when prompted
npa workbench token-factory verify    # confirm auth against the current endpoint
npa workbench token-factory models | grep -F 'MiniMaxAI/MiniMax-M3'
```

If `MiniMaxAI/MiniMax-M3` is not listed for your key, stop before inference and
select a key-accessible supported vision model only after validating its response
contract. Do not try to restore the retired public Cosmos3 model by changing the
endpoint or model name.

## Run the loop (local)

1. Walk to a scene and capture one or more images into a folder, e.g. `./scene/`.
2. Ask the reasoner what to do:

```bash
npa workbench token-factory reason \
  --input-path ./scene \
  --output-path ./out \
  --task "What is in this scene and what steps should the robot take to <YOUR TASK>?" \
  --output json
```

3. Read `./out/scene_reasoning.json` — it contains the model, the task, the
   image list, and the `analysis` (scene understanding + plan of action).

## Run at scale (SkyPilot, S3 in/out)

Push scenes to S3 and submit the CPU-only workflow; the key rides in as a secret:

```bash
npa workbench workflow submit \
  workflows/testing/token-factory-cosmos-reason.yaml \
  --run-id physical-reasoning-<new-run-id> \
  --var bucket=<your-bucket> \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

The historical filename is retained for compatibility; the toolRef resolves the
current Token Factory reasoner default. This reference derives its scene and
plan paths from `bucket` and `prefix`, and intentionally retains the reason
CLI's task/frame defaults. For a custom task, model, or image count, use the
direct command above or a user-owned spec that explicitly exposes those fields.

## Suggested scoring rubric

- **Scene understanding** — did it correctly identify objects, layout, hazards?
- **Plan quality** — are the steps ordered, executable, and safe?
- **Grounding** — does the plan reference what is actually visible?
- **Robustness** — try the same task across 3+ scenes (indoor, outdoor, cluttered).

Teams can compare models with `--model`, tune the framing with `--task` /
`--system-prompt`, and send multiple angles of a scene with `--max-images`.

## Is one reasoner enough for a challenge?

Yes for a focused, fun challenge: a single hosted reasoner removes all infra
friction and still exercises real physical common sense. Caveats to set
expectations with teams:

- **Model availability and rate limits** are key-dependent — verify early and
  share one organizer key budget or per-team keys.
- **Stills vs. video**: this cookbook sends images. Multi-image (several angles)
  improves spatial grounding; capture 2–4 frames per scene.
- **It plans, it does not act** — pair it with a teleoperated or scripted robot;
  the reasoner is the "brain", not the controller.
