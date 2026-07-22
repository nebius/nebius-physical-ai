# Physical Reasoning Challenge (Token Factory + Cosmos3-Super-Reasoner)

A zero-GPU challenge built on Nebius Token Factory's hosted
`nvidia/Cosmos3-Super-Reasoner`. Teams walk a robot (or just a phone camera) to
different scenes around campus, capture images, and use the reasoner to extract
**scene understanding** and a **plan of action** for a task at that scene.

No GPUs, no servers to manage: inference is hosted, so a team needs only a
`NEBIUS_TOKEN_FACTORY_KEY` and the `npa` CLI.

## Why this works

`nvidia/Cosmos3-Super-Reasoner` is a vision-language model post-trained for
physical-AI common sense and embodied reasoning. Given scene images plus a task,
it reasons over objects, spatial layout, motion, and physical interactions and
returns a concrete, ordered plan — exactly the "what do I see, what should I do"
loop this challenge needs.

## Setup (once per team)

```bash
npa configure                         # enter the Token Factory API key when prompted
npa workbench token-factory verify    # confirm auth + that the model is available
npa workbench token-factory models | grep -i cosmos
```

If `nvidia/Cosmos3-Super-Reasoner` is not listed for your key, ask an organizer
to enable it (or use whatever Cosmos reasoner id `models` reports via `--model`).

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

Push scenes to S3 and launch the CPU-only workflow; the key rides in as a secret:

```bash
sky jobs launch \
  --secret NEBIUS_TOKEN_FACTORY_KEY \
  --secret AWS_ACCESS_KEY_ID \
  --secret AWS_SECRET_ACCESS_KEY \
  npa/src/npa/workflows/skypilot/token-factory-cosmos-reason.yaml
```

Set `INPUT_URI`, `OUTPUT_URI`, and `TASK` in the YAML's `envs:` (or override at
launch).

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
