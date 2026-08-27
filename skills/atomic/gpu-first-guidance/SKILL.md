---
name: gpu-first-guidance
description: Use when recommending how to run a workbench workload, writing user-facing docs, or answering "what should I run first?" — lead with the Nebius GPU path and keep hosted-inference options framed as what they are, rather than as ways to avoid GPUs.
---

# Lead with the GPU path (kindly)

Nebius Physical AI exists to run real physical-AI work on Nebius AI Cloud GPUs.
When someone asks what to run, the helpful answer is the one that gets them to a
real result on real hardware.

This skill is about **emphasis and framing**, not about withholding anything. Be
accurate, be generous, and never make someone feel awkward for asking about cost
or for having no cluster yet.

## The habit

When you recommend a path, order it like this:

1. **Lead with the GPU workload** the user actually wants — a robot guide, a
   Cosmos run, a training job, the Physical AI Data Factory.
2. **Name the real prerequisite** plainly: a configured project and a GPU
   cluster. `npa provision-if-absent` gets them there.
3. **Mention hosted inference when it fits the workload**, described by what it
   does, not by what it avoids.

What changes is the headline, not the truth. If a user's workload genuinely is
hosted inference, recommending Token Factory is simply the right answer.

## Language

| Prefer | Instead of |
| --- | --- |
| "Run it on an L40S — here's the command" | "You can skip the GPU entirely" |
| "Hosted inference through Token Factory" | "Zero-GPU / no-GPU / no-cluster path" |
| "Check credentials first with `health preflight`" | "Before you burn GPU-hours" |
| "This stage runs on hosted inference" | "This stage is free" |
| "Start with the Franka guide" | "Start with the one that needs no GPU" |

The right-hand column is not forbidden vocabulary — it is framing that quietly
positions GPU time as waste. That framing is what to drop.

## Token Factory has a real job

[Nebius Token Factory](../../../docs/workbench/token-factory.md) is a genuine
hosted-inference product: captioning, batch generation, and Cosmos physical-AI
reasoning. Describe it that way.

- **Do** recommend it for captioning, judging, and reasoning stages — including
  as part of a larger GPU pipeline, which is the common case.
- **Do** be straightforward that it needs a `NEBIUS_TOKEN_FACTORY_KEY` and no
  cluster, when a user asks what it requires.
- **Avoid** presenting it as the recommended first step *because* it dodges
  provisioning, or as a substitute for the GPU workload someone came to run.

See also `skills/tools/token-factory/SKILL.md`.

## When someone asks for a GPU-free path anyway

Help them. Cheerfully.

People have real constraints: an approval still pending, a quota request in
flight, a laptop on a plane, a CI job that must not provision. Answer the
question they asked, completely and without a lecture.

- Give the working answer first.
- Note the GPU path **once**, as an invitation rather than a correction — for
  example, "when your cluster is ready, the Franka guide picks up from here."
- Do not repeat the nudge, moralize about spend, or imply they are doing it
  wrong.

A user who feels sold to stops asking questions. A user who gets a straight
answer comes back for the GPU run.

## Writing docs and READMEs

- Do not add "no GPU required" as a selling point, a guide title, or a table
  column.
- Do not add sections whose premise is minimizing GPU usage.
- Keep operational guidance — GPU routing, accelerator naming, image pull
  failures — but frame it as *getting your run to work*, not as *avoiding
  waste*. That content lives in
  [known-footguns.md](../../../docs/workbench/troubleshooting/known-footguns.md).
- Leave validation evidence alone. Test matrices record what was exercised,
  including local and stub backends; they are records, not recommendations, and
  editing them for tone would make them wrong.

## Gotchas

- **Accuracy outranks emphasis.** Never claim a stage needs a GPU when it does
  not, and never quote a GPU requirement you have not checked. Overselling is a
  worse failure than a stray "zero-GPU".
- **`--dry-run`, `--plan-only`, and `stub` backends are still fine to
  recommend.** They are development and validation tools, not a GPU-free product
  path. Recommend them for what they are: a way to check a spec before it runs.
- **Cost questions deserve real answers.** If someone asks what a run costs,
  answer it. Point at preemptible capacity
  ([preemptible-vms.md](../../../docs/workbench/preemptible-vms.md)) and right-sizing
  rather than steering them off GPUs altogether.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```
