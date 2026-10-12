---
name: cosmos-data-factory
description: Work on the separate Early Access IAA and EVG NPA workflows in workflows/cosmos-data-factory, choosing task-appropriate seeds and reviewing their generated media and labels.
---

# Cosmos Data Factory IAA and EVG

Keep CDF-specific work in this directory. Leave
`workflows/main/paidf-cosmos3.yaml` in place. This package adapts NVIDIA's
IAA/EVG reference implementation to the existing NPA CLI and SkyPilot runtime;
the [README](README.md) records the pinned source and orchestration mapping.

For robotics requests, use the README's [task-fit and reuse map](README.md#robotics-fit-and-reuse).
Reuse the existing main PAIDF/VDA path for robot-video augmentation and existing
LeRobot/Sim2Real tools for training/evaluation. IAA supports human-appearance
perception; EVG supports generated-event perception. Do not duplicate those
existing graphs or treat a physical-AI label as proof of robot-policy usefulness.

Load [shared setup](../../skills/cosmos-data-factory/cosmos-data-factory-setup/SKILL.md),
then the matching workflow skill:

- **IAA:** [skill](../../skills/cosmos-data-factory/physical-ai-image-attribute-augmentation/SKILL.md)
  and [guide](image-attribute-augmentation.md). Use person images to evaluate
  clothing, color, and footwear edits with verification/search labels.
- **EVG:** [skill](../../skills/cosmos-data-factory/physical-ai-event-video-generation/SKILL.md)
  and [guide](event-video-generation.md). Use scene images to evaluate generated
  event videos with detection, tracking, captioning, QA, and search annotations.

Use the two checked-in specs through generic `npa workbench workflow` commands.
Retain their nine-state IAA and twelve-state EVG graphs and real component
adapters. The existing canonical skills stay registered in `skills/index.yaml`;
this local entry point links them without creating another implementation.
Check the guides' current sampling coverage before promising an attribute or
event. IAA exposes six clothing/footwear fields; EVG's fixed event list omits
fighting and shoplifting. Upstream distribution JSON is not a direct NPA override.

Choose inputs using the guides before treating a run as a representative demo.
For requested visual proof, follow [proof.md](proof.md), including actual media,
source hashes, sampled attributes/events, annotation coverage, and quality
verdicts. Keep historical integration fixtures identified as such. Robot dataset
frames used for EVG test image-to-event-video synthesis; they do not establish
robot action or policy generalization. Video-to-video edge conditioning remains
part of the separate main PAIDF workflow.
