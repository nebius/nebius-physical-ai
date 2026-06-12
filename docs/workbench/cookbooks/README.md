# Cookbooks

Technical recipes for reproducing benchmark, demo, and customer evaluation
workflows with Nebius Physical AI Workbench.

> Looking for the SkyPilot YAML behind a cookbook? The
> [workflow catalog](../../../npa/workflows/workbench/skypilot/README.md) maps
> every workflow YAML to its command and guide.

## Available Cookbooks

- [BDD100K SkyPilot Pipeline](bdd100k-pipeline.md): provision the object store,
  Kubernetes cluster, GPU node groups, and in-cluster LanceDB/detection-training
  services, then run the BDD100K ingest, UDF backfill, CLIP embedding,
  materialized-view, training, and evaluation pipeline.
- [Sim-To-Real Pipeline](sim-to-real-pipeline.md): CLI, SDK, raw SkyPilot, BYO
  policy image, BYO S3 endpoint, eval, feedback, artifact, and teardown details
  behind the one-command H100 quickstart.
- [VLM-Eval Loop Runbook](vlm-eval-loop-runbook.md): serve a VLM with vLLM,
  score rollout directories with `vlm-eval`, and write a task-success report.
- [Token Factory + Nebius compute combos](tokenfactory-compute-combos.md): two
  workflows that pair real Nebius GPU compute with hosted Token Factory
  inference — a serverless GPU train run triaged by a text model, and a
  Kubernetes GPU rollout judged by a hosted VLM.
- [LeRobot GPU Benchmarks](lerobot-gpu-benchmarks.md): reproduce the May 2026
  LeRobot GPU benchmark research across L40S, H200, B300, and RTX PRO 6000.
- [LeRobot GPU Benchmarks Runbook](lerobot-gpu-benchmarks-runbook.md): exact
  terminal steps for serverless Jobs, seconds/step measurements, artifact
  checks, validation, and cleanup.
- [Isaac Lab BYOF](byof-isaac-lab/README.md): layer a custom Isaac Lab image
  over the digest-pinned Workbench base and run it through the SkyPilot image
  override surface.
- [SONIC Locomotion Fine-Tuning](sonic-locomotion-finetuning.md): retarget
  motion data, run SONIC fine-tuning, and evaluate with MJLab through SkyPilot
  YAML.
- [SONIC G1 Fine-Tune to MuJoCo MVP](sonic-mvp-g1-mujoco.md): first milestone
  G1 warm-start fine-tune from the released SONIC checkpoint and headless
  MuJoCo checkpoint evaluation.
- [SONIC Export and Eval Runbook](sonic-eval-runbook.md): export a trained
  SONIC checkpoint to ONNX, run reference locomotion eval, and swap in a
  config-driven external eval container.
