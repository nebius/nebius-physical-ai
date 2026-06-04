# Cookbooks

Technical recipes for reproducing benchmark, demo, and customer evaluation
workflows with Nebius Physical AI Workbench.

## Available Cookbooks

- [VLM-Eval Loop Runbook](vlm-eval-loop-runbook.md): serve a VLM with vLLM,
  score rollout directories with `vlm-eval`, and write a task-success report.
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
- [SONIC Export and Eval Runbook](sonic-eval-runbook.md): export a trained
  SONIC checkpoint to ONNX, run reference locomotion eval, and swap in a
  config-driven external eval container.
- [Cosmos 3 Augmentation and Reasoning](cosmos3-augmentation-reasoning.md):
  run controlled-generation augmentation and structured VLM reasoning through
  the CLI, SDK, or standalone raw SkyPilot YAML.
