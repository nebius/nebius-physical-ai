# Cookbooks

Technical recipes for reproducing benchmark, demo, and customer evaluation
workflows with Nebius Physical AI Workbench.

## Available Cookbooks

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
