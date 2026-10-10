#!/usr/bin/env bash
# Run the generated reference workload through a real operator-provisioned Slurm job.
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
set -euo pipefail
: "${NPA_POLICY_PYTHON:?Set NPA_POLICY_PYTHON to the worker Python with npa installed}"
exec "$NPA_POLICY_PYTHON" -m npa.workflows.policy_training.reference "$@"
