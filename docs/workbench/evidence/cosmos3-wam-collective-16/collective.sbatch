#!/usr/bin/env bash
#SBATCH --job-name=wam-collective
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
set -euo pipefail
umask 077
: "${WAM_SHARED_ROOT:?Set the prepared shared root}"
: "${WAM_COLLECTIVE_SCRIPT:?Set the staged diagnostic script path}"
: "${WAM_COLLECTIVE_OUTPUT:?Set a fresh diagnostic result path}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export LD_LIBRARY_PATH=
export OMP_NUM_THREADS=4
export PATH="$WAM_SHARED_ROOT/framework/.venv/bin:$PATH"
unset SLURM_TRES_PER_TASK
exec srun --ntasks=2 --ntasks-per-node=1 --cpus-per-task=128 \
  --gpus-per-task=8 --gpu-bind=none --kill-on-bad-exit=1 bash -c \
  'exec "$WAM_SHARED_ROOT/framework/.venv/bin/python" -m torch.distributed.run \
    --nproc_per_node=8 --nnodes=2 --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --max_restarts=0 \
    "$WAM_COLLECTIVE_SCRIPT" --output-path "$WAM_COLLECTIVE_OUTPUT"'
