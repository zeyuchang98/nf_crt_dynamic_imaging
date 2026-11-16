#!/bin/bash
# One streamed motion_static training run over several Vista GH200 nodes, one process per node.
#
# Submit from the repository root on Vista, choosing queue, node count and wall time:
#   sbatch -p gh -N 4 -t 02:00:00 motion_static/experiments/vista/train_multinode.sbatch CONFIG NAME [run.py options]
# CONFIG must read the source phase files (data.source_root); see experiments/README.md,
# "Streamed source data and several processes". Process 0 writes the run to
# motion_static/results/NAME. This job's log (with every process's peak host memory from
# /usr/bin/time -v) and the first node's per-minute GPU record are moved there.
#SBATCH -J nf_train_mn
#SBATCH --ntasks-per-node=1
#SBATCH -A ASC25089
#SBATCH -o motion_static/results/slurm-%j.log

# The venv's python needs this module's libpython; load gcc first, in a separate call.
module load gcc/15.1.0
module load python3/3.11.8
set -eo pipefail

config=$1
name=$2
shift 2
out=motion_static/results/$name
gpu_log=motion_static/results/gpu-$SLURM_JOB_ID.csv

# torch.distributed meets at process 0's node (motion_static/distributed.py).
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export PYTHONDONTWRITEBYTECODE=1

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 60 > "$gpu_log" &
monitor=$!
status=0
# RANK and WORLD_SIZE come from Slurm; a failing process stops the others.
srun --kill-on-bad-exit=1 bash -c \
    'RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID exec /usr/bin/time -v \
     "$WORK/envs/nf-crt/bin/python" -m motion_static.run --distributed "$@"' \
    run --config "$config" --out "$out" "$@" || status=$?
kill "$monitor" || true

if [ -d "$out" ]; then
    mv "$gpu_log" "$out/"
    cp "motion_static/results/slurm-$SLURM_JOB_ID.log" "$out/"
fi
exit "$status"
