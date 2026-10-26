#!/bin/bash
# One motion_static training run on one Vista GH200 node.
#
# Submit from the repository root on Vista, choosing queue and wall time:
#   sbatch -p gh-dev -t 02:00:00 motion_static/experiments/vista/train.sbatch CONFIG NAME [run.py options]
# The run goes to motion_static/results/NAME. This job's log (with the peak host memory
# from /usr/bin/time -v) and a per-minute GPU memory/utilization record are moved there.
#SBATCH -J nf_train
#SBATCH -N 1
#SBATCH -n 1
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

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 60 > "$gpu_log" &
monitor=$!
status=0
PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v "$WORK/envs/nf-crt/bin/python" -m motion_static.run \
    --config "$config" --out "$out" "$@" || status=$?
kill "$monitor" || true

if [ -d "$out" ]; then
    mv "$gpu_log" "$out/"
    cp "motion_static/results/slurm-$SLURM_JOB_ID.log" "$out/"
fi
exit "$status"
