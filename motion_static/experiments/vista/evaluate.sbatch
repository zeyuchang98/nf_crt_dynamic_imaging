#!/bin/bash
# One motion_static evaluation script on one Vista node (CPU work, e.g. the gg queue).
#
# Submit from the repository root on Vista, choosing queue and wall time:
#   sbatch -p gg -t 06:00:00 motion_static/experiments/vista/evaluate.sbatch MODULE [options]
# MODULE is an evaluation module, e.g. motion_static.evaluation.moby_svd_baseline, and the
# options are its own. The job log, with the code commit, the thread count and the peak host
# memory from /usr/bin/time -v, is motion_static/results/slurm-JOBID.log; copy it next to
# the outputs when they are recorded.
#SBATCH -J nf_eval
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A ASC25089
#SBATCH -o motion_static/results/slurm-%j.log

# The venv's python needs this module's libpython; load gcc first, in a separate call.
module load gcc/15.1.0
module load python3/3.11.8
set -eo pipefail

# TACC sets OMP_NUM_THREADS=1 in jobs, which limits numpy's BLAS (the SVD) and torch to one
# core; give them every core of the allocated node instead.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

module_name=$1
shift
echo "code commit: $(git rev-parse HEAD); threads: $OMP_NUM_THREADS"
PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v "$WORK/envs/nf-crt/bin/python" -m "$module_name" "$@"
