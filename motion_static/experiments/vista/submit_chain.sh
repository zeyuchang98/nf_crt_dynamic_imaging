#!/bin/bash
# Submit one streamed run as a chain of short gh jobs that continue one another.
#
# From the repository root on Vista:
#   motion_static/experiments/vista/submit_chain.sh JOBS NODES TIME MINUTES CONFIG NAME [run.py options]
# for example
#   motion_static/experiments/vista/submit_chain.sh 3 32 02:00:00 100 motion_static/experiments/dce_moby_box64_full_series.json NAME
# Each job runs train_multinode.sbatch with `run.py --continue --segment-minutes MINUTES`:
# the first starts the run in motion_static/results/NAME, each later one (afterok on the one
# before) continues it, and a job that finds the run complete exits at once. MINUTES must
# leave time within TIME for the checkpoint and, in the last job, for the scoring.
# Do not pull on Vista until every job of the chain has finished: each job checks the code.
set -eo pipefail

if [ $# -lt 6 ]; then
    echo "usage: $0 JOBS NODES TIME MINUTES CONFIG NAME [run.py options]" >&2
    exit 2
fi
jobs=$1
nodes=$2
limit=$3
minutes=$4
shift 4

previous=""
for k in $(seq 1 "$jobs"); do
    dependency=()
    if [ -n "$previous" ]; then
        dependency=(--dependency="afterok:$previous")
    fi
    previous=$(sbatch --parsable -p gh -N "$nodes" -t "$limit" "${dependency[@]}" \
        motion_static/experiments/vista/train_multinode.sbatch "$@" --continue --segment-minutes "$minutes")
    echo "segment $k of $jobs: job $previous"
done
