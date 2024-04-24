#!/usr/bin/env bash
#SBATCH --partition=cpu-prio
#SBATCH --mem=14GB
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=oc-compute03
#SBATCH --time=2-00:00:00
#SBATCH --output=output/%x-%A-%a.txt


echo "$SLURM_JOB_ID"
echo "$@"
srun nix develop . --impure --command julia --project=. scripts/2024-gecco/run.jl optparams --name-run="$SLURM_JOB_ID" "$@"
