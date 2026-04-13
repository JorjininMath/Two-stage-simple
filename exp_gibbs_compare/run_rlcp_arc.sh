#!/bin/bash
# run_rlcp_arc.sh
#
# Single SLURM job: run RLCP on Gibbs DGPs (Setting 1 & 2), 50 reps.
#
# Usage (from project root):
#   sbatch exp_gibbs_compare/run_rlcp_arc.sh

#SBATCH -J rlcp_gibbs
#SBATCH -A xchen6_lab
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -o exp_gibbs_compare/slurm_logs/slurm-%x-%j.out
#SBATCH -e exp_gibbs_compare/slurm_logs/slurm-%x-%j.err

set -euo pipefail

module load R/4.4.2-gfbf-2024a

cd "$SLURM_SUBMIT_DIR"

echo "=== RLCP Gibbs job ${SLURM_JOB_ID} ==="
date

Rscript exp_gibbs_compare/run_rlcp_gibbs.R --nrep 50 --h 0.05

echo "=== Done ==="
date
