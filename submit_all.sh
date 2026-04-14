#!/bin/bash
# submit_all.sh
#
# Submit all ARC experiments from the project root.
# Run on the login node (not as a SLURM job):
#   bash submit_all.sh
#
# Comment out experiments you don't want to run.

set -euo pipefail

echo "=== Submitting ARC jobs ==="

# --- Gibbs comparison: CKME-CP (50 macroreps, adaptive h) ---
sbatch exp_gibbs_compare/run_all_gibbs_arc.sh
echo "  [submitted] CKME-CP Gibbs (50 macroreps)"

# --- Gibbs comparison: RLCP (50 reps, single job) ---
sbatch exp_gibbs_compare/run_rlcp_arc.sh
echo "  [submitted] RLCP Gibbs (50 reps)"

# --- Add future experiments below ---
# sbatch exp_nongauss/run_nongauss_arc.sh
# sbatch exp_onesided/run_onesided_arc.sh

echo ""
echo "=== All jobs submitted. Check status with: squeue -u zjin20 ==="
