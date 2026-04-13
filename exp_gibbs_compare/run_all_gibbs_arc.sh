#!/bin/bash
# run_all_gibbs_arc.sh
#
# SLURM array job: run CKME-CP Gibbs comparison on Tinkercliffs (ARC).
# Each task = one macrorep (both gibbs_s1 and gibbs_s2).
#
# Test run (1 task):
#   sbatch --array=0-0 exp_gibbs_compare/run_all_gibbs_arc.sh
#
# Full run (50 tasks):
#   sbatch exp_gibbs_compare/run_all_gibbs_arc.sh
#
# Must be submitted from project root:
#   cd ~/path/to/Two-stage-simple
#   sbatch exp_gibbs_compare/run_all_gibbs_arc.sh

#SBATCH -J gibbs_compare
#SBATCH -A xchen6_lab
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH --array=0-49
#SBATCH -o exp_gibbs_compare/slurm_logs/slurm-%x-%A_%a.out
#SBATCH -e exp_gibbs_compare/slurm_logs/slurm-%x-%A_%a.err

set -euo pipefail

# ---- environment ----
module purge
module load Miniforge3
source activate ckme_env   # change to your conda env name if different

# ---- paths ----
cd "$SLURM_SUBMIT_DIR"

H_MODE="adaptive"        # "fixed" or "adaptive"
C_SCALE="2.0"
OUTPUT_DIR="exp_gibbs_compare/output_adaptive_c${C_SCALE}"
BASE_SEED=42

echo "=== Task ${SLURM_ARRAY_TASK_ID} / job ${SLURM_ARRAY_JOB_ID} ==="
echo "    h_mode=${H_MODE}, c_scale=${C_SCALE}"
echo "    output=${OUTPUT_DIR}"
date

python exp_gibbs_compare/run_gibbs_compare.py \
    --macrorep_id "$SLURM_ARRAY_TASK_ID" \
    --base_seed   "$BASE_SEED" \
    --h_mode      "$H_MODE" \
    --c_scale     "$C_SCALE" \
    --output_dir  "$OUTPUT_DIR"

echo "=== Done (task ${SLURM_ARRAY_TASK_ID}) ==="
date
