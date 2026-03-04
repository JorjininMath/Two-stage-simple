#!/bin/bash
#SBATCH -J stage2
#SBATCH -A xchen6_lab
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH --array=0-49
#SBATCH -o exp_stage2_impact_arc/slurm_logs/slurm-%x-%A_%a.out
#SBATCH -e exp_stage2_impact_arc/slurm_logs/slurm-%x-%A_%a.err

# ARC (Tinkercliffs): one macrorep per array task.
# Upload project to /home/zjin20/CKME/Two-stage-simple, then:
#   cd /home/zjin20/CKME/Two-stage-simple && sbatch exp_stage2_impact_arc/run_all_stage2_arc.sh

PROJECT_ROOT=/home/zjin20/CKME/Two-stage-simple
CONFIG=${PROJECT_ROOT}/exp_stage2_impact_arc/config.txt
OUTPUT_DIR=${SCRATCH:-${PROJECT_ROOT}}/exp_stage2_impact_arc/output

# Use conda env if it exists; else use default python (e.g. system + ~/.local from pip install --user).
CONDA_ENV=${CONDA_ENV:-$HOME/ckme_stage2_env}

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

if [ -d "$CONDA_ENV" ]; then
  module load Miniconda3
  source activate "$CONDA_ENV"
fi
module load R

python exp_stage2_impact_arc/run_stage2_influence.py \
  --config "$CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --macrorep_id "$SLURM_ARRAY_TASK_ID"
