# Stage 2 impact experiment on ARC (Tinkercliffs)

One macrorep per node via SLURM array job.

## 0. One-time: Python and R environment

The job needs **Python** (numpy, pandas, openpyxl, scipy, scikit-learn) and **R** (for DCP-DR and hetGP). Do this once on ARC:

1. Get an interactive compute node (so the env matches job nodes):
   ```bash
   interact -A xchen6_lab -p normal_q -c 4 --mem=8G -t 1:00:00
   ```
2. Load Miniconda, create a env, install Python deps:
   ```bash
   cd /home/zjin20/CKME/Two-stage-simple
   module load Miniconda3
   conda create -p $HOME/ckme_stage2_env python=3.10 -y
   source activate $HOME/ckme_stage2_env
   pip install -r exp_stage2_impact_arc/requirements_arc.txt
   ```
   **Alternative (no conda):** on the same interact session run  
   `pip install --user -r exp_stage2_impact_arc/requirements_arc.txt`  
   and do not create the env. The job script will then use the default Python (and `~/.local` packages) when `$HOME/ckme_stage2_env` does not exist.
3. Load R (to confirm it’s available for jobs):
   ```bash
   module load R
   which Rscript
   exit
   ```

The script uses `CONDA_ENV=$HOME/ckme_stage2_env` by default. If you use another path, set it before sbatch: `export CONDA_ENV=$HOME/your_env_path`.

## 1. Create folder and upload to ARC

**On ARC** (ssh into tinkercliffs), create the target folder:

```bash
mkdir -p /home/zjin20/CKME/Two-stage-simple
```

**On your Mac** (in Terminal), upload the whole project into that folder:

```bash
cd /Users/zhaojin/Dropbox/Jin/CKME
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' \
  Two-stage-simple/ \
  zjin20@tinkercliffs1.arc.vt.edu:/home/zjin20/CKME/Two-stage-simple/
```

(If you use a different login node than `tinkercliffs1.arc.vt.edu`, replace it. Same for `zjin20` if your ARC username differs.)

After upload, on ARC you should have:

```
/home/zjin20/CKME/Two-stage-simple/
├── CKME/                   # for Params (used by config_utils)
├── Two_stage/
├── dcp_r.R
└── exp_stage2_impact_arc/
    ├── config.txt
    ├── config_utils.py     # config load + X_cand (no Experiment/ needed)
    ├── run_stage2_influence.py
    ├── run_benchmarks_one_case.R
    └── run_all_stage2_arc.sh
```

Minimum required on ARC:

| Path on ARC | Purpose |
|-------------|---------|
| `CKME/` (whole dir) | `Params` used by config_utils |
| `Two_stage/` (whole dir) | Stage1/Stage2, evaluation, test_data, design, sim_functions |
| `dcp_r.R` (at project root) | Sourced by R script |
| `exp_stage2_impact_arc/config.txt` | Config |
| `exp_stage2_impact_arc/config_utils.py` | Config load + candidate design (replaces Experiment/) |
| `exp_stage2_impact_arc/run_stage2_influence.py` | Main Python |
| `exp_stage2_impact_arc/run_benchmarks_one_case.R` | DCP-DR + hetGP |
| `exp_stage2_impact_arc/run_all_stage2_arc.sh` | SLURM submit script |

You do **not** need `Experiment/` or `experiment_stage2_influence/` on ARC.

## 2. Run on ARC

1. SSH to ARC and go to the project:
   ```bash
   cd /home/zjin20/CKME/Two-stage-simple
   ```
2. Create the SLURM log folder (so .out/.err go there, not the project root):
   ```bash
   mkdir -p exp_stage2_impact_arc/slurm_logs
   ```
3. (Optional) Edit `exp_stage2_impact_arc/run_all_stage2_arc.sh` if needed: partition, time.
4. Submit the array job:
   ```bash
   sbatch exp_stage2_impact_arc/run_all_stage2_arc.sh
   ```

**Where things go:**
- **Results:** `macrorep_0.xlsx` … `macrorep_49.xlsx` and `log_macrorep_*.txt` in `$OUTPUT_DIR` (`$SCRATCH/exp_stage2_impact_arc/output` or project `exp_stage2_impact_arc/output` if no SCRATCH).
- **SLURM .out / .err:** `exp_stage2_impact_arc/slurm_logs/stage2_<jobid>_<0-49>.out` and `.err` (under `Two-stage-simple/`).
