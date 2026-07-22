# HPC Entry Points

Run SLURM submission commands from the repository root. The experiment-level
scripts remain beside the code they execute; `submit_all.sh` is only a small
dispatcher.

```bash
bash hpc/submit_all.sh
```

Before submitting, review account, partition, email, and output settings in
each experiment's `.sh` file for the target cluster.
