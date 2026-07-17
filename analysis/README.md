# Analysis Index

This folder is the project-level index for post-hoc analyses, diagnostic
summaries, and interpretation-ready result maps.

It follows the Rasi Lab style separation:

- experiment code and raw/generated outputs stay with the experiment folder,
  such as `exp_adaptive_h/`;
- day-to-day run notes stay in `experiment_logs/`;
- post-hoc analysis summaries and artifact maps are indexed here;
- polished paper-facing reports stay in `manuscript/`.

Do not move existing `output_*` folders into this directory. Most scripts assume
their current relative paths. Instead, add analysis READMEs that point to the
source outputs, generated tables, figures, and manuscript destinations.
