# Daily Experiment Logs

This folder is for day-to-day experiment records, quick diagnostics, and
run-level interpretation notes. It is separate from `manuscript/reports/`,
which is reserved for polished paper-facing reports.

Daily logs should be useful for answering:

- What did we run?
- Why did we run it?
- Which outputs were produced?
- What looked normal?
- What looked suspicious?
- What should be checked or rerun next?

## Local-Only Policy

This folder is intentionally local-first. Generated logs, figures, CSV copies,
and scratch notes stay untracked by default. Only this README and the templates
under `templates/` are intended to be tracked.

Do not put secrets, credentials, private machine paths, or personal notes that
should not be public into tracked files.

## Suggested Layout

Use one subfolder per experiment family:

```text
experiment_logs/
  exp_adaptive_h/
    2026-06-24_existing_dgp_diagnostics.md
  ckme_dcp_mm1/
    2026-06-15_kme_feasibility.md
  templates/
    daily_experiment_report.md
```

## Naming Convention

Use date-first names so logs sort naturally:

```text
YYYY-MM-DD_short_experiment_name.md
```

For multiple runs on the same day, add a suffix:

```text
2026-06-24_existing_dgp_diagnostics_v2.md
```

## Template

Start from:

```text
experiment_logs/templates/daily_experiment_report.md
```

Copy the template into the relevant experiment subfolder, then fill in the
sections that matter for that run. It is fine to leave a section as `N/A` if it
does not apply.

## External Template References

This folder follows two public academic/research-template ideas:

- Rasi Lab's GitHub laboratory research template keeps experiments, analyses,
  presentations, grants, and manuscripts in separate folders and recommends
  issue-linked experiment/lab-notebook records.
- DrivenData's Cookiecutter Data Science template is a high-use data-science
  project template with a clean split between code, data, reports, and figures.

For this project, daily experiment logs stay in `experiment_logs/`, formal
paper-facing reports stay in `manuscript/reports/`, and generated outputs stay
beside each experiment under `output_*` folders.
