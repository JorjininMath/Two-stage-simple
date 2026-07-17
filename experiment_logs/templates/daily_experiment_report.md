# Daily Experiment Report: <short title>

Date: <YYYY-MM-DD>

Status: <draft | checked | rerun-needed | ready-to-use>

Experiment family: `<exp_folder_or_module>`

Experiment ID / issue: `<optional issue number or run label>`

Owner / reviewer: `<name or N/A>`

Related outputs:

- `<relative/path/to/output_dir_or_file>`

## 1. Purpose

State the reason for this run in one or two sentences.

- Main question:
- Expected pattern:
- What would count as suspicious:

## 2. Run Setup

Commands:

```bash
<command used to run the experiment or diagnostic>
```

Key settings:

| setting | value |
| --- | --- |
| random seeds / macroreps |  |
| simulator / DGP |  |
| methods / arms |  |
| sample sizes |  |
| alpha / coverage target |  |
| important tuning parameters |  |

Input files:

- `<relative/path>`

Output files:

- `<relative/path>`

Artifact inventory:

| artifact type | path | tracked? | note |
| --- | --- | --- | --- |
| raw output |  | no |  |
| summary table |  | no |  |
| figure |  | no |  |
| formal report link |  | yes/no |  |

## 3. Main Results

Use compact tables. Prefer mean, standard error, and paired comparisons when
available.

| metric | fixed / baseline | adaptive / oracle | plug-in / other | note |
| --- | ---: | ---: | ---: | --- |
| marginal coverage |  |  |  |  |
| worst-bin deviation |  |  |  |  |
| mean interval width |  |  |  |  |
| interval score |  |  |  |  |

## 4. Diagnostic Checks

Coverage and calibration:

- Marginal coverage:
- Group/bin coverage:
- Calibration or quantile issues:

Bandwidth / scale:

- Effective ratio `h(x)/s(x)`:
- Plug-in scale error:
- Scale floor or clamping:

Grid and numerical behavior:

- Response-grid boundary hits:
- Nonmonotone CDF behavior:
- Missing or unusable outputs:

## 5. Figures To Inspect

| figure | what to look for | conclusion |
| --- | --- | --- |
| `<relative/path/to/figure.png>` |  |  |

## 6. Interpretation

Short conclusion in paper-safe language.

- What the results support:
- What they do not prove:
- Caveats:

Recommended paper/report wording:

> `<one or two sentences that are safe to reuse later>`

## 7. Next Actions

- [ ] Check:
- [ ] Rerun:
- [ ] Add to manuscript/report:
- [ ] Archive or ignore:

## 8. Notes For Future Runs

Record small lessons that will save time later, such as parameter choices,
runtime, plotting issues, or diagnostics that should be added to the runner.
