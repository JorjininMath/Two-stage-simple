# C-MAPSS Experiment Spec for CKME+CP (Interview-Oriented)

## Goal

Build a **small but convincing experiment** that adapts the existing **CKME+CP / two-stage uncertainty quantification framework** to a **complex equipment degradation** setting using the **NASA C-MAPSS** dataset.

This experiment is **not** intended to be a full paper-level benchmark.
Its purpose is to generate:
1. a clean technical story for interview slides,
2. a few interpretable figures/tables,
3. evidence that the method can transfer from simulation-style settings to **industrial/robotics-adjacent dynamic systems**.

---

## High-Level Story

We want to present CKME+CP as:

> A two-stage framework for **trustworthy prediction under uncertainty** in complex dynamic systems.

In C-MAPSS, this becomes:

- **Stage 1**: learn a global predictive distribution / uncertainty model for engine degradation.
- **Stage 2**: focus on **high-risk / high-uncertainty regions**, especially **late-stage degradation**, to improve reliability where deployment decisions matter most.

This should support an interview narrative like:

> In industrial or embodied systems, point prediction alone is insufficient.  
> We care about **how reliable the prediction is**, especially in high-risk regimes.

---

## Dataset Choice

Use **C-MAPSS FD001 only**.

### Why FD001
- simplest subset,
- easiest to explain,
- enough for a strong demo,
- avoids too much complexity from multiple operating conditions / fault modes.

### Data Setting
Each engine has:
- unit id,
- cycle index,
- operating settings,
- multiple sensor measurements.

The original task is often framed as **RUL prediction**.

---

## Prediction Target

### Main target
Use **RUL (Remaining Useful Life)** as the target.

For each engine at cycle `t`:

`RUL_t = failure_cycle - t`

Optionally cap RUL at a threshold (common in the literature), for example:
- `RUL_cap = 125`

This is recommended because:
- it stabilizes training,
- makes plots cleaner,
- is standard enough,
- still preserves the degradation story.

### Why RUL
This is the most natural target for:
- uncertainty quantification,
- interval prediction,
- high-risk decision support,
- maintenance/deployment interpretation.

---

## Input Representation

Do **not** start with a complex sequence model.

Instead, build a **window-based tabular representation** so the existing CKME+CP pipeline can be reused more easily.

### Recommended approach
For each engine and each valid cycle `t`, define a lookback window of length `L`.

Recommended:
- `L = 20` initially

For each sample `(engine, t)`, use information from cycles `[t-L+1, ..., t]`.

### Features
Construct a feature vector from:

1. **Current operating settings**
2. **Selected sensor values at current time**
3. **Window summary statistics** for each selected sensor:
   - mean
   - std
   - min
   - max
   - slope / linear trend
   - last-minus-first
4. **Current cycle or normalized life progress**
   - e.g. `t / failure_cycle`

### Sensor selection
Do not use all sensors blindly if many are constant or uninformative.

Recommended workflow:
- identify sensors with very low variance and drop them,
- start from the commonly informative sensors in FD001,
- keep feature dimension moderate.

A practical first version:
- keep operating settings,
- keep a subset of informative sensors,
- compute summary features over the window.

---

## Train / Validation / Test Split

Use the standard FD001 train/test files if available.

### Recommended split
- Training engines: use official training set
- Internal validation: split some training engines for tuning / calibration design
- Test: official test set

Important:
- split by **engine id**, not by rows
- avoid leakage across cycles of the same engine

### Suggested internal partition
Within training engines:
- proper training subset
- calibration / validation subset

This is important for conformal-style evaluation.

---

## Baseline Task Framing

The experiment should answer:

> Can the two-stage CKME+CP framework produce **more reliable RUL uncertainty estimates**, especially in **late-stage degradation** or other high-risk regions?

This is more important than squeezing the best RMSE.

---

## Experimental Design

## Stage 0: Data preprocessing

### Steps
1. Load FD001 train/test
2. Compute RUL labels
3. Cap RUL if desired
4. Remove near-constant sensors
5. Build window-based features
6. Standardize features using training statistics only
7. Split engines into train / calibration / validation / test

### Deliverables
- final feature matrix
- target vector
- engine ids
- degradation stage labels if needed

---

## Stage 1: Global model

Train the first-stage CKME-based model (or your current practical approximation of the CKME+CP pipeline) on the full training distribution.

### Goal
Obtain:
- predictive distribution surrogate,
- predictive intervals,
- uncertainty scores.

### Required outputs per sample
For each validation/test sample, produce:
- point prediction for RUL
- interval prediction (e.g. 90%)
- uncertainty score
- residual / conformity-related quantity if needed

### Notes
This stage should represent:
- global lifecycle modeling,
- overall degradation awareness,
- broad coverage across different engine states.

---

## Stage 2: Focused refinement

This is the key part for the interview story.

Use Stage 1 outputs to identify **high-risk / high-uncertainty regions**, then refine the model or calibration there.

### Recommended high-risk region definitions
Try one or two of the following:

#### Option A: Late-stage degradation
Define high-risk samples as:
- low RUL region, e.g. `RUL <= 30`

This is the simplest and most interpretable choice.

#### Option B: High predictive uncertainty
Use Stage 1 outputs:
- large predicted interval width,
- large estimated uncertainty score,
- large conformity score,
- large residual on validation data.

#### Option C: Combined rule
For example:
- low RUL OR top quantile of uncertainty score

### Refinement strategy
Keep it simple. Possible approaches:
1. allocate more modeling emphasis to high-risk subset,
2. fit a second-stage local correction / recalibration,
3. use local conformal refinement,
4. compare stage-1 vs stage-2 intervals specifically on high-risk subset.

### Interview-oriented principle
Even if Stage 2 is implemented in a simplified way, it must clearly support this message:

> We first build a global uncertainty-aware model, then improve reliability in the regimes that matter most for deployment decisions.

---

## Evaluation Metrics

We care about **reliability**, not just average point accuracy.

### Core metrics
Evaluate at least:

1. **Empirical coverage**
   - overall
   - high-risk subset
2. **Average interval width**
   - overall
   - high-risk subset
3. **Interval score**
   - e.g. Winkler score or similar interval quality metric
4. **Point prediction metric**
   - RMSE or MAE as a secondary metric

### Strongly recommended subset evaluations
Report metrics on:
- **all test samples**
- **late-stage subset** (e.g. `RUL <= 30`)
- optionally **high-uncertainty subset**

### Why subset evaluation matters
This is a central story point:
- average performance alone is not enough,
- high-risk region reliability is more relevant for industrial deployment.

---

## Optional Extension: One-Sided Quantile View

If implementation time allows, add a lightweight extension motivated by your recent one-sided experiments.

### Suggested interpretation
Instead of only two-sided intervals, also estimate a **lower confidence bound on RUL**.

This can be interpreted as:
- a conservative maintenance threshold,
- a safety-aware estimate of “at least how much life remains”.

### This is optional
Do this only if the main two-stage experiment is already working.

Do not let this extension delay the core pipeline.

---

## Baselines

Keep baselines simple and relevant.

### Minimum baseline set
1. **Point predictor only**
   - e.g. simple regressor with no uncertainty
2. **Split conformal baseline**
   - standard conformal interval on top of a point predictor
3. **Stage-1 only**
   - your global CKME+CP-style model without focused refinement
4. **Stage-1 + Stage-2**
   - your proposed method

### Goal of comparison
Show:
- Stage-2 improves reliability in high-risk regimes,
- not necessarily best average RMSE overall.

---

## Plots to Produce

Keep figures minimal and slide-friendly.

### Figure 1: Problem illustration
A simple degradation plot showing:
- engine cycles,
- RUL target,
- late-stage risk region,
- why uncertainty matters.

### Figure 2: Two-stage framework diagram
Include:
- windowed sensor input,
- Stage 1 global modeling,
- high-risk region identification,
- Stage 2 focused refinement,
- output intervals for RUL.

### Figure 3: Coverage vs width summary
Bar chart or table comparing:
- baseline,
- Stage 1,
- Stage 2,
for:
- all samples,
- late-stage subset.

### Figure 4: Example engine trajectory
For one representative engine:
- x-axis: cycle
- y-axis: true RUL and predicted interval
- highlight late-stage behavior

Optional:
- compare Stage 1 vs Stage 2 visually.

---

## Tables to Produce

### Table 1: Overall performance
Columns:
- method
- RMSE / MAE
- coverage
- avg interval width
- interval score

### Table 2: High-risk subset performance
Same columns, but restricted to:
- `RUL <= 30`
or another clearly defined risk subset.

This table is especially important for slides.

---

## Success Criteria

The experiment is successful if it produces evidence for at least one of these claims:

1. Stage-2 improves coverage in late-stage degradation.
2. Stage-2 achieves a better coverage-width tradeoff in high-risk regions.
3. The method provides more decision-relevant uncertainty information than point prediction alone.
4. The framework naturally supports industrial/robotics-adjacent deployment narratives.

It is **not necessary** to beat every baseline on every metric.

---

## Slide-Oriented Interpretation

The results should support these interview messages:

### Message 1
Complex dynamic systems require **trustworthy prediction**, not only point estimation.

### Message 2
A **two-stage** strategy is useful because high-risk regions deserve extra modeling attention.

### Message 3
The same methodology can transfer from simulation/statistical settings to **equipment health / industrial deployment** scenarios.

### Message 4
This thinking is relevant to robotics and embodied systems, where uncertainty and deployment reliability matter.

---

## Expected Folder Structure

Recommended output structure:

- `data/`
- `notebooks/` or `scripts/`
- `results/figures/`
- `results/tables/`
- `slides_material/`
- `spec.md`

---

## Implementation Priorities

### Priority 1
Build a working FD001 pipeline:
- preprocess
- construct window features
- train stage-1
- produce intervals
- evaluate overall + late-stage

### Priority 2
Add Stage-2 refinement:
- define risk subset
- refine / recalibrate
- compare against Stage-1

### Priority 3
Produce polished figures/tables for slides

### Priority 4
Optional one-sided RUL lower bound extension

---

## Guardrails

1. Do not over-engineer the model.
2. Do not spend too much time on sequence architectures.
3. Do not try to benchmark every C-MAPSS subset.
4. Prioritize **clarity + interpretable results + interview story**.
5. If something is too heavy, simplify rather than expanding scope.

---

## Final Deliverables

Please generate:

1. a runnable experiment pipeline for FD001,
2. one config for the default experiment,
3. overall and late-stage evaluation tables,
4. 3–4 slide-ready figures,
5. a short markdown summary explaining:
   - task setup,
   - stage-1 and stage-2 definitions,
   - key results,
   - how this connects to industrial / robotics deployment.

---

## One-Sentence Summary

Build a **small, interpretable, two-stage CKME+CP experiment on C-MAPSS FD001 for RUL uncertainty prediction**, with special emphasis on **late-stage/high-risk reliability**, so the results can be used directly in an interview presentation for industrial or robotics-oriented roles.