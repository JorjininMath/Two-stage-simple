# Experiment Report Writing Skill

## Purpose
This skill helps write, polish, and evaluate a strong experiment report or experiment section for academic writing, especially for papers, dissertations, technical reports, and simulation studies.

It is designed to turn raw experiment notes, code outputs, tables, and figures into clear, structured, academically persuasive writing.

This skill is especially suitable for:
- simulation and metamodeling experiments
- uncertainty quantification experiments
- conformal prediction experiments
- machine learning benchmark comparisons
- dissertation chapter experiments
- conference-paper experiment sections

This skill is specifically calibrated for papers that:
- propose a method for conditional distribution estimation or predictive interval construction
- compare CKME-CP against benchmarks such as DCP-DR, hetGP, or QR
- use a two-stage adaptive design with Stage 1 training and Stage 2 calibration
- report coverage, width, and interval score across multiple macroreplications
- involve per-point output files (per_point.csv) and aggregated summary CSVs

---

## When to Use
Use this skill when the task is to:
- draft a new experiment section from notes
- rewrite a weak or messy experiment section
- polish the structure and language of experimental writing
- interpret tables and figures
- connect empirical results to theory
- shorten an experiment section for page limits
- check whether an experiment report is complete and convincing

---

## Core Goal
A good experiment report should answer these questions clearly:

1. What is the experimental question?
2. Why is this experiment needed?
3. How was the experiment designed?
4. What methods and baselines were compared?
5. What metrics were used?
6. What happened in the results?
7. Why do the results matter?
8. What limitations remain?

The writing should not merely list settings and numbers. It should tell a logical story from objective to evidence to interpretation.

---

## Main Principles

### 1. Be question-driven
Every experiment subsection should revolve around a concrete question, such as:
- Does the proposed method achieve nominal coverage?
- Does the method improve interval efficiency?
- How does performance change with budget allocation?
- Does the method remain robust under heavy-tailed noise?
- How sensitive is the method to grid size or tuning parameters?

Do not present experiments as disconnected collections of plots.

### 2. Separate roles of sections
Keep the function of each part distinct:
- motivation explains why the experiment exists
- setup explains how it was conducted
- results state what was observed
- discussion explains what the observations mean

Avoid mixing all four everywhere.

### 3. Report facts before interpretation
State the empirical pattern first, then explain it.

Preferred pattern:
- observation
- comparison
- implication

Example:
"CKME-CP maintains nominal coverage across all tested settings and achieves lower interval scores than DCP-DR. This suggests that improving the conditional CDF estimator can yield more efficient conformal sets without sacrificing validity."

### 4. Match claims to evidence
Do not overclaim.
- If the experiment shows marginal coverage, do not claim conditional validity.
- If the result is only on two examples, do not claim universal superiority.
- If performance improves in one regime and not another, say so directly.
- The two-stage design claim (improved efficiency while maintaining asymptotic conditional coverage) should be stated as an empirical trend, not a proven guarantee, in finite-sample experiments.

### 5. Reproducibility matters
The setup should be specific enough that another researcher can understand and approximately reproduce the study.

---

## Default Structure Template

A complete experiment report or section should usually contain the following parts.

### A. Experimental Objective
State what the experiment is trying to test.

Questions to answer:
- What is the purpose of this experiment?
- What specific property is being evaluated?
- What hypothesis or expectation is being examined?

Useful sentence patterns:
- "This experiment evaluates whether ..."
- "We study how ... affects ..."
- "The goal is to examine whether the proposed method ..."

### B. Experimental Setup
Describe the design clearly and concretely.

Include:
- problem setting
- data-generating process or simulator
- input space
- true mean / variance / conditional distribution if known
- training / calibration / test split
- number of design points
- number of replications per site
- number of macroreplications
- budget allocation (n_0, r_0 for Stage 1; n_1, r_1 for Stage 2)
- benchmark methods
- tuning procedure (fixed parameters vs. cross-validation)
- implementation details that affect comparison
- whether baselines use the same total budget

Do not overload this part with interpretation.

### C. Evaluation Metrics
Define what is measured and why.

Common examples:
- empirical marginal coverage
- interval width
- interval score (Winkler score)
- group conditional coverage (by x-bin)
- CRPS
- quantile error
- runtime

For each metric, explain:
- what it measures
- whether higher or lower is better
- why it is appropriate for the study
- how it is aggregated across macroreplications (mean, SD)

### D. Results
Present the main findings clearly.

Include:
- major trends
- direct comparison with baselines
- sensitivity to settings
- reference to tables and figures
- important non-results if relevant
- any available result figures placed immediately after the relevant table or opening summary sentence; include a descriptive caption

Preferred style:
- point out the main pattern first
- then discuss details
- then mention exceptions or caveats

### E. Interpretation and Discussion
Explain what the results imply.

Possible focuses:
- why the method performs better or worse
- whether results align with theory
- tradeoffs between validity and efficiency
- where the method is most useful
- what the results reveal about design choices

### F. Limitations
Acknowledge boundaries honestly.

Examples:
- limited number of test examples
- finite-grid approximation
- only marginal rather than conditional coverage studied
- limited dimensionality
- hyperparameter sensitivity
- computational overhead not fully explored
- R-based baselines may fail on some macroreplications

### G. Takeaway
End the section or subsection with one clean conclusion.

Example:
"Overall, the results show that CKME-CP preserves nominal marginal coverage while improving interval efficiency, especially in non-Gaussian settings where Gaussian-based baselines become conservative."

---

## Writing Rules by Section

### 1. Objective / Intro to Experiments
This part should explain why the experiment is included.

Include:
- the property being tested
- why that property matters
- the role of the experiment in the paper

Do not include:
- raw numbers
- too many implementation details
- long literature review unless necessary

Good pattern:
- practical motivation
- study question
- brief description of evaluation approach

Example template:
"Since the proposed method is intended to provide valid and efficient predictive uncertainty quantification under heteroscedastic noise, we evaluate both coverage and interval efficiency across several simulation settings."

---

### 2. Setup
This section should be concrete and readable.

Always specify:
- examples or datasets
- methods compared
- sample sizes / budgets
- design strategy
- replication scheme
- tuning approach
- number of macroreplications
- test set generation

Good setup writing should answer:
- what was generated?
- how much data was used?
- how were methods trained?
- how was fairness ensured?

For simulation papers, also consider including:
- whether the design is space-filling, adaptive, weighted, or mixed
- whether outputs are Gaussian, heavy-tailed, skewed, or multimodal
- whether the true conditional mean or CDF is known
- whether the test set is independent

**For two-stage adaptive design specifically:**
Fairness requires that all methods receive the same total observation budget. State the allocation explicitly. For example: "Stage 1 uses n_0 sites with r_0 replications each; Stage 2 uses n_1 sites with r_1 replications each. The total budget is n_0 r_0 + n_1 r_1 observations, matched across all compared methods." If a baseline method does not use a two-stage design, explain how it uses the equivalent budget.

---

### 3. Metrics
Do not merely list metric names.

For each metric, provide:
- definition or concise explanation
- intuition
- optimization direction

Example:
"The three primary metrics are empirical coverage, average interval width, and interval score (IS). Empirical coverage is the proportion of test points for which the true response falls inside the predicted interval; it evaluates finite-sample validity. Average width measures interval sharpness; lower is better given valid coverage. IS summarizes the joint tradeoff: IS(L, U, Y) = (U − L) + (2/α)(L − Y)_+ + (2/α)(Y − U)_+, where lower values are preferred. Because IS can be reduced by sacrificing coverage, IS must always be reported alongside empirical coverage and interpreted only among methods that achieve at least nominal coverage."

---

### 4. Results
The results section should be descriptive first.

Recommended order:
1. overall pattern (with a reference to the main figure if one exists)
2. strongest comparison
3. subgroup or setting-level detail
4. exceptions or nuances

If result figures exist, embed them directly in the Results section — immediately after the opening summary or after the relevant table. Do not place figures in a standalone section separate from the prose. Each figure must have a caption that states what is shown, what the lines/bands represent, and how many macroreplications were used.

**Reporting across macroreplications:**
All reported numbers should be means across macroreplications, with standard deviations shown either in the table (mean ± SD format) or as shaded bands in figures. The number of macroreplications must be stated. Do not report a single macrorep result as the main finding.

Good result verbs:
- achieves
- maintains
- outperforms
- degrades
- remains stable
- becomes conservative
- improves with
- is insensitive to
- is more pronounced under

Avoid vague phrases like:
- "works well"
- "very good"
- "obviously better"
- "significantly improved"
unless numerical evidence or statistical evidence supports them

Good result sentence pattern:
- "Table X shows that ..."
- "Figure Y indicates that ..."
- "Compared with Method B, Method A ..."
- "This gap is larger under ..."
- "The benefit becomes more pronounced when ..."

---

### 5. Discussion
The discussion should explain the results, not simply restate them.

Useful discussion angles:
- theoretical consistency
- robustness to misspecification
- sensitivity to noise distribution
- effect of design points vs replications
- influence of tuning
- tradeoff between exploration and exploitation
- relation between estimator quality and final conformal efficiency

Good discussion pattern:
- what was observed
- why it may happen
- what it implies for practice or theory

Example:
"The stronger gains under heavy-tailed noise suggest that the flexibility of CKME in modeling the full conditional distribution can translate into more efficient conformal calibration when Gaussian assumptions fail."

---

### 6. Limitations
Always include limitations when writing a full report, thesis chapter, or serious discussion subsection.

Possible limitation sentence patterns:
- "These results are limited to ..."
- "We do not claim ..."
- "The current study only considers ..."
- "An important direction for future work is ..."

This improves credibility.

---

## Project-Specific Sections

### Writing Group Conditional Coverage Results

Group conditional coverage evaluates whether the constructed intervals achieve nominal coverage not just on average, but within local regions of the input space. This is a stricter validity criterion than marginal coverage and is central to the paper's claim about asymptotic conditional coverage.

**How to describe the grouping:**
State how the input space is partitioned into x-bins:
- equal-width bins are simpler but may yield unequal sample counts per bin
- equal-frequency (quantile) bins ensure roughly equal test-point counts per bin
- report the number of bins and the approximate number of test points per bin

Example:
"To assess conditional coverage, we partition the input domain [0, 2π] into K = 10 equal-width bins and compute the empirical coverage within each bin using the 1000 held-out test points. Each bin contains approximately 100 test points."

**How to interpret deviations:**
Local undercoverage below the nominal level may reflect genuine conditional invalidity or finite-sample variability. To distinguish these, note:
- how far below the nominal level the local coverage falls
- whether the deviation is consistent across macroreplications or appears in isolated bins
- whether the deviation is larger for a specific baseline

Example:
"Figure X shows that CKME-CP achieves within-bin coverage close to the 90% nominal level across all bins, while DCP-DR exhibits systematic undercoverage in the high-variance region (x near 0.9). This pattern is consistent across all macroreplications, suggesting a genuine conditional validity gap rather than sampling noise."

**Claim hedging for finite samples:**
Do not write "CKME-CP achieves conditional coverage." Write instead:
"The group conditional coverage results show that CKME-CP maintains near-nominal local coverage across all tested bins, consistent with the theoretical prediction of asymptotic conditional validity."

---

### Writing Interval Score Results

The interval score (Winkler score) is the primary efficiency metric in this paper. It penalizes both wide intervals and miscoverage symmetrically, making it a joint measure of sharpness and validity.

**Always report IS jointly with empirical coverage:**
A method with lower IS but below-nominal coverage is not superior. Always report empirical coverage and IS in the same table row or figure panel. If any method's coverage falls below nominal, state this explicitly before comparing IS values — an IS comparison is only meaningful among methods that are valid.

**Standard reporting format:**
Report mean ± SD across macroreplications:

"Table 1 shows empirical coverage and interval score (mean ± SD across 20 macroreplications) for all methods. CKME-CP achieves coverage of 0.901 ± 0.012, which is close to the nominal 90% level, and an interval score of 1.83 ± 0.11. DCP-DR achieves coverage of 0.893 ± 0.018 and an interval score of 2.14 ± 0.19. Compared with DCP-DR, CKME-CP reduces the interval score by approximately 14%, while maintaining comparable or slightly better coverage."

**IS improvement claim template:**
"Among methods that achieve at least nominal coverage, CKME-CP attains the lowest interval score in [X out of Y] simulation settings, indicating superior interval efficiency."

**When IS results are mixed:**
"While CKME-CP achieves lower IS under [regime A], the advantage is smaller and not consistent under [regime B]. This may reflect [explanation]."

---

### Writing the Two-Stage Design Experiment Fairly

The two-stage design is the core methodological contribution. Its empirical evaluation must address both the benefit of adaptive site selection and the fairness of the comparison.

**State the total budget explicitly:**
"Stage 1 uses n_0 = 250 sites with r_0 = 20 replications each (5000 observations). Stage 2 uses n_1 = 500 sites with r_1 = 10 replications each (5000 observations). The total budget is 10000 observations."

**Fairness argument:**
All comparison methods must receive the same total observation budget. State this directly:
"To ensure a fair comparison, DCP-DR and hetGP are trained on all 10000 observations allocated using the same LHS design. CKME-CP uses the two-stage allocation described above."

**Motivating the adaptive design:**
Explain the S^0 score and why it guides Stage 2 allocation:
"Stage 2 sites are selected based on the S^0 score, defined as the Stage 1 estimated quantile interval width q_{1-α/2}(x) − q_{α/2}(x). Locations with higher S^0 scores are more uncertain and therefore more informative for calibrating conformal intervals. This adaptive allocation focuses calibration data where uncertainty is highest."

**Comparing site-selection strategies:**
If comparing lhs, sampling, and mixed strategies, explain each:
"We compare three site-selection methods: LHS (space-filling, ignores S^0), sampling (sites drawn proportional to S^0 scores), and mixed (a convex combination of LHS and sampling weights). The mixed strategy balances space-filling coverage with adaptive concentration."

**Reporting Stage 2 benefit:**
"Figure X compares interval width under two-stage adaptive allocation versus a one-stage LHS baseline with equal total budget. The two-stage design achieves lower average width by [X%] across [Y] settings, with the gain concentrated in the high-variance region of the input domain."

---

### Paired Reporting with CRN

When methods are compared under Common Random Numbers (CRN) — i.e. all methods share
the same Stage 1 data, candidate sites, and test set within each macrorep — the report
should leverage paired analysis rather than independent-mean comparison.

**Preferred reporting format:**

"Table X reports per-macrorep paired differences in interval score (two-stage minus
one-stage). Across 20 macroreplications, the two-stage adaptive method achieves lower
IS in 18 out of 20 cases, with a mean paired reduction of 0.20 ± 0.03 (paired SD).
A Wilcoxon signed-rank test confirms the difference is significant (p < 0.001)."

**Why this is stronger than unpaired means:**
Paired differences cancel out macrorep-level randomness (test-point difficulty, Stage 1
model quality), isolating the effect of the method itself. This makes small but real
effects detectable without needing 50+ macroreps.

**What to report:**
- Mean ± SD of paired differences (not raw per-method means alone)
- Win count: "method A is better in K out of M macroreps"
- A paired test (Wilcoxon or paired t) if reviewers expect p-values
- Optionally, the raw per-method means as supplementary context

**When NOT to use paired analysis:**
When methods cannot share the same random world (e.g. comparing against an external
baseline whose outputs come from a different codebase with its own seeds), fall back to
unpaired reporting with independent means and SDs.

---

### Factorial Presentation (Method × Simulator)

When multiple methods are compared across multiple simulators or settings, present
results in a **factorial layout** rather than separate per-simulator tables.

**Main table**: rows = simulators, columns = methods, cells = metric (mean ± SD).
This immediately reveals **interaction effects** — e.g. "adaptive selection helps
most under heavy-tailed noise but adds little under Gaussian noise."

**Supplementary figure**: method × budget curves (IS vs total N), one line per method.
This shows whether the advantage grows, shrinks, or stays constant with budget.

These factorial layouts are standard in DOE-informed experimental writing and help
reviewers see patterns at a glance rather than hunting through multiple disjoint tables.

**DOE narrative hook (for motivation section):**
The two-stage adaptive design can be positioned as a modern extension of the classical
DOE "screen-then-refine" paradigm: Stage 1 screens the input space to identify
high-uncertainty regions; Stage 2 concentrates calibration data there. This framing
connects the method to a well-established experimental design tradition.

---

### Handling Baseline Failures (R Errors and NaN Results)

R-based benchmarks (hetGP, DCP-DR) may fail due to numerical issues in some macroreplications. This is an honest feature of the comparison and should be reported transparently.

**What to track:**
- number of macroreplications in which each baseline method produced valid results
- whether failures are random or systematic (e.g., always fail at large n or specific simulator settings)
- whether CKME-CP results are reported only for macroreps where all methods succeeded, or for all macroreps

**Preferred practice:**
Report results for all macroreplications where CKME-CP succeeds. For baselines with failures, report the subset mean alongside the failure count:

"Over 20 macroreplications, hetGP produced valid results in 17 cases; the remaining 3 failed due to numerical issues in the R implementation and are excluded from hetGP's reported statistics. CKME-CP results are reported over all 20 macroreplications."

**Sentence templates:**
- "hetGP failed to converge in [K] out of [N] macroreplications due to [reason if known]; these runs are excluded from the hetGP averages."
- "DCP-DR produced NaN outputs in [K] macroreplications, likely due to quantile crossing under [setting]. These are excluded from the reported IS and width statistics."
- "To avoid selection bias, we also report CKME-CP statistics restricted to the macroreplications where all baselines succeeded; the results are consistent with the full-sample analysis."

**Do not silently drop failures.** Always state the failure count and the exclusion rule.

---

## Recommended Tone and Style

### Tone
Use a formal, statement-based, evidence-driven tone.

Preferred characteristics:
- concise
- precise
- neutral
- analytical
- modest in claims

### Preferred style
- write in complete academic sentences
- use explicit logical transitions
- distinguish clearly between empirical fact and interpretation
- keep wording compact but not cryptic

### Avoid
- hype
- casual language
- unsupported adjectives
- excessive repetition
- long vague paragraphs with no main point

---

## Simulation-Oriented Adaptation
For simulation, metamodeling, and UQ papers, make sure to address the following when relevant:

### Design description
- number of design points
- number of replications per site
- fixed budget or varying budget
- site-selection strategy
- exploration vs exploitation mechanism

### Distributional assumptions
- Gaussian or non-Gaussian noise
- homoscedastic or heteroscedastic setting
- heavy-tailed (Student-t), skewed (Gamma), or multimodal (Gaussian mixture) outputs

### Uncertainty metrics
- coverage (marginal and conditional)
- width
- interval score (Winkler)
- CRPS
- quantile loss / pinball loss
- calibration diagnostics

### Macrorep aggregation
- report mean and standard deviation across macroreplications
- figures should show shaded SD bands, not confidence intervals (the SD band reflects variability across independent simulation runs, not estimation uncertainty within a run)
- always state the number of macroreplications in the caption or table header

### Theory-experiment connection
Explicitly say whether the experiment is intended to validate:
- consistency
- nominal coverage
- robustness
- efficiency
- convergence trend
- design tradeoff
- sensitivity to discretization or hyperparameters

For coverage claims: marginal coverage can be verified with finite samples; conditional coverage can only be assessed approximately through group conditional coverage experiments, and results should be described as "consistent with" or "suggestive of" asymptotic conditional validity, not as proof.

---

## Interpretation Patterns Library

### Comparing methods
- "CKME-CP outperforms DCP-DR on interval score across all tested settings, while maintaining equivalent marginal coverage."
- "Both methods achieve nominal coverage, but CKME-CP produces narrower intervals on average."
- "Although both methods maintain nominal coverage, CKME-CP attains lower interval score, indicating more efficient calibration."

### Trend with budget
- "Performance improves as the total budget increases."
- "Increasing the number of design sites yields larger gains than increasing replications per site."
- "The two-stage method benefits more from the adaptive Stage 2 allocation than from increasing the Stage 1 budget."

### Robustness
- "The proposed method remains stable under heavy-tailed noise."
- "The advantage is more pronounced when the benchmark model is misspecified."
- "Results suggest that the proposed estimator is less sensitive to non-Gaussian outputs."

### Validity-efficiency tradeoff
- "While all methods attain nominal marginal coverage, interval widths differ substantially."
- "The baseline is valid but conservative, producing wider intervals than necessary."
- "The proposed method improves efficiency without sacrificing validity."

### Sensitivity or ablation
- "The method is robust to moderate changes in t_grid size."
- "Performance deteriorates when the bandwidth h is too small relative to the noise scale."
- "The benefit of the adaptive Stage 2 design depends on sufficient Stage 2 site coverage; the gain diminishes under LHS allocation."

### Asymptotic conditional coverage hedging
- "The group conditional coverage results are consistent with the theoretical prediction of asymptotic conditional validity."
- "These finite-sample experiments cannot establish conditional coverage guarantees, but they indicate that local deviations from the nominal level are small and unsystematic."
- "Conditional coverage in the high-variance region remains near-nominal, suggesting that the adaptive calibration set effectively covers the most uncertain regions of the input space."

---

## Common Mistakes to Avoid

### Structural mistakes
- no clear experiment question
- setup, results, and discussion mixed together
- figures dropped in without interpretation
- results listed without telling the main story

### Content mistakes
- missing benchmark description
- missing metric definitions
- unclear train/calibration/test split
- unfair comparisons due to inconsistent tuning or budget
- ignoring negative or mixed results
- failing to report baseline failure rates

### Statistical reporting mistakes
- reporting a single macrorep result as the main finding
- showing SD bands labeled as confidence intervals
- comparing IS without reporting coverage
- claiming conditional coverage from marginal coverage results
- using unequal bin sizes for conditional coverage without noting the per-bin sample count
- silently dropping NaN results from baselines without disclosure

### Writing mistakes
- vague claims
- overclaiming generality
- repeating figure captions in prose without adding interpretation
- saying "significant" without evidence
- using too many raw details in the middle of interpretation

---

## Quality Checklist

Before finalizing an experiment section, check:

### Objective
- Is the experimental goal explicit?
- Is the main question clear?

### Setup
- Are the examples or datasets clearly described?
- Are methods and baselines identified?
- Are budgets, sample sizes, and splits stated?
- Is the tuning procedure specified?
- Is the total budget matched across compared methods?
- Is the number of macroreplications stated?

### Metrics
- Are metrics defined and motivated?
- Is it clear what better performance means?
- Is interval score reported jointly with coverage?
- Are means and SDs across macroreplications reported?

### Results
- Are the main patterns stated before details?
- Are tables and figures discussed rather than merely referenced?
- Are important comparisons made directly?
- Are figure captions complete (what is shown, what bands represent, how many macroreps)?

### Baseline failures
- Is the number of successful macroreplications reported for each baseline?
- Is the exclusion rule for NaN results stated explicitly?
- Is there a sensitivity check showing that the main findings hold on the subset of macroreps where all baselines succeeded?

### Conditional coverage
- If group conditional coverage is reported, is the binning strategy described?
- Is the per-bin test-point count stated?
- Are conditional coverage claims appropriately hedged as empirical trends, not guarantees?

### Discussion
- Are results interpreted rather than just repeated?
- Are findings linked back to the experiment objective?
- Are theory and empirical evidence connected where appropriate?

### Credibility
- Are limitations acknowledged?
- Are claims appropriately scoped?

---

## Output Modes

### Draft Mode
Use when the user has notes and wants a first complete write-up.

Output:
- polished subsection draft
- structured paragraphs
- smooth academic transitions

### Polish Mode
Use when the user already has a draft.

Output:
- improved clarity
- tighter logic
- more professional tone
- reduced redundancy

### Critique Mode
Use when the goal is evaluation rather than rewriting.

Output:
- strengths
- weaknesses
- missing experimental details
- possible reviewer concerns
- suggestions for revision

### Compression Mode
Use when the section must be shortened.

Output:
- shorter version preserving main logic
- removal of low-value detail
- merging redundant observations
- for Conference Mode: prioritize marginal coverage and IS; move group conditional coverage to a supplementary paragraph or appendix reference

### Thesis Mode
Use when more explanation and narrative are appropriate.

Output:
- fuller background
- clearer transitions
- slightly more interpretation
- full group conditional coverage subsection with per-bin discussion
- complete disclosure of baseline failure rates and exclusion logic

### Conference Mode
Use when space is limited.

Output:
- compact setup
- result-focused wording
- minimal but sufficient detail
- report marginal coverage and IS as primary metrics; note conditional coverage trend in one sentence

---

## Input Template
When using this skill, try to collect the following inputs if available:

- experiment goal
- problem setting
- data/simulator description (including noise type: Gaussian / Student-t / Gamma / mixture)
- design points and replications (n_0, r_0, n_1, r_1)
- training/calibration/test split
- benchmark methods
- hyperparameters or tuning method (fixed vs. CV)
- site-selection method (lhs / sampling / mixed)
- evaluation metrics (coverage, width, IS, group conditional coverage)
- table/figure references
- main numerical findings (mean ± SD format preferred)
- baseline failure counts if any
- observations or interpretation notes
- known limitations

---

## Output Templates

### Template 1: Experiment Intro
"This experiment evaluates whether [method/property] under [setting]. In particular, we examine [main question], with emphasis on [coverage/efficiency/robustness/etc.]."

### Template 2: Setup
"We consider [examples/settings]. For each setting, [data generation / simulator description]. The total observation budget is [N], split across Stage 1 (n_0 sites, r_0 replications each) and Stage 2 (n_1 sites, r_1 replications each). We compare [method list] under [budget/split/tuning details]. All baseline methods receive the same total budget using [LHS / random design]. Performance is evaluated over [M] independent macroreplications using [metrics], where [brief metric explanation]."

### Template 3: Results (Coverage and IS)
"Table [X] reports empirical marginal coverage and interval score (mean ± SD across [M] macroreplications) for all compared methods. CKME-CP achieves coverage of [value] ± [SD], close to the nominal [1-alpha] level, and an interval score of [value] ± [SD]. Compared with DCP-DR, CKME-CP reduces the interval score by approximately [pct]% while maintaining comparable coverage. This advantage is more pronounced under [heavy-tailed / skewed / multimodal] noise, while [exception or nuance if any]."

### Template 4: Results (Group Conditional Coverage)
"Figure [Y] shows empirical coverage within each of [K] equal-[width/frequency] input bins, computed over [M] macroreplications. Each bin contains approximately [n_per_bin] test points. CKME-CP achieves within-bin coverage close to the nominal [1-alpha] level across all bins, with deviations of at most [max_dev]. In contrast, [baseline] exhibits systematic undercoverage in [region], suggesting [interpretation]. These results are consistent with the theoretical expectation of asymptotic conditional validity, though we emphasize that finite-sample experiments cannot establish conditional coverage guarantees."

### Template 5: Baseline Failure Disclosure
"Over [M] macroreplications, [baseline] produced valid results in [k] cases; the remaining [M-k] runs failed due to [reason if known] and are excluded from [baseline]'s reported statistics. CKME-CP results are reported over all [M] macroreplications. Results restricted to the [k] macroreplications where [baseline] succeeded are qualitatively consistent with the full analysis."

### Template 6: Discussion
"These results suggest that [interpretation]. A likely reason is that [mechanism]. This is consistent with [theoretical expectation / design intuition]. The practical implication is that [actionable conclusion]."

### Template 7: Limitation
"Although the results are encouraging, the current study is limited to [restriction]. The experiment evaluates marginal and group conditional coverage but does not provide exact conditional coverage guarantees; the observed trends are consistent with asymptotic theory. Extending the analysis to [future direction] remains important."

---

## Default Preferences
Unless the user requests otherwise:
- prefer formal academic tone
- prefer concise but complete writing
- prefer statement-based interpretation
- prefer explicit discussion of what each experiment is testing
- prefer linking empirical findings back to the paper's methodological contribution
- prefer honest limitation statements
- prefer mean ± SD format for macrorep-aggregated numerical results
- always report IS jointly with empirical coverage; IS comparisons are only meaningful among methods that achieve at least nominal coverage

---

## Special Guidance for Result Paragraphs
A good result paragraph usually follows this pattern:

1. State the figure/table and the main observation.
2. Compare against baselines.
3. Explain what the comparison means.
4. Note any caveat or regime dependence.

Example pattern:
"Figure 3 shows that CKME-CP maintains nominal marginal coverage across all tested settings (mean coverage 0.903 ± 0.011 over 20 macroreplications). Compared with DCP-DR, it consistently achieves lower interval scores, indicating better efficiency at the same validity level. The gain is especially noticeable under heavy-tailed (Student-t) noise, where DCP-DR produces wider intervals due to Gaussian model misspecification. In the Gaussian setting, both methods perform similarly, which is expected given that the Gaussian assumption underlying DCP-DR is satisfied."

---

## Final Standard
The final experiment writing should be:
- logically structured
- easy to follow
- reproducible in setup
- precise in reporting (means and SDs, macrorep counts, bin specifications)
- insightful in interpretation
- honest about baseline failures
- appropriately hedged on conditional coverage claims
- aligned with the paper's actual claims about two-stage adaptive design
