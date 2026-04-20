# Section Note Writing Skill

## Purpose
This skill guides the writing of **section notes** — detailed working papers that expand a single result or topic from `paper_strategy_memo.tex` into a self-contained document with full mathematical derivations, experiment details, and discussion.

Each note corresponds to one section of the eventual full paper but contains more detail than a journal submission would allow. Notes serve as:
- Drafting grounds for paper sections
- Archives of derivations and proof sketches
- Detailed experiment reports with full numerical tables
- Internal references for co-authors

## When to Use
- Writing a new section note from scratch
- Expanding a result from `paper_strategy_memo.tex` into a detailed note
- Combining theory derivations with experiment validation in one document

## Relationship to Other Skills
- **`experiment_report_skills.md`**: Use its guidelines for the Experiment portion of a note (Setup → Metrics → Results → Discussion). This skill adds theory-writing structure around it.
- **`paper_strategy_memo.tex`**: Each note expands one section/result from this memo.

---

## Structure Template

A section note follows this structure:

### 1. Abstract (5–8 sentences)
- What result is this note about?
- What is the main theoretical claim?
- What experiment validates it?
- What is the key finding?

### 2. Setup and Notation
- Define all symbols used in the note
- State the model assumptions (e.g., location-scale family)
- Reference the CKME estimator and conformal prediction framework
- Keep consistent with the symbol table below

### 3. Theory
Organized as a sequence of:

**Definition → Proposition/Theorem → Proof sketch → Remark**

Guidelines:
- **Definitions** come before first use; group them at the start of the theory section
- **Propositions/Theorems**: state the claim formally, then provide intuition in 1–2 sentences immediately after
- **Proof sketch**: not a full proof; show the key calculation step and state what is omitted. Use "Proof sketch." not "Proof."
- **Remarks**: connect the formal result to practical implications, numerical examples, or experiment predictions
- **Intuition paragraph**: after each major result, write a short paragraph explaining "what this means" in plain language
- Number all equations that are referenced later

### 4. Experiment Design
Follow `experiment_report_skills.md` structure:
- Objective (what theoretical prediction is being tested)
- Setup (simulator, budget, bandwidth configs, macroreps)
- Metrics (what is measured and why)

### 5. Experiment Results
Follow `experiment_report_skills.md` structure:
- Main pattern first, then details
- Tables with mean ± SD across macroreps
- Reference figures by number
- State findings as numbered items

### 6. Discussion
- Do results match theoretical predictions? Where do they diverge?
- What are the finite-sample limits of the theory?
- Connection to other results in the paper

### 7. Connection to Full Paper
- Which section of the paper does this note feed into?
- What should be kept vs. condensed for the paper version?
- Open questions that the paper need not resolve

---

## Theory Writing Guidelines

### Proposition Structure
```latex
\begin{proposition}[Short descriptive name]
\label{prop:tag}
Under [assumptions], [conclusion].
\end{proposition}

\paragraph{Intuition.} [1--2 sentences explaining why this is true without equations.]

\begin{proof}[Proof sketch]
[Key calculation. State what is routine and omitted.]
\end{proof}

\begin{remark}
[Practical consequence, numerical example, or connection to experiment.]
\end{remark}
```

### Derivation Style
- Show the key algebraic step explicitly; don't skip the step that carries the insight
- Use aligned environments for multi-line derivations
- After a derivation, state the conclusion in words: "Therefore, ..."
- If a step uses a standard result (e.g., Taylor expansion), cite it inline rather than re-deriving

### What to Prove vs. State
- **Prove**: results that are novel to this work (score homogeneity, over-smoothed collapse, effective bandwidth)
- **State without proof**: standard results from literature (CP exchangeability, kernel consistency rates, Taylor expansions)
- **Proof sketch**: results that are provable but whose full proof is routine or long (perturbation bounds, plug-in stability)

---

## Symbol Table (Shared Across All Notes)

| Symbol | Meaning |
|--------|---------|
| $f(x)$ | True conditional mean $E[Y \mid X=x]$ |
| $\sigma(x)$ | True conditional noise scale |
| $\varepsilon$ | Standardized noise: $Y = f(x) + \sigma(x)\varepsilon$ |
| $G(\cdot)$ | CDF of $\varepsilon$ (noise distribution) |
| $\Psi(u)$ | Logistic CDF: $1/(1+e^{-u})$ |
| $h$ | Bandwidth of smooth indicator in CKME |
| $h(x) = c\,\sigma(x)$ | Adaptive bandwidth |
| $h_{\mathrm{eff}}(x)$ | Effective bandwidth in standardized scale: $h/\sigma(x)$ |
| $c$ | Adaptive bandwidth constant |
| $\hat{F}(y \mid x)$ | CKME CDF estimate |
| $\hat{G}_c(z)$ | Standardized CDF estimate (common across $x$ under adaptive $h$) |
| $z$ | Standardized variable: $(y - f(x))/\sigma(x)$ |
| $c_i(x)$ | CKME kernel weights (signed, sum $\approx 1$) |
| $s(x, y)$ | Conformal score |
| $\hat{q}$ | CP calibration quantile |
| $\alpha$ | CP significance level |
| $d_{\mathrm{TV}}$ | Total variation distance |
| $n_0, r_0$ | Stage 1: number of sites, replications per site |
| $n_1, r_1$ | Stage 2: number of sites, replications per site |
| $M$ | Number of evaluation grid points |
| $B$ | Number of test points per evaluation point |

---

## LaTeX Conventions

- Use `\newtheorem` for proposition, theorem, lemma, remark, corollary, definition
- Use `\paragraph{Intuition.}` after formal statements
- Use `\begin{proof}[Proof sketch]` for proof sketches
- Number equations only if referenced later
- Use `\texttt{}` for code/experiment names (e.g., `\texttt{exp\_score\_homogeneity}`)
- Use `\cite{}` placeholders for references: `\cite{CP2}`, `\cite{CKME}`, etc.

---

## Quality Checklist

### Theory
- [ ] All symbols defined before first use
- [ ] Assumptions stated explicitly
- [ ] Each proposition has intuition paragraph
- [ ] Proof sketches show the key step
- [ ] Remarks connect to experiments or practice
- [ ] No circular reasoning (don't use the result to prove itself)

### Theory–Experiment Connection
- [ ] Each theoretical prediction has a corresponding experiment test
- [ ] Experiment results explicitly reference which proposition they validate
- [ ] Discrepancies between theory and experiment are discussed (finite-sample effects, violated assumptions)

### Experiment (defer to `experiment_report_skills.md`)
- [ ] Setup fully specified (simulator, budget, configs, macroreps)
- [ ] Metrics defined with optimization direction
- [ ] Results reported as mean ± SD
- [ ] Figures referenced and described
- [ ] Limitations acknowledged

### Writing
- [ ] Abstract is self-contained
- [ ] Connection to full paper is clear
- [ ] Note is readable independently of other notes
- [ ] Consistent notation with symbol table

---

## Note Index (Planned)

| # | Title | Memo Section | Status |
|---|-------|-------------|--------|
| 1 | Natural Rectification & Finite-Sample Bound | Result 1 (§3) | Planned |
| 2 | Score Homogeneity Mechanism | Result 2 (§4) | In progress |
| 3 | CKME Consistency Rate | Result 3 (§5) | Planned |
| 4 | Non-Gaussian Robustness | §6 | Planned |
| 5 | Two-Stage Adaptive Design | Paper 2 material | Planned |
