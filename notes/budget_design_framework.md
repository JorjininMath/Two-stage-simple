# Budget Design Framework for CKME-DCP

## Total Budget Decomposition

Total budget B is fixed:

```
B = B_train + B_cal = n₀·r₀ + n₁·r₁
```

Since we do NOT refit the model in Stage 2:
- **Stage 1 = Training**: D₀ trains CKME model
- **Stage 2 = Calibration**: D₁ calibrates CP (computes q̂)

## Three Layers of Design

### Layer 1: Stage 1 internal (n₀ vs r₀)

Affects: **model quality** (CDF estimation accuracy)

- n₀ large: better spatial coverage, but fewer reps per site → CDF estimate at each x relies on fewer Y samples
- r₀ large: more stable per-site CDF, but fewer sites → interpolation/extrapolation pressure
- Tradeoff: spatial resolution vs local sample size (classic kernel method tradeoff)

### Layer 2: Stage 2 internal (n₁ vs r₁)

Affects: **calibration quality** (q̂ precision and representativeness)

- n₁ large: scores from more diverse x locations → q̂ more representative, conditional coverage more uniform
- r₁ large: multiple scores per site, but these are highly correlated (same F̂(·|x)) → diminishing information gain
- Conclusion: calibration needs diversity, not redundancy → n₁ > r₁ preferred
- Supported by exp_allocation empirical results

### Layer 3: Stage 1 vs Stage 2 ratio (B₁/B₂)

Affects: **model quality vs calibration quality tradeoff**

- B₁ too small → poor model → scores large and noisy → q̂ large → wide intervals
- B₂ too small → q̂ high variance → coverage unstable
- Sweet spot: once model is "good enough", additional training budget has diminishing returns; shift to calibration

## What "Adaptive" Means

In our framework, "adaptive" has ONE meaning:

> **Stage 2 site locations are adaptively selected** based on S⁰ scores from Stage 1 model.

Specifically:
- Model is NOT adaptive (not refitted on D₁)
- Budget allocation is NOT adaptive (n₀, r₀, n₁, r₁ are preset)
- Only the WHERE of D₁ collection is adaptive

## Why Not Refit?

1. CP calibration is designed to correct model bias — even imperfect models produce valid intervals
2. Splitting D₁ for refit reduces calibration set → wider intervals (higher q̂ variance)
3. Score exchangeability is clean: model depends only on D₀, all D₁ scores are fresh
4. Adaptive site selection already targets high-uncertainty regions

## Connection to Width

Since coverage is guaranteed by CP (marginal guarantee), Stage 2 design primarily affects **width**:
- Better calibration data → more representative score distribution → tighter q̂ → narrower intervals
- S⁰-guided selection trades average width for conditional coverage uniformity

## Connection to Adaptive h

Two mechanisms both reduce score heterogeneity, but at different levels:

| Mechanism | Level | What it does |
|-----------|-------|-------------|
| Adaptive h(x) = cσ(x) | Model/score function | Makes h_eff(x) = c constant → score distribution x-independent |
| Adaptive design (S⁰) | Calibration data | Concentrates calibration data where scores are most variable |

Both improve width by reducing the "worst-case penalty" that a global q̂ pays for heterogeneity.

Key insight: if adaptive h already achieves perfect score homogeneity, adaptive design becomes less critical (and vice versa). They are **complementary but partially redundant**.

## 2×2 Factorial: Adaptive h × Adaptive Design

| | no S⁰ (uniform LHS) | S⁰ design |
|---|---|---|
| **fixed h** | baseline (worst) | data-level compensation |
| **adaptive h** | model-level homogeneity | double insurance |

**Predicted interaction**: under adaptive h, S⁰ design effect diminishes
(scores already homogeneous → sampling location doesn't matter much).
This interaction would directly validate score homogeneity as the core mechanism.

**Priority**: nice-to-show, not must-show. Ablation 1 and 7 individually suffice.
Can be added in revision if reviewers ask.

## Plug-in σ̂(x) and Stage 2

Under our no-refit framework, σ̂(x) is estimated from Stage 1 only:

```
Stage 1 (D₀) → CKME model → σ̂(x) → h(x) = c·σ̂(x)    [fixed after Stage 1]
                           → S⁰(x) → Stage 2 sites
                                           ↓
                                    D₁ → CP calibration (q̂)
```

**Stage 2 does NOT influence σ̂(x).** Consequence:

| | Stage 1 burden | Stage 2 burden |
|---|---|---|
| **fixed h** | train model | S⁰ design needed to compensate score heterogeneity |
| **adaptive h** | train model + estimate σ̂(x) | light (scores homogeneous) |

Adaptive h shifts burden from Stage 2 → Stage 1. The total budget may be similar,
but the optimal B₁/B₂ split shifts toward more Stage 1.

This means: **adaptive h demands a richer Stage 1** (more sites/reps for σ̂ quality),
while **fixed h demands a richer Stage 2** (more calibration data for score diversity).

## Which is More Natural? Adaptive Design vs Adaptive h

In real-world simulation experiments, the practitioner's situation is:

**Adaptive design is the natural choice.** Reasons:

1. **σ(x) is unknown** — adaptive h requires h(x) = c·σ(x), but in a real problem
   we don't know σ(x). We must estimate σ̂(x) from Stage 1, which requires enough
   Stage 1 data AND a variance estimation method. This is an extra modeling step
   that may itself be unreliable.

2. **Two-stage data collection is standard practice** — in simulation experiments
   (manufacturing, engineering, operations research), sequential data collection
   is the norm. Running a pilot study (Stage 1), then deciding where to sample
   more (Stage 2) is a natural workflow. The S⁰ score just formalizes "where is
   my model most uncertain?"

3. **Design is model-agnostic** — S⁰-guided site selection works regardless of
   whether you use fixed or adaptive h, CKME or another CDF estimator. It's a
   general principle: calibrate where you're least confident.

4. **Adaptive h is a model-internal optimization** — it requires knowledge of the
   noise structure (σ(x) or at least σ̂(x)), and the theoretical benefit depends
   on the location-scale assumption Y = f(x) + σ(x)ε. If the noise isn't
   location-scale (e.g., shape changes with x), adaptive h's homogeneity
   guarantee breaks down.

5. **Robustness** — adaptive design degrades gracefully (if S⁰ is uninformative,
   it reduces to near-uniform sampling). Adaptive h with bad σ̂(x) can actively
   hurt (wrong bandwidth → worse CDF estimate → worse scores).

**Implication for the paper**: Position adaptive design (two-stage + S⁰) as the
primary contribution. Adaptive h is a theoretical insight that explains WHY
CDF-based scores are structurally better, and points toward future improvement
when σ(x) can be reliably estimated.

## Ablation Mapping

| Ablation | Tests which layer? |
|----------|--------------------|
| Ablation 1 (S⁰ vs LHS) | Adaptive site selection value |
| Ablation 2 (two-stage vs one-stage) | Train/cal separation value |
| Ablation 5 (B₁/B₂ ratio) | Optimal split point |
| Ablation 7 (adaptive h) | Score homogeneity at model level |
