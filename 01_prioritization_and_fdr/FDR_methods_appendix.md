# Methods appendix: empirical FDR — formula, theory, and comparison with the previous z-score approach

## 1. Definition (Benjamini & Hochberg 1995)

For a set of selected variants, let R = number selected and V = number of those
that are truly null (false positives). The false discovery rate is

    FDR = E[ V / R ]  — the expected proportion of discoveries that are false.

## 2. Estimator used here

Each variant has an add-one **test MSE** (lower = stronger predictive
contribution). Selecting variants with MSE ≤ t:

    FDR_hat(t) = [ M · π0 · p0(t) ] / R(t)  ≈  M · p0(t) / R(t)

- **M** = 510,274  (number of variants screened; the multiple-testing burden)
- **R(t)** = number of real variants with MSE ≤ t  (observed discoveries)
- **p0(t)** = probability a *null* variant reaches MSE ≤ t  (per-test false-positive prob.)
- **π0** = proportion of truly null variants; set to 1 (conservative → over-estimates FDR)

**Derivation.** Treating all M variants as null, each independently crosses the
threshold with probability p0(t); the expected number of null crossings is the
binomial mean M·p0(t) = E[V(t)]. Dividing by the observed number of crossings
R(t) gives the expected false-discovery proportion.

## 3. Estimating p0(t) by permutation (distribution-free)

The add-one ΔMSE has **no parametric null distribution**, so p0(t) is estimated
non-parametrically. For a variant, permuting its genotype across individuals
(while the phenotype and all covariates, including principal components, are held
fixed) breaks only the variant–phenotype association, yielding a draw of the
statistic under H0. Pooling permuted statistics across sampled variants and
permutations gives the empirical null; then

    p0(t) = #{ permuted MSE ≤ t } / (total permuted values).

(Here: 3.0×10⁶ pooled null values in females, 9.0×10⁵ in males, pooled across the
three ranking strata; a shuffled variant is exchangeable, so pooling is valid.)

**Note — p0(t) is not a per-variant p-value.** Two related quantities should not
be conflated. A *per-variant empirical p-value* is computed for one variant from
its own permutations — the fraction of that variant's permuted statistics at
least as extreme as its observed value — and quantifies that single variant's
significance. In contrast, **p0(t) is one population-level number**: the fraction
of the *pooled* permutation null (across all sampled variants and permutations)
that reaches threshold t. The FDR estimator uses **p0(t) — the pooled null tail
probability at the selection threshold — not an average of per-variant p-values.**
Pooling is valid because a permuted variant carries no signal and is exchangeable
across variants, and it yields a far larger null sample, giving the tail
resolution needed at the 5% FDR threshold.

**Selection procedure and monotonicity.** There is no closed-form relationship
between the selection depth, the MSE threshold t, and p0(t); the threshold, the
number of real discoveries R(t), and the null tail probability p0(t) are linked
and are all read empirically at each depth (t is simply the R-th smallest observed
MSE, so choosing R fixes t and vice versa). FDR_hat(t) is therefore evaluated
across the full range of thresholds by sweeping R = 1, 2, 3, …. Because p0(t) is an
empirical step function — its value can rise abruptly when a cluster of permuted
null statistics falls just below a slightly relaxed threshold — FDR_hat is not
guaranteed to be monotone in R. Selection was accordingly **not** based on the
first threshold at which the estimate exceeds 0.05; following the
Benjamini–Hochberg / q-value convention, we use the monotonised quantity
q(R) = min over R' ≥ R of FDR_hat(R') and retain all variants with q ≤ 0.05. In
this study FDR_hat increased monotonically through the 5% boundary in both sexes
(females: 4.4% at rank 84, 5.2% at rank 85; males: 3.0% at rank 19, ≈1 at rank 20),
so the "largest R with FDR ≤ 0.05" and the q-value rules coincide, yielding 84
(female) and 19 (male) high-confidence variants.

## 4. Theory / relationship to standard methods

This is the resampling-based FDR estimator of **Storey (2002, 2003)**, the
plug-in used by **SAM (Tusher et al. 2001)**, and the permutation analogue of
**Benjamini–Hochberg (1995)**: BH selects the largest k with p(k) ≤ (k/M)α,
i.e. the FDR at rank k equals M·p(k)/k. Substituting the permutation-estimated
null tail probability p0(t) for the theoretical p-value p(k), and R(t) for k,
gives M·p0(t)/R(t). In short: **the standard BH/Storey FDR estimator, with the
parametric p-value replaced by a permutation empirical tail probability.**

## 5. Worked example (this study, females)

| Selected R | threshold t | p0(t) | expected false M·p0 | FDR |
|---|---|---|---|---|
| 1,048 (knee) | ≈ baseline | 0.0066 | ≈ 3,367 | ≈ 1.0 (capped) |
| 84 | stricter | ≈ 8×10⁻⁶ | ≈ 4 | ≈ 0.05 |

At the knee the null alone is expected to produce ~3,367 crossings versus 1,048
selected → FDR ≈ 1; only the ~84 strongest contributions reach a threshold
strict enough for FDR ≤ 5%.

## 6. Assumptions
One-sided (improvement side); π0 = 1 (conservative); the permutation null is the
correct null for "this variant has no effect" (covariates and structure
preserved, Freedman–Lane style); exchangeability for pooling; tail resolution
≈ 1/(pooled null size) ≈ 3×10⁻⁷.

---

# 7. Comparison with the previous z-score-based approach

The earlier version of this analysis derived significance by **transforming the
add-one MSE to a z-score and mapping it to a p-value under the standard normal
distribution**, then applied the knee point to that distribution and compared
selection counts against Bonferroni / Benjamini–Hochberg. That approach yielded a
much larger and more favourable set (≈1,580 female / 381 male). The two approaches
differ fundamentally, and the discrepancy is expected:

| | Previous (MSE → z-score → p) | Current (permutation FDR) |
|---|---|---|
| Null distribution | Assumed **normal**; reference = the observed across-SNP MSE distribution | **Empirical**, from permuting each SNP's genotype (external, distribution-free) |
| Statistic → significance | z = (MSE − mean)/SD → p from normal tail | direct empirical tail probability p0(t) |
| Multiple-testing treatment | Selection-**count** comparison only (Fig. 5B); the knee set's FDR was never computed | FDR explicitly estimated as M·p0(t)/R(t) |
| FDR of the knee set | implicitly assumed low (knee = significance) | ≈ 1 |
| Result | ~1,580 / 381 "significant" | 84 / 19 at FDR ≤ 5% |

**Why the previous result was over-optimistic (and "too clean").**

1. **Circular / self-referential null.** Each SNP's z-score was computed relative
   to the mean and SD of *all SNPs' MSE values* — a distribution that is itself
   dominated by real signal plus noise, not a genuine no-effect null. Scoring
   ML-EAT's output against ML-EAT's own output distribution cannot fail to look
   internally consistent; there was no independent null against which the method
   could be found wanting.

2. **Unjustified normality → inflated tail p-values.** The MSE statistic is not
   normal under the null (the manuscript's own Kolmogorov–Smirnov test rejected
   normality). Standardising a near-zero, tightly-spread set of improvements and
   extrapolating into the normal tail turns modest improvements into
   astronomically small p-values. Empirically, the lead 12q13.2 variant received
   p ≈ 1.6×10⁻²⁴ under the transform, whereas its honest permutation-based FDR at
   the depth at which it is selected is ≈ 35% — an overstatement of roughly twenty
   orders of magnitude.

3. **No genome-wide multiple-testing accounting.** The "FDR comparison" contrasted
   how many variants each method selects; a smaller count was interpreted as
   higher specificity. But count ≠ FDR: a method can select few variants that are
   nonetheless mostly false. The knee set's actual false-discovery rate was never
   estimated.

The permutation FDR removes all three problems by introducing an **external,
non-parametric null** (genotype permutation) and by **explicitly accounting for
the 510,274-variant multiple-testing burden**. The harsher result (84 / 19) is
therefore the correct, conservative one; the previous z-score result was inflated
by the discredited normal-transform that reviewers identified, and is not retained.

**References.** Benjamini & Hochberg, *J. R. Stat. Soc. B* 1995; Storey,
*J. R. Stat. Soc. B* 2002; Storey & Tibshirani, *PNAS* 2003; Tusher, Tibshirani &
Chu, *PNAS* 2001.
