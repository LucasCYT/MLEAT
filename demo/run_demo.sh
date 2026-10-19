#!/bin/bash
# End-to-end ML-EAT demo on SYNTHETIC data (no Taiwan Biobank data needed).
# Runs Steps 1-3 (00_baseline_addone_permutation) and the Step 3-4 core
# (01_prioritization_and_fdr) for both sexes, then compares the results with
# the simulated truth (compare with demo/expected_summary.txt).
# Run time: about 8 minutes on a laptop.
#
# Usage:  bash demo/run_demo.sh            (from the release root or demo/)
# Requires: Python 3 with numpy, pandas, scipy, scikit-learn.
set -euo pipefail
export PYTHONWARNINGS=ignore   # silence sklearn ConvergenceWarnings at very small alpha

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
UP="$ROOT/00_baseline_addone_permutation"
CORE="$ROOT/01_prioritization_and_fdr"
DATA="$HERE/demo_data"
OUT="$HERE/demo_output"
N_STRATUM=20        # SNPs per permutation stratum (paper: 1,000 F / 300 M)
N_PERM=50           # permutations per SNP     (paper: 1,000)
mkdir -p "$OUT"

echo "== 0. Generate synthetic data"
python "$HERE/make_synthetic_data.py" --out-dir "$DATA"

for SEX in female male; do
  SUB=$([ "$SEX" = female ] && echo 200 || echo 166)   # test-partition subset size
  echo; echo "########## $SEX ##########"

  echo "== Step 1. Baseline-model alpha scan (S01)"
  python "$UP/S01_baseline_model_alpha_scan.py" --input "$DATA/${SEX}_bm_candidates.csv" \
         --subset-size "$SUB" --out-prefix "$OUT/${SEX}_bm"

  echo "== Step 2. Genome-wide add-one screen (S02)"
  python "$UP/S02_addone_screen.py" --data-dir "$DATA/matrices_$SEX" \
         --out-dir "$OUT/lassoLoop_RES_$SEX" > "$OUT/${SEX}_S02.log"
  tail -n 1 "$OUT/${SEX}_S02.log"
  python "$HERE/make_ranking_and_strata.py" --addone-dir "$OUT/lassoLoop_RES_$SEX" \
         --sex "$SEX" --n "$N_STRATUM" --ranking-out "$OUT/${SEX}_ranking.csv" \
         --strata-dir "$OUT/strata"

  echo "== Step 3a. Kneedle prioritization set"
  python "$CORE/kneedle_prioritization.py" --ranking "$OUT/${SEX}_ranking.csv" \
         --out-prefix "$OUT/$SEX"
  KNEE=$(wc -l < "$OUT/${SEX}_prioritization_set.txt" | tr -d ' ')

  echo "== Step 3b. Genotype-permutation null (S03 + S04)"
  S=$([ "$SEX" = female ] && echo Female || echo Male)
  NULLS=()
  for ST in Top Middle Last; do
    python "$UP/S03_genotype_permutation_null.py" --data-dir "$DATA/matrices_$SEX" \
           --snp-list "$OUT/strata/New_Knee_Point_SNP_List_${S}_${ST}${N_STRATUM}.csv" \
           --results-dir "$OUT/perm_${ST}" --sex "$SEX" --permutations "$N_PERM" \
           > "$OUT/${SEX}_S03_${ST}.log"
    python "$UP/S04_merge_permutation_null.py" --perm-dir "$OUT/perm_${ST}/permutation_$SEX" \
           --out "$OUT/${ST}${N_STRATUM}_${SEX}_snps_merged.csv"
    NULLS+=("$OUT/${ST}${N_STRATUM}_${SEX}_snps_merged.csv")
  done

  echo "== Step 3b/4. Empirical FDR, high-confidence set, enrichment"
  M=$(($(wc -l < "$OUT/${SEX}_ranking.csv") - 2))     # minus header and 'basic' row
  python "$CORE/mleat_downstream_analysis.py" --ranking "$OUT/${SEX}_ranking.csv" \
         --perm-null "${NULLS[@]}" --yengo "$DATA/reference_gwas_demo.tsv.gz" \
         --knee "$KNEE" --M "$M" --out-prefix "$OUT/$SEX"
done

echo; echo "== Comparison with the simulated truth"
python "$HERE/summarize_demo.py" --work-dir "$OUT" --data-dir "$DATA"
