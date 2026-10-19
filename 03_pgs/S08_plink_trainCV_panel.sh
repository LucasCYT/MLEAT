#!/usr/bin/env bash
# S08_plink_trainCV_panel.sh — run on the NCHC cluster (paths as in the existing PLINK logs).
# Builds the training-only ML-EAT panel PGS with the SAME LD handling as the GWAS-significant panel
# (--clump r2<0.1, 250 kb; index SNP = best dMSE_CV per locus), weights = training-GWAS BETA, scored in the 20 % test set.
# Copy R11_*_clumpinput.txt (from 03_results) and S09_make_weights_from_clumped.py next to this script first.
set -euo pipefail
PLINK=/path/to/plink
BASE=/path/to/project
declare -A BFILE=( [female]=$BASE/GTprobe_height_female/GTprobe_female        [male]=$BASE/GTprobe_height_male/GTprobe_male )
declare -A KEEP=(  [female]=$BASE/GTprobe_height_female/_0807_new_gwas/Merged_Female_Data_withTWB2ID_C37_test20.txt
                   [male]=$BASE/GTprobe_height_female/_0807_new_gwas/PRS/Merged_Male_Data_withTWB2ID_C33_test20.txt )
declare -A ASSOC=( [female]=$BASE/GTprobe_height_female/_0807_new_gwas/PRS/female_GWAS_BODY_HEIGHT_c33_80_ADD.assoc.linear
                   [male]=$BASE/GTprobe_height_female/_0807_new_gwas/PRS/male_GWAS_BODY_HEIGHT_c29_80_ADD.assoc.linear )
declare -A K=( [female]=1048 [male]=378 )

for sex in female male; do
  IN=R11_${sex}_MLEAT_trainCV_top${K[$sex]}_clumpinput.txt
  # 1) LD clumping on the panel itself (p1 = 1 keeps every listed SNP as a candidate; pseudo-P only sets the index-SNP order)
  $PLINK --bfile ${BFILE[$sex]} --clump $IN --clump-p1 1 --clump-p2 1 --clump-r2 0.1 --clump-kb 250 \
         --out ${sex}_MLEAT_trainCV_clumped
  # 2) weights: SNP, A1, BETA from the training-partition GWAS, restricted to the clumped index SNPs
  python3 S09_make_weights_from_clumped.py ${ASSOC[$sex]} ${sex}_MLEAT_trainCV_clumped.clumped ${sex}_prs_weights_MLEAT_trainCV_clumped.txt
  # 3) PGS in the held-out 20 % test partition (never used for selection or weights)
  $PLINK --bfile ${BFILE[$sex]} --score ${sex}_prs_weights_MLEAT_trainCV_clumped.txt 1 2 3 header \
         --keep ${KEEP[$sex]} --out ${sex}_prs_test20_MLEAT_trainCV_clumped_result
done
# Return to the Drive:  *_MLEAT_trainCV_clumped.{clumped,log}  *_prs_weights_MLEAT_trainCV_clumped.txt
#                       *_prs_test20_MLEAT_trainCV_clumped_result.{profile,log,nopred}
# -> put them in NewData/Figure5/ (or output/01_inputs/) and re-run S01 -> S05 with the extra panel.
