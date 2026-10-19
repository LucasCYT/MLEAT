#!/bin/bash
# SLURM job template for S03_genotype_permutation_null.py (one sex x one stratum x one
# block of analysis matrices).  Resources as used on the NCHC "ngs186G" partition
# (28 CPU cores, 175 GB RAM).  Edit the account, paths and stratum before submitting.
#
#   SEX       female | male
#   STRATUM   Top | Middle | Last      (female 1,000 SNPs per stratum, male 300)
#   DATA_DIR  directory with the {sex} *.mtx.csv analysis matrices for this block
#
#SBATCH -A <ACCOUNT>
#SBATCH -J mleat_perm
#SBATCH -p ngs186G
#SBATCH -c 28
#SBATCH --mem=175G
#SBATCH --time=240:00:00
#SBATCH -o log/%j.log
#SBATCH -e log/%j.err

SEX=female
STRATUM=Top
N=$([ "$SEX" = female ] && echo 1000 || echo 300)
SEX_CAP=$([ "$SEX" = female ] && echo Female || echo Male)
DATA_DIR=/path/to/mergerdataFolder/${SEX}_S0_S51/
OUT_DIR=/path/to/results_filter_v4_${STRATUM}_NoFilter_SnpTrainShuff/

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyenv
cd "$(dirname "$0")/.."

python S03_genotype_permutation_null.py \
  --data-dir    "$DATA_DIR" \
  --snp-list    "strata/New_Knee_Point_SNP_List_${SEX_CAP}_${STRATUM}${N}.csv" \
  --results-dir "$OUT_DIR" \
  --sex         "$SEX" \
  --permutations 1000

# after all blocks of a stratum have finished:
# python S04_merge_permutation_null.py \
#   --perm-dir "$OUT_DIR/permutation_${SEX}" \
#   --out      "$OUT_DIR/${STRATUM}${N}_${SEX}_snps_merged.csv"
