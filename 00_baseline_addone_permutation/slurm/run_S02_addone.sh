#!/bin/bash
# SLURM job template for S02_addone_screen.py (one sex x one block of analysis matrices).
# The 510,274 SNPs are split across several matrix directories; submit one job per
# block and concatenate the per-file outputs into {sex}_cv_5_lasso_aligned_merged_*.csv.
#
#SBATCH -A <ACCOUNT>
#SBATCH -J mleat_addone
#SBATCH -p ngs186G
#SBATCH -c 28
#SBATCH --mem=175G
#SBATCH --time=240:00:00
#SBATCH -o log/%j.log
#SBATCH -e log/%j.err

SEX=female
BLOCK=S0_S51
DATA_DIR=/path/to/mergerdataFolder/${SEX}_${BLOCK}/

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyenv
cd "$(dirname "$0")/.."

python S02_addone_screen.py --data-dir "$DATA_DIR" --out-dir lassoLoop_RES/
