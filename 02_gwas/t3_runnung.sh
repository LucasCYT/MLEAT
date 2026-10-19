#!/bin/bash
#SBATCH -A <ACCOUNT>        # Account name/project number
#SBATCH -J female_c37        # Job name
#SBATCH -p ngs92G           # Partition Name 等同PBS裡面的 -q Queue name
#SBATCH -c 14               # 使用的core數  請參考Queue資源設定 
#SBATCH --mem=92G           # 使用的記憶體量 請參考Queue資源設定
#SBATCH -o female_c37.log          # Path to the standard output file 
#SBATCH -e female_c37_err.log          # Path to the standard error ouput file
#SBATCH --mail-user=<EMAIL>              # email
#SBATCH --mail-type=BEGIN,END              # 指定送出email時機 可為NONE, BEGIN, END, FAIL, REQUEUE, ALL


source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyenv #進入 conda 環境

cd /path/to/project/GTprobe_height_female/_0807_new_gwas/


python -u GWAS_female_c37.py
echo -e "GWAS_female_37_done"