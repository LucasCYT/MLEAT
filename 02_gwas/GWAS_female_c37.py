#!/usr/bin/env python3

import os
import pandas as pd
import subprocess


# 設定 PLINK 可執行檔路徑
plink_path = r"/path/to/plink"

current_directory = os.getcwd()
print(f'current_directory: {current_directory}')

# PLINK 輸入文件 (不含副檔名)
input_dir = r"/path/to/project/GTprobe_height_female/"
print(f'input_dir: {input_dir}')

input_file = os.path.join(input_dir, f'GTprobe_female')
print(f'input_file: {input_file}')

output_dir = r"/path/to/project/GTprobe_height_female/_0807_new_gwas/"


# 設定輸入文件
covar_file = r"/path/to/project/GTprobe_height_female/_0807_new_gwas/Merged_Female_Data_withTWB2ID_C37.txt"
out_prefix = os.path.join(output_dir, "female_GWAS_BODY_HEIGHT_c37")


# 讀取文件（假設是 tab 分隔）
df = pd.read_csv(covar_file, sep='\t')

# 取得所有欄位名稱
column_names = df.columns.tolist()

# 顯示欄位名稱
print(column_names)

columns_to_remove = ['FID', 'IID', 'TWB2_ID', 'BODY_HEIGHT']

pca_columns = [col for col in df.columns if col.startswith('pca')]

# 合并要删除的列
columns_to_remove.extend(pca_columns)

covar_name = [col for col in column_names if col not in columns_to_remove]
print(len(covar_name))

covar_name_str = ",".join(covar_name)

print(covar_name_str)


print(input_file)
print(covar_file)
print(covar_name_str)
print(out_prefix)

# 建立 PLINK 指令
plink_command = [
    plink_path,
    "--bfile", input_file,
    "--covar", covar_file,
    "--covar-name", covar_name_str,
    "--linear",
    "--out", out_prefix,
    "--pheno", covar_file,  # 確保是正確的表型檔案
    "--chr", "1-22",
    "--pheno-name", "BODY_HEIGHT"  # 確保這個欄位在表型檔案中
]

print(plink_command)

print("Running PLINK GWAS analysis...")
try:
    result = subprocess.run(plink_command, check=True)
    print("PLINK command executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error occurred during PLINK execution: {e}")
