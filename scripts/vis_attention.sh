#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

source ~/.bashrc
conda activate job_new
cd ..

TRANSFORMER_PATHS_ALL=(
"transformer/models/dino_sa_all_split_1_0907_1147_full/best_sa_model.pth"
"transformer/models/dino_sa_all_split_2_0907_1203_full/best_sa_model.pth"
"transformer/models/dino_sa_all_split_3_0907_1224_full/best_sa_model.pth"
"transformer/models/dino_sa_all_split_4_0907_1314_full/best_sa_model.pth"
"transformer/models/dino_sa_all_split_5_0907_1330_full/best_sa_model.pth"
)

TRANSFORMER_PATHS=(
"transformer/models/dino_sa_binary_split_1_0907_1613_full/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_2_0907_1524_full/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_3_0907_1500_full/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_4_0907_1442_full/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_5_0907_1352_full/best_sa_model.pth"
)

python -u -m dino.vis_attention \
  --trans_paths "${TRANSFORMER_PATHS[@]}" \
  --complex_augs \
  --binary_classification \
  --output_dir 'binary_transformer_attention_analysis'

python -u -m dino.vis_attention \
  --trans_paths "${TRANSFORMER_PATHS_ALL[@]}" \
  --img_size 512 \
  --complex_augs  \
  --output_dir 'all_transformer_attention_analysis'
