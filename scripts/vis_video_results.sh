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

CLASSIFIER_PATHS=(
"classifier/models/dino_binary_split_1_0906_2310_full/best_model.pth"
"classifier/models/dino_binary_split_2_0906_2340_full/best_model.pth"
"classifier/models/dino_binary_split_3_0907_0010_full/best_model.pth"
"classifier/models/dino_binary_split_4_0906_2208_full/best_model.pth"
"classifier/models/dino_binary_split_5_0906_2239_full/best_model.pth"
)

CLASSIFIER_PATHS_ALL=(
"classifier/models/dino_all_split_1_0906_1934_full/best_model.pth"
"classifier/models/dino_all_split_2_0906_2005_full/best_model.pth"
"classifier/models/dino_all_split_3_0906_2036_full/best_model.pth"
"classifier/models/dino_all_split_4_0906_2107_full/best_model.pth"
"classifier/models/dino_all_split_5_0906_2138_full/best_model.pth"
)

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

python -u -m dino.vis_video_results \
  --dataset_name "2025" \
  --annotations_path "data/stratified_splits" \
  --videos_path "data/ensemble_results_paxos2025/cleaned_videos" \
  --output_dir "data/multi_eval_paxos2025_binary_2" \
  --classifier_paths "${CLASSIFIER_PATHS[@]}" \
  --trans_paths "${TRANSFORMER_PATHS[@]}" \
  --img_size 512 \
  --complex_augs \
  --binary_classification

python -u -m dino.vis_video_results \
  --dataset_name "2025" \
  --annotations_path "data/stratified_splits" \
  --videos_path "data/ensemble_results_paxos2025/cleaned_videos" \
  --output_dir "data/multi_eval_paxos2025_all_2" \
  --classifier_paths "${CLASSIFIER_PATHS_ALL[@]}" \
  --trans_paths "${TRANSFORMER_PATHS_ALL[@]}" \
  --img_size 512 \
  --complex_augs
