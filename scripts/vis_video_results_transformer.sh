#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

source ~/.bashrc
conda activate job_new
cd ..

TRANSFORMER_PATHS=(
"transformer/models/dino_sa_1_split_1_False_0815_0042/best_sa_model.pth"
"transformer/models/dino_sa_1_split_2_False_0815_0051/best_sa_model.pth"
"transformer/models/dino_sa_1_split_3_False_0815_0051/best_sa_model.pth"
"transformer/models/dino_sa_1_split_4_False_0815_0051/best_sa_model.pth"
"transformer/models/dino_sa_1_split_5_False_0815_0355/best_sa_model.pth"
)

TRANSFORMER_PATHS_ROPE_SMALL=(
"transformer/models/dino_sa_split_1_0819_2348_rope/best_sa_model.pth"
"transformer/models/dino_sa_split_2_0819_2348_rope/best_sa_model.pth"
"transformer/models/dino_sa_split_3_0819_2348_rope/best_sa_model.pth"
"transformer/models/dino_sa_split_4_0819_2349_rope/best_sa_model.pth"
"transformer/models/dino_sa_split_5_0820_0301_rope/best_sa_model.pth"
)

TRANSFORMER_PATHS_ROPE_LARGE=(
"transformer/models/dino_sa_split_1_0825_0007_rope_100_300/best_sa_model.pth"
"transformer/models/dino_sa_split_2_0825_0010_rope_100_300/best_sa_model.pth"
"transformer/models/dino_sa_split_3_0825_0011_rope_100_300/best_sa_model.pth"
"transformer/models/dino_sa_split_4_0825_0011_rope_100_300/best_sa_model.pth"
"transformer/models/dino_sa_split_5_0825_0322_rope_100_300/best_sa_model.pth"
)

python -u -m dino.vis_video_results_transformer \
  --dataset_name "2020" \
  --annotations_path "data/stratified_splits" \
  --videos_path "data/ensemble_results_paxos2020/cleaned_videos" \
  --output_dir "data/multi_eval_paxos2020_transformer" \
  --trans_paths "${TRANSFORMER_PATHS[@]}" \
  --trans_paths_rope_small "${TRANSFORMER_PATHS_ROPE_SMALL[@]}" \
  --trans_paths_rope_large "${TRANSFORMER_PATHS_ROPE_LARGE[@]}" \
  --img_size 512 \
  --complex_augs

python -u -m dino.vis_video_results_transformer \
  --dataset_name "2025" \
  --annotations_path "data/stratified_splits" \
  --videos_path "data/ensemble_results_paxos2025/cleaned_videos" \
  --output_dir "data/multi_eval_paxos2025_transformer" \
  --trans_paths "${TRANSFORMER_PATHS[@]}" \
  --trans_paths_rope_small "${TRANSFORMER_PATHS_ROPE_SMALL[@]}" \
  --trans_paths_rope_large "${TRANSFORMER_PATHS_ROPE_LARGE[@]}" \
  --img_size 512 \
  --complex_augs
