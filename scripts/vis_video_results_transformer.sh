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

TRANSFORMER_PATHS_OLD=(
"transformer/models/dino_sa_complex_split_1_segmented_0729_1551/best_mil_model.pth"
"transformer/models/dino_sa_complex_split_2_segmented_0729_2058/best_mil_model.pth"
"transformer/models/dino_sa_complex_split_3_segmented_0730_0126/best_mil_model.pth"
"transformer/models/dino_sa_complex_split_4_segmented_0730_0601/best_mil_model.pth"
"transformer/models/dino_sa_complex_split_5_segmented_0730_1040/best_mil_model.pth"
)

TRANSFORMER_PATHS_POS_ENC=(
"transformer/models/dino_sa_1_split_1_True_0803_2148/best_mil_model.pth"
"transformer/models/dino_sa_1_split_2_True_0804_0244/best_mil_model.pth"
"transformer/models/dino_sa_1_split_3_True_0804_0715/best_mil_model.pth"
"transformer/models/dino_sa_1_split_4_True_0804_1200/best_mil_model.pth"
"transformer/models/dino_sa_1_split_5_True_0804_1656/best_mil_model.pth"
)

TRANSFORMER_PATHS_DEEPER=(
"transformer/models/dino_sa_4_split_1_False_0804_2306/best_mil_model.pth"
"transformer/models/dino_sa_4_split_2_False_0804_2307/best_mil_model.pth"
"transformer/models/dino_sa_4_split_3_False_0805_1128/best_mil_model.pth"
"transformer/models/dino_sa_4_split_4_False_0804_2307/best_mil_model.pth"
"transformer/models/dino_sa_4_split_5_False_0805_1126/best_mil_model.pth"
)

python -u -m dino.vis_video_results_transformer \
  --dataset_name "paxos2020" \
  --annotations_path "data/stratified_splits" \
  --videos_path "data/ensemble_results_paxos2020/cleaned_videos" \
  --output_dir "data/multi_eval_paxos2020_transformer" \
  --trans_paths_old "${TRANSFORMER_PATHS_OLD[@]}" \
  --trans_paths_pos_enc "${TRANSFORMER_PATHS_POS_ENC[@]}" \
  --trans_paths_deeper "${TRANSFORMER_PATHS_DEEPER[@]}" \
  --img_size 512 \
  --complex_augs

python -u -m dino.vis_video_results_transformer \
  --dataset_name "paxos2025" \
  --annotations_path "data/stratified_splits" \
  --videos_path "data/ensemble_results_paxos2025/cleaned_videos" \
  --output_dir "data/multi_eval_paxos2025_transformer" \
  --trans_paths_old "${TRANSFORMER_PATHS_OLD[@]}" \
  --trans_paths_pos_enc "${TRANSFORMER_PATHS_POS_ENC[@]}" \
  --trans_paths_deeper "${TRANSFORMER_PATHS_DEEPER[@]}" \
  --img_size 512 \
  --complex_augs
