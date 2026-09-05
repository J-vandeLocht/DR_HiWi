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

TRANSFORMER_PATHS=(
"transformer/models/dino_sa_all_split_1_0831_2146_full/best_sa_model.pth"
"transformer/models/dino_sa_all_split_2_0901_0055_full/best_sa_model.pth"
"transformer/models/dino_sa_all_split_3_0901_0341_full/best_sa_model.pth"
"transformer/models/dino_sa_all_split_4_0901_0401_full/best_sa_model.pth"
"transformer/models/dino_sa_all_split_5_0901_0653_full/best_sa_model.pth"
)

python -u -m dino.vis_video_results_all \
  --dataset_name "2020" \
  --annotations_path "data/stratified_splits" \
  --videos_path "data/ensemble_results_paxos2020/cleaned_videos" \
  --output_dir "data/multi_eval_paxos2020_all" \
  --trans_paths "${TRANSFORMER_PATHS[@]}" \
  --img_size 512 \
  --complex_augs

python -u -m dino.vis_video_results_all \
  --dataset_name "2025" \
  --annotations_path "data/stratified_splits" \
  --videos_path "data/ensemble_results_paxos2025/cleaned_videos" \
  --output_dir "data/multi_eval_paxos2025_all" \
  --trans_paths "${TRANSFORMER_PATHS[@]}" \
  --img_size 512 \
  --complex_augs
