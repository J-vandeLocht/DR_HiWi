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

CLASSIFIER_2020_PATHS=(
"classifier/models/dino_split_1_0815_1421_2020/best_model.pth"
"classifier/models/dino_split_2_0815_1509_2020/best_model.pth"
"classifier/models/dino_split_3_0815_1556_2020/best_model.pth"
"classifier/models/dino_split_4_0815_1719_2020/best_model.pth"
"classifier/models/dino_split_5_0815_1805_2020/best_model.pth"
)

CLASSIFIER_2025_PATHS=(
"classifier/models/dino_split_1_0815_1719_2025/best_model.pth"
"classifier/models/dino_split_2_0815_1751_2025/best_model.pth"
"classifier/models/dino_split_3_0815_1720_2025/best_model.pth"
"classifier/models/dino_split_4_0815_1751_2025/best_model.pth"
"classifier/models/dino_split_5_0815_1720_2025/best_model.pth"
)

CLASSIFIER_FUSED_PATHS=(
"classifier/models/dino_split_1_0811_1512/best_model.pth"
"classifier/models/dino_split_2_0811_1634/best_model.pth"
"classifier/models/dino_split_3_0811_1756/best_model.pth"
"classifier/models/dino_split_4_0811_1918/best_model.pth"
"classifier/models/dino_split_5_0811_2040/best_model.pth"
)

TRANSFORMER_2020_PATHS=(
"transformer/models/dino_sasplit_1_0815_1914_2020/best_sa_model.pth"
"transformer/models/dino_sasplit_2_0815_2056_2020/best_sa_model.pth"
"transformer/models/dino_sasplit_3_0815_1917_2020/best_sa_model.pth"
"transformer/models/dino_sasplit_4_0815_2059_2020/best_sa_model.pth"
"transformer/models/dino_sasplit_5_0815_1917_2020/best_sa_model.pth"
)

TRANSFORMER_2025_PATHS=(
"transformer/models/dino_sasplit_1_0815_2058_2025/best_sa_model.pth"
"transformer/models/dino_sasplit_2_0815_1917_2025/best_sa_model.pth"
"transformer/models/dino_sasplit_3_0815_2051_2025/best_sa_model.pth"
"transformer/models/dino_sasplit_4_0815_2225_2025/best_sa_model.pth"
"transformer/models/dino_sasplit_5_0815_2355_2025/best_sa_model.pth"
)

TRANSFORMER_FUSED_PATHS=(
"transformer/models/dino_sa_1_split_1_False_0815_0042/best_sa_model.pth"
"transformer/models/dino_sa_1_split_2_False_0815_0051/best_sa_model.pth"
"transformer/models/dino_sa_1_split_3_False_0815_0051/best_sa_model.pth"
"transformer/models/dino_sa_1_split_4_False_0815_0051/best_sa_model.pth"
"transformer/models/dino_sa_1_split_5_False_0815_0355/best_sa_model.pth"
)

python -u -m dino.vis_video_results_comparison \
  --dataset_name "2020" \
  --annotations_path "data/stratified_splits" \
  --videos_path "data/ensemble_results_paxos2020/cleaned_videos" \
  --output_dir "data/multi_eval_paxos2020_comparison" \
  --clf_2020_paths "${CLASSIFIER_2020_PATHS[@]}" \
  --clf_2025_paths "${CLASSIFIER_2025_PATHS[@]}" \
  --clf_fused_paths "${CLASSIFIER_FUSED_PATHS[@]}" \
  --trans_2020_paths "${TRANSFORMER_2020_PATHS[@]}" \
  --trans_2025_paths "${TRANSFORMER_2025_PATHS[@]}" \
  --trans_fused_paths "${TRANSFORMER_FUSED_PATHS[@]}" \
  --img_size 512 \
  --complex_augs

#python -u -m dino.vis_video_results_comparison \
#  --dataset_name "2025" \
#  --annotations_path "data/stratified_splits" \
#  --videos_path "data/ensemble_results_paxos2025/cleaned_videos" \
#  --output_dir "data/multi_eval_paxos2025_comparison" \
#  --clf_2020_paths "${CLASSIFIER_2020_PATHS[@]}" \
#  --clf_2025_paths "${CLASSIFIER_2025_PATHS[@]}" \
#  --clf_fused_paths "${CLASSIFIER_FUSED_PATHS[@]}" \
#  --trans_2020_paths "${TRANSFORMER_2020_PATHS[@]}" \
#  --trans_2025_paths "${TRANSFORMER_2025_PATHS[@]}" \
#  --trans_fused_paths "${TRANSFORMER_FUSED_PATHS[@]}" \
#  --img_size 512 \
#  --complex_augs
