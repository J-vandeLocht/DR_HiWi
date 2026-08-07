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

MIL_PATHS=(
"mil/models/dino_mil_complex_split_1_segmented_0728_1551/best_mil_model.pth"
"mil/models/dino_mil_complex_split_2_segmented_0728_2128/best_mil_model.pth"
"mil/models/dino_mil_complex_split_3_segmented_0729_0219/best_mil_model.pth"
"mil/models/dino_mil_complex_split_4_segmented_0729_0641/best_mil_model.pth"
"mil/models/dino_mil_complex_split_5_segmented_0729_1103/best_mil_model.pth"
)

CLASSIFIER_PATHS=(
"classifier/models/dino_complex_split_1_imagenet_0725_0545/best_model.pth"
"classifier/models/dino_complex_split_2_imagenet_0725_0715/best_model.pth"
"classifier/models/dino_complex_split_3_imagenet_0725_0846/best_model.pth"
"classifier/models/dino_complex_split_4_imagenet_0725_1016/best_model.pth"
"classifier/models/dino_complex_split_5_imagenet_0725_1142/best_model.pth"
)

#MIL_PATHS=(
#"mil/models/dino_mil_complex_split_1_segmented_0729_1101/best_mil_model.pth"
#"mil/models/dino_mil_complex_split_2_segmented_0729_1620/best_mil_model.pth"
#"mil/models/dino_mil_complex_split_3_segmented_0729_1102/best_mil_model.pth"
#"mil/models/dino_mil_complex_split_4_segmented_0729_1704/best_mil_model.pth"
#"mil/models/dino_mil_complex_split_5_segmented_0729_1102/best_mil_model.pth"
#)
#
#CLASSIFIER_PATHS=(
#"classifier/models/dino_complex_split_1_imagenet_0512_1423/best_model.pth"
#"classifier/models/dino_complex_split_2_imagenet_0512_1453/best_model.pth"
#"classifier/models/dino_complex_split_3_imagenet_0512_1522/best_model.pth"
#"classifier/models/dino_complex_split_4_imagenet_0512_1550/best_model.pth"
#"classifier/models/dino_complex_split_5_imagenet_0512_1619/best_model.pth"
#)

TRANSFORMER_PATHS=(
"transformer/models/dino_sa_complex_split_1_segmented_0729_1551/best_mil_model.pth"
"transformer/models/dino_sa_complex_split_2_segmented_0729_2058/best_mil_model.pth"
"transformer/models/dino_sa_complex_split_3_segmented_0730_0126/best_mil_model.pth"
"transformer/models/dino_sa_complex_split_4_segmented_0730_0601/best_mil_model.pth"
"transformer/models/dino_sa_complex_split_5_segmented_0730_1040/best_mil_model.pth"
)

python -u -m dino.vis_video_results \
  --dataset_name "paxos2020" \
  --annotations_path "data/stratified_splits" \
  --videos_path "data/ensemble_results_paxos2020/cleaned_videos" \
  --output_dir "data/multi_eval_paxos2020" \
  --mil_paths "${MIL_PATHS[@]}" \
  --classifier_paths "${CLASSIFIER_PATHS[@]}" \
  --trans_paths "${TRANSFORMER_PATHS[@]}" \
  --img_size 512 \
  --complex_augs

python -u -m dino.vis_video_results \
  --dataset_name "paxos2025" \
  --annotations_path "data/stratified_splits" \
  --videos_path "data/ensemble_results/cleaned_videos" \
  --output_dir "data/multi_eval_paxos2025" \
  --mil_paths "${MIL_PATHS[@]}" \
  --classifier_paths "${CLASSIFIER_PATHS[@]}" \
  --trans_paths "${TRANSFORMER_PATHS[@]}" \
  --img_size 512 \
  --complex_augs
