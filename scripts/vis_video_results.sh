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

MIL_PATHS=(
"mil/models/dino_mil_complex_split_1_segmented_0512_1705/best_mil_model.pth"
"mil/models/dino_mil_complex_split_2_segmented_0512_1824/best_mil_model.pth"
"mil/models/dino_mil_complex_split_3_segmented_0512_1945/best_mil_model.pth"
"mil/models/dino_mil_complex_split_4_segmented_0512_2107/best_mil_model.pth"
"mil/models/dino_mil_complex_split_5_segmented_0620_1709/best_mil_model.pth"
)

CLASSIFIER_PATHS=(
"classifier/models/dino_complex_split_1_imagenet_0512_1423/best_model.pth"
"classifier/models/dino_complex_split_2_imagenet_0512_1453/best_model.pth"
"classifier/models/dino_complex_split_3_imagenet_0512_1522/best_model.pth"
"classifier/models/dino_complex_split_4_imagenet_0512_1550/best_model.pth"
"classifier/models/dino_complex_split_5_imagenet_0512_1619/best_model.pth"
)

TRANSFORMER_PATHS=(
"transformer/models/dino_sa_complex_split_1_segmented_0619_1814/best_mil_model.pth"
"transformer/models/dino_sa_complex_split_2_segmented_0619_1935/best_mil_model.pth"
"transformer/models/dino_sa_complex_split_3_segmented_0619_2056/best_mil_model.pth"
"transformer/models/dino_sa_complex_split_4_segmented_0619_2225/best_mil_model.pth"
"transformer/models/dino_sa_complex_split_5_segmented_0619_2357/best_mil_model.pth"
)

python -u -m dino.vis_video_results \
  --output_dir "data/multi_eval_paxos2020" \
  --mil_paths "${MIL_PATHS[@]}" \
  --classifier_paths "${CLASSIFIER_PATHS[@]}" \
  --trans_paths "${TRANSFORMER_PATHS[@]}" \
  --img_size 512 \
  --complex_augs
