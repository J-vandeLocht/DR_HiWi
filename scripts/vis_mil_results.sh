#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --nodelist=node-04
#SBATCH --cpus-per-task=8

source ~/.bashrc
conda activate job_new
cd ..

#MODEL_PATHS=(
#"mil/models/dino_mil_complex_split_1_segmented_0512_1705/best_mil_model.pth"
#"mil/models/dino_mil_complex_split_2_segmented_0512_1824/best_mil_model.pth"
#"mil/models/dino_mil_complex_split_3_segmented_0512_1945/best_mil_model.pth"
#"mil/models/dino_mil_complex_split_4_segmented_0512_2107/best_mil_model.pth"
#"mil/models/dino_mil_complex_split_5_segmented_0514_1205/best_mil_model.pth"
#)
MODEL_PATHS=(
"mil/models/dino_mil_complex_split_5_segmented_0514_1205/best_mil_model.pth"
)

python -u -m dino.vis_mil_results \
  --model_path "${MODEL_PATHS[@]}" \
  --img_size 512 \
  --complex_augs
