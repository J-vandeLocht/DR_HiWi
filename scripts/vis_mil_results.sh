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

REPO=dino/dinov3
WEIGHTS=dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
MODEL_PATHS=(
mil_runs/dino_mil_complex_0426_1242/best_mil_model.pth
mil_runs/dino_mil_complex_0426_1409/best_mil_model.pth
mil_runs/dino_mil_complex_0426_1535/best_mil_model.pth
mil_runs/dino_mil_complex_0426_1701/best_mil_model.pth
)

python -u -m dino.vis_mil_results \
  --repo_dir $REPO \
  --weight_path $WEIGHTS \
  --model_path "${MODEL_PATHS[@]}" \
  --use_cls \
  --img_size 512 \
  --complex_augs
