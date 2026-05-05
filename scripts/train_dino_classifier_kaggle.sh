#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --nodelist=node-03
#SBATCH --cpus-per-task=8

source ~/.bashrc
conda activate job_new
cd ..

python -u -m dino.train_dino_classifier \
  --split_path "data/stratified_splits/split_1" \
  --weight_path "fine_tuning/2026-05-01_12-46-50_DINOv3_ep15_size512_unfrozen/best_weights.pth" \
  --lr 5e-6 \
  --complex_augs

python -u -m dino.train_dino_classifier \
  --split_path "data/stratified_splits/split_1" \
  --weight_path "fine_tuning/2026-05-01_12-46-50_DINOv3_ep15_size512_unfrozen/best_weights.pth" \
  --lr 5e-6 \
  --complex_augs

python -u -m dino.train_dino_classifier \
  --split_path "data/stratified_splits/split_1" \
  --weight_path "fine_tuning/2026-05-01_12-46-50_DINOv3_ep15_size512_unfrozen/best_weights.pth" \
  --lr 5e-6 \
  --complex_augs

python -u -m dino.train_dino_classifier \
  --split_path "data/stratified_splits/split_1" \
  --weight_path "fine_tuning/2026-05-01_12-46-50_DINOv3_ep15_size512_unfrozen/best_weights.pth" \
  --lr 5e-6 \
  --complex_augs
