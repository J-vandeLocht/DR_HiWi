#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A100medium
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

source ~/.bashrc
conda activate job_new
cd ..

python -u -m dino.train_dino_classifier \
  --split_path "data/stratified_splits/split_1" \
  --lr 5e-6 \
  --complex_augs

python -u -m dino.train_dino_classifier \
  --split_path "data/stratified_splits/split_2" \
  --lr 5e-6 \
  --complex_augs

python -u -m dino.train_dino_classifier \
  --split_path "data/stratified_splits/split_3" \
  --lr 5e-6 \
  --complex_augs

python -u -m dino.train_dino_classifier \
  --split_path "data/stratified_splits/split_4" \
  --lr 5e-6 \
  --complex_augs

python -u -m dino.train_dino_classifier \
  --split_path "data/stratified_splits/split_5" \
  --lr 5e-6 \
  --complex_augs
