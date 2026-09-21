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
  --lr 5e-6 \
  --complex_augs \
  --binary_classification

python -u -m dino.train_dino_classifier \
  --split_path "data/stratified_splits/split_2" \
  --lr 5e-6 \
  --complex_augs \
  --binary_classification

python -u -m dino.train_dino_classifier \
  --split_path "data/stratified_splits/split_3" \
  --lr 5e-6 \
  --complex_augs \
  --binary_classification

#python -u -m dino.train_dino_classifier \
#  --split_path "data/stratified_splits/split_4" \
#  --lr 5e-6 \
#  --complex_augs \
#  --binary_classification
#
#python -u -m dino.train_dino_classifier \
#  --split_path "data/stratified_splits/split_5" \
#  --lr 5e-6 \
#  --complex_augs \
#  --binary_classification
