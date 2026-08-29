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

#python -u -m dino.train_dino_classifier \
#  --split_path "data/stratified_splits/split_1" \
#  --train_json "frame_train_2020.json" \
#  --val_json "frame_val_2020.json" \
#  --lr 5e-6 \
#  --complex_augs
#
#python -u -m dino.train_dino_classifier \
#  --split_path "data/stratified_splits/split_2" \
#  --train_json "frame_train_2020.json" \
#  --val_json "frame_val_2020.json" \
#  --lr 5e-6 \
#  --complex_augs
#
#python -u -m dino.train_dino_classifier \
#  --split_path "data/stratified_splits/split_3" \
#  --train_json "frame_train_2020.json" \
#  --val_json "frame_val_2020.json" \
#  --lr 5e-6 \
#  --complex_augs

#python -u -m dino.train_dino_classifier \
#  --split_path "data/stratified_splits/split_4" \
#  --train_json "frame_train_2020.json" \
#  --val_json "frame_val_2020.json" \
#  --lr 5e-6 \
#  --complex_augs
#
#python -u -m dino.train_dino_classifier \
#  --split_path "data/stratified_splits/split_5" \
#  --train_json "frame_train_2020.json" \
#  --val_json "frame_val_2020.json" \
#  --lr 5e-6 \
#  --complex_augs

#python -u -m dino.train_dino_classifier \
#  --split_path "data/stratified_splits/split_1" \
#  --train_json "frame_train_2025.json" \
#  --val_json "frame_val_2025.json" \
#  --lr 5e-6 \
#  --complex_augs
#
#python -u -m dino.train_dino_classifier \
#  --split_path "data/stratified_splits/split_2" \
#  --train_json "frame_train_2025.json" \
#  --val_json "frame_val_2025.json" \
#  --lr 5e-6 \
#  --complex_augs

#python -u -m dino.train_dino_classifier \
#  --split_path "data/stratified_splits/split_3" \
#  --train_json "frame_train_2025.json" \
#  --val_json "frame_val_2025.json" \
#  --lr 5e-6 \
#  --complex_augs
#
#python -u -m dino.train_dino_classifier \
#  --split_path "data/stratified_splits/split_4" \
#  --train_json "frame_train_2025.json" \
#  --val_json "frame_val_2025.json" \
#  --lr 5e-6 \
#  --complex_augs

python -u -m dino.train_dino_classifier \
  --split_path "data/stratified_splits/split_5" \
  --train_json "frame_train_2025.json" \
  --val_json "frame_val_2025.json" \
  --lr 5e-6 \
  --complex_augs
