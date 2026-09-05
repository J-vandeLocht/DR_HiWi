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
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_all_split_1_0829_2040_full/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_1" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset

#CLASSIFIER_WEIGHTS=classifier/models/dino_all_split_2_0829_2158_full/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_2" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_all_split_3_0829_2316_full/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_3" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_all_split_4_0829_2040_full/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_4" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset
#
CLASSIFIER_WEIGHTS=classifier/models/dino_all_split_5_0829_2158_full/best_model.pth
python -u -m dino.train_dino_transformer \
  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
  --split_path "data/stratified_splits/split_5" \
  --freeze_backbone \
  --lr 1e-4 \
  --epochs 5 \
  --lr_step 3 \
  --complex_augs \
  --random_segment_sample \
  --fused_dataset
