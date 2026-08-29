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

#CLASSIFIER_WEIGHTS=classifier/models/dino_split_1_0815_1421_2020/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_1" \
#  --train_json "mil_train_2020.json" \
#  --val_json "mil_val_2020.json" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_split_2_0815_1509_2020/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_2" \
#  --train_json "mil_train_2020.json" \
#  --val_json "mil_val_2020.json" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset

#CLASSIFIER_WEIGHTS=classifier/models/dino_split_3_0815_1556_2020/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_3" \
#  --train_json "mil_train_2020.json" \
#  --val_json "mil_val_2020.json" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_split_4_0815_1719_2020/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_4" \
#  --train_json "mil_train_2020.json" \
#  --val_json "mil_val_2020.json" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_split_5_0815_1805_2020/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_5" \
#  --train_json "mil_train_2020.json" \
#  --val_json "mil_val_2020.json" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_split_1_0815_1719_2025/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_1" \
#  --train_json "mil_train_2025.json" \
#  --val_json "mil_val_2025.json" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_split_2_0815_1751_2025/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_2" \
#  --train_json "mil_train_2025.json" \
#  --val_json "mil_val_2025.json" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_split_3_0815_1720_2025/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_3" \
#  --train_json "mil_train_2025.json" \
#  --val_json "mil_val_2025.json" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset
#
CLASSIFIER_WEIGHTS=classifier/models/dino_split_4_0815_1751_2025/best_model.pth
python -u -m dino.train_dino_transformer \
  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
  --split_path "data/stratified_splits/split_4" \
  --train_json "mil_train_2025.json" \
  --val_json "mil_val_2025.json" \
  --freeze_backbone \
  --lr 1e-4 \
  --epochs 5 \
  --lr_step 3 \
  --complex_augs \
  --random_segment_sample \
  --fused_dataset

CLASSIFIER_WEIGHTS=classifier/models/dino_split_5_0815_1720_2025/best_model.pth
python -u -m dino.train_dino_transformer \
  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
  --split_path "data/stratified_splits/split_5" \
  --train_json "mil_train_2025.json" \
  --val_json "mil_val_2025.json" \
  --freeze_backbone \
  --lr 1e-4 \
  --epochs 5 \
  --lr_step 3 \
  --complex_augs \
  --random_segment_sample \
  --fused_dataset
