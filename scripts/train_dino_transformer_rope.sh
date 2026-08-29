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

#CLASSIFIER_WEIGHTS=classifier/models/dino_split_1_0811_1512/best_model.pth
#python -u -m dino.train_dino_transformer_rope \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_1" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset \
#  --use_rope

#CLASSIFIER_WEIGHTS=classifier/models/dino_split_2_0811_1634/best_model.pth
#python -u -m dino.train_dino_transformer_rope \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_2" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset \
#  --use_rope
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_split_3_0811_1756/best_model.pth
#python -u -m dino.train_dino_transformer_rope \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_3" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset \
#  --use_rope
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_split_4_0811_1918/best_model.pth
#python -u -m dino.train_dino_transformer_rope \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_4" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset \
#  --use_rope

CLASSIFIER_WEIGHTS=classifier/models/dino_split_5_0811_2040/best_model.pth
python -u -m dino.train_dino_transformer_rope \
  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
  --split_path "data/stratified_splits/split_5" \
  --freeze_backbone \
  --lr 1e-4 \
  --epochs 5 \
  --lr_step 3 \
  --complex_augs \
  --random_segment_sample \
  --fused_dataset \
  --use_rope
