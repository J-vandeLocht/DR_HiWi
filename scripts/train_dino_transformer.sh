#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --nodelist=node-[02-03]
#SBATCH --cpus-per-task=8

source ~/.bashrc
conda activate job_new
cd ..

#CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_1_imagenet_0725_0545/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_1" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset \
#  --use_pos_embedding
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_2_imagenet_0725_0715/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_2" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset \
#  --use_pos_embedding
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_3_imagenet_0725_0846/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_3" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset \
#  --use_pos_embedding
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_4_imagenet_0725_1016/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_4" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset \
#  --use_pos_embedding
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_5_imagenet_0725_1142/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_5" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset \
#  --use_pos_embedding

#CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_1_imagenet_0725_0545/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_1" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset \
#  --num_blocks 4

#CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_2_imagenet_0725_0715/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_2" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset \
#  --num_blocks 4

CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_3_imagenet_0725_0846/best_model.pth
python -u -m dino.train_dino_transformer \
  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
  --split_path "data/stratified_splits/split_3" \
  --freeze_backbone \
  --lr 1e-4 \
  --epochs 5 \
  --lr_step 3 \
  --complex_augs \
  --random_segment_sample \
  --fused_dataset \
  --num_blocks 4

#CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_4_imagenet_0725_1016/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_4" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset \
#  --num_blocks 4

#CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_5_imagenet_0725_1142/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_5" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --fused_dataset \
#  --num_blocks 4
