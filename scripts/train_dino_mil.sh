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

#CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_1_kaggle_0502_1258/best_model.pth
#python -u -m dino.train_dino_mil \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_1" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_1_kaggle_0502_1828/best_model.pth
#python -u -m dino.train_dino_mil \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_1" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_1_kaggle_0502_1856/best_model.pth
#python -u -m dino.train_dino_mil \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_1" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_1_kaggle_0502_1925/best_model.pth
#python -u -m dino.train_dino_mil \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_1" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs

CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_2/best_model.pth
python -u -m dino.train_dino_mil \
  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
  --split_path "data/stratified_splits/split_2" \
  --freeze_backbone \
  --lr 1e-4 \
  --epochs 5 \
  --lr_step 3 \
  --complex_augs \
  --random_segment_sample

CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_2/best_model.pth
python -u -m dino.train_dino_mil \
  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
  --split_path "data/stratified_splits/split_2" \
  --freeze_backbone \
  --lr 1e-4 \
  --epochs 5 \
  --lr_step 3 \
  --complex_augs \
  --random_segment_sample

CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_2/best_model.pth
python -u -m dino.train_dino_mil \
  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
  --split_path "data/stratified_splits/split_2" \
  --freeze_backbone \
  --lr 1e-4 \
  --epochs 5 \
  --lr_step 3 \
  --complex_augs \
  --random_segment_sample

CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_2/best_model.pth
python -u -m dino.train_dino_mil \
  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
  --split_path "data/stratified_splits/split_2" \
  --freeze_backbone \
  --lr 1e-4 \
  --epochs 5 \
  --lr_step 3 \
  --complex_augs \
  --random_segment_sample

#CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_2/best_model.pth
#python -u -m dino.train_dino_mil \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_2" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs

#CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_3/best_model.pth
#python -u -m dino.train_dino_mil \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_3" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_4/best_model.pth
#python -u -m dino.train_dino_mil \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_4" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_5/best_model.pth
#python -u -m dino.train_dino_mil \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_5" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs
