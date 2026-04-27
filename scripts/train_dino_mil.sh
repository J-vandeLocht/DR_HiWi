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

REPO=dino/dinov3
WEIGHTS=dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

# -----------------------------
# RUN 2: Complex Augementations
# -----------------------------

CLASSIFIER_WEIGHTS=classifier/dino_complex_0424_1900/best_model.pth
python -u -m dino.train_dino_mil \
  --repo_dir $REPO \
  --weight_path $WEIGHTS \
  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
  --freeze_backbone \
  --use_cls \
  --img_size 512 \
  --lr 1e-4 \
  --epochs 5 \
  --complex_augs

CLASSIFIER_WEIGHTS=classifier/dino_complex_0424_1920/best_model.pth
python -u -m dino.train_dino_mil \
  --repo_dir $REPO \
  --weight_path $WEIGHTS \
  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
  --freeze_backbone \
  --use_cls \
  --img_size 512 \
  --lr 1e-4 \
  --epochs 5 \
  --complex_augs

CLASSIFIER_WEIGHTS=classifier/dino_complex_0424_1940/best_model.pth
python -u -m dino.train_dino_mil \
  --repo_dir $REPO \
  --weight_path $WEIGHTS \
  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
  --freeze_backbone \
  --use_cls \
  --img_size 512 \
  --lr 1e-4 \
  --epochs 5 \
  --complex_augs

CLASSIFIER_WEIGHTS=classifier/dino_complex_0424_2000/best_model.pth
python -u -m dino.train_dino_mil \
  --repo_dir $REPO \
  --weight_path $WEIGHTS \
  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
  --freeze_backbone \
  --use_cls \
  --img_size 512 \
  --lr 1e-4 \
  --epochs 5 \
  --complex_augs
