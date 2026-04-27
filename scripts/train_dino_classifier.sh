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

python -u -m dino.train_dino_classifier \
  --repo_dir $REPO \
  --weight_path $WEIGHTS \
  --use_cls \
  --img_size 512 \
  --batch_size 16 \
  --lr 5e-6 \
  --epochs 10 \
  --complex_augs

python -u -m dino.train_dino_classifier \
  --repo_dir $REPO \
  --weight_path $WEIGHTS \
  --use_cls \
  --img_size 512 \
  --batch_size 16 \
  --lr 5e-6 \
  --epochs 10 \
  --complex_augs

python -u -m dino.train_dino_classifier \
  --repo_dir $REPO \
  --weight_path $WEIGHTS \
  --use_cls \
  --img_size 512 \
  --batch_size 16 \
  --lr 5e-6 \
  --epochs 10 \
  --complex_augs

python -u -m dino.train_dino_classifier \
  --repo_dir $REPO \
  --weight_path $WEIGHTS \
  --use_cls \
  --img_size 512 \
  --batch_size 16 \
  --lr 5e-6 \
  --epochs 10 \
  --complex_augs

python -u -m dino.train_dino_classifier \
  --repo_dir $REPO \
  --weight_path $WEIGHTS \
  --use_cls \
  --img_size 512 \
  --batch_size 16 \
  --lr 5e-6 \
  --epochs 10

python -u -m dino.train_dino_classifier \
  --repo_dir $REPO \
  --weight_path $WEIGHTS \
  --use_cls \
  --img_size 512 \
  --batch_size 16 \
  --lr 5e-6 \
  --epochs 10

python -u -m dino.train_dino_classifier \
  --repo_dir $REPO \
  --weight_path $WEIGHTS \
  --use_cls \
  --img_size 512 \
  --batch_size 16 \
  --lr 5e-6 \
  --epochs 10

python -u -m dino.train_dino_classifier \
  --repo_dir $REPO \
  --weight_path $WEIGHTS \
  --use_cls \
  --img_size 512 \
  --batch_size 16 \
  --lr 5e-6 \
  --epochs 10

## -----------------------------
## RUN 1: frozen + CLS
## -----------------------------
#python -u -m dino.train_dino_classifier \
#  --repo_dir $REPO \
#  --weight_path $WEIGHTS \
#  --freeze_backbone \
#  --use_cls \
#  --img_size 512 \
#  --batch_size 32 \
#  --lr 1e-3 \
#  --epochs 10
#
## -----------------------------
## RUN 2: frozen + mean pooling
## -----------------------------
#python -u -m dino.train_dino_classifier \
#  --repo_dir $REPO \
#  --weight_path $WEIGHTS \
#  --freeze_backbone \
#  --img_size 512 \
#  --batch_size 32 \
#  --lr 1e-3 \
#  --epochs 10
#
## -----------------------------
## RUN 3: finetune + CLS
## -----------------------------
#python -u -m dino.train_dino_classifier \
#  --repo_dir $REPO \
#  --weight_path $WEIGHTS \
#  --use_cls \
#  --img_size 512 \
#  --batch_size 16 \
#  --lr 5e-6 \
#  --epochs 10
#
## -----------------------------
## RUN 4: finetune + mean pooling
## -----------------------------
#python -u -m dino.train_dino_classifier \
#  --repo_dir $REPO \
#  --weight_path $WEIGHTS \
#  --img_size 512 \
#  --batch_size 16 \
#  --lr 5e-6 \
#  --epochs 10
