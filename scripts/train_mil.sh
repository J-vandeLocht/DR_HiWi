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

python -u -m mil.train_mil --model 'b4' --epochs 10 --lr 0.00001 --lr_step 3 --frame_model_path 'classifier/b4_lr0.0001_bs32_sz512_0416_1718/best_model.pth'
python -u -m mil.train_mil --model 'b4' --epochs 10 --lr 0.000001 --lr_step 3 --frame_model_path 'classifier/b4_lr0.0001_bs32_sz512_0416_1718/best_model.pth'

python -u -m mil.train_mil --model 'b4' --epochs 10 --lr 0.00001 --lr_step 3
python -u -m mil.train_mil --model 'b4' --epochs 10 --lr 0.000001 --lr_step 3
