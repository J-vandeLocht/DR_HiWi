#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A100medium
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

source ~/.bashrc
conda activate job_new
cd ..
python -u -m fine_tuning.train_dino --lr 5e-6
