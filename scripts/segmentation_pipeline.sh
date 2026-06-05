#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A100short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

source ~/.bashrc
conda activate job_new
cd ..

python -u -m cropping.segmentation_pipeline
