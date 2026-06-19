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

python -u -m temp.extract_inf_frames --gpu_id 3 --total_gpus 4
