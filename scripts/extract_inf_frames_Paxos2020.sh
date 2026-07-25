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
python -u -m informative_frames.extract_inf_frames \
  --gpu_id 3 \
  --total_gpus 4 \
  --video_dir "experiments/Paxos_2020/data" \
  --seg_model_dir "experiments/Paxos_2020/cropping/models" \
  --cls_model_dir "informative_frames/models" \
  --out_base "experiments/Paxos_2020/data/ensemble_results"

