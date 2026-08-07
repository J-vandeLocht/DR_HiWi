#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

source ~/.bashrc
conda activate job_new
cd ..
python -u -m informative_frames.extract_inf_frames \
  --gpu_id 0 \
  --total_gpus 1 \
  --seg_model_dir "cropping/models" \
  --cls_model_dir "informative_frames/models" \
  --video_dir "data/dr_videos" \
  --out_base "data/ensemble_results_paxos2025"
