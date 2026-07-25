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
python -u -m informative_frames.extract_inf_frames \
  --gpu_id 3 \
  --total_gpus 4 \
  --video_dir "data/Paxos_2020_videos" \
  --out_base "data/ensemble_results_paxos2020"
