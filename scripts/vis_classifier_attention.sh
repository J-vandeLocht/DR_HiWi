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

CLASSIFIER_WEIGHTS=classifier/models/dino_complex_split_1_imagenet_0512_1423/best_model.pth
python -u -m dino.vis_classifier_attention \
  --input_dir "data/2024_Paxos_Frames/cropped_frames" \
  --weights $CLASSIFIER_WEIGHTS \
  --complex_augs
