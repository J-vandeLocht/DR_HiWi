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

python -u -m classifier.train_classifier --weight_path '2026-04-15_13-19-56_Classifier_ep25_size512.pth'
python -u -m classifier.train_classifier --weight_path '2026-04-15_13-19-56_Classifier_ep25_size512.pth'
python -u -m classifier.train_classifier --weight_path '2026-04-15_13-19-56_Classifier_ep25_size512.pth'
python -u -m classifier.train_classifier --weight_path '2026-04-15_13-19-56_Classifier_ep25_size512.pth'

python -u -m classifier.train_classifier
python -u -m classifier.train_classifier
python -u -m classifier.train_classifier
python -u -m classifier.train_classifier
