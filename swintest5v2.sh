#!/bin/bash
#SBATCH -J swintest5v2
#SBATCH --gres=gpu:0
#SBATCH -N 1
#SBATCH -n 1                 
#SBATCH -c 32           
#SBATCH --mem-per-cpu=4096         
#SBATCH --time=48:00:00

CUDA_VISIBLE_DEVICES=0

source /home/nathan/miniconda3/etc/profile.d/conda.sh

conda activate swinunetr2

# Run your Python script
python train.py /raid/test_5_2_torch/imagesTr/ "0"