#!/bin/bash
#SBATCH -J swintest4
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1                 
#SBATCH -c 32           
#SBATCH --mem-per-cpu=4096         
#SBATCH --time=48:00:00

CUDA_VISIBLE_DEVICES=1

source /home/nathan/miniconda3/etc/profile.d/conda.sh

conda activate swinunetr2

# Run your Python script
python train.py /raid/test_3_2_torch/imagesTr/ "0" --c