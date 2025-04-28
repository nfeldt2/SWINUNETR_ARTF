#!/usr/bin/env bash
# Exit on error, undefined var, or pipefail
set -euo pipefail

# Activate the conda environment for SwinUNETR (ensure conda is initialized)
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate swinunetr2

# Script to run predictArtf.py across different model checkpoints and window sizes
# Assumes this script is run from the repository root

# S1: use S2 model with 3/4 window size
echo "Running S1 (3/4 window)"
python predictArtf.py \
    --model_path results_segmentation_S2/run_ep200_bs6_lr0.0001_optimadamw/checkpoint_latest.pt \
    --device cuda:2 \
    --window_size 48 144 168 \
    --overlap 0.25 \
    --output_dir /hot/4DCT_datasets/SyntheticArtifactsDatasets_v2/Results/swin_trueArt_S1_ROI

# S1_LR: use S2 model with 2/5 window size
echo "Running S1_LR (2/5 window)"
python predictArtf.py \
    --model_path results_segmentation_S2/run_ep200_bs6_lr0.0001_optimadamw/checkpoint_latest.pt \
    --device cuda:2 \
    --window_size 26 77 90 \
    --overlap 0.25 \
    --output_dir /hot/4DCT_datasets/SyntheticArtifactsDatasets_v2/Results/swin_trueArt_S1_LR

# S2: default window, save to /hot/.../swin_trueArt_S2_ROI
echo "Running S2 (default window)"
python predictArtf.py \
    --model_path results_segmentation_S2/run_ep200_bs6_lr0.0001_optimadamw/checkpoint_latest.pt \
    --device cuda:2 \
    --window_size 64 192 224 \
    --overlap 0.25 \
    --output_dir /hot/4DCT_datasets/SyntheticArtifactsDatasets_v2/Results/swin_trueArt_S2_ROI

# S2_LR: use S2 model with half window size, save to swin_trueArt_S2_LR
echo "Running S2_LR (half window)"
python predictArtf.py \
    --model_path results_segmentation_S2/run_ep200_bs6_lr0.0001_optimadamw/checkpoint_latest.pt \
    --device cuda:2 \
    --window_size 32 96 112 \
    --overlap 0.25 \
    --output_dir /hot/4DCT_datasets/SyntheticArtifactsDatasets_v2/Results/swin_trueArt_S2_LR

# S3: default window using S3 model
echo "Running S3 (default window)"
python predictArtf.py \
    --model_path results_segmentation_S3/run_ep200_bs6_lr0.0001_optimadamw/checkpoint_best.pt \
    --device cuda:2 \
    --window_size 64 192 224 \
    --overlap 0.25 \
    --output_dir /hot/4DCT_datasets/SyntheticArtifactsDatasets_v2/Results/swin_trueArt_S3_ROI

# S3_LR: use S3 model with half window size
echo "Running S3_LR (half window)"
python predictArtf.py \
    --model_path results_segmentation_S3/run_ep200_bs6_lr0.0001_optimadamw/checkpoint_best.pt \
    --device cuda:2 \
    --window_size 32 96 112 \
    --overlap 0.25 \
    --output_dir /hot/4DCT_datasets/SyntheticArtifactsDatasets_v2/Results/swin_trueArt_S3_LR

echo "All predictions completed." 