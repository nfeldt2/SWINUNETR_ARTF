#!/usr/bin/env bash
set -euo pipefail

# Activate conda environment
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate swinunetr2

# Common training parameters
COMMON_ARGS=(
    --epochs 50
    --batch_size 3
    --initial_lr 1e-4
    --optimizer adamw
    --weight_decay 1e-5
    --feature_size 12
    --min_artifact_pixels 5
    --foreground_prob 0.8
    --train_steps_per_epoch 500
)

# 1) Base models: S1, S2, S3 on GPUs 1,2,3 in parallel
python train_segmentation.py /raid/addedArtifacts_S1 \
    --output_dir ./results_segmentation_S1 \
    --device 1 "${COMMON_ARGS[@]}" &
python train_segmentation.py /raid/addedArtifacts_S2 \
    --output_dir ./results_segmentation_S2 \
    --device 2 "${COMMON_ARGS[@]}" &
python train_segmentation.py /raid/addedArtifacts_S3 \
    --output_dir ./results_segmentation_S3 \
    --device 3 "${COMMON_ARGS[@]}" &
wait

echo "Base model training complete."

# 2) ROI models (input_size [64,192,224]) in parallel
ROI_SIZE=(64 192 224)

python train_segmentation.py /raid/addedArtifacts_S1 \
    --output_dir ./results_segmentation_S1_ROI \
    --device 3 --input_size "${ROI_SIZE[@]}" "${COMMON_ARGS[@]}" &
python train_segmentation.py /raid/addedArtifacts_S2 \
    --output_dir ./results_segmentation_S2_ROI \
    --device 0 --input_size "${ROI_SIZE[@]}" "${COMMON_ARGS[@]}" &
python train_segmentation.py /raid/addedArtifacts_S3 \
    --output_dir ./results_segmentation_S3_ROI \
    --device 1 --input_size "${ROI_SIZE[@]}" "${COMMON_ARGS[@]}" &
wait

echo "ROI model training complete."

# 3) LR models (input_size [64,160,192]) in parallel on remaining GPUs
LR_SIZE=(64 160 192)

python train_segmentation.py /raid/addedArtifacts_S1 \
    --output_dir ./results_segmentation_S1_LR \
    --device 2 --input_size "${LR_SIZE[@]}" "${COMMON_ARGS[@]}" &
python train_segmentation.py /raid/addedArtifacts_S2 \
    --output_dir ./results_segmentation_S2_LR \
    --device 0 --input_size "${LR_SIZE[@]}" "${COMMON_ARGS[@]}" &
python train_segmentation.py /raid/addedArtifacts_S3 \
    --output_dir ./results_segmentation_S3_LR \
    --device 1 --input_size "${LR_SIZE[@]}" "${COMMON_ARGS[@]}" &
wait

echo "LR model training complete."

# 4) Inference using run_predict_all.sh
 echo "Starting inference tasks..."
 bash run_predict_all.sh
 echo "All inference tasks completed."

# Done
 echo "All training tasks completed." 