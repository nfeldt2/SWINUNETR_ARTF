#!/usr/bin/env bash
set -euo pipefail

# Activate conda environment for SwinUNETR
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate swinunetr2

# Base output path
BASE_OUT=/hot/4DCT_datasets/SyntheticArtifactsDatasets_v2/Results

# Device
DEVICE=cuda:3

# Model checkpoints
S1_MODEL=/home/user/Nathan/Desktop/modif/SWINUNETR_ARTF/results_segmentation_S1/run_ep50_bs3_lr0.0001_optimadamw/checkpoint_latest.pt
S2_MODEL=/home/user/Nathan/Desktop/modif/SWINUNETR_ARTF/results_segmentation_S2/run_ep50_bs3_lr0.0001_optimadamw/checkpoint_latest.pt
S3_MODEL=/home/user/Nathan/Desktop/modif/SWINUNETR_ARTF/results_segmentation_S3/run_ep50_bs3_lr0.0001_optimadamw/checkpoint_latest.pt

# Limits
LIMIT_S2=1107
LIMIT_S1=1124
LIMIT_S3=1084

# Function to run prediction
run_pred() {
    local variant=$1
    local model_path=$2
    local win=$3
    local inp_dir=$4
    local out_dir=$5
    local limit=$6

    echo "Running ${variant} on ${inp_dir}, window ${win}, limit ${limit}"
    python predictArtf.py \
        --model_path ${model_path} \
        --device ${DEVICE} \
        --window_size ${win} \
        --overlap 0.25 \
        --input_dir ${inp_dir} \
        --output_dir ${BASE_OUT}/${out_dir} \
        --limit ${limit}
    echo "Completed ${variant} for ${inp_dir}";
    echo
}

## S2 test and train (default models)
run_pred S2_test  ${S2_MODEL} "64 224 256" "/raid/addedArtifacts_S2/test"  "swin_test_S2"  ${LIMIT_S2} &
run_pred S2_train ${S2_MODEL} "64 224 256" "/raid/addedArtifacts_S2/train" "swin_train_S2" ${LIMIT_S2} &

## S2 ROI models (batch size 4)
run_pred S2_ROI_test  ${S2_MODEL} "64 192 224" "/raid/addedArtifacts_S2/test"  "swin_test_S2_ROI"  ${LIMIT_S2} &
run_pred S2_ROI_train ${S2_MODEL} "64 192 224" "/raid/addedArtifacts_S2/train" "swin_train_S2_ROI" ${LIMIT_S2} &

## S2 LR models (batch size 5)
run_pred S2_LR_test  ${S2_MODEL} "64 160 192" "/raid/addedArtifacts_S2/test"  "swin_test_S2_LR"  ${LIMIT_S2} &
run_pred S2_LR_train ${S2_MODEL} "64 160 192" "/raid/addedArtifacts_S2/train" "swin_train_S2_LR" ${LIMIT_S2} &

## S1 test and train (uses S2 model, default bs3)
run_pred S1_test  ${S2_MODEL} "64 224 256" "/raid/addedArtifacts_S1/test"  "swin_test_S1"  ${LIMIT_S1} &
run_pred S1_train ${S2_MODEL} "64 224 256" "/raid/addedArtifacts_S1/train" "swin_train_S1" ${LIMIT_S1} &

## S1 ROI (bs4)
run_pred S1_ROI_test  ${S2_MODEL} "64 192 224" "/raid/addedArtifacts_S1/test"  "swin_test_S1_ROI"  ${LIMIT_S1} &
run_pred S1_ROI_train ${S2_MODEL} "64 192 224" "/raid/addedArtifacts_S1/train" "swin_train_S1_ROI" ${LIMIT_S1} &

## S1 LR (bs5)
run_pred S1_LR_test  ${S2_MODEL} "64 160 192" "/raid/addedArtifacts_S1/test"  "swin_test_S1_LR"  ${LIMIT_S1} &
run_pred S1_LR_train ${S2_MODEL} "64 160 192" "/raid/addedArtifacts_S1/train" "swin_train_S1_LR" ${LIMIT_S1} &

## S3 test and train (default)
run_pred S3_test  ${S3_MODEL} "64 224 256" "/raid/addedArtifacts_S3/test"  "swin_test_S3"  ${LIMIT_S3} &
run_pred S3_train ${S3_MODEL} "64 224 256" "/raid/addedArtifacts_S3/train" "swin_train_S3" ${LIMIT_S3} &

## S3 ROI (bs4)
run_pred S3_ROI_test  ${S3_MODEL} "64 192 224" "/raid/addedArtifacts_S3/test"  "swin_test_S3_ROI"  ${LIMIT_S3} &
run_pred S3_ROI_train ${S3_MODEL} "64 192 224" "/raid/addedArtifacts_S3/train" "swin_train_S3_ROI" ${LIMIT_S3} &

## S3 LR (bs5)
run_pred S3_LR_test  ${S3_MODEL} "64 160 192" "/raid/addedArtifacts_S3/test"  "swin_test_S3_LR"  ${LIMIT_S3} &
run_pred S3_LR_train ${S3_MODEL} "64 160 192" "/raid/addedArtifacts_S3/train" "swin_train_S3_LR" ${LIMIT_S3} &

## TrueArtifacts predictions for all model configs
# S2 default TrueArtifacts
run_pred S2_trueArt_test  ${S2_MODEL} "64 224 256" "/raid/trueArtifacts"  "swin_trueArt_S2"  ${LIMIT_S2} &
run_pred S2_trueArt_train ${S2_MODEL} "64 224 256" "/raid/trueArtifacts" "swin_trueArt_S2" ${LIMIT_S2} &

# S2 ROI TrueArtifacts
run_pred S2_ROI_trueArt_test  ${S2_MODEL} "64 192 224" "/raid/trueArtifacts"  "swin_trueArt_S2_ROI"  ${LIMIT_S2} &
run_pred S2_ROI_trueArt_train ${S2_MODEL} "64 192 224" "/raid/trueArtifacts" "swin_trueArt_S2_ROI" ${LIMIT_S2} &

# S2 LR TrueArtifacts
run_pred S2_LR_trueArt_test  ${S2_MODEL} "64 160 192" "/raid/trueArtifacts"  "swin_trueArt_S2_LR"  ${LIMIT_S2} &
run_pred S2_LR_trueArt_train ${S2_MODEL} "64 160 192" "/raid/trueArtifacts" "swin_trueArt_S2_LR" ${LIMIT_S2} &

# S1 default TrueArtifacts
run_pred S1_trueArt_test  ${S2_MODEL} "64 224 256" "/raid/trueArtifacts"  "swin_trueArt_S1"  ${LIMIT_S1} &
run_pred S1_trueArt_train ${S2_MODEL} "64 224 256" "/raid/trueArtifacts" "swin_trueArt_S1" ${LIMIT_S1} &

# S1 ROI TrueArtifacts
run_pred S1_ROI_trueArt_test  ${S2_MODEL} "64 192 224" "/raid/trueArtifacts"  "swin_trueArt_S1_ROI"  ${LIMIT_S1} &
run_pred S1_ROI_trueArt_train ${S2_MODEL} "64 192 224" "/raid/trueArtifacts" "swin_trueArt_S1_ROI" ${LIMIT_S1} &

# S1 LR TrueArtifacts
run_pred S1_LR_trueArt_test  ${S2_MODEL} "64 160 192" "/raid/trueArtifacts"  "swin_trueArt_S1_LR"  ${LIMIT_S1} &
run_pred S1_LR_trueArt_train ${S2_MODEL} "64 160 192" "/raid/trueArtifacts" "swin_trueArt_S1_LR" ${LIMIT_S1} &

# S3 default TrueArtifacts
run_pred S3_trueArt_test  ${S3_MODEL} "64 224 256" "/raid/trueArtifacts"  "swin_trueArt_S3"  ${LIMIT_S3} &
run_pred S3_trueArt_train ${S3_MODEL} "64 224 256" "/raid/trueArtifacts" "swin_trueArt_S3" ${LIMIT_S3} &

# S3 ROI TrueArtifacts
run_pred S3_ROI_trueArt_test  ${S3_MODEL} "64 192 224" "/raid/trueArtifacts"  "swin_trueArt_S3_ROI"  ${LIMIT_S3} &
run_pred S3_ROI_trueArt_train ${S3_MODEL} "64 192 224" "/raid/trueArtifacts" "swin_trueArt_S3_ROI" ${LIMIT_S3} &

# S3 LR TrueArtifacts
run_pred S3_LR_trueArt_test  ${S3_MODEL} "64 160 192" "/raid/trueArtifacts"  "swin_trueArt_S3_LR"  ${LIMIT_S3} &
run_pred S3_LR_trueArt_train ${S3_MODEL} "64 160 192" "/raid/trueArtifacts" "swin_trueArt_S3_LR" ${LIMIT_S3} &

wait
echo "All inference tasks completed." 