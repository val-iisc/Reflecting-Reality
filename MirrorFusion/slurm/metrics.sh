#!/bin/bash
#SBATCH --job-name=metrics_brushnet
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH -w cn7

module load conda/24.1.2 cuda/cuda12.4

# first compute the selection metric for all 4 seed validation images
accelerate launch \
    metrics/evaluate_metrics.py \
    --infer_dir runs/logs/sd15_depth_20percent_drop_constant_auto_caption_train_unet/checkpoint-17000/inference \
    --mode calc \
    --metrics mask_SSIM

# compute required metrics based on best selected image. You can add IoU and obj also.
accelerate launch \
    metrics/evaluate_metrics.py \
    --infer_dir runs/logs/sd15_depth_20percent_drop_constant_auto_caption_train_unet/checkpoint-17000/inference \
    --mode best \
    --metrics mask full mirror text_align

# calling using python metrics/evaluate_metrics.py gives module not found error. so use accelerate and 1 gpu process
accelerate launch --num_processes 1 metrics/evaluate_metrics.py \
    --infer_dir runs/logs/sd15_depth_20percent_drop_constant_auto_caption_train_unet/checkpoint-17000/inference \
    --mode avg

# Construct log file names using SLURM environment variables
OUTPUT_FILE="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
ERROR_FILE="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

# Move the log files to LOG_DIR after job execution
echo "Moving log files to ${LOG_DIR}"
mv ${OUTPUT_FILE} ${LOG_DIR}/
mv ${ERROR_FILE} ${LOG_DIR}/