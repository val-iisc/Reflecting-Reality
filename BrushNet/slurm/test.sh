#!/bin/bash
#SBATCH --job-name=inference_brushnet
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=9:00:00
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --gres=gpu:7
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH -w cn3

module load conda/24.1.2 cuda/cuda12.4

# using defualt ground truth depth maps.
# To use monocular depth maps, set --geometric_input_data_dir and --depth_source
accelerate launch examples/brushnet/test_brushnet.py \
        --brushnet_path runs/logs/sd15_depth_20percent_drop_constant_auto_caption_train_unet/checkpoint-17000 \
        --output_dir inference \
        --caption_column auto_caption \
        --depth_conditioning_mode concat

# Construct log file names using SLURM environment variables
OUTPUT_FILE="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
ERROR_FILE="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

# Move the log files to LOG_DIR after job execution
echo "Moving log files to ${LOG_DIR}"
mv ${OUTPUT_FILE} ${LOG_DIR}/
mv ${ERROR_FILE} ${LOG_DIR}/