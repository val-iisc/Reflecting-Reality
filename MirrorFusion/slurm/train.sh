#!/bin/bash
#SBATCH --job-name=train_brushnet
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=11:59:00
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH -w cn7

module load conda/24.1.2 cuda/cuda12.4

# Initialize conda
# source ~/miniconda3/etc/profile.d/conda.sh
# # Activate the conda environment
# conda activate brushnet
# pip install loguru


accelerate launch --num_processes=4 examples/brushnet/train_brushnet_mirror.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --output_dir runs/logs/sd15_depth_20percent_drop_constant_auto_caption_train_unet \
    --train_data_dir data/blenderproc \
    --resolution 512 \
    --seed 42 \
    --learning_rate 1e-5 \
    --train_batch_size 4 \
    --max_train_steps 20000 \
    --tracker_project_name brushnet \
    --report_to wandb \
    --resume_from_checkpoint latest \
    --checkpointing_steps 30000 \
    --validation_steps 1000 \
    --custom_checkpoints 15000 17000 18000 20000 \
    --num_validation_images 10 \
    --caption_column auto_caption \
    --depth_conditioning_mode concat \
    --proportion_empty_prompts 0.2 \
    --train_base_unet


# Construct log file names using SLURM environment variables
OUTPUT_FILE="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
ERROR_FILE="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

# Move the log files to LOG_DIR after job execution
echo "Moving log files to ${LOG_DIR}"
mv ${OUTPUT_FILE} ${LOG_DIR}/
mv ${ERROR_FILE} ${LOG_DIR}/
