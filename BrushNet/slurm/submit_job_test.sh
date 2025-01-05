#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=00:01:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH -w cn7

module load conda/24.1.2 cuda/cuda12.4

pwd
echo "JOB_NAME: ${SLURM_JOB_NAME}"
echo "LOG_DIR: $LOG_DIR"

# Print which Python
which python
nvcc --version

# python -c "import torch; print(torch.cuda.nccl.version())"

nvidia-smi

# Construct log file names using SLURM environment variables
OUTPUT_FILE="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
ERROR_FILE="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

# Move the log files to LOG_DIR after job execution
echo "Moving log files to ${LOG_DIR}"
mv ${OUTPUT_FILE} ${LOG_DIR}/
mv ${ERROR_FILE} ${LOG_DIR}/