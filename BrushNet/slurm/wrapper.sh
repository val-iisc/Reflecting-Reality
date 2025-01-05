#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <slurm_script> [log_directory]"
    echo "Example: $0 slurm/train.sh runs/logs/sd15_full_1"
    exit 1
fi

# Assign the arguments to variables
SLURM_SCRIPT=$1

LOG_DIR=${2:-"slurm/logs"}
SLURM="$LOG_DIR/slurm"

# Ensure the directory exists
mkdir -p $SLURM

# prefix slurm script with slurm/ if not already
if [[ ! $SLURM_SCRIPT =~ ^slurm/ ]]; then
    SLURM_SCRIPT="slurm/$SLURM_SCRIPT"
fi

# Check if the SLURM script exists and is executable
if [[ ! -f $SLURM_SCRIPT && ! -f $SLURM_SCRIPT.sh ]]; then
    echo "Error: SLURM script '$SLURM_SCRIPT' or '$SLURM_SCRIPT.sh' not found."
    exit 1
fi

# Add .sh extension if not provided
if [[ ! $SLURM_SCRIPT =~ \.sh$ ]]; then
    SLURM_SCRIPT="$SLURM_SCRIPT.sh"
fi

# Submit the SLURM job with the directory as an environment variable
sbatch --export=ALL,LOG_DIR=$SLURM $SLURM_SCRIPT