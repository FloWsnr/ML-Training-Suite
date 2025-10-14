#!/bin/bash

### Task name
#SBATCH --account=your_account_here

### Job name
#SBATCH --job-name=name_of_your_job

### Output file
#SBATCH --output=results/00_slrm_logs/name_of_your_job_%j.out

### Number of nodes
#SBATCH --nodes=1

### How many CPU cores to use
#SBATCH --ntasks-per-node=70

### How much memory in total (MB)
#SBATCH --mem=600G

### Mail notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email_here

### Maximum runtime per task
#SBATCH --time=24:00:00

### set number of GPUs per task (v100, a100, h200)
#SBATCH --gres=gpu:a100:4
##SBATCH --constraint=a100_80gb


### Partition
#SBATCH --partition=gpu

### create time series, i.e. 100 jobs one after another. Each runs for 24 hours
##SBATCH --array=1-10%1


#####################################################################################
############################# Setup #################################################
#####################################################################################
# Set the time limit for the job, allows for graceful shutdown
# Should be lower than the time limit of the partition
# Format: HH:MM:SS
time_limit="24:00:00"

# activate conda environment
export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate train_env

# Load environment variables from .env file
# Script is located at results/sim_name/train.sh, .env is at repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../../.env"
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

######################################################################################
############################# Set paths ##############################################
######################################################################################

sim_name="train" # name of the folder where you placed the yaml config

python_exec="${BASE_DIR}/ml_suite/train/run_training.py"
config_file="${RESULTS_DIR}/${sim_name}/train.yaml"

nnodes=1
ngpus_per_node=4
export OMP_NUM_THREADS=1


#####################################################################################
############################# Training GPM ##########################################
#####################################################################################
echo "--------------------------------"
echo "Starting training..."
echo "config_file: $config_file"
echo "--------------------------------"

exec_args="--config_path $config_file"

# Capture Python output and errors in a variable and run the script

torchrun --standalone --nproc_per_node=$ngpus_per_node $python_exec $exec_args
