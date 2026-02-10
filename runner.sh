#!/bin/bash
#SBATCH --output=/scratch/htc/mwagner/output/%j
# NOTE: Adjust username in the SBATCH command above, this cannot be dynamically inferred from $USER

codebase_path=$1
sweep_id=$2
conda_env=$3

# Switch to the cwd
cd $codebase_path

# Acquire node information
echo 'Getting node information'
date;hostname;id;pwd

# Setup environment
echo 'Activating virtual environment'
bash
source ~/.bashrc
conda activate $conda_env
which python
nvidia-smi

# Setup internet access and set environment variables
echo 'Enabling Internet Access'
export https_proxy=http://squid.zib.de:3128
export http_proxy=http://squid.zib.de:3128

echo 'Set the wandb directory variable'
export WANDB_DIR=/home/htc/$USER/SCRATCH


# Setup temporary directory (this avoids the problem that wandb fills up the git repo with 100k files)
if [ -d "/scratch/local" ]; then
  # Create /scratch/local/$USER and /scratch/local/$USER/tmp if they do not exist
  mkdir -p "/scratch/local/$USER/tmp"
  chown -R :3331 "/scratch/local/$USER"
  # Set TMPDIR to the temporary directory
  export TMPDIR="/scratch/local/$USER/tmp"
fi

# Execute the job
sg login_ais2t -c "srun wandb agent --count 1 $sweep_id"