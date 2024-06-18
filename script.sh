#!/bin/bash
#SBATCH --job-name=python_gpu_job  # Job name
#SBATCH --partition=gpu            # Request GPU partition
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00        # Time limit hh:mm:ss
#SBATCH --output=result.out        # Standard output and error log

cd /home/username/img_denoising/
# module load cuda/9.0      # Load the module for Python with GPU support

# Activate conda environment
# create your environment and add necessary libraries and then this will work
source activate myenv

# Run the Python script
# python train.py
python main.py
