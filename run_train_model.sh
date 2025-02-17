#!/bin/bash
#SBATCH --job-name=bfm_run          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=16    # Request 8 CPU cores per GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=256G
#SBATCH -t 12:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=0-0      # 14 jobs (108/8 rounded up)
#SBATCH --output r/%A_%a.out # STDOUT
#SBATCH --error r/%A_%a.err # STDERR
#SBATCH -p yanglab

source .venv/bin/activate
python -u train_model.py