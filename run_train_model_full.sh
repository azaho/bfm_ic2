#!/bin/bash
#SBATCH --job-name=bfm_run          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=16    # Request 8 CPU cores per GPU
#SBATCH --gres=gpu:a100:1 #  --gres=shard:1 --constraint=any-A100
#SBATCH --mem=300G
#SBATCH -t 24:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=1-9      # 14 jobs (108/8 rounded up)
#SBATCH --output logs/%A_%a.out # STDOUT
#SBATCH --error logs/%A_%a.err # STDERR
#SBATCH -p normal # use-everything
source .venv/bin/activate
export TMPDIR=/om2/scratch/tmp

# Define parameter arrays
learning_rates=(0.002) #(0.0015 0.0012 0.0018) # 0.001 0.0007)
d_models=(192) # 240 288) 
n_layers=(5) # 10 20)
weight_decays=(0.0 0.01 0.001)
random_strings=("X9_1" "X9_2" "X9_3")

# Calculate index into parameter combinations
idx=$SLURM_ARRAY_TASK_ID-1
n_lr=${#learning_rates[@]}
n_d=${#d_models[@]}
n_l=${#n_layers[@]}
n_ws=${#weight_decays[@]}
n_rs=${#random_strings[@]}

# Convert single index into parameter indices
layer_idx=$(( idx % n_l ))
d_idx=$(( (idx / n_l) % n_d ))
lr_idx=$(( (idx / (n_l * n_d)) % n_lr ))
ws_idx=$(( (idx / (n_l * n_d * n_lr)) % n_ws ))
rs_idx=$(( (idx / (n_l * n_d * n_lr * n_ws)) % n_rs ))

# Get parameter values
LR=${learning_rates[$lr_idx]}
D_MODEL=${d_models[$d_idx]}
N_LAYERS=${n_layers[$layer_idx]}
WS=${weight_decays[$ws_idx]}
RS=${random_strings[$rs_idx]}

echo "LR: $LR"
echo "D_MODEL: $D_MODEL"
echo "N_LAYERS: $N_LAYERS"
echo "WS: $WS"
echo "RS: $RS"

python -u train_model.py --learning_rate $LR --d_model $D_MODEL --n_layers_electrode $N_LAYERS --n_layers_time $N_LAYERS --weight_decay $WS --random_string $RS