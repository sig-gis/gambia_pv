#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --partition=ewg
#SBATCH --output=/home/ceoas/truongmy/emapr/gambia_pv/logs/02_train_model.log
#SBATCH --error=/home/ceoas/truongmy/emapr/gambia_pv/logs/02_train_model.err
#SBATCH --job-name=02_train_model
#SBATCH --chdir=/home/ceoas/truongmy/emapr/gambia_pv/src
#SBATCH --export=ALL,EXPERIMENT_NAME=dice_lr1e-5,BASE_PATH=/home/ceoas/truongmy/emapr/gambia_pv

conda activate sundial
srun python -u 02_train_model.py && python -u 03_test_model.py
