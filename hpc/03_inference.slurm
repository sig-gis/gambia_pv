#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --mem=1024G
#SBATCH --cpus-per-task=32
#SBATCH --partition=ewg
#SBATCH --output=/home/ceoas/truongmy/emapr/gambia_pv/logs/04_inference.log
#SBATCH --error=/home/ceoas/truongmy/emapr/gambia_pv/logs/04_inference.err
#SBATCH --job-name=04_inference
#SBATCH --chdir=/home/ceoas/truongmy/emapr/gambia_pv/src
#SBATCH --export=ALL,EXPERIMENT_NAME=dice_lr1e-5,BASE_PATH=/home/ceoas/truongmy/emapr/gambia_pv

conda activate sundial

srun python -u 04_inference.py