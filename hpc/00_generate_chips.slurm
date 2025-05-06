#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --partition=ceoas
#SBATCH --output=/home/ceoas/truongmy/emapr/gambia_pv/logs/00_generate_chips.log
#SBATCH --error=/home/ceoas/truongmy/emapr/gambia_pv/logs/00_generate_chips.err
#SBATCH --job-name=00_generate_chips
#SBATCH --chdir=/home/ceoas/truongmy/emapr/gambia_pv/src
#SBATCH --export=ALL,BASE_PATH=/home/ceoas/truongmy/emapr/gambia_pv

conda activate sundial

srun python -u 00_generate_chips.py && python -u 01_calculate_stats.py