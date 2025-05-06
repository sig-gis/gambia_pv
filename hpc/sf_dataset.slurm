#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --mem=700000
#SBATCH --cpus-per-task=64
#SBATCH --partition=ceoas
#SBATCH --output=/home/ceoas/truongmy/emapr/gambia_pv/logs/sf_dataset.log
#SBATCH --error=/home/ceoas/truongmy/emapr/gambia_pv/logs/sf_dataset.err
#SBATCH --job-name=sf_dataset
#SBATCH --chdir=/home/ceoas/truongmy/emapr/gambia_pv


conda activate sit_fuse
python -u /home/ceoas/truongmy/emapr/SIT_FUSE/src/sit_fuse/datasets/sf_dataset.py -y /home/ceoas/truongmy/emapr/gambia_pv/sit_fuse/gambia_pv_dbn.yaml