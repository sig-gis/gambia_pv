#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --mem=2051000
#SBATCH --cpus-per-task=256
#SBATCH --partition=ceoas
#SBATCH --output=/home/ceoas/truongmy/emapr/gambia_pv/logs/sf_dataset.log
#SBATCH --error=/home/ceoas/truongmy/emapr/gambia_pv/logs/sf_dataset.err
#SBATCH --job-name=sf_dataset
#SBATCH --chdir=/home/ceoas/truongmy/emapr/gambia_pv


conda activate sit_fuse
python -u /home/ceoas/truongmy/emapr/SIT_FUSE/src/sit_fuse/train/pretrain_encoder.py -y /home/ceoas/truongmy/emapr/gambia_pv/sit_fuse/gambia_pv_dbn.yaml