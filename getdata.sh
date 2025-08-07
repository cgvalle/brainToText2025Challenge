#!/bin/bash
#SBATCH --job-name getdata                        # Custom name
#SBATCH -t 03:00:00                                   # Max runtime of 3 hours
#SBATCH -p batch                                      # Choose partition (interactive or batch)
#SBATCH -q batch                                      # Choose QoS, must be same as partition
#SBATCH --cpus-per-task 2                             # Request 2 cores
#SBATCH --mem=32G                                      # Request RAM (memory)
#SBATCH --gpus=0                                      # Request 0 GPU
#SBATCH -o /mnt/workspace/%u/slurm-out/example-%j.out # Write output to this file
#SBATCH --mail-type=END                               # Notify when it ends

## Load conda and activate your environment
clear
module load conda
conda activate base

/mnt/workspace/cgvallea/intentionally-disabled/bin/kaggle  competitions download -c brain-to-text-25 -p /mnt/workspace/cgvallea/brain