#!/bin/bash
#SBATCH --job-name b_train                        # Custom name
#SBATCH -t 72:00:00                                   # Max runtime of 3 hours
#SBATCH --nodelist=ih-condor
#SBATCH -p batch                                      # Choose partition (interactive or batch)
#SBATCH -q batch                                      # Choose QoS, must be same as partition
#SBATCH --cpus-per-task 1                             # Request 2 cores
#SBATCH --mem=1G                                      # Request RAM (memory)
#SBATCH --gpus=0                                      # Request 0 GPU
#SBATCH -o /mnt/workspace/%u/slurm-out/example-%j.out # Write output to this file
#SBATCH --mail-type=END                               # Notify when it ends

## Load conda and activate your environment
clear
module load conda

conda activate b2txt25



python slurm.py

