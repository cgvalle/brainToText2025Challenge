#!/bin/bash
#SBATCH --job-name b_train                        # Custom name
#SBATCH -t 72:00:00                                   # Max runtime of 3 hours
#SBATCH -p batch                                      # Choose partition (interactive or batch)
#SBATCH -q batch    
#SBATCH --nodelist=ih-condor                             # Choose a specific node
#SBATCH --cpus-per-task 12                             # Request 12 cores
#SBATCH --mem=40G                                      # Request RAM (memory)
#SBATCH --gpus=1                                      # Request 1 GPU
#SBATCH -o /mnt/workspace/%u/slurm-out/example-%j.out # Write output to this file
#SBATCH --mail-type=END                               # Notify when it ends

## Load conda and activate your environment
clear
module load conda

conda activate b2txt25



(cd model_training && python train_model.py)


# time warp: 74571