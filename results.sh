#!/bin/bash
#SBATCH --job-name results
#SBATCH -t 03:00:00 # Time limit (D-HH:MM:SS)
#SBATCH -p interactive
#SBATCH -q interactive
#SBATCH --cpus-per-task 1 # 2 cores
#SBATCH -o /mnt/workspace/%u/slurm-out/example-%j.out # Write output to this file
#SBATCH --mem=1G
#SBATCH --gpus=0

module load conda
conda activate b2txt25

python results.py