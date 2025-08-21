#!/bin/bash
#SBATCH --job-name jupyter
#SBATCH -t 03:00:00 # Time limit (D-HH:MM:SS)
#SBATCH -p interactive
#SBATCH -q interactive
#SBATCH --cpus-per-task 2 # 2 cores
#SBATCH -o /mnt/workspace/%u/slurm-out/example-%j.out # Write output to this file
#SBATCH --mem=2G
#SBATCH --gpus=0

module load conda
conda activate uci

python -m jupyter notebook --no-browser --port 30651