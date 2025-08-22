#!/bin/bash
#SBATCH --job-name b_infe                        # Custom name
#SBATCH -t 03:00:00                                   # Max runtime of 3 hours
#SBATCH --nodelist=ih-condor
#SBATCH -p batch                                      # Choose partition (interactive or batch)
#SBATCH -q batch                                      # Choose QoS, must be same as partition
#SBATCH --cpus-per-task 4                             # Request 2 cores
#SBATCH --mem=40G                                      # Request RAM (memory)
#SBATCH --gpus=1                                      # Request 0 GPU
#SBATCH -o /mnt/workspace/%u/slurm-out/example-%j.out # Write output to this file
#SBATCH --mail-type=END                               # Notify when it ends

## Load conda and activate your environment
clear
module load conda
module load redis
conda activate b2txt25_lm

#30655

(redis-server  --port 30655 & echo "hi")

#(cd nejm-brain-to-text && python language_model/language-model-standalone.py --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil --do_opt --nbest 100 --acoustic_scale 0.325 --blank_penalty 90 --alpha 0.55 --gpu_number 0)




 