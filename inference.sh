#!/bin/bash
#SBATCH --job-name b_infe                        # Custom name
#SBATCH -t 03:00:00                                   # Max runtime of 3 hours
#SBATCH --nodelist=ih-condor
#SBATCH -p batch                                      # Choose partition (interactive or batch)
#SBATCH -q batch                                      # Choose QoS, must be same as partition
#SBATCH --cpus-per-task 4                             # Request 2 cores
#SBATCH --mem=100G                                      # Request RAM (memory)
#SBATCH --gpus=2                                      # Request 0 GPU
#SBATCH -o /mnt/workspace/%u/slurm-out/example-%j.out # Write output to this file
#SBATCH --mail-type=END                               # Notify when it ends

## Load conda and activate your environment
clear
module load conda
module load redis


redis-server --port 30655 &

# Run first Python script
/mnt/workspace/cgvallea/.conda/envs/b2txt25_lm/bin/python language_model/language-model-standalone.py \
    --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil \
    --do_opt \
    --nbest 100 \
    --acoustic_scale 0.325 \
    --blank_penalty 90 \
    --alpha 0.55 \
    --gpu_number 0 &

 

(cd model_training && /mnt/workspace/cgvallea/.conda/envs/b2txt25/bin/python evaluate_model.py \
    --model_path data/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline \
    --data_dir data/t15_copyTask_neuralData/hdf5_data_final \
    --eval_type test \
    --gpu_number 1)


/mnt/workspace/cgvallea/intentionally-disabled/bin/kaggle  competitions submit -c brain-to-text-25  \
    -f data/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline/baseline_rnn_test_predicted_sentences_20250822_114145.csv  \
    -m "base model"