#!/bin/bash
#SBATCH --job-name b_infe                        # Custom name
#SBATCH -t 05:00:00                                   # Max runtime of 3 hours
#SBATCH -p batch                                      # Choose partition (interactive or batch)
#SBATCH -q batch                                      # Choose QoS, must be same as partition
#SBATCH --cpus-per-task 12                             # Request 2 cores
#SBATCH --mem=80G                                      # Request RAM (memory)
#SBATCH --gpus=1                                      # Request 0 GPU
#SBATCH -o /mnt/workspace/%u/slurm-out/example-%j.out # Write output to this file
#SBATCH --mail-type=END                               # Notify when it ends

## Load conda and activate your environment
clear
module load conda
module load redis


model_path=trained_models/time_warp
lm_path=language_model/pretrained_language_models/openwebtext_1gram_lm_sil 
lm_path=data/n3gram


redis-server --port 30655 &


# Run first Python script
/mnt/workspace/cgvallea/.conda/envs/b2txt25_lm/bin/python language_model/language-model-standalone.py \
    --lm_path $lm_path \
    --do_opt \
    --nbest 100 \
    --acoustic_scale 0.325 \
    --blank_penalty 90 \
    --alpha 0.55 \
    --gpu_number 0 &

# if 3gram wait for 300 seconds
if [[ $lm_path == *"3gram"* ]]; then
    sleep 300
fi

(cd model_training && /mnt/workspace/cgvallea/.conda/envs/b2txt25/bin/python evaluate_model.py \
    --model_path $model_path \
    --data_dir ../data/t15_copyTask_neuralData/hdf5_data_final \
    --eval_type test \
    --gpu_number 0)


/mnt/workspace/cgvallea/intentionally-disabled/bin/kaggle  competitions submit -c brain-to-text-25  \
   -f model_training/$model_path/rnn_test_predicted_sentences.csv  \
   -m $model_path


# /mnt/workspace/cgvallea/intentionally-disabled/bin/kaggle  competitions submit -c brain-to-text-25 -f



