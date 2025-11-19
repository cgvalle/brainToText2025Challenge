#!/bin/bash
#SBATCH --job-name b_val                        # Custom name
#SBATCH -t 12:00:00                                   # Max runtime of 4 hours
#SBATCH -p batch                                      # Choose partition (interactive or batch)
#SBATCH -q batch                                      # Choose QoS, must be same as partition
#SBATCH --cpus-per-task 4                             # Request 2 cores
#SBATCH --mem=65G                                      # Request RAM (memory)
#SBATCH --nodelist=ih-condor                             # Choose a specific node
#SBATCH --gpus=1                                      # Request 0 GPU
#SBATCH -o /mnt/workspace/%u/slurm-out/example-%j.out # Write output to this file
#SBATCH --mail-type=END                               # Notify when it ends
#SBATCH --array=1-3


## Load conda and activate your environment
clear
module load conda
module load redis


model_path=/mnt/workspace/cgvallea/brain/model_weights/time_warp_010_2000
lm_path=language_model/pretrained_language_models/openwebtext_1gram_lm_sil 
lm_path=data/n3gram

port=$((30652 + SLURM_ARRAY_TASK_ID))



redis-server --port $port &


# Run first Python script
/mnt/workspace/cgvallea/.conda/envs/b2txt25_lm/bin/python language_model/language-model-standalone.py \
    --lm_path $lm_path \
    --do_opt \
    --nbest 100 \
    --acoustic_scale 0.325 \
    --blank_penalty 90 \
    --alpha 0.55 \
    --redis_port $port \
    --gpu_number 0 &

# if 3gram wait for 300 seconds
if [[ $lm_path == *"3gram"* ]]; then
    sleep 300
fi

if [[ $lm_path == *"language_model/pretrained_language_models/openwebtext_1gram_lm_sil "* ]]; then
    sleep 30
    echo "Waited for 30 seconds for 1gram LM"
fi


touch $model_path/val_summary.csv
touch $model_path/test_summary.csv


start=$(( (SLURM_ARRAY_TASK_ID-1) * 250 ))
end=10000
step=750

echo "Task $SLURM_ARRAY_TASK_ID -> start=$start, end=$end, step=$step"

for (( i=start; i<=end; i+=step )); do
  sleep "$SLURM_ARRAY_TASK_ID"   # (optional) beware: task 0 sleeps 0s
  echo "Evaluating checkpoint_batch_$i"
  full_path=$model_path/rnn_val_predicted_sentences_checkpoint_batch_$i.csv
  echo "Full path: $full_path"

  # if path exists skip
    if [ -f "$full_path" ]; then
        echo "File $full_path exists. Skipping evaluation for checkpoint_batch_$i."
        continue
    fi

(cd model_training && /mnt/workspace/cgvallea/.conda/envs/b2txt25/bin/python evaluate_model.py \
    --model_path $model_path \
    --checkpoint_name checkpoint_batch_$i \
    --data_dir ../data/t15_copyTask_neuralData/hdf5_data_final \
    --eval_type val \
    --redis_port $port \
    --gpu_number 0)


done


#/mnt/workspace/cgvallea/intentionally-disabled/bin/kaggle  competitions submit -c brain-to-text-25  \
#   -f model_training/$model_path/rnn_test_predicted_sentences.csv  \
#   -m $model_path


# /mnt/workspace/cgvallea/intentionally-disabled/bin/kaggle  competitions submit -c brain-to-text-25 -f



