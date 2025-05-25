#!/bin/bash
export PYTHONPATH='.'

base_model="baffo32/decapoda-research-llama-7B-hf"
target_module="qkvupdown"
tune_ckpt_path="bslora"

step=800
gpu_id=2

lora_r=2
intra_r=4
inter_r=16

echo "Lora Config: lora_r=($lora_r), intra_r=($intra_r), inter_r=($inter_r)"

current_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start tuning on gpu: $gpu_id, $current_time"

CUDA_VISIBLE_DEVICES=$gpu_id python sharelora_fine_tune.py \
   --base_model $base_model \
   --data_path yahma/alpaca-cleaned \
   --output_dir tune_log/$target_module/$tune_ckpt_path/${lora_r}_${intra_r}_${inter_r}/ \
   --share_mode kron \
   --lora_r $lora_r \
   --intra_r $intra_r \
   --inter_r $inter_r \
   --num_epochs 2 \
   --learning_rate 1e-4 \
   --batch_size 64

current_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "End tuning on gpu: $gpu_id, $current_time"


current_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start evaluation on gpu: $gpu_id, $current_time"

CUDA_VISIBLE_DEVICES=$gpu_id lm_eval --model hf \
    --model_args peft=tune_log/$target_module/$tune_ckpt_path/${lora_r}_${intra_r}_${inter_r}/checkpoint-$step,pretrained=$base_model \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,social_iqa,boolq \
    --device cuda:0 \
    --output_path results/$target_module/reasoning/${tune_ckpt_path}_${lora_r}_${intra_r}_${inter_r}_$step.json

current_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "End evaluation on gpu: $gpu_id, $current_time"