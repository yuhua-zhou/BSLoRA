#!/bin/bash
export PYTHONPATH='.'

base_model="baffo32/decapoda-research-llama-7B-hf"
target_module="qkvupdown"

step=800
modes=(
  "gate"
  "gate"
  "gate"
  "gate"
  "kron"
  "kron"
  "kron"
  "kron"

#  "kron"
#  "gate"
#  "slice"
#  "kron"
#  "slice"
#  "gate"
#  "slice"
#  "gate"
)

gpu_ids=(7 6 5 4 3 2 1 0)

#local_rs=(16 16 8 8 4 4 2 0)
#intra_rs=(32 8 16 16 8 16 1 4)
#inter_rs=(32 8 32 16 32 32 2 8)

local_rs=(2 2 2 2 4 2 2 2)
intra_rs=(4 4 4 8 8 4 8 8)
inter_rs=(8 16 16 8 8 16 16 32)


run_tuning_and_evaluation(){

  local mode=$1
  local local_r=$2
  local intra_r=$3
  local inter_r=$4
  local gpu_id=$5

  tune_ckpt_path="share_$mode"

  echo "mode: $mode"
  echo "Lora Config: local_r=($local_r), intra_r=($intra_r), inter_r=($inter_r)"

  current_time=$(date "+%Y-%m-%d %H:%M:%S")
  echo "Start tuning on gpu: $gpu_id, $current_time"

  CUDA_VISIBLE_DEVICES=$gpu_id python sharelora_fine_tune.py \
     --base_model $base_model \
     --data_path yahma/alpaca-cleaned \
     --output_dir tune_log/$target_module/$tune_ckpt_path/${local_r}_${intra_r}_${inter_r}/ \
     --local_r $local_r \
     --intra_r $intra_r \
     --inter_r $inter_r \
     --share_mode $mode \
     --num_epochs 2 \
     --learning_rate 1e-4 \
     --batch_size 64

  current_time=$(date "+%Y-%m-%d %H:%M:%S")
  echo "End tuning on gpu: $gpu_id, $current_time"

  current_time=$(date "+%Y-%m-%d %H:%M:%S")
  echo "Start evaluation on gpu: $gpu_id, $current_time"

  CUDA_VISIBLE_DEVICES=$gpu_id lm_eval --model hf \
      --model_args peft=tune_log/$target_module/$tune_ckpt_path/${local_r}_${intra_r}_${inter_r}/checkpoint-$step,pretrained=$base_model \
      --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,social_iqa,boolq \
      --device cuda:0 \
      --output_path results/$target_module/reasoning/${tune_ckpt_path}_${local_r}_${intra_r}_${inter_r}_$step.json

  current_time=$(date "+%Y-%m-%d %H:%M:%S")
  echo "End evaluation on gpu: $gpu_id, $current_time"
}

for ((j=0; j<${#modes[@]}; j+=4)); do
  for i in "${!gpu_ids[@]}"; do

      idx=$((j+i))
      if ((idx>=${#modes[@]}));then
        break
      fi

      mode=${modes[$idx]}
      local_r=${local_rs[$idx]}
      intra_r=${intra_rs[$idx]}
      inter_r=${inter_rs[$idx]}
      gpu_id=${gpu_ids[$i]}

      run_tuning_and_evaluation "$mode" "$local_r" "$intra_r" "$inter_r" "$gpu_id"&
  done
  wait  # Wait for all tuning and evaluation processes to finish
done