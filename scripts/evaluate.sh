#!/bin/bash

DIRECTORY="/workspace/mount_dir/tied-lora/experiments"
base_model="meta-llama/Llama-2-7b-hf"
cache_dir="/workspace/mount_dir/tied-lora/huggingface/hub/"
output_dir="/workspace/mount_dir/tied-lora/results"
dataset_dir="/workspace/mount_dir/tied-lora/datasets"
max_new_tokens=8

# Check if the directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Directory $DIRECTORY does not exist!"
    exit 1
fi

# Find all subdirectories named "training"
find "$DIRECTORY" -type d -name "training" | while read -r line; do
   	peft_model=$line
	echo "PEFT model: $peft_model"
    # Extract the desired portion of the path using awk
    experiment=$(echo "$line" | awk -F'/' '{print $(NF-2)}' | awk -F'_' '{for(i=1; i<=4; i++) printf $i"_"}' | sed 's/_$//')
    echo "Experiment: $experiment"
    
    # Extract the task name which appears right after /experiments/
    task=$(echo "$line" | awk -F'/experiments/' '{print $2}' | awk -F'/' '{print $1}' | awk -F'_' '{print $1}')
    echo "Task: $task"

	python peft_model_evaluate.py \
		--output_file=${output_dir}/llama2_7b_hf_${experiment}_predictions.jsonl \
		--cache_dir=${cache_dir} \
		--model_name=${base_model} \
		--peft_model_name=${peft_model} \
		--test_ds=${dataset_dir}/${task}/test.jsonl \
		--max_new_tokens=${max_new_tokens} \
		--use_flash_attn=True
done

