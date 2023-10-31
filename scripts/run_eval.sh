task=cb
peft_model='/workspace/mount_dir/tied-lora/experiments/cb_llama2_7b_hf/superglue_lora_cb_1e-4_8_1_10/2023-10-30_20-17-50/training/'
python peft_model_evaluate.py \
--output_file=/workspace/mount_dir/tied-lora/results/test.jsonl \
--model_name="meta-llama/Llama-2-7b-hf" \
--peft_model_name=${peft_model} \
--test_ds=/workspace/mount_dir/tied-lora/datasets/${task}/test.jsonl \
--cache_dir=/workspace/mount_dir/tied-lora/huggingface/hub/ \
--max_new_tokens=4 \
--bf16=True \
--use_flash_attn True
