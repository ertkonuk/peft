task=cb
peft_model='/workspace/mount_dir/tied-lora/experiments/cb_lora/peft.lora.llama-2-7b-hf/2023-10-30_15-31-22/training/'
python eval.py \
--model_name="meta-llama/Llama-2-7b-hf" \
--peft_model_name=${peft_model} \
--test_ds=/workspace/mount_dir/tied-lora/datasets/${task}/test.jsonl \
--cache_dir=/workspace/mount_dir/tied-lora/huggingface/hub/ \
--max_seq_len=2048 \
--logging_steps=1 \
--bf16=True \
--packing=False \
--output_dir=/workspace/mount_dir/tied-lora/experiments/${task}_lora/interactive \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=1 \
--use_flash_attn True
