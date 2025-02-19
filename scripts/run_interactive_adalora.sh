torchrun --nproc_per_node=8 train.py \
--model_name "meta-llama/Llama-2-7b-hf" \
--train_ds "/workspace/mount_dir/tied-lora/datasets/boolq/train.jsonl" \
--validation_ds "/workspace/mount_dir/tied-lora/datasets/boolq/validation.jsonl" \
--cache_dir "/workspace/mount_dir/tied-lora/huggingface/hub" \
--max_seq_len 2048 \
--max_steps 1000 \
--logging_steps 25 \
--eval_steps 100 \
--save_steps 500 \
--bf16 True \
--packing True \
--output_dir "peft-lora-llama-2-7b-hf-squad-v1" \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--use_peft_adalora True \
--adalora_init_r 12 \
--adalora_target_r 4 \
--adalora_beta1 0.85 \
--adalora_beta2 0.85 \
--adalora_tinit 200 \
--adalora_tfinal 1000 \
--adalora_delta_t 10 \
--adalora_orth_reg_weight 0.5 \
--lora_alpha 32 \
--lora_qkv_mlp_dense \
--use_4bit_quantization False \
--use_nested_quant True \
--bnb_4bit_compute_dtype "bfloat16" \
--use_flash_attn False
