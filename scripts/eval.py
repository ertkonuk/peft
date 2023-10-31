# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
import os
import subprocess
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments, Trainer, DataCollatorForSeq2Seq, EvalPrediction
from utils import *
from evaluate import load
import numpy as np
from datasets import load_dataset
########################################################################
# This is a fully working simple example to use trl's RewardTrainer.
#
# This example fine-tunes any causal language model (GPT-2, GPT-Neo, etc.)
# by using the RewardTrainer from trl, we will leverage PEFT library to finetune
# adapters on the model.
#
########################################################################


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})

    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=1e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[float] = field(default=0.001)
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": "The model that you want to evaluate from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    peft_model_name: Optional[str] = field(
        default="awesome_lora_model",
        metadata={
            "help": "The trained PEFT model."
        },
    )
    use_4bit: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_torch_fused",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=10000, metadata={"help": "How many optimizer update steps to take"})
    warmup_steps: int = field(default=50, metadata={"help": "Number of warmup steps"})
    save_steps: int = field(default=10, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=10, metadata={"help": "Eval model every X steps."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    output_dir: str = field(default="results", metadata={"help": "Where to store the final model."})
    wandb_run_name: str = field(default="draco_oci.llama_2_7b.lora", metadata={"help": "The name of the wandb job."})
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Gradient Checkpointing."},
    )
    dataset_text_field: str = field(default="text", metadata={"help": "Dataset field to use as input text."})
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, pushes the model to the HF Hub"},
    )
    num_workers: int = field(default=16, metadata={"help": "Number of dataset workers to use."})
    debug: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tests things like proper saving/loading/logging of model"},
    )
    prompt_template: Optional[str] = field(
        default="{input} {output}",
        metadata={
            "help": "Template for formatting data."
        },
    )
    label_key: Optional[str] = field(
        default="output",
        metadata={
            "help": "Dataset key for labels."
        },
    )
    answer_only_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Compute loss only on answers"},
    )
    cache_dir: Optional[str] = field(
        default="/models",
        metadata={"help": "The Hugging Face cache dir for hub. Usuall ~/.cache/huggingface/hub for local."},
    )
    test_ds: Optional[str] = field(
        default="None",
        metadata={"help": "The training dataset in JSONL format"},
    )
    add_eos_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Add EOS token at the end of each traininf sequence."},
    )
    # PEFT parameters
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_peft_adalora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT AdaLoRA for training."},
    )
    # Parameters for LoRA and AdaLoRA
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    
    # AdaLoRA parameters
    adalora_init_r: Optional[int] = field(
            default=12,
            metadata={"help": "Initial AdaLoRA rank."},
    )
    adalora_target_r: Optional[int] = field(
            default=4,
            metadata={"help": "Target AdaLoRA rank."},
    )
    adalora_tinit: Optional[int] = field(
            default=200,
            metadata={"help": "Number of warmup steps for AdaLoRA wherein no pruning is performed."},
    )
    adalora_tfinal: Optional[int] = field(
            default=1000,
            metadata={"help": "Fix the resulting budget distribution and fine-tune the model for tfinal steps when using AdaLoRA."},
    )
    adalora_delta_t: Optional[int] = field(
            default=10,
            metadata={"help": "Interval of steps for AdaLoRA to update rank."},
    )
    adalora_orth_reg_weight: Optional[float] = field(
            default=0.5,
            metadata={"help": "Orthogonal regularization weight."},
    )

    adalora_beta1: Optional[float] = field(default=0.85)
    adalora_beta2: Optional[float] = field(default=0.85)

    # Convenience arguments
    lora_qkv: Optional[bool] = field(default=False)
    lora_qkv_mlp: Optional[bool] = field(default=False)
    lora_qkv_mlp_dense: Optional[bool] = field(default=False)
    
    # Evaluation arguments
    task: str = field(default="squad", metadata={"help": "The name of the task to evaluate."})
    

def main(args):
    # training arguments
    is_deepspeed_peft_enabled = (
        os.environ.get("ACCELERATE_USE_DEEPSPEED", "False").lower() == "true" and args.use_peft_lora
    )
    save_strategy = "no" if is_deepspeed_peft_enabled else "steps"
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        do_eval=True,
    )

    # model
    model, tokenizer = restore_peft_model(args)
    model.config.use_cache = False

    # datasets
    test_data = load_dataset("json", data_files=args.test_ds, split="train")
    def tokenize_function(examples):
        return tokenizer(examples['input'], truncation=True, padding='max_length', max_length=args.max_seq_length)

    test_dataset = test_data.map(tokenize_function, batched=True)

    # data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    
    # metrics
    if args.task == "squad":
        metric = load('squad')
    else:
        metric = load('super_glue_benchmark', args.task)
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        print(f'predictions: {predictions.shape} labels={labels.shape}')
        return metric.compute(predictions=predictions, references=labels)

    # trainer
    #trainer = Trainer(model=model, args=training_arguments, eval_dataset=test_dataset, data_collator=data_collator, compute_metrics=compute_metrics)
    trainer = Trainer(model=model, args=training_arguments, eval_dataset=test_dataset, compute_metrics=compute_metrics)
    trainer.accelerator.print(f"{trainer.model}")

    if args.use_peft_lora or args.use_peft_adalora:
        peft_module_casting_to_bf16(trainer.model, args)

    # evaluate
    metrics = trainer.evaluate()
    print(f'metrics: {metrics}')

def sanity_check(args):
    # Allow either just LoRA or AdaLoRA
    if args.use_peft_lora and args.use_peft_adalora:
        raise ValueError("Both use_peft_lora and use_peft_adalora cannot be set to True simultaneously.")
    # Allow only one configuration
    lora_args = [args.lora_qkv, args.lora_qkv_mlp, args.lora_qkv_mlp_dense]
    if sum(lora_args) > 1:
        raise ValueError("Only one (or none) of --lora_qkv, --lora_qkv_mlp, and --lora_qkv_mlp_dense can be set to True.")
    # Set target modules based on the configuration
    if args.lora_qkv:
        args.lora_target_modules='q_proj,k_proj,v_proj'
    if args.lora_qkv_mlp:
        args.lora_target_modules='q_proj,k_proj,v_proj,down_proj,up_proj,gate_proj'
    if args.lora_qkv_mlp_dense:
        args.lora_target_modules='q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj'

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    sanity_check(args)
    main(args)
