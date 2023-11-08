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
from transformers import HfArgumentParser, TrainingArguments, Trainer, DataCollatorForSeq2Seq, EvalPrediction, pipeline, GenerationConfig
from utils import *
from evaluate import load
import numpy as np
from datasets import load_dataset
import json


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})

    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
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
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training."},
    )
    use_gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Gradient Checkpointing."},
    )
    output_file: str = field(default="results/predictions.json", metadata={"help": "The output predictions file."})
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
    dataset_text_field: str = field(default="text", metadata={"help": "Dataset field to use as input text."})
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
    
    # Evaluation arguments
    task: str = field(default="squad", metadata={"help": "The name of the task to evaluate."}) 
    max_new_tokens: Optional[int] = field(default=8)

def main(args):

    # model
    model, tokenizer = restore_peft_model(args)
    model.config.use_cache = False

    # datasets
    test_data = load_dataset("json", data_files=args.test_ds, split="train")
    
    generator = pipeline('text-generation', model=model.cuda(), tokenizer=tokenizer, device='cuda:0')
    model = model.cuda()
    #peft_module_casting_to_bf16(model, args)

    output_file=args.output_file
    with open(output_file, 'w') as file:
        for idx, sample in enumerate(test_data):
            if idx%10==0:
                print(f'Predicting sample: {idx}')
            input_text = sample["input"]
            output = generator(input_text, max_new_tokens=args.max_new_tokens, num_return_sequences=1, do_sample=False, temperature=0., eos_token_id=tokenizer.eos_token_id)[0]["generated_text"]
            response = output.replace(input_text, "")
            sample['pred'] = response
            file.write(json.dumps(sample) + '\n')


    print(f'Finished generation.')

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
