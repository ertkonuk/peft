import random
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset
from tqdm import tqdm
import warnings
from peft import LoraConfig, AdaLoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)


class SaveDeepSpeedPeftModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (state.global_step + 1) % self.save_steps == 0:
            self.trainer.accelerator.wait_for_everyone()
            state_dict = self.trainer.accelerator.get_state_dict(self.trainer.deepspeed)
            unwrapped_model = self.trainer.accelerator.unwrap_model(self.trainer.deepspeed)
            if self.trainer.accelerator.is_main_process:
                unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
            self.trainer.accelerator.wait_for_everyone()
        return control

class SFTDataset(Dataset):
    def __init__(
        self, 
        tokenizer,
        dataset,
        max_length=4096,
        add_eos_token=True,
        answer_only_loss=True,
        prompt_template="{input} {output}",
        label_key="output",
    ):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length
        self.add_eos_token = add_eos_token
        self.answer_only_loss = answer_only_loss
        self.prompt_template = prompt_template
        self.label_key = label_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        prompt = self.prompt_template.format(**sample)
        tokenized_prompt = self._tokenize(prompt)

        if self.answer_only_loss:
            user_prompt = sample["input"]
            tokenized_user_prompt = self._tokenize(user_prompt, add_eos_token=self.add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
    
            if self.add_eos_token:
                user_prompt_len -= 1
    
            tokenized_prompt["labels"] = [-100] * user_prompt_len + tokenized_prompt["labels"][user_prompt_len:]

        return tokenized_prompt


    def _tokenize(self, text, add_eos_token=True):
        result = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None)
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.max_length
            and self.add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
    
        result["labels"] = result["input_ids"].copy()
        return result

class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            shuffle (bool): If true, the samples in each buffer are suffled. Default is `True`.
            add_eos_token (bool): If true, each buffer is delimited with eos token. Default is `True`.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        shuffle=True,
        add_eos_token=True,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.shuffle = shuffle
        self.add_eos_token = add_eos_token

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                if self.add_eos_token:
                    tokenized_input = tokenized_input + [self.concat_token_id]
                all_token_ids.extend(tokenized_input)
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens


def create_datasets(tokenizer, args):
    if args.train_ds.endswith(".json") or args.train_ds.endswith(".jsonl"):
        train_data = load_dataset("json", data_files=args.train_ds, split="train")
    else:
        train_data = load_dataset(args.train_ds, split="train")
    
    if args.validation_ds.endswith(".json") or args.validation_ds.endswith(".jsonl"):
        valid_data = load_dataset("json", data_files=args.validation_ds, split="train")
    elif script_args.validation_ds is None:
        valid_data = None
    else:
        valid_data = load_dataset(args.validation_ds, split="validation")

    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    train_dataset = SFTDataset(
        tokenizer=tokenizer,
        dataset=train_data,
        max_length=args.max_seq_length,
        add_eos_token=args.add_eos_token,
        answer_only_loss=args.answer_only_loss,
        prompt_template=args.prompt_template,
        label_key=args.label_key,
    )
    valid_dataset = SFTDataset(
        tokenizer=tokenizer,
        dataset=valid_data,
        max_length=args.max_seq_length,
        add_eos_token=args.add_eos_token,
        answer_only_loss=args.answer_only_loss,
        prompt_template=args.prompt_template,
        label_key=args.label_key,
    )

    return train_dataset, valid_dataset

def create_constant_length_datasets(tokenizer, args):
    dataset = load_dataset(args.dataset_name, use_auth_token=True, num_proc=args.num_workers)
    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    chars_per_token = chars_token_ratio(train_data, tokenizer, args.dataset_text_field)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=args.max_seq_length,
        chars_per_token=chars_per_token,
        content_field=args.dataset_text_field,
        shuffle=True,
        add_eos_token=False,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.max_seq_length,
        chars_per_token=chars_per_token,
        content_field=args.dataset_text_field,
        shuffle=False,
        add_eos_token=False,
    )

    return train_dataset, valid_dataset


def create_and_prepare_model(args):
    device_map = None
    bnb_config = None
    load_in_8bit = args.use_8bit_quantization

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)

    if args.use_4bit_quantization or args.use_8bit_quantization:
        device_map = "auto"  # {"": 0}
    
    from accelerate import Accelerator
    device_index = Accelerator().process_index
    device_map = {"": device_index}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map=device_map,
        use_cache=not args.use_gradient_checkpointing,
        use_auth_token=True,
        use_flash_attention_2=args.use_flash_attn,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    peft_config = None
    if args.use_peft_lora:
        print('Using LoRA')
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            target_modules=args.lora_target_modules.split(","),
        )
    elif args.use_peft_adalora:
        print('Using AdaLoRA')
        peft_config = AdaLoraConfig(
            init_r=args.adalora_init_r,
            target_r=args.adalora_target_r,
            beta1=args.adalora_beta1,
            beta2=args.adalora_beta2,
            tinit=args.adalora_tinit,
            tfinal=args.adalora_tfinal,
            deltaT=args.adalora_delta_t,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(","),
            orth_reg_weight=args.adalora_orth_reg_weight,
        )
    else:
        peft_config = None
    if (args.use_4bit_quantization or args.use_8bit_quantization) and args.use_peft_lora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.use_gradient_checkpointing)

    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=True, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer


def peft_module_casting_to_bf16(model, args):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
