from datasets import load_dataset
import json
import os

#main_dataset = load_dataset("tatsu-lab/alpaca") 
main_dataset = load_dataset("yahma/alpaca-cleaned")
isExist = os.path.exists("alpaca_cleaned")
if not isExist:
    os.makedirs("alpaca_cleaned")


# Loop through the splits and save them as JSONL files
splits= ["train"]
save_splits = {}
for split in splits:
    dataset = main_dataset.get(split, None)
    if dataset is None:
        continue

    #split_dataset = dataset.train_test_split(test_size=0.15, seed=1234)
    save_splits['train'] = dataset
    #resplit_dataset = split_dataset['test'].train_test_split(test_size=0.66, seed=1234)
    #save_splits['validation'] = resplit_dataset['train']
    #save_splits['test'] = resplit_dataset['test']

for split_name, dataset in save_splits.items():
    output_file = f"alpaca_cleaned/{split_name}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            # Write each example as a JSON line in the output file
            example["input"] = example["instruction"] + " " + example["input"]
            example["input"] = "User: " + example["input"].strip() + " Answer:"
            f.write(json.dumps(example)  + "\n")

    print(f"{split_name} split saved to {output_file}")



