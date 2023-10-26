from datasets import load_dataset
import json

folder="multi_news"
main_dataset = load_dataset(folder)

# Loop through the splits and save them as JSONL files
splits= ["train", "test", "validation"]
save_splits = {}
for split in splits:
    dataset = main_dataset.get(split, None)
    if dataset is None:
        continue
    save_splits[split] = dataset

for split_name, dataset in save_splits.items():
    output_file = f"{folder}/{split_name}.jsonl"


    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            # Write each example as a JSON line in the output file
            f.write(json.dumps(example)  + "\n")

    print(f"{split_name} split saved to {output_file}")



