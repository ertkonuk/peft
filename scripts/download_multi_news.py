from datasets import load_dataset
import json
import os

folder="multi_news"
main_dataset = load_dataset(folder)
isExist = os.path.exists("multi_news")
if not isExist:
    os.makedirs("multi_news")


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
            e = {}
            e['input'] = 'Document: ' + example['document'] + ' Summary:'
            e['output'] = example['summary']
            f.write(json.dumps(e)  + "\n")

    print(f"{split_name} split saved to {output_file}")



