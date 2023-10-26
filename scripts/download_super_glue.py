from datasets import load_dataset
import json
import random
import math
import os

subsets =  ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc', 'wsc.fixed', 'axb', 'axg']
for subset in subsets:
    main_dataset = load_dataset("super_glue", subset)
    isExist = os.path.exists(subset)
    if not isExist:
        os.makedirs(subset)

    # Loop through the splits and save them as JSONL files
    splits= ["train", "test", "validation"]
    save_splits = {}
    for split in splits:
        dataset = main_dataset.get(split, None)
        if dataset is None:
            continue

        if split == "train":
            split_dataset = dataset.train_test_split(test_size=0.05, seed=1234)
            save_splits['train'] = split_dataset['train']
            save_splits['validation'] = split_dataset['test']
        elif split == "validation":
            save_splits['test'] = dataset
        else:
            save_splits['test_no_gt'] = dataset
            

    for split_name, dataset in save_splits.items():
        output_file = f"{subset}/{split_name}.jsonl"


        with open(output_file, "w", encoding="utf-8") as f:
            for example in dataset:
                # Write each example as a JSON line in the output file
                f.write(json.dumps(example)  + "\n")

        print(f"{split_name} split saved to {output_file}")


