from datasets import load_dataset
import json
import random
import math
import os

def converted_example(e, s, t):
    if s in t:
        inputs = t[s][0]
        outputs = t[s][1]
        ce = ""
        for i in inputs.split():
            ce = ce + i + ": " + str(e[i]) + " "
        ce = ce.strip()
        ce = ce + " answer:"
        if isinstance(e[outputs], list):
            if len(e[outputs]) > 0:
                oce = str(e[outputs][0])
            else:
                oce = ""
            e["original_answers"] = e[outputs]
        elif isinstance(e[outputs], float):
            oce = str(round(e[outputs]))
        else:
            oce = str(e[outputs])
        e["input"] = ce
        e["output"] = oce
        return e
    else:
        return e

templates = {"qnli": ["question sentence", "label"],
            "sst2": ["sentence", "label"],
            "stsb": ["sentence1 sentence2", "label"],
            "cola": ["sentence", "label"],
            "wnli": ["sentence1 sentence2", "label"],
            "mrpc": ["sentence1 sentence2", "label"]}
#TODO: template for record

#subsets =  ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc', 'wsc.fixed', 'axb', 'axg']
subsets = ["cola", "mrpc", "qnli", "qqp", "sst2", "stsb", "wnli"]
for subset in subsets:
    print(subset)
    main_dataset = load_dataset("glue", subset)
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
                ce = converted_example(example, subset, templates)
                f.write(json.dumps(ce)  + "\n")

        print(f"{split_name} split saved to {output_file}")


