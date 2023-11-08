import numpy as np
import matplotlib.pyplot as plt
#import tikzplotlib
import re

def extract_dim_from_key(key):
    match = re.search(r'dim(\d+)', key)
    return int(match.group(1)) if match else None

def process_data(file_path):
    # Read the lines from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize the main dictionary
    main_dict = {}

    for line in lines:
        # Split the line into components
        components = line.strip().split()
        
        # The first word is the task name
        task = components[0]
        
        # The last component is the experiment name which contains the dimension and kf_ information
        experiment = components[-1]
        experiment_name_parts = components[-1].split('_')

        # Find the part that starts with 'seed'
        seed_parts = [part for part in experiment.split('_') if part.startswith('seed')] 
        if not seed_parts:
            # If there is no 'seed' part, we can choose to skip this line or handle it differently
            #print(f"Skipping line, no 'seed' found: {line.strip()}")
            continue
        seed = 'seed' + seed_parts[0][4:]

        # Find the part that starts with 'dim'
        dim_parts = [part for part in experiment.split('_') if part.startswith('dim')]
        if not dim_parts:
            # If there is no 'dim' part, we can choose to skip this line or handle it differently
            #print(f"Skipping line, no 'dim' found: {line.strip()}")
            continue
        dim = 'dim' + dim_parts[0][3:]

        # Initialize sub-dictionaries if they don't exist
        if task not in main_dict:
            main_dict[task] = {}
    
        # Extract experiment name starting with kf_       
        dimension_part = next(part for part in experiment_name_parts if 'dim' in part)
        experiment_name = '_'.join(experiment_name_parts[experiment_name_parts.index(dimension_part) + 1:])  # Experiment details        

        # Initialize the experiment sub-dictionary
        if experiment_name not in main_dict[task]:
            main_dict[task][experiment_name] = {}    

        # Initialize the seed sub-dictionary
        if seed not in main_dict[task][experiment_name]:
            main_dict[task][experiment_name][seed] = {}    

        # Initialize the dim sub-dictionary
        if dim not in main_dict[task][experiment_name][seed]:
            main_dict[task][experiment_name][seed][dim] = {}

        # Extract the metrics and scores
        for i in range(1, len(components) - 1, 2):
            metric = components[i]
            score = float(components[i + 1])
            main_dict[task][experiment_name][seed][dim][metric] = score

    return main_dict

# Assuming the file path is "/mnt/data/data.txt"
file_path = "rte.out.try20.txt"
exp_dict = process_data(file_path)

d_rank = 8 # the delta increment of the horizontal axis values
for task in exp_dict.keys():
    experiments = []
    for experiment in exp_dict[task].keys():
        if experiment not in experiments:
            experiments.append(experiment)
        metric = []
        dims = []        
        for seed in exp_dict[task][experiment].keys(): 
            for dim in exp_dict[task][experiment][seed].keys():                
                metric_value = exp_dict[task][experiment][seed][dim]['exact_match']
                print(f" {task = } {experiment = } {seed = } {dim = } {metric_value = }")
                metric.append(metric_value)
                rank = extract_dim_from_key(dim)
                if rank not in dims:
                    dims.append(rank)                   
        avg_metric = np.array(metric).reshape(len(list(exp_dict[task][experiment].keys())), -1).mean(axis=0)
        plt.plot(dims, avg_metric, '-', marker='o')
        plt.xticks(np.arange(np.amin(dims), np.amax(dims)+d_rank, d_rank))
        plt.yticks(np.arange(40, 100, 10))
        plt.xlabel('Rank')
        plt.ylabel('Accuracy (%)')
        plt.grid("on")
        plt.title(task)
    plt.legend(experiments, loc="lower right", fontsize=8)
plt.savefig(f"{task}.png", dpi=300)
#tikzplotlib.save("test.tex")
