import sys
import torch
import numpy as np
import gc
import sys
import time
import os
from train_and_validation.trials import multiple_trials_optuna
from dill.source import getsource
from suit import search_space_creator, info_creator, arguments

def save_json(obj, name):
    import json
    with open(name, "w") as f:
        json.dump(obj, f, skipkeys=True, indent=True)

def save_analyzed_names(analyzed_names, base_path):
    save_json(analyzed_names, base_path + 'analyzed_names')

def save_pure_info(info, base_path):
    chunks = info.split('\n')
    with open(base_path + 'info', 'a') as file:
        for chunk in chunks:
            file.write(chunk + '\n')

def save_search_space_creator(search_space_str, base_path):
    chunks = search_space_str.split('\n')
    with open(base_path + 'search_space', 'a') as file:
        for chunk in chunks:
            file.write(chunk + '\n')

def save_fixed_parameters(fixed_items, base_path):
    s = fixed_items.copy()
    for i in range(len(s)):
        if s[i][0] == "device":
            del(s[i])
            break
    save_json(s, base_path + 'fixed_parameters')

info = info_creator()

analyzed_names, prefix = arguments()
info['prefix'] = prefix
info['analyzed_names'] = analyzed_names

# creation of directory
# Parent Directory path
parent_dir = "./Experiments"
info['parent_dir'] = parent_dir

# Path
path = os.path.join(parent_dir, prefix)
os.mkdir(path)

save_analyzed_names(analyzed_names, 'Experiments/' + prefix + '/')
save_pure_info(getsource(info_creator), 'Experiments/' + prefix + '/')
save_search_space_creator(getsource(search_space_creator), 'Experiments/' + prefix + '/')

time1 = time.process_time()
multiple_trials_optuna(search_space_creator, info)

time2 = time.process_time()
print("total time of this cell in seconds", time2 - time1)

gc.collect()