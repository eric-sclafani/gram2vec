import json
import spacy
import numpy as np
import pickle
from time import time

# this module is due for cleanup

def timer_func(func):
    # This function shows the execution time of the function object passed
    # Credits: https://www.geeksforgeeks.org/timing-functions-with-decorators-python/
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

def load_json(path) -> dict[str, list[str]]:
    """Loads a JSON as a dict"""
    with open (path, "r") as fin:
        data = json.load(fin)
        return data
    
def save_json(data:dict, path, mode="w"):
    """Saves a dict as a JSON"""
    with open(path, mode) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)
        
def load_vocab(path) -> tuple[str]:
    """Loads in a txt file delimited by newlines as a tuple of strings"""
    with open (path, "r") as fin:
        return tuple(map(lambda x: x.strip("\n"), fin.readlines()))

def load_spacy(model:str):

    nlp = spacy.load(model)
    for name in ["lemmatizer", "ner"]:
        nlp.remove_pipe(name)
    return nlp

def save_pkl(data, path):
    with open (path, "ab") as fout:
        pickle.dump(data, fout)
        
def load_pkl(path):
    with open (path, "rb") as fin:
        return pickle.load(fin)

def remove_dupes(iterable):
    """Removes duplicates from an iterable"""
    checked = []
    for n in iterable:
        if n not in checked:
            checked.append(n)
    return checked

def get_dataset_name(train_path:str) -> str:
    """
    Gets the dataset name from training data path.  
    Needed to generate path for vocab per dataset
    NOTE: This function needs to be manually updated when new datasets are used.
    """
    if "pan" in train_path:
        dataset_name = "pan"
    elif "mud" in train_path:
        dataset_name = "mud"
    # add other dataset names here following the same condition-checking format
    else:
        raise ValueError(f"Dataset name unrecognized in path: {train_path}")
    return dataset_name 