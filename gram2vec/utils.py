import json
import spacy
from nltk import Tree
import numpy as np
import pickle
from time import time
import subprocess


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

def load_json(path) -> dict:
    """Loads a JSON as a dict"""
    with open (path, "r") as fin:
        data = json.load(fin)
        return data
    
def save_json(data:dict, path, mode="w") -> None:
    """Saves a dict as a JSON"""
    with open(path, mode) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)

def load_spacy(model:str):
    
    try:
        nlp = spacy.load(model)
    except OSError:
        print(f"{model} not detected. Downloading now...")
        subprocess.run(f"python -m spacy download {model}") # untested
        
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

def _to_nltk_tree(node):
    """
    Converts a spacy parse tree into nltk Tree (for visualization purposes w/o using displacy)
    Credits: https://stackoverflow.com/questions/36610179/how-to-get-the-dependency-tree-with-spacy
    """
    tok_format = lambda tok: "_".join([tok.orth_, tok.tag_, tok.dep_])      
    if node.n_lefts + node.n_rights > 0:
        return Tree(tok_format(node), [_to_nltk_tree(child) for child in node.children])
    else:
        return tok_format(node)
    
def tree(root):
    """Print a NLTK style parse tree given the root"""
    _to_nltk_tree(root).pretty_print()
    