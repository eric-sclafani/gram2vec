#!/usr/bin/env python3

from metric_learn import MMC
import numpy as np
import pickle
import os
import jsonlines
import argparse
from dataclasses import dataclass
from tqdm import tqdm
from time import time

# project imports
from featurizers import GrammarVectorizer


@dataclass
class Pair:
    docs:tuple[str, str]
    same:bool


def load_metric_data(path) -> list[Pair]:
    """Reads in a jsonlines file and returns a list of Pair objects"""
    data:list[Pair] = []
    with jsonlines.open(path, "r") as reader:
        for obj in reader:
            data.append(Pair(obj["pair"], obj["same"]))
    return data

def vectorize_pair(g2v:GrammarVectorizer, pair:Pair) -> np.ndarray:
    """Applies the GrammarVectorizer to pairs of documents and returns a matrix of docuemnt vector pairs"""
    vec1 = g2v.vectorize_document(pair.docs[0])
    vec2 = g2v.vectorize_document(pair.docs[1])
    return np.array([vec1, vec2])

def get_pair_matrix(g2v:GrammarVectorizer, data:list[Pair]) -> np.ndarray:
    """Returns a 3-d matrix representing a collection of vector pairs"""
    vector_pairs = []
    for pair in tqdm(data, desc="Vectorizing document pairs"):
        vector_pairs.append(vectorize_pair(g2v, pair))
    return np.stack(vector_pairs)

def encode_labels(train:list[Pair]) -> np.ndarray:
    """
    Encodes True or False labels indicating same or different author pairs, respectively
    1 = True
    -1 = False
    """
    truth_labels = [1 if pair.same == True else -1 for pair in train]
    return np.array(truth_labels)

def timer(func):
    """This decorator shows the execution time of the function object passed"""
    # Credits: https://www.geeksforgeeks.org/timing-functions-with-decorators-python/
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func
          
@timer 
def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-t",
                        "--train_path",
                        type=str,
                        help="path to metric learning train data",
                        default="data/pan/train_dev_test/pairs/metric_train.jsonl")
    
    args = parser.parse_args()
    
    g2v = GrammarVectorizer()
    mmc = MMC()
    
    train = load_metric_data(args.train_path)
    pair_matrix = get_pair_matrix(g2v, train)
    truths = encode_labels(train)

    print("Fitting metric learning model...")
    mmc.fit(pair_matrix, truths)
    

if __name__ == "__main__":
    main()