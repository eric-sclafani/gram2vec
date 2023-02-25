#!/usr/bin/env python3

import argparse
import os
import csv
import jsonlines
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from datetime import datetime
from time import time
import re

# project imports
from gram2vec.featurizers import GrammarVectorizer

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

def iter_author_jsonls(author_files_dir:str) -> str:
    """Yields each {author_id}.jsonl from a given dir"""
    for author_file in Path(author_files_dir).glob("*.jsonl"):
        yield author_file
        
def iter_author_entries(author_file):
    """Yields each JSON object from an {author_id}.jsonl file"""
    with jsonlines.open(author_file) as author_entries:
        for entry in author_entries:
            yield entry
            
def get_all_documents(train_dir:str, text_type="fixed_text") -> list[str]:
    """Aggregates all documents into one list"""
    all_documents = []
    for author_file in iter_author_jsonls(train_dir):
        for entry in iter_author_entries(author_file):
            all_documents.append(entry[text_type])
    return all_documents

def get_authors(train_dir:str) -> list[str]:

    all_authors = []
    for author_file in iter_author_jsonls(train_dir):
        for entry in iter_author_entries(author_file):
            all_authors.append(entry["author_id"])
    return all_authors

def get_result_path(eval_dir:str, dataset_name:str, dev_or_test:str):
    """Determines whether result path should be for bins or overall eval data"""
    if "bin" in eval_dir:
        result_path = f"eval/results/{dataset_name}_{dev_or_test}_bin_results.csv"
    else:
        result_path = f"eval/results/{dataset_name}_{dev_or_test}_results.csv"
    return result_path

def get_dataset_name(train_dir:str) -> str:
    """
    Gets the dataset name from training data path which is needed to generate paths
    NOTE: This function needs to be manually updated when new datasets are used.
    """
    if "pan" in train_dir:
        dataset_name = "pan"
    else:
        raise ValueError(f"Dataset name unrecognized in path: {train_dir}")
    return dataset_name 

def write_results_entry(path, to_write:list):
    
    if not os.path.exists(path):
        with open(path, "w") as fout:
            writer = csv.writer(fout)
            writer.writerow(["Datetime", "Accuracy", "Vector_length", "k", "Distance function", "config"])
            
    with open(path, "a") as fout:
        writer = csv.writer(fout)
        writer.writerow(to_write)
        
def majority_vote(model:KNeighborsClassifier, X_eval:np.ndarray, y_eval_encoded:np.ndarray) -> float:
    """
    Evaluates a KNN model using the classic majority vote algorithm
    
    :param model: KNN classifier instance
    :param X_eval: matrix of document vectors from eval set
    :param: y_eval_encoded: vector of encoded author_id eval labels
    :returns: accuracy score
    """
    predictions = model.predict(X_eval)
    accuracy = metrics.accuracy_score(y_eval_encoded, predictions)
    return accuracy

def recall_at_n(n:int):
    pass   
 

        
G2V_CONFIG = {
    "pos_unigrams":1,
    "pos_bigrams":1,
    "func_words":1,
    "punc":1,
    "letters":1,
    "common_emojis":1,
    "embedding_vector":0,
    "document_stats":1,
    "dep_labels":1,
    "mixed_bigrams":1,
}        
   
@timer
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", 
                        "--k_value", 
                        type=int, 
                        help="k value for K-NN", 
                        default=7)
    
    parser.add_argument("-ef", 
                        "--eval_function", 
                        type=str, 
                        help="Evaluation function to calculate nearest neighbors metric",
                        choices=["majority_vote", "R@1", "R@8"],
                        default="majority_vote")
    
    parser.add_argument("-train", 
                        "--train_dir", 
                        type=str, 
                        help="Path to train directory",
                        default="eval/pan22_splits/knn/train/") 
    
    parser.add_argument("-eval", 
                        "--eval_dir", 
                        type=str,
                        help="Path to eval directory",
                        default="eval/pan22_splits/knn/dev/") 
    
    args = parser.parse_args()
    
    g2v = GrammarVectorizer(config=G2V_CONFIG)
    le  = LabelEncoder()
    scaler = StandardScaler()
    
    X_train = g2v.vectorize_episode(get_all_documents(args.train_dir))
    y_train = get_authors(args.train_dir)

    X_eval = g2v.vectorize_episode(get_all_documents(args.eval_dir))
    y_eval = get_authors(args.eval_dir)
    
    y_train_encoded = le.fit_transform(y_train)
    y_eval_encoded  = le.transform(y_eval)
    
    X_train = scaler.fit_transform(X_train)
    X_eval = scaler.transform(X_eval)
    
    model = KNeighborsClassifier(n_neighbors=int(args.k_value), metric="cosine")
    model.fit(X_train, y_train_encoded)
    
    if args.eval_function == "majority_vote":
        eval_score = majority_vote(model, X_eval, y_eval_encoded)
        
    elif re.search(r"R@\d", args.eval_function):
        
        eval_score = recall_at_n()


   
    
    activated_feats = [feat.__name__ for feat in g2v.config]
    dev_or_test     = "dev" if "dev" in args.eval_dir else "test"
    dataset_name    = get_dataset_name(args.train_dir)
    result_path     = get_result_path(args.eval_dir, dataset_name, dev_or_test)
    
    print(f"Eval set: {dev_or_test}")
    print(f"Features: {activated_feats}")
    print(f"Feature vector size: {len(X_train[0])}")
    print(f"k: {args.k_value}")
    print(f"Eval function: {args.eval_function}")
    print(f"Evaluation score: {eval_score}")
    
    write_results_entry(result_path, [datetime.now().strftime("%c"),
                                      eval_score,
                                      len(X_train[0]),
                                      args.k_value,
                                      args.eval_function,
                                      str(activated_feats)])
 
if __name__ == "__main__":
    main()