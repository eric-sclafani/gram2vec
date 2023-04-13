#!/usr/bin/env python3


###############################################################################################################
#                      THIS EVAL SCRIPT IS DEPRECATED. PLEASE SEE PAUSIT-EVAL REPO 
###############################################################################################################
raise DeprecationWarning("Module deprecated for now")

import argparse
import os
import csv
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from datetime import datetime
from typing import List, Dict
import pickle
from time import time

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

def load_data(data_path:str) -> Dict[str, List[dict]]:
    """Loads in a JSON consisting of author_ids mapped to lists of dict entries as a dict"""
    with open(data_path) as fin:
        data = json.load(fin)
    return data

def get_all_documents(data_path:str, text_type="fixed_text") -> List[str]:
    """Aggregates all documents from a json file into one list"""
    all_documents = []
    for author_entries in load_data(data_path).values():
        for entry in author_entries:
            all_documents.append(entry[text_type])
            
    return all_documents

def get_authors(data_path:str) -> List[str]:
    """Aggregates all authors from a json file into one list"""
    all_authors = []
    for author_entries in load_data(data_path).values():
        for entry in author_entries:
            all_authors.append(entry["author_id"])
    return all_authors

def get_result_path(eval_path:str, dataset_name:str, dev_or_test:str) -> str:
    """Determines whether result path should be for bins or overall eval data"""
    if "bin" in eval_path:
        result_path = f"eval/results/{dataset_name}_{dev_or_test}_bin_results.csv"
    else:
        result_path = f"eval/results/{dataset_name}_{dev_or_test}_results.csv"
    return result_path

def get_dataset_name(train_path:str) -> str:
    """
    Gets the dataset name from training data path which is needed to generate paths
    NOTE: This function needs to be manually updated when new datasets are used.
    """
    if "pan" in train_path:
        dataset_name = "pan"
    else:
        raise ValueError(f"Dataset name unrecognized in path: {train_path}")
    return dataset_name 

def write_results_entry(path:str, to_write:List):
    
    if not os.path.exists(path):
        with open(path, "w") as fout:
            writer = csv.writer(fout)
            writer.writerow(["Datetime", "Score", "Vector length", "Metric", "Has content vector", "config"])
            
    with open(path, "a") as fout:
        writer = csv.writer(fout)
        writer.writerow(to_write)
            
def fetch_labels_from_indices(indices:np.ndarray, encoded_labels:np.ndarray) -> np.ndarray:
    """Fetches labels from given array of index positions"""
    return encoded_labels[indices]

def get_first_8_authors(predicted_labels:np.ndarray) -> List[int]:
    """Retrieves the first 8 unique labels from an array of predicted labels"""
    candidates = []
    for label in predicted_labels:
        if label not in candidates and not len(candidates) == 8:
            candidates.append(label)
    assert len(candidates) == 8, "Not enough candidates found while calculating R@8"
    return candidates

def load_metric(path:str):
    """Loads a trained MMC model from disk"""
    with open(path, "rb") as fin:
        return pickle.load(fin)
    
        
def recall_at_1(k:int, X_train:np.ndarray, X_eval:np.ndarray, y_train_encoded:np.ndarray, y_eval_encoded:np.ndarray) -> float:
    """
    Evaluates a KNN model using the majority vote algorithm to calculate R@1
    
    :param k: kNN k value
    :param X_train: vectorized training matrix
    :param X_eval: vectorized evaluation matrix
    :param y_train_encoded: array of encoded training labels
    :param y_eval_encoded: array of encoded evaluation labels
    
    :returns: R@1 score
    """
    model = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    model.fit(X_train, y_train_encoded)
    
    predictions = model.predict(X_eval)
    accuracy = metrics.accuracy_score(y_eval_encoded, predictions)
    return accuracy

def recall_at_8(X_train:np.ndarray, X_eval:np.ndarray, y_train_encoded:np.ndarray, y_eval_encoded:np.ndarray) -> float:
    """
    Evaluates a KNN model using R@8. Checks to see if the 
    query author is in the top 8 nearest authors predicted by the model
    
    :param X_train: vectorized training matrix
    :param X_eval: vectorized evaluation matrix
    :param y_train_encoded: array of encoded training labels
    :param y_eval_encoded: array of encoded evaluation labels
    
    :returns: R@8 score
    """
    correct_pred = 0
    all_pred = 0
    model = KNeighborsClassifier(n_neighbors=50, metric="cosine")
    model.fit(X_train, y_train_encoded)
    
    _, prediction_indices = model.kneighbors(X_eval)
    for i in range(len(prediction_indices)):
        
        predicted_authors = fetch_labels_from_indices(prediction_indices[i], y_train_encoded)
        first_eight_authors = get_first_8_authors(predicted_authors)
        
        if y_eval_encoded[i] in first_eight_authors:
            correct_pred += 1
            
        all_pred += 1
    
    return correct_pred / all_pred

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
    "morph_tags":1
}        
   
@timer
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", 
                        "--k_value", 
                        type=int, 
                        help="k value to calculate R@1. Is ignored when --metric == R@8", 
                        default=6)
    
    parser.add_argument("-m", 
                        "--metric", 
                        type=str, 
                        help="Metric to calculate",
                        choices=["R@1", "R@8"],
                        default="R@1")
    
    parser.add_argument("-train", 
                        "--train_path", 
                        type=str, 
                        help="Path to train directory",
                        default="eval/pan22_splits/knn/train.json") 
    
    parser.add_argument("-eval", 
                        "--eval_path", 
                        type=str,
                        help="Path to eval directory",
                        default="eval/pan22_splits/knn/dev.json") 
    
    args = parser.parse_args()
    
    g2v = GrammarVectorizer(config=G2V_CONFIG)
    le  = LabelEncoder()
    scaler = StandardScaler()
    
    train_docs = get_all_documents(args.train_path, text_type="raw_text")
    X_train = g2v.vectorize_episode(train_docs)
    y_train = get_authors(args.train_path)

    eval_docs = get_all_documents(args.eval_path, text_type="raw_text")
    X_eval = g2v.vectorize_episode(eval_docs)
    y_eval = get_authors(args.eval_path)
    
    y_train_encoded = le.fit_transform(y_train)
    y_eval_encoded  = le.transform(y_eval)
    
    X_train = scaler.fit_transform(X_train)
    X_eval = scaler.transform(X_eval)
    
    
    if args.metric == "R@1":
        eval_score = recall_at_1(args.k_value, X_train, X_eval, y_train_encoded, y_eval_encoded)
        
    elif args.metric == "R@8":
        eval_score = recall_at_8(X_train, X_eval, y_train_encoded, y_eval_encoded)
        
    dev_or_test  = "dev" if "dev" in args.eval_path else "test"
    dataset_name = get_dataset_name(args.train_path)
    result_path  = get_result_path(args.eval_path, dataset_name, dev_or_test)
    has_content_vector = "embedding_vector" in g2v.get_config()
    
    print(f"Eval set: {dev_or_test}")
    print(f"Features: {g2v.get_config()}")
    print(f"Has content vector: {has_content_vector}")
    print(f"Feature vector size: {len(X_train[0])}")
    print(f"Metric: {args.metric}")
    print(f"Evaluation score: {eval_score}")
    
    write_results_entry(result_path, [datetime.now(),
                                      eval_score,
                                      len(X_train[0]),
                                      args.metric,
                                      has_content_vector,
                                      str(g2v.get_config())])
 
if __name__ == "__main__":
    main()