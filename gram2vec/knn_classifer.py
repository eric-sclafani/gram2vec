#!/usr/bin/env python3

import argparse
import numpy as np
import os
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from datetime import datetime

# project imports
import utils
from featurizers import GrammarVectorizer

def vectorize_all_data(data:dict, g2v:GrammarVectorizer) -> np.ndarray:
    """Vectorizes a dict of documents. Returns a matrix from all documents"""
    vectors = []
    for author_id in data.keys():
        for text in data[author_id]:
            grammar_vector = g2v.vectorize(text)
            vectors.append(grammar_vector)
    return np.stack(vectors)

def get_authors(data:dict) -> list[int]:
    """Get all instances of authors from data"""
    authors = []
    for author_id in data.keys():
        for _ in data[author_id]:
            authors.append(author_id)
    return authors

def get_result_path(eval_path, dataset_name, dev_or_test):

    if "bin" in eval_path:
        result_path = f"results/{dataset_name}_{dev_or_test}_bin_results.csv"
    else:
        result_path = f"results/{dataset_name}_{dev_or_test}_results.csv"
    return result_path

def write_results_entry(path, to_write:list):
    
    if not os.path.exists(path):
        with open(path, "w") as fout:
            writer = csv.writer(fout)
            writer.writerow(["Datetime", "Accuracy","Vector_length","k","Metric","config"])
            
    with open(path, "a") as fout:
        writer = csv.writer(fout)
        writer.writerow(to_write)
      
      
@utils.timer_func
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", 
                        "--k_value", 
                        type=int, 
                        help="k value for K-NN", 
                        default=7)
    
    parser.add_argument("-m", 
                        "--metric", 
                        type=str, 
                        help="distance metric", 
                        default="cosine")
    
    parser.add_argument("-train", 
                        "--train_path", 
                        type=str, 
                        help="Path to train data",
                        default="data/pan/train_dev_test/train.json") 
    
    parser.add_argument("-eval", 
                        "--eval_path", 
                        type=str,
                        help="Path to eval data",
                        default="data/pan/train_dev_test/dev.json") 
    
    args = parser.parse_args()
    
    g2v = GrammarVectorizer()
    le  = LabelEncoder()
    scaler = StandardScaler()
    
    train = utils.load_json(args.train_path)
    eval  = utils.load_json(args.eval_path)
    
    X_train = vectorize_all_data(train, g2v) 
    Y_train = get_authors(train)
    
    X_eval = vectorize_all_data(eval, g2v)
    Y_eval = get_authors(eval)
    
    Y_train_encoded = le.fit_transform(Y_train)
    Y_eval_encoded  = le.transform(Y_eval)
    
    X_train = scaler.fit_transform(X_train)
    X_eval = scaler.transform(X_eval)
    
    model = KNeighborsClassifier(n_neighbors=int(args.k_value), metric=args.metric)
    model.fit(X_train, Y_train_encoded)
    
    predictions = model.predict(X_eval)
    accuracy = metrics.accuracy_score(Y_eval_encoded, predictions)
    activated_feats = [feat.__name__ for feat in g2v.config]
    
    dev_or_test = "dev" if args.eval_path.endswith("dev.json") else "test"
    dataset_name = utils.get_dataset_name(args.train_path)
    result_path = get_result_path(args.eval_path, dataset_name, dev_or_test)
    
    print(f"Eval set: {dev_or_test}")
    print(f"Features: {activated_feats}")
    print(f"Feature vector size: {len(X_train[0])}")
    print(f"k: {args.k_value}")
    print(f"Metric: {args.metric}")
    print(f"Accuracy: {accuracy}")
    
    write_results_entry(result_path, [datetime.now().strftime("%c"),
                                      accuracy,
                                      len(X_train[0]),
                                      args.k_value,
                                      args.metric,
                                      str(activated_feats)])
 
if __name__ == "__main__":
    main()