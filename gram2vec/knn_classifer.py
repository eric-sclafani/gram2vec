#!/usr/bin/env python3

import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from datetime import datetime

# project imports
import utils
from featurizers import GrammarVectorizer


def vectorize_data(data, g2v) -> np.ndarray:
    """Vectorizes a dict of documents. Returns a matrix from all documents"""
    vectors = []
    authors = []
    for id in data.keys():
        for text in data[id]:
            grammar_vector = g2v.vectorize(text)
            vectors.append(grammar_vector)
            authors.append(id)
    
    return np.stack(vectors), authors



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
    
    g2v = GrammarVectorizer(args.train_path, logging=True)
    le  = LabelEncoder()
    scaler = StandardScaler()
    
    
    train = utils.load_json(args.train_path)
    eval  = utils.load_json(args.eval_path)
    
    X_train, Y_train = vectorize_data(train, g2v) 
    X_eval,  Y_eval  = vectorize_data(eval, g2v)
    
    Y_train_encoded = le.fit_transform(Y_train)
    Y_eval_encoded  = le.transform(Y_eval)
    
    X_train = scaler.fit_transform(X_train)
    X_eval = scaler.transform(X_eval)
    
    model = KNeighborsClassifier(n_neighbors=int(args.k_value), metric=args.metric)
    model.fit(X_train, Y_train_encoded)
    
    predictions = model.predict(X_eval)
    accuracy = metrics.accuracy_score(Y_eval_encoded, predictions)

    feats = [feat.__name__ for feat in g2v._config()]
    eval_set = "dev" if args.eval_path.endswith("dev.json") else "test"
    result_path = f"results/{eval_set}_results.json" if "bin" not in args.eval_path else f"results/{eval_set}_bin_results.json"
    
    print(f"Eval set: {eval_set}")
    print(f"Features: {feats}")
    print(f"Feature vector size: {len(X_train[0])}")
    print(f"k: {args.k_value}")
    print(f"Metric: {args.metric}")
    print(f"Accuracy: {accuracy}")
    
    try:
        results = utils.load_json(result_path)
    except:
        utils.save_json({"results":[]}, result_path)
        results = utils.load_json(result_path)
    
    results["results"].append({"datetime": str(datetime.now()),
                               "acc": accuracy, 
                               "vector_length":f"{len(X_train[0])}",
                               "k": f"{args.k_value}",
                               "Metric": args.metric,
                               "config":feats})
    
    utils.save_json(data=results, path=result_path)
           

if __name__ == "__main__":
    main()