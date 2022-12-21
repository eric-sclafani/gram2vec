#!/usr/bin/env python3

import argparse
import utils
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from featurizers import GrammarVectorizer


def vectorize_data(data, g2v, config) -> np.ndarray:
    """Vectorizes a dict of documents. Returns a matrix from all documents"""
    vectors = []
    authors = []
    for id in data.keys():
        for text in data[id]:
            grammar_vector = g2v.vectorize(text, config=config)
            vectors.append(grammar_vector)
            authors.append(id)
    #import ipdb;ipdb.set_trace()
    return np.stack(vectors), authors

@utils.timer_func
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--k_value", type=int, help="k value for K-NN", default=7)
    parser.add_argument("-cfg", "--config", type=str, required=True, help="Configuration for featurizers")
    parser.add_argument("-train", "--train_path", type=str, required=True, help="Path to train data") 
    parser.add_argument("-eval", "--eval_path", type=str, required=True, help="Path to eval data") 
    args = parser.parse_args()
    
    config = list(map(lambda x: int(x), args.config.split()))
    g2v    = GrammarVectorizer()
    le     = LabelEncoder()
    scalar = StandardScaler()
    
    train  = utils.load_json(args.train_path)
    eval   = utils.load_json(args.eval_path)
    
    # vectorize the data
    
    X_train, Y_train = vectorize_data(train, g2v, config) 
    X_eval,  Y_eval  = vectorize_data(eval, g2v, config)

    # scale the vectors to behave for sklearn. 
    X_train_scaled = scalar.fit_transform(X_train)
    X_eval_scaled  = scalar.transform(X_eval)
    
    # encode the labels
    Y_train_encoded = le.fit_transform(Y_train)
    Y_eval_encoded  = le.transform(Y_eval)
    
    
    model = KNeighborsClassifier(n_neighbors=int(args.k_value))
    model.fit(X_train_scaled, Y_train_encoded)
    
    predictions = model.predict(X_eval_scaled)
    accuracy = metrics.accuracy_score(Y_eval_encoded, predictions)

    print(accuracy)
    print(f"Features: {[g2v.featurizers[cfg].__name__ for cfg in config]}")


    #! save putput

if __name__ == "__main__":
    main()