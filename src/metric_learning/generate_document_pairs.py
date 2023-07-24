#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import time
import pickle 
from typing import Iterable, List, Tuple
from dataclasses import dataclass
from more_itertools import distinct_combinations

from gram2vec import vectorizer

@dataclass
class Document:
    
    entry:pd.Series

    @property
    def text(self):
        return self.entry["fullText"].values[0]
    
    @property
    def author_id(self):
        return self.entry["authorIDs"].values[0]

def measure_time(func):
    """Debugging function for measuring function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds.")
        return result
    return wrapper

def load_data(path:str) -> pd.DataFrame:
    df = vectorizer.load_jsonlines(path)
    df["authorIDs"] = df["authorIDs"].apply(lambda x: "".join(x))
    return df
    
def get_unique_author_ids(df:pd.DataFrame) -> List[str]:
    return df["authorIDs"].unique().tolist()

def get_author_docs(df:pd.DataFrame, author_id:str) -> pd.Series:
    return df["fullText"].loc[df["authorIDs"] == author_id]

def apply_vectorizer(documents:Iterable) -> np.ndarray:
    return vectorizer.from_documents(documents).values

def to_array(iter:List[float]) -> np.ndarray:
    return np.array(iter)

def get_string(series:pd.Series) -> str:
    return series.fullText.values[0]

def get_author(series:pd.Series) -> str:
    return series.authorIDs.values[0]

def difference(pair:Tuple[List,List]) -> np.ndarray:
    """|a-b|"""
    return np.abs(to_array(pair[0]) - to_array(pair[1]))

def calculate_difference(pairs:Iterable[Tuple]) -> np.ndarray:
    return np.array([difference(pair) for pair in pairs])

def create_same_author_vectors(author_ids:List[str], data:pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
    """
    For each author, creates all possible distinct combinations of vector pairs and calculate their element-wise similarity
    """
    X_train = []
    y_train = []
    for author_id in author_ids:
        documents = get_author_docs(data, author_id)
        vectors = apply_vectorizer(documents)
        
        same_author_vector_pairs = distinct_combinations(vectors.tolist(), r=2)
        similarity_vectors = 1 - calculate_difference(same_author_vector_pairs)
        
        for vector in similarity_vectors:
            X_train.append(vector)
            y_train.append(1)
            
    return np.array(X_train), y_train

def create_different_author_vectors(data:pd.DataFrame, same_vectors_shape:Tuple[int,int]) -> Tuple[np.ndarray, List[str]]:
    """Creates similarity vectors using documents from different authors. The amount is equal to the # of same author vectors"""
    
    X_train = []
    y_train = []
    for _ in range(same_vectors_shape[0]):
        random_doc_1 = Document(data[["authorIDs", "fullText"]].sample(n=1))
        random_doc_2 = Document(data[["authorIDs", "fullText"]].sample(n=1)) 
        vector1 = apply_vectorizer([random_doc_1.text]).squeeze()
        vector2 = apply_vectorizer([random_doc_2.text]).squeeze()
        
        if random_doc_1.author_id != random_doc_2.author_id:
            similarity = 1 - difference([vector1, vector2])
            X_train.append(similarity)
            y_train.append(0)
                        
    return np.array(X_train), y_train

def write_to_file(obj, path:str):
    with open(path, "wb") as writer:
        pickle.dump(obj, writer)
    

@measure_time
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--dataset_dir",
                        default="../../data/hrs_release_May23DryRun")
    
    args = parser.parse_args()
    data = load_data(args.dataset_dir)
    author_ids = get_unique_author_ids(data)
    
    same_author_vectors, same_author_labels = create_same_author_vectors(author_ids, data)
    diff_author_vectors, diff_author_labels = create_different_author_vectors(data, same_author_vectors.shape)
    
    X_train = np.concatenate([same_author_vectors, diff_author_vectors], axis=0)
    y_train = same_author_labels + diff_author_labels
    

    write_to_file(X_train, "metric_learn_data/X_train.pkl")
    write_to_file(y_train, "metric_learn_data/y_train.pkl")
        
        
    
        
        

   

    

    


if __name__ == "__main__":
    main()