#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from typing import Iterable, List, Tuple
from more_itertools import distinct_combinations

from gram2vec import vectorizer


def load_data(path:str) -> pd.DataFrame:
    df = vectorizer.load_jsonlines(path)
    df["authorIDs"] = df["authorIDs"].apply(lambda x: "".join(x))
    return df
    
def get_unique_author_ids(df:pd.DataFrame) -> np.ndarray[str]:
    return df["authorIDs"].unique()

def get_author_docs(df:pd.DataFrame, author_id:str) -> pd.Series:
    return df["fullText"].loc[df["authorIDs"] == author_id]

def apply_vectorizer(documents:Iterable) -> np.ndarray:
    return vectorizer.from_documents(documents).values

def to_array(iter:List[float]) -> np.ndarray:
    return np.array(iter)

def difference(pair:Tuple[List,List]) -> np.ndarray:
    """|a-b|"""
    return np.abs(to_array(pair[0]) - to_array(pair[1]))

def calculate_difference(pairs:Iterable[Tuple]) -> np.ndarray:
    return np.array([difference(pair) for pair in pairs])


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--dataset_dir",
                        default="../../data/hrs_release_May23DryRun")
    
    args = parser.parse_args()
    data = load_data(args.dataset_dir)
    author_ids = get_unique_author_ids(data)
    

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
        
        import ipdb;ipdb.set_trace()
        
        
        
    
        
        

   

    

    


if __name__ == "__main__":
    main()