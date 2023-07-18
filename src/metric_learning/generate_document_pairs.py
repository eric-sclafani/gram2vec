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

def calculate_difference(pair:Tuple[List,List]):
    """|a-b|/2 """
    ar1 = to_array(pair[0])
    ar2 = to_array(pair[1])
    return np.abs(ar1 - ar2) / 2

def calculate_similarity(pair:Tuple[List,List]):
    """1-(|a-b|/2)"""
    return 1 - calculate_difference(pair)






def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--dataset_dir",
                        default="../../data/hrs_release_May23DryRun")
    
    args = parser.parse_args()
    data = load_data(args.dataset_dir)
    author_ids = get_unique_author_ids(data)
    
    for author_id in author_ids:
        documents = get_author_docs(data, author_id)
        vectors = apply_vectorizer(documents).tolist() # more_itertools.distinct_combinations crashes when working with np arrays
        same_author_vector_pairs = distinct_combinations(vectors, r=2)
        break
    
    a = [15,5,5]
    b = [2,2,2]
    
    result = calculate_difference((a,b))
    print(result)
    
    


if __name__ == "__main__":
    main()