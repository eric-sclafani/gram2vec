#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from itertools import combinations
from typing import Iterable

from gram2vec import vectorizer

def load_data(path) -> pd.DataFrame:
    df = vectorizer.load_jsonlines(path)
    df["authorIDs"] = df["authorIDs"].apply(lambda x: "".join(x))
    return df
    
def get_unique_author_ids(df:pd.DataFrame) -> np.ndarray[str]:
    return df["authorIDs"].unique()

def get_author_docs(df:pd.DataFrame, author_id:str) -> pd.Series:
    return df["fullText"].loc[df["authorIDs"] == author_id]

def apply_vectorizer(documents) -> pd.DataFrame:
    return vectorizer.from_documents(documents).values

def get_all_possible_combinations(vectors:Iterable[np.ndarray]):
    return combinations(vectors, r=2)




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
        vectors = apply_vectorizer(documents)
        combos = list(combinations(vectors, 2))
        
        break


if __name__ == "__main__":
    main()