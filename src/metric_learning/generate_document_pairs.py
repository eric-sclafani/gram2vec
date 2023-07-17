#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np

from gram2vec import vectorizer


def get_unique_author_ids(df:pd.DataFrame) -> np.ndarray[str]:
    return df["authorIDs"].unique()


def get_author_docs(df:pd.DataFrame, author_id:str) -> pd.Series:
    return df.loc[df["authorIDs"] == author_id]


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--dataset_dir",
                        default="../../data/hrs_release_May23DryRun")
    
    args = parser.parse_args()
    data = vectorizer.load_jsonlines(args.dataset_dir)
    


if __name__ == "__main__":
    main()