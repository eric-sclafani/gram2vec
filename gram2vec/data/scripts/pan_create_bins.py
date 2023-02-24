
import os
import numpy as np
import jsonlines
from pathlib import Path
from collections import defaultdict
import re
from more_itertools import chunked
import pandas as pd

def iter_author_jsonls(author_files_dir:str) -> str:
    """Yields each {author_id}.jsonl from a given dir"""
    for author_file in Path(author_files_dir).glob("*.jsonl"):
        yield author_file
        
def iter_author_entries(author_file):
    """Yields each JSON object from an {author_id}.jsonl file"""
    with jsonlines.open(author_file) as author_entries:
        for entry in author_entries:
            yield entry

def count_num_entries(author_file:str) -> int:
    """Counts how many documents are in a given {author_id}.jsonl"""
    return sum([1 for _ in iter_author_entries(author_file)])

def clean_file_name(file_name:str) -> str:
    """Removes extra path information from jsonl file name"""
    return re.sub(r"(eval/pan22_splits/knn/train/)|.jsonl", "", str(file_name))

def author_doc_counts_df(train_dir_path:str) -> pd.DataFrame:
    """
    Stores each unique author_id with the document count
    
    :param train_dir_path: training data directory
    :returns: dataframe consisting of author_ids and corresponding document counts
    """
    author_docfreq_map = defaultdict(lambda:[])
    for author_file in iter_author_jsonls(train_dir_path):
        author_docfreq_map["author_id"].append(clean_file_name(author_file)) 
        author_docfreq_map["doc_count"].append(count_num_entries(author_file))
    return pd.DataFrame(author_docfreq_map)

def get_sorted_authors_ids(train_dir:str) -> list[str]:
    """
    Sorts a dataframe by doc_count and returns the sorted author_id labels
    
    :param train_dir_path: training data directory
    :returns: list of author_ids sorted by document frequency
    """
    df = author_doc_counts_df(train_dir)
    df.sort_values("doc_count", inplace=True)
    return df.author_id.values

def create_bins_by_docfreq(source_dir:str, target_dir:str, bins:list[list]):
    """
    Writes each bin to file given source and target directories, and list of bins
    
    :param source_dir: directory to create bins from
    :param target_dir: directory to place the bins
    """
    for i, author_id_bin in enumerate(bins):
        for author_id in author_id_bin:
            os.system(f"cp {source_dir}/{author_id}.jsonl {target_dir}/bin{i+1}/")
            
            
def main():
    
    os.chdir("../../")
    authors_sorted_by_docfreq = get_sorted_authors_ids("eval/pan22_splits/knn/train")
    bins_sorted_by_docfreq = list(chunked(authors_sorted_by_docfreq, 7))
    create_bins_by_docfreq("eval/pan22_splits/knn/dev", "eval/eval_bins", bins_sorted_by_docfreq)
    
    
    
            
    
    
    
   
    
    



if __name__ == "__main__":
    main()

        
        
