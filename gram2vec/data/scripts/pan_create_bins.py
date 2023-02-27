
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

def make_author_bins(iterable):
    """Makes 7 bins given an iterable"""
    return list(chunked(iterable, 7))

def author_doc_counts_df(train_dir:str) -> pd.DataFrame:
    """
    Stores each unique author_id with the document count
    
    :param train_dir_path: training data directory
    :returns: dataframe consisting of author_ids and corresponding document counts
    """
    author_docfreq_map = defaultdict(lambda:[])
    for author_file in iter_author_jsonls(train_dir):
        author_docfreq_map["author_id"].append(clean_file_name(author_file)) 
        author_docfreq_map["doc_count"].append(count_num_entries(author_file))
    return pd.DataFrame(author_docfreq_map)

def get_authors_sorted_by_docfreq(train_dir:str) -> list[str]:
    """Sorts a dataframe by doc_count and returns the sorted author_id labels"""
    df = author_doc_counts_df(train_dir)
    df.sort_values("doc_count", inplace=True)
    return df.author_id.values

def write_bins(source_dir:str, target_dir:str, bins:list[list]):
    """
    Writes each bin to file given source and target directories, and list of bins
    :param source_dir: directory to create bins from
    :param target_dir: directory to place the bins
    """
    for i, author_id_bin in enumerate(bins):
        for author_id in author_id_bin:
            bin_path = f"{target_dir}/devbin{i+1}"
            if not os.path.exists(bin_path):
                os.mkdir(bin_path)
            
            os.system(f"cp {source_dir}/{author_id}.jsonl {bin_path}")

def count_author_avg_tokens(author_file:str) -> float:
    """Aggregates the token counts of an author and takes the average"""
    author_token_counts = [len(doc["fixed_text"].split()) for doc in iter_author_entries(author_file)]
    return np.mean(author_token_counts)

def author_avg_tokens_df(train_dir:str) -> pd.DataFrame:
    """
    Stores each unique author_id with corresponding average token count
    
    :param train_dir_path: training data directory
    :returns: dataframe consisting of author_ids and corresponding average token count
    """
    author_avg_tokens_map = defaultdict(lambda:[])
    for author_file in iter_author_jsonls(train_dir):
        author_avg_tokens_map["author_id"].append(clean_file_name(author_file))
        author_avg_tokens_map["avg_token_count"].append(count_author_avg_tokens(author_file))
    return pd.DataFrame(author_avg_tokens_map)          
            
def get_authors_sorted_by_avg_tokens(train_dir:str) -> list[str]:
    """Sorts a dataframe by avg_token_count and returns the sorted author_id labels"""
    df = author_avg_tokens_df(train_dir)
    df.sort_values("avg_token_count", inplace=True)
    return df.author_id.values
            
def main():
    
    os.chdir("../../")
    train_dir = "eval/pan22_splits/knn/train"
    
    authors_sorted_by_docfreq = get_authors_sorted_by_docfreq(train_dir)
    bins_sorted_by_docfreq = make_author_bins(authors_sorted_by_docfreq)
    #write_bins("eval/pan22_splits/knn/dev", "eval/sorted_by_doc_freq/eval_bins", bins_sorted_by_docfreq)
    
    authors_sorted_by_avg_tokens = get_authors_sorted_by_avg_tokens(train_dir)
    bins_sorted_by_avg_tokens = make_author_bins(authors_sorted_by_avg_tokens)
    write_bins("eval/pan22_splits/knn/dev", "eval/eval_bins/sorted_by_avg_tokens", bins_sorted_by_avg_tokens)
            
     
    
    
   
    
    



if __name__ == "__main__":
    main()

        
        
