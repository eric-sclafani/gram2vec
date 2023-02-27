
import os
import numpy as np
from collections import defaultdict
import json
from more_itertools import chunked
import pandas as pd

def load_data(data_path:str) -> dict[str, list[dict]]:
    """Loads in a JSON consisting of author_ids mapped to lists of dict entries as a dict"""
    with open(data_path) as fin:
        data = json.load(fin)
    return data

def count_num_entries(author_dict:dict) -> int:
    """Counts how many documents an authr has by taking the length of their list of dict objects"""
    return len(author_dict)

def make_author_bins(iterable):
    """Makes 7 bins given an iterable"""
    return tuple(chunked(iterable, 7))

def author_doc_counts_dict(train_path:str) -> dict:
    """
    Stores each unique author_id with their document count
    
    :param train_dir_path: training data directory
    :returns: dict consisting of author_ids mapped to document count
    """
    author_docfreq_map = defaultdict(list)
    for author_id, author_dict in load_data(train_path).items():
        author_docfreq_map[author_id] = count_num_entries(author_dict)
    return author_docfreq_map

def get_train_authors_sorted_by_docfreq(train_path:str) -> list[str]:
    """Sorts a dict by document count and returns the sorted author_id labels"""
    df = author_doc_counts_dict(train_path)
    sorted_by_docfreq = dict(sorted(df.items(), key=lambda items: items[1]))
    return tuple(sorted_by_docfreq.keys())

def sort_dev_by_train_docfreq(train_sorted_bins:tuple[list], dev_path:str) -> list[tuple]:
    """
    Sorts the dev set based on the train set sorted by document frequency. 
    Creates a list of tuples of the format: (bin_num, data), where data is a list of 
    author_ids mapped to a list of document entries
    """
    dev_set = load_data(dev_path)
    sorted_dev_bins = []
    
    for i, bin in enumerate(train_sorted_bins):
        current_bin_data = []
        for author_id in bin:
            current_bin_data.append({author_id:dev_set[author_id]})
        
        sorted_dev_bins.append((f"devbin{i+1}", current_bin_data))
            
    return sorted_dev_bins
    
    
def write_sorted_bins(sorted_bins:tuple, target_dir:str):
    """Writes the sorted bins to a target directory"""
    for bin in sorted_bins:
        with open(f"{target_dir}/{bin[0]}.json", "w") as fout:
            for entry in bin[1]:
                json.dump(entry, fout, indent=2, ensure_ascii=False)


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
    train_path = "eval/pan22_splits/knn/train.json"
    
    train_authors_sorted_by_docfreq = get_train_authors_sorted_by_docfreq(train_path)
    bin_labels_sorted_by_train = make_author_bins(train_authors_sorted_by_docfreq)
    dev_sorted_by_train = sort_dev_by_train_docfreq(bin_labels_sorted_by_train,"eval/pan22_splits/knn/dev.json")
    write_sorted_bins(dev_sorted_by_train, "eval/eval_bins/sorted_by_doc_freq")


    #! CODE DEPRECATED. NEEDS TO BE UPDATED
    #write_bins("eval/pan22_splits/knn/dev.json", "eval/eval_bins/sorted_by_doc_freq/", bins_sorted_by_docfreq)
    #authors_sorted_by_avg_tokens = get_authors_sorted_by_avg_tokens(train_path)
    #write_bins("eval/pan22_splits/knn/dev", "eval/eval_bins/sorted_by_avg_tokens", bins_sorted_by_avg_tokens)
            
     
    
    
   
    
    



if __name__ == "__main__":
    main()

        
        
