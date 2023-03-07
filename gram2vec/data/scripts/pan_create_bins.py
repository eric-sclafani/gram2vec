
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
    """Counts how many documents an author has by taking the length of their list of dict objects"""
    return len(author_dict)

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

def get_train_authors_sorted_by_docfreq(train_path:str) -> dict:
    """Sorts a dict by document count and returns the sorted author_id labels"""
    count_dict = author_doc_counts_dict(train_path)
    sorted_by_docfreq = dict(sorted(count_dict.items(), key=lambda items: items[1]))
    return sorted_by_docfreq



# #! deprecated
# def count_author_avg_tokens(author_file:str) -> float:
#     """Aggregates the token counts of an author and takes the average"""
#     author_token_counts = [len(doc["fixed_text"].split()) for doc in iter_author_entries(author_file)]
#     return np.mean(author_token_counts)

# #! deprecated
# def author_avg_tokens_df(train_dir:str) -> pd.DataFrame:
#     """
#     Stores each unique author_id with corresponding average token count
    
#     :param train_dir_path: training data directory
#     :returns: dataframe consisting of author_ids and corresponding average token count
#     """
#     author_avg_tokens_map = defaultdict(lambda:[])
#     for author_file in iter_author_jsonls(train_dir):
#         author_avg_tokens_map["author_id"].append(clean_file_name(author_file))
#         author_avg_tokens_map["avg_token_count"].append(count_author_avg_tokens(author_file))
#     return pd.DataFrame(author_avg_tokens_map)     
     
# #! deprecated         
# def get_authors_sorted_by_avg_tokens(train_dir:str) -> list[str]:
#     """Sorts a dataframe by avg_token_count and returns the sorted author_id labels"""
#     df = author_avg_tokens_df(train_dir)
#     df.sort_values("avg_token_count", inplace=True)
#     return df.author_id.values

     
     
         
def bin_authors(sorted_dict:dict) -> tuple[list[str], ...]:
    """Makes bins of 7 authors given a sorted dict"""
    authors = list(sorted_dict.keys())
    return tuple(chunked(authors, 7))       
         
def write_sorted_bins(sorted_bins:tuple, target_dir:str):
    """Writes the sorted bins to a target directory"""
    for bin in sorted_bins:
        with open(f"{target_dir}/{bin[0]}.json", "w") as fout:
            json.dump(bin[1], fout, indent=2, ensure_ascii=False)
            
def create_dev_bins(train_sorted_bins:tuple[list], dev_path:str) -> list[tuple]:
    """
    Sorts the dev set based on the train-set-sorted labels
    
    :param train_sorted_bins: an eight tuple of author bins consisting of seven authors each
    :param dev_path: path to dev set to bin data from
    :returns: list of tuples of the format: (bin_num, data), where data is a dict of author_ids mapped to lists of document entry dictionaries
    """
    dev_set = load_data(dev_path)
    sorted_dev_bins = []
    for i, bin in enumerate(train_sorted_bins):
        current_bin_data = {}
        for author_id in bin:
            current_bin_data[author_id] = dev_set[author_id]
        
        sorted_dev_bins.append((f"devbin{i+1}", current_bin_data))
    return sorted_dev_bins
               
def main():
    
    os.chdir("../../")
    train_path = "eval/pan22_splits/knn/train.json"
    
    train_authors_dict = get_train_authors_sorted_by_docfreq(train_path)
    bin_labels_sorted_by_docfreq = bin_authors(train_authors_dict)
    #dev_sorted_by_train = create_dev_bins(bin_labels_sorted_by_docfreq,"eval/pan22_splits/knn/dev.json")
    #write_sorted_bins(dev_sorted_by_train, "eval/eval_bins/sorted_by_doc_freq")


    #! CODE DEPRECATED. NEEDS TO BE UPDATED
    #write_bins("eval/pan22_splits/knn/dev.json", "eval/eval_bins/sorted_by_doc_freq/", bins_sorted_by_docfreq)
    #authors_sorted_by_avg_tokens = get_authors_sorted_by_avg_tokens(train_path)
    #write_bins("eval/pan22_splits/knn/dev", "eval/eval_bins/sorted_by_avg_tokens", bins_sorted_by_avg_tokens)
            
     
    
    
   
    
    



if __name__ == "__main__":
    main()

        
        
