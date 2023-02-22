
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import jsonlines
import os
import re

@dataclass
class Partition:
    """Simple way to categorize which parition is train, dev, or test"""
    data:dict
    set_type:str


def iter_author_jsonls(author_files_dir:str) -> str:
    """Yields each {author_id}.jsonl from a given dir"""
    for author_file in Path(author_files_dir).glob("*.jsonl"):
        yield author_file

def extract_knn_splits_from_authors(authors_file_dir:str) -> tuple[Partition, Partition, Partition]:
    """
    Splits the given PAN22 training set into training, development, and testing partitions.
    This is only necessary until the official PAN22 testing set has been released.
    """
    train = defaultdict(lambda:[])
    dev = defaultdict(lambda:[])
    test = defaultdict(lambda:[])

    for file in iter_author_jsonls(authors_file_dir):
        with jsonlines.open(file) as author_entries:
            for i, entry in enumerate(author_entries):
                author_id = entry["author_id"]
                if i <= 4:
                    test[author_id].append(entry)
                elif i <= 9:
                    dev[author_id].append(entry)
                else:
                    train[author_id].append(entry)
    return Partition(train, "train"), Partition(dev,"dev"), Partition(test, "test")
                
            
def write_knn_splits(partitions:tuple[Partition], knn_splits_dir:str):
    
    for parition in partitions:
        for author_id, documents in parition.data.items():
            path = f"{knn_splits_dir}{parition.set_type}/{author_id}.jsonl"
            
            with jsonlines.open(path, "w") as author_file:
                author_file.write_all(documents)
    
    
                
            
def prepare_metric_learn_splits(raw_train, raw_dev, raw_test) -> tuple[list, list, list]:
    """
    Using the raw train, dev, test splits, sort the raw pairs into their own splits
    for the metric learning setup. Also applies the same text fixes as done to the regular eval data
    """
    doc_pairs = get_data("pan/raw/pairs.jsonl")
    doc_truths = get_data("pan/raw/truth.jsonl")
    assert len(doc_pairs) == len(doc_truths)
    
    metric_train, metric_dev, metric_test = [],[],[]
    
    for doc_entry, truth_entry in zip(doc_pairs, doc_truths):
        
        truth:bool = truth_entry["same"]
        pair:tuple[str,str] = tuple(doc_entry["pair"])
        entry = {"same":truth, "pair": fix_pair(pair)}
        
        if pair[0] in raw_train and pair[1] in raw_train:
            metric_train.append(entry)
            
        elif pair[0] in raw_train and pair[1] in raw_dev:
            metric_dev.append(entry)
            
        elif pair[0] in raw_dev and pair[1] in raw_train:
            metric_dev.append(entry)
            
        elif pair[0] in raw_train and pair[1] in raw_test:
            metric_test.append(entry)
            
        elif pair[0] in raw_test and pair[1] in raw_train:
            metric_test.append(entry)
            
        elif pair[0] in raw_dev and pair[1] in raw_dev:
            metric_dev.append(entry)
            
        elif pair[0] in raw_test and pair[1] in raw_test:
            metric_test.append(entry)
            
        elif pair[0] in raw_dev and pair[1] in raw_test:
            metric_dev.append(entry)
            
        elif pair[0] in raw_test and pair[1] in raw_dev:
            metric_test.append(entry) 
        else:
            raise Exception(f"Document unclassified: ({pair[0].split()[0:10]}, {pair[1].split()[0:10]})")
        
    return metric_train, metric_dev, metric_test

def write_metric_data_to_file(data:list[dict], out_path):
    """Write a list of dictionaries to jsonl file"""
    with jsonlines.open(out_path, "w") as fout:
        for entry in data:
            fout.write(entry)



    
def main():
    
    

    os.chdir("../")
    
    print("Partitioning KNN splits...")
    train, dev, test = extract_knn_splits_from_authors("pan22/preprocessed/")
    write_knn_splits((train, dev, test), "pan22/splits/knn/")
    print("Done!")
    
    
    
    
                
    


if __name__ == "__main__":
    main()