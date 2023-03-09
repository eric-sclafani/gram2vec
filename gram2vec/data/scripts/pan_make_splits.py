
from collections import defaultdict
from dataclasses import dataclass
import json
import jsonlines
import os
from pan_preprocess import apply_all_fixes
from typing import Union

@dataclass
class Partition:
    """Simple way to categorize which parition is train, dev, or test"""
    data:Union[dict, list]
    set_type:str
    
def load_preprocessed_data(preprocessed_file_path:str) -> dict[str, list[dict]]:
    """Loads in the preprocessed PAN data as a dict"""
    with open(preprocessed_file_path) as fin:
        data = json.load(fin)
    return data
    
def extract_knn_splits_from_authors(preprocessed_data:dict[str, list[dict]]) -> tuple[Partition, Partition, Partition]:
    """
    Splits the given PAN22 training set into training, development, and testing partitions.
    This is only necessary until the official PAN22 testing set has been released.
    """
    train = defaultdict(list)
    dev = defaultdict(list)
    test = defaultdict(list)

    author_ids = preprocessed_data.keys()
    for author_id in author_ids:
        for i, entry in enumerate(preprocessed_data[author_id]):
            author_id = entry["author_id"]
            if i <= 4:
                test[author_id].append(entry)
            elif i <= 9:
                dev[author_id].append(entry)
            else:
                train[author_id].append(entry)
    return Partition(train, "train"), Partition(dev,"dev"), Partition(test, "test")
                
def write_knn_splits(partitions:tuple[Partition, Partition, Partition], knn_splits_dir:str):
    """
    Writes a tuple of Partition data splits to disk
    
    :param partitions: train, dev, test paritions to save
    :param knn_splits_dir: directory to write the {author}.json files
    """
    for parition in partitions:
        path = f"{knn_splits_dir}{parition.set_type}.json"
        with open(path, "w") as fout:
            json.dump(parition.data, fout, indent=2, ensure_ascii=False)
            
          
def get_which_partition_docs(partition:Partition) -> list[str]:
    """
    Gets the raw documents that belong to given parition. Needed to partition metric learning 
    splits correctly according to already established KNN train, dev, test splits
    """
    docs = []
    for parition_objs in partition.data.values():
        for obj in parition_objs:
            docs.append(obj["raw_text"])
    return docs
       
def get_data(path) -> list[dict]:
    """Reads a series of JSON objects into a list"""
    return [json.loads(line) for line in open(path, "r")]
             
def apply_fixes_to_pair(pair:tuple) -> tuple:
    return apply_all_fixes(pair[0]), apply_all_fixes(pair[1])      
       
def prepare_metric_learn_splits(raw_train:list[str], 
                                raw_dev:list[str], 
                                raw_test:list[str]) -> tuple[Partition, Partition, Partition]:
    """
    Using the raw train, dev, test splits, sort the raw pairs into their own splits
    for the metric learning setup. Also applies the same text fixes as done to the regular eval data
    """
    doc_pairs = get_data("data/pan22/raw/pairs.jsonl")
    doc_truths = get_data("data/pan22/raw/truth.jsonl")
    assert len(doc_pairs) == len(doc_truths)
    
    metric_train, metric_dev, metric_test = [],[],[]
    for doc_entry, truth_entry in zip(doc_pairs, doc_truths):
        
        truth:bool = truth_entry["same"]
        pair:tuple[str,str] = tuple(doc_entry["pair"])
        entry = {"same":truth, "pair": apply_fixes_to_pair(pair)}
        
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
        
    return Partition(metric_train, "train"), Partition(metric_dev,"dev"), Partition(metric_test, "test")

def write_metric_splits(partitions:tuple[Partition, Partition, Partition], out_dir:str):
    """Write a list of paritions to jsonl file"""
    for partition in partitions:
        with jsonlines.open(f"{out_dir}{partition.set_type}.jsonl", "w") as metric_file:
            metric_file.write_all(partition.data)
  
def main():
    

    os.chdir("../../")
    pan_preprocessed = load_preprocessed_data("data/pan22/preprocessed/author_doc_mappings.json")
    train, dev, test = extract_knn_splits_from_authors(pan_preprocessed)
    write_knn_splits((train, dev, test), "eval/pan22_splits/knn/")

    print("Partitioning metric learning splits...")
    raw_train_docs = get_which_partition_docs(train)
    raw_dev_docs   = get_which_partition_docs(dev)
    raw_test_docs  = get_which_partition_docs(test)
    metric_train, metric_dev, metric_test = prepare_metric_learn_splits(raw_train_docs, 
                                                                        raw_dev_docs, 
                                                                        raw_test_docs)
    write_metric_splits((metric_train, metric_dev, metric_test), "eval/pan22_splits/metric_learn/")
    
    
                
    


if __name__ == "__main__":
    main()