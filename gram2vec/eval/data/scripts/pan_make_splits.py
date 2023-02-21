
from collections import defaultdict
import json
import jsonlines

def get_data(path) -> list[dict]:
    """Reads a series of JSON objects into a list"""
    return [json.loads(line) for line in open(path, "r")]


def get_raw_document_splits(sorted_authors_path) -> tuple[list,list,list]:
    """
    Retrieves the pre-fixed versions (raw) of the documents sorted into train, dev, test
    """
    #! THIS FUNCTION WILL BECOME IRRELEVANT WHEN DATA IS REGENERATED
    data = load_json(sorted_authors_path)
    
    train, dev, test = [],[],[]
    for author_id in data.keys():
        for idx, text in enumerate(data[author_id]):
            if idx <= 4:  
                test.append(text)
            elif idx <= 9: 
                dev.append(text)
            else:         
                train.append(text)
    return train, dev, test



def make_knn_splits(data:dict):
    """
    Splits the fixed_sorted_authors.json into train, dev, test splits.
    CURRENT SPLITS: 5 for dev and test, the rest for train. These numbers are specific to the PAN22 dataset.
    When test set us released, this function will change
    """
    train = defaultdict(list)
    dev   = defaultdict(list)
    test  = defaultdict(list)
    for author_id in data.keys():
        for idx, text in enumerate(data[author_id]):
            if idx <= 4:  
                test[author_id].append(text)
            elif idx <= 9: 
                dev[author_id].append(text)
            else:         
                train[author_id].append(text)
    return train, dev, test

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
    # print("Dividing data into splits...")
    # train, dev, test = train_dev_test_splits(fixed_sorted_authors)
    # for split, path in [(train, "pan/train_dev_test/train.json"), (dev, "pan/train_dev_test/dev.json"), (test, "pan/train_dev_test/test.json")]:
    #     utils.save_json(split, path)
    # print("Done!")
    pass


if __name__ == "__main__":
    main()