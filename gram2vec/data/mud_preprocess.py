#!/usr/bin/env python3

import ijson
import re
from collections import defaultdict

# project imports
from gram2vec import utils


def clean_doc(doc:str):
    
    to_remove = [r"\(?http[s]?://\S+", r"\n"]
    for pattern in to_remove:
        doc = re.sub(pattern, "", doc)
        
    doc = doc.replace("&gt;", ">").replace("&lt;", "<").replace("\'", "'").replace("&amp;", "&")
    
    return doc

def extract_n_authors(raw_path, n) -> dict:
    """This function extracts n amount of authors from MUD with k amount of documents"""
    raw_data = ijson.parse(open(raw_path), multiple_values=True)

    seen_authors = 0
    data = {}
    posts = []
    for prefix, _, value in raw_data:
        
        if prefix == "syms.item":
            posts.append(clean_doc(value))
            
        if prefix.startswith("author_id") and len(posts) in range(3000, 4000):
            #print(f"Author found: {value}")
            data[value] = posts
            posts = []
            seen_authors += 1
            
            
        if seen_authors == n:
            break
        
    return data


def train_dev_test_splits(data:dict):
    """
    Splits the fixed_sorted_authors.json into train, dev, test splits.
    CURRENT SPLITS: 5 for dev and test, the rest for train. These numbers are specific to the PAN22 dataset.
    """
    train = defaultdict(list)
    dev   = defaultdict(list)
    test  = defaultdict(list)
    
    for author_id in data.keys():
        for idx, text in enumerate(data[author_id]):
            if idx <=99: # includes 0
                test[author_id].append(text)
            elif idx <= 199: 
                dev[author_id].append(text)
            else:         
                train[author_id].append(text)
    
             
    for author_id in data.keys():
        assert len(test[author_id]) == 100, f"{len(test[author_id])} =! 100"
        assert len(dev[author_id]) == 100, f"{len(dev[author_id])} =! 100"
        
    return train, dev, test

def main():
    
    data = extract_n_authors("mud/full/data.jsonl", n=50)
    
    train, dev, test = train_dev_test_splits(data)
    
    utils.save_json(train, "mud/train_dev_test/train.json")
    utils.save_json(dev, "mud/train_dev_test/dev.json")
    utils.save_json(test, "mud/train_dev_test/test.json")

if __name__ == "__main__":
    main()