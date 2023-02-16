#!/usr/bin/env python3
import json
import jsonlines
import re
from copy import deepcopy
from nltk.corpus import names
import numpy as np
from random import choice, randint, seed
import re
from collections import defaultdict
import csv

seed(80085)

# project import
from gram2vec import utils

AuthorToDocsMapping = tuple[str,list[str]]

def sort_by_token_avg(author_to_avg_tokens:dict[str, float]) -> dict[str, int]:
    """Sorts a dictionary of author id to average token count mappings"""
    author_avg_token_pairs = author_to_avg_tokens.items()
    sorted_pairs:tuple[str,int] = sorted(author_avg_token_pairs, key=lambda x: x[1])
    return dict(sorted_pairs)

def sort_by_doc_freq(train:dict) -> list[tuple]:
    """Sorts a dictionary by the amount of documents of each author"""
    author_docs_pairs:list[AuthorToDocsMapping] = train.items()
    return sorted(author_docs_pairs, key=lambda pairs: len(pairs[1]))

def get_data(path) -> list[dict]:
    """Reads a series of JSON objects into a list"""
    return [json.loads(line) for line in open(path, "r")]

def load_raw_data(pairs_path:str, truths_path:str) -> tuple[list]:
    """This function loads the raw json data as a list of dicts and extracts each pair"""
    pairs = get_data(pairs_path) 
    truths = get_data(truths_path)
     
    id_pairs = [tuple(entry["authors"]) for entry in truths]
    text_pairs = [tuple(entry["pair"]) for entry in pairs]
    #discourse_pairs = [tuple(entry["discourse_types"]) for entry in pairs] 
    
    return id_pairs, text_pairs

def replace_tag(tag:str):
    """
    FIX: detects a functional tag and returns a different string instead.
    
    This is done so the dependency parser doesn't get tripped up
    """
    
    to_remove = ["<data_extract>", "<data_excerpt>", "<link>", "<line_break>", "<tab>", "<table>", "<image>", "<images>", "<nl>", "<new>", "<figure>", "<susiness>"]
    if re.search(r"<question(\d)?>", tag) or tag in to_remove: 
        new_string = " "
  
    # tags that need to be replaced 
    elif re.search(r"(<addr?(\d+)?_.*>)|(<pers(\d)?.*>)", tag):
        new_string = choice(names.words())
        
    elif re.search(r"<part_.*>", tag):
        new_string = "."
    
    elif re.search(r"<city(\d)?>", tag):
        new_string = choice(["New York City", "Seattle", "Los Angelos", "San Fransisco", "Chicago", "Houston", "Pheonix", "Philadelphia", "San Antonio", "San Jose", "Dallas"])
    
    elif re.search(r"<condition(\d)?>", tag):
        new_string = choice(["hypothermia", "flu", "covid", "cancer", "asthma", "monkey pox"])
    
    elif re.search(r"(<continent(\d)?_adj>)|(<condition(\d)?_adj>)|(<country(\d)_adj>)", tag):
        new_string = choice(["happy", "dense", "loud", "large", "small", "populated", "amazing"])
    
    elif re.search(r"(<country(\d)?>)|(<counr?ty>)|(<continent>)", tag):
        new_string = choice(["America", "Britain", "Brazil", "Russia", "Mexico", "Iran", "Iraq"])
    
    elif re.search(r"<course(\d)>", tag):
        new_string = choice(["math", "linguistics", "computer science", "biology", "physics", "chemistry"])
    
    elif re.search(r"(<day(\d)?>)|(<day_abbr>)", tag):
        new_string = choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    
    elif re.search(r"(<month(\d)?>)|(<month_abbr>)", tag):
        new_string = choice(["December", "November", "October", "September", "August", "July", "June"])

    elif re.search(r"(<.*_number(\d)?>)|(<.*code(\d)?>)", tag) or tag in ["<DD>", "<DD_MM_YY>", "<DDth>", "<YY>","<YYYY>", "<age>"]: # NUMBERS
        new_string = str(randint(0,10000))

    elif re.search(r"<language(\d)?>", tag):
        new_string = choice(["Spanish", "English", "Arabic", "Russian", "Mandarin", "French", "Hebrew"])
    
    elif re.search(r"<station(\d)?>", tag):
        new_string = choice(["Penn Station", "Grand Central Terminal", "Huntington Station", "Port Jefferson Station", "Stony Brook Station"])
    
    elif re.search(r"<town(\d)?>", tag):
        new_string = choice(["Stony Brook", "Port Jefferson", "East Setauket", "Huntington", "Patchogue"])
    
    elif tag == "<band>":
        new_string = choice(["Nirvana", "Queen", "Pink Floyd", "The Beatles"])
     
    else: # strip the tag of <> and numbers
        new_string = re.sub(r"[<>\d]", "", tag)
    
    return new_string

def normalize(text):
    """
    This function normalizes spaces and fixes BOS cut offs (if applicable)
    """
    text = text.split()
                  
    first = text[0]  # BOS fix     
    if first == "r":
        text[0] = "Dear"
    elif first == "nk":
        text[0] = "Thank"
    elif first == "nks":
        text[0] = "Thanks"
    elif first == "d":
        text[0] = "Good"
    elif first in ["lo", "lo,"]:
        text[0] = "Hello"
    elif first in ["t's", "t`s"]:
        text[0] = "It's"
    elif first == "t":
        text[0] = "It" 
    elif first == "py":
        text[0] = "Happy"
    elif first == "ning":
        text[0] = "Morning"
    elif first == "ry":
        text[0] = "Sorry"
    elif first in ["y>", "nl>"]:
        text[0] = ""
            
    return " ".join(text)

def fix_html(text):
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "and")

def remove_parenthesis(text):
    return re.sub(r"\(.*\)", "", text)  

def get_tags(text):
    """Gets list of redaction tags from a text document"""
    return list(map(lambda x: f"<{x}>", re.findall(r"<(.*?)>", text))) # regex doesnt include the <> in the tags for findall, so they need to be re-added
    
    
def sort_data(id_pairs, text_pairs) -> dict:
    """This function maps authors or discourses to their texts."""
    data = {}
    # the pair lists are one to one corresponding, so iterate over them at once
    for ids, texts in zip(id_pairs, text_pairs):
        for id, text in zip(ids, texts): 
            if id not in data:
                data[id] = []
            if text not in data[id]: # avoid duplicates
                data[id].append(text)   
    return data 

def fix_data(data):
    """Applies the data fixes to the vanilla PAN22 data"""
    data = deepcopy(data) # prevent data mutation
    
    for id in data.keys(): 
        for idx, text in enumerate(data[id]):
            
            # FIX: fix &gt;, &lt;, and &amp;
            text = fix_html(text)
            
            # FIX: remove anything wrapped in parenthesis (for some reason, parenthesis mess with the function tag replacing regexes, causing a crash)
            text = remove_parenthesis(text)
            
            # FIX: replace function tags
            for tag in get_tags(text):
                text = re.sub(tag, replace_tag(tag), text, count=1)  
            
            # FIX: normalize spaces and fix beginning of string cut offs
            text = normalize(text)
            
            data[id][idx] = text
    return data

def sort_authors_by_avg_tokens(dev:dict, train:dict) -> list[tuple]:
    """Sorts the authors in DEV by average token count in TRAIN"""
    author_to_avg_tokens = {}
    for author_id, documents in train.items():
        token_count = lambda x: len(x.split())  
        author_to_avg_tokens[author_id] = np.mean([token_count(doc) for doc in documents], axis=0)
    
    train_sorted = sort_by_token_avg(author_to_avg_tokens) # (low -> high)
    train_sorted_authors = train_sorted.keys()
    
    # https://stackoverflow.com/questions/21773866/how-to-sort-a-dictionary-based-on-a-list-in-python
    # sorting DEV by sorted list of authors in TRAIN
    index_map = {author_id: i for i, author_id in enumerate(train_sorted_authors)}
    dev_sorted = sorted(dev.items(), key=lambda pair: index_map[pair[0]])
    return dev_sorted

def sort_authors_by_doc_freq(dev:dict, train:dict):
    """Sorts the authors in DEV by document frequency in TRAIN"""
    train_sorted = sort_by_doc_freq(train)
    
    # sorts dev by document occurence in train
    dev_sorted = sorted(dev.items(), key=lambda x:len(train[x[0]]))
    
    assert [i[0] for i in train_sorted] == [i[0] for i in dev_sorted], "Sorting incorrect"
    for devitems, trainitems in zip(dev_sorted, train_sorted):
        assert trainitems[0] == devitems[0]
    
    return dev_sorted
        
def save_dev_bins(sorted_data:list[tuple]):
    """Saves dev bins to directory"""
    i = 0
    for bin_num in range(1,9):
        partition = dict(sorted_data[i:i+7])
        utils.save_json(data=partition, path=f"pan/dev_bins/sorted_by_docfreq/bin_{bin_num}_dev.json")
        i += 7

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
            if idx <= 4:  
                test[author_id].append(text)
            elif idx <= 9: 
                dev[author_id].append(text)
            else:         
                train[author_id].append(text)
    return train, dev, test

def get_raw_document_splits(sorted_authors_path) -> tuple[list,list,list]:
    """
    Retrieves the pre-fixed versions (raw) of the documents sorted into train, dev, test
    """
    data = utils.load_json(sorted_authors_path)
    
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


def fix_pair(pair:tuple[str, str]) -> tuple[str, str]:
    """Applies the same changes as the fix_data() function, except to pairs of texts"""

    fixed_pair = []
    for text in pair:
        
        text = fix_html(text)
        text = remove_parenthesis(text)
        for tag in get_tags(text):
            text = re.sub(tag, replace_tag(tag), text, count=1)  
        text = normalize(text)
        
        fixed_pair.append(text)
        
    return tuple(fixed_pair)


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
        entry = {"same":truth, "pair":fix_pair(pair)}
        
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
    
def write_metric_eval_to_file(data:list[dict], out_path):
    """Write a list of dictionaries to jsonl file"""
    with jsonlines.open(out_path, "w") as fout:
        for entry in data:
            fout.write(entry)
                                 
def main(): 

    print("Loading raw data...")
    id_pairs, text_pairs = load_raw_data("pan/raw/pairs.jsonl", "pan/raw/truth.jsonl")
    print("Done!")
    
    print("Sorting and fixing data...")
    sorted_authors = sort_data(id_pairs, text_pairs)
    fixed_sorted_authors = fix_data(sorted_authors)
    print("Done!")
    
    # print("Saving sorted datasets...")
    # utils.save_json(sorted_authors, "pan/preprocessed/sorted_authors.json")
    # print("Done!")
    
    # print("Saving preprocessed datasets...")
    # utils.save_json(fixed_sorted_authors, "pan/preprocessed/fixed_sorted_author.json")
    # print("Done!")

    print("Dividing data into splits...")
    train, dev, test = train_dev_test_splits(fixed_sorted_authors)
    # for split, path in [(train, "pan/train_dev_test/train.json"), (dev, "pan/train_dev_test/dev.json"), (test, "pan/train_dev_test/test.json")]:
    #     utils.save_json(split, path)
    print("Done!")
    
    # print("Saving development bins...")
    # save_dev_bins(sort_authors_by_doc_freq(dev, train))
    # print("Done!")
    
    print("Creating metric learning evaluation splits...")
    raw_train, raw_dev, raw_test = get_raw_document_splits("pan/preprocessed/sorted_authors.json")
    metric_train, metric_dev, metric_test = prepare_metric_learn_splits(raw_train, raw_dev, raw_test)
    
    write_metric_eval_to_file(metric_train, "pan/train_dev_test/pairs/metric_train.jsonl")
    write_metric_eval_to_file(metric_dev, "pan/train_dev_test/pairs/metric_dev.jsonl")
    write_metric_eval_to_file(metric_test, "pan/train_dev_test/pairs/metric_test.jsonl")
    
    print("Done!")
    
    
    
if __name__ == "__main__":
    main()
