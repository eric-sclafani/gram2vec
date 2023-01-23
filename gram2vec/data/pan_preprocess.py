#!/usr/bin/env python3
import json
import re
from copy import deepcopy
from nltk.corpus import names
import numpy as np
from random import choice, randint
import re
from collections import defaultdict
import csv

# project import
from gram2vec import utils

def load_raw_data(pairs_path:str, truths_path:str) -> tuple[list]:
    """This function loads the raw json data as a list of dicts and extracts each pair"""
    get_data = lambda x: [json.loads(line) for line in open(x, "r")]
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
    
    get_tags = lambda text: list(map(lambda x: f"<{x}>", re.findall(r"<(.*?)>", text))) # regex doesnt include the <> in the tags for findall, so they need to be re-added
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


# function under heavy construction
def save_dev_bins(dev:dict, train:dict):
    
    #! put under new func?
    # train_sorted = sorted(train.items(), key=lambda x: len(x[1]), reverse=True)
    # dev_sorted = sorted(dev.items(), key=lambda x:len(train[x[0]]), reverse=True)
    # assert [i[0] for i in train_sorted] == [i[0] for i in dev_sorted], "Sorting incorrect"
    # for devitems, trainitems in zip(dev_sorted, train_sorted):
    #     assert trainitems[0] == devitems[0]
    
    
    author_to_avg_tokens = {}
    for author_id, documents in train.items():  
        author_to_avg_tokens[author_id] = np.mean([len(doc.split()) for doc in documents], axis=0)
    
    # sort TRAIN by avg num of tokens
    train_sorted = dict(sorted(author_to_avg_tokens.items(),key = lambda x: x[1]))
    index_map = {v: i for i, v in enumerate(train_sorted.keys())}
    
    # sort DEV by avg num of tokens in TRAIN
    dev_sorted = sorted(dev.items(), key=lambda pair: index_map[pair[0]])
    
    i = 0
    for bin_num in range(1,9):
        partition = dict(dev_sorted[i:i+7])
        utils.save_json(data=partition, path=f"pan/dev_bins/sorted_by_avg_tokens/bin_{bin_num}_dev.json")
        i += 7
    
    
                                 
def main(): 

    print("Loading raw data...")
    id_pairs, text_pairs = load_raw_data("pan/raw/pairs.jsonl", "pan/raw/truth.jsonl")
    print("Done!")
    
    print("Sorting and fixing data...")
    sorted_authors = sort_data(id_pairs, text_pairs)
    fixed_sorted_authors = fix_data(sorted_authors)
    print("Done!")
    
    # print("Saving sorted datasets...")
    # utils.save_json(sorted_authors, "data/pan/preprocessed/sorted_author.json")
    # print("Done!")
    
    # print("Saving preprocessed datasets...")
    # utils.save_json(fixed_sorted_authors, "data/pan/preprocessed/fixed_sorted_author.json")
    # print("Done!")

    print("Dividing data into splits...")
    train, dev, test = train_dev_test_splits(fixed_sorted_authors)
    # for split, path in [(train, "data/pan/train_dev_test/train.json"), (dev, "data/pan/train_dev_test/dev.json"), (test, "data/pan/train_dev_test/test.json")]:
    #     utils.save_json(split, path)
    # print("Done!")
    
    save_dev_bins(dev, train)
    
    
    
    
if __name__ == "__main__":
    main()
