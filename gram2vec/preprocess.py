#!/usr/bin/env python3
import json
import re
from copy import deepcopy
from nltk.corpus import names
from random import choice, randint
import re
from collections import defaultdict
import csv

# project import
import utils

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
    FIX: detects a functional tag and returns a different string instead
    """
    
    to_remove = ["<data_extract>", "<data_excerpt>", "<link>", "<line_break>", "<tab>", "<table>", "<image>", "<images>", "<nl>", "<new>", "<figure>", "<susiness>"]
    if re.search(r"<question(\d)?>", tag) or tag in to_remove: 
        t = " "
  
    # tags that need to be replaced 
    elif re.search(r"(<addr?(\d+)?_.*>)|(<pers(\d)?.*>)", tag):
        t = choice(names.words())
        
    elif re.search(r"<part_.*>", tag):
        t = "."
    
    elif re.search(r"<city(\d)?>", tag):
        t = choice(["New York City", "Seattle", "Los Angelos", "San Fransisco", "Chicago", "Houston", "Pheonix", "Philadelphia", "San Antonio", "San Jose", "Dallas"])
    
    elif re.search(r"<condition(\d)?>", tag):
        t = choice(["hypothermia", "flu", "covid", "cancer", "asthma", "monkey pox"])
    
    elif re.search(r"(<continent(\d)?_adj>)|(<condition(\d)?_adj>)|(<country(\d)_adj>)", tag):
        t = choice(["happy", "dense", "loud", "large", "small", "populated", "amazing"])
    
    elif re.search(r"(<country(\d)?>)|(<counr?ty>)|(<continent>)", tag):
        t = choice(["America", "Britain", "Brazil", "Russia", "Mexico", "Iran", "Iraq"])
    
    elif re.search(r"<course(\d)>", tag):
        t = choice(["math", "linguistics", "computer science", "biology", "physics", "chemistry"])
    
    elif re.search(r"(<day(\d)?>)|(<day_abbr>)", tag):
        t = choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    
    elif re.search(r"(<month(\d)?>)|(<month_abbr>)", tag):
        t = choice(["December", "November", "October", "September", "August", "July", "June"])

    elif re.search(r"(<.*_number(\d)?>)|(<.*code(\d)?>)", tag) or tag in ["<DD>", "<DD_MM_YY>", "<DDth>", "<YY>","<YYYY>", "<age>"]: # NUMBERS
        t = str(randint(0,10000))

    elif re.search(r"<language(\d)?>", tag):
        t = choice(["Spanish", "English", "Arabic", "Russian", "Mandarin", "French", "Hebrew"])
    
    elif re.search(r"<station(\d)?>", tag):
        t = choice(["Penn Station", "Grand Central Terminal", "Huntington Station", "Port Jefferson Station", "Stony Brook Station"])
    
    elif re.search(r"<town(\d)?>", tag):
        t = choice(["Stony Brook", "Port Jefferson", "East Setauket", "Huntington", "Patchogue"])
    
    elif tag == "<band>":
        t = choice(["Nirvana", "Queen", "Pink Floyd", "The Beatles"])
     
    else: # strip the tag of <> and numbers
        t = re.sub(r"[<>\d]", "", tag)
    
    return t

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
    """TODO: simplify code"""
    
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

#! this func is deprecatd and its functionality is due to be moved into a separate script
def save_dev_bins(dev, train):
    """Sort dev by author frequency in train, and split into bins"""
    
    train_ = sorted(train.items(), key=lambda x: len(x[1]), reverse=True)
    dev_ = sorted(dev.items(), key=lambda x:len(train[x[0]]), reverse=True)
    assert [i[0] for i in train_] == [i[0] for i in dev_], "Sorting incorrect"

    # manually checking that dev_ is sorted by frequency in train
    # for devitems, trainitems in zip(dev_, train_):
    #     print(f"TRAIN AUTHOR: {trainitems[0]} : {len(trainitems[1])}")
    #     print(f"DEV AUTHOR:   {devitems[0]}   : {len(devitems[1])}\n")
    
    utils.save_json({k:v for k, v in dev_[0:7]},   "data/dev_bins/dev_bin_1.json")
    utils.save_json({k:v for k, v in dev_[7:14]},  "data/dev_bins/dev_bin_2.json")
    utils.save_json({k:v for k, v in dev_[14:21]}, "data/dev_bins/dev_bin_3.json")
    utils.save_json({k:v for k, v in dev_[21:28]}, "data/dev_bins/dev_bin_4.json")
    utils.save_json({k:v for k, v in dev_[28:35]}, "data/dev_bins/dev_bin_5.json")
    utils.save_json({k:v for k, v in dev_[35:42]}, "data/dev_bins/dev_bin_6.json")
    utils.save_json({k:v for k, v in dev_[42:49]}, "data/dev_bins/dev_bin_7.json")
    utils.save_json({k:v for k, v in dev_[49:56]}, "data/dev_bins/dev_bin_8.json")
        
                                   
def generate_vocabularies(data):
    """This function aggregates all tokens and pos tags into .txt files. Needed for ngram featurizers."""
    
    nlp = utils.load_spacy("en_core_web_md")
    # stuffs all texts into one list
    all_docs = [entry for id in data.keys() for entry in data[id]]
    
    for filename in ["all_tokens.txt", "all_pos_tags.txt"]:
        with open(f"resources/{filename}", "w") as fout:
            for text in all_docs:
                doc = nlp(text)
                for token in doc:
                    to_write = token.text if filename == "all_tokens.txt" else token.pos_
                    fout.write(to_write)
                    fout.write("\n")
    
def save_dataset_stats(data:dict):
    
    authors    = []
    doc_counts = []
    for author_id in data.keys():
        if author_id not in authors:
            authors.append(author_id)
        doc_counts.append(len(data[author_id]))   
         
    with open("resources/stats.tsv", "w") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(["author", "num_counts"])
        for author, count in zip(authors, doc_counts):
            writer.writerow([author, count])
                                 
def main(): 

    print("Loading raw data...")
    id_pairs, text_pairs = load_raw_data("data/raw/pairs.jsonl", "data/raw/truth.jsonl")
    print("Done!")
    
    print("Sorting and fixing data...")
    sorted_authors = sort_data(id_pairs, text_pairs)
    fixed_sorted_authors = fix_data(sorted_authors)
    print("Done!")
    
    print("Saving preprocessed datasets...")
    utils.save_json(sorted_authors, "data/preprocessed/sorted_authors.json")
    utils.save_json(fixed_sorted_authors, "data/preprocessed/fixed_sorted_author.json")
    print("Done!")
    
    print("Generating vocabularies...")
    generate_vocabularies(fixed_sorted_authors)
    print("Done!")
    
    print("Dividing data into splits...")
    train, dev, test = train_dev_test_splits(fixed_sorted_authors)
    for split, path in [(train, "data/train_dev_test/train.json"), (dev, "data/train_dev_test/dev.json"), (test, "data/train_dev_test/test.json")]:
        utils.save_json(split, path)
    print("Done!")
    
    print("Generating dataset statistics...")
    save_dataset_stats(sorted_authors)
    print("Done!")
    
    print("Generating dev bins...")
    save_dev_bins(dev, train)
    print("Done!")
    
    
    
if __name__ == "__main__":
    main()
