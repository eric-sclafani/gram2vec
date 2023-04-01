#!/usr/bin/env python3

from collections import defaultdict
import jsonlines
from typing import List, Dict
import json
import re
from nltk.corpus import names
import random
import re
import os

random.seed(42)


def replace_tag(tag:str) -> str:
    """
    Given a tag, either deletes it, or replaces it with semantic information corresponding to that tag.
    This information is chosen "randomly" (seeded) from lists of possible strings
    
    The format for all mappings is pattern : [list of possible selections]
    
    NOTE: there is no official tagset description for these tags, so I replace them 
    based off observed context clues from the data. Also does not take the tag numbers into account.
    
    :param tag: tag to be replaced
    :returns: new information in place of the tag
    """
    new_string = None
    tag_fix_mappings = {
        r"<question(\d)?>" : [" "],
        
        r"(<addr?(\d+)?_.*>)|(<pers(\d)?.*>)" : names.words(),
        
        r"<part_.*>" : ["."],
        
        r"<city(\d)?>" : ["New York City", "Seattle", "Los Angelos", "San Fransisco", "Chicago", "Houston", "Pheonix", "Philadelphia", "San Antonio", "San Jose", "Dallas"],
        
        r"<condition(\d)?>" : ["hypothermia", "flu", "covid", "cancer", "asthma", "monkey pox"],
        
        r"(<continent(\d)?_adj>)|(<condition(\d)?_adj>)|(<country(\d)_adj>)" : ["happy", "dense", "loud", "large", "small", "populated", "amazing"],
        
        r"(<country(\d)?>)|(<counr?ty>)|(<continent>)" : ["America", "Britain", "Brazil", "Russia", "Mexico", "Iran", "Iraq"],
        
        r"<course(\d)>" : ["math", "linguistics", "computer science", "biology", "physics", "chemistry"],
        
        r"(<day(\d)?>)|(<day_abbr>)" : ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        
        r"(<month(\d)?>)|(<month_abbr>)" : ["December", "November", "October", "September", "August", "July", "June"],
        
        r"<language(\d)?>" : ["Spanish", "English", "Arabic", "Russian", "Mandarin", "French", "Hebrew"],
        
        r"<station(\d)?>" : ["Penn Station", "Grand Central Terminal", "Huntington Station", "Port Jefferson Station", "Stony Brook Station"],
        
        r"<town(\d)?>" : ["Stony Brook", "Port Jefferson", "East Setauket", "Huntington", "Patchogue"],
        
        r"<band>" : ["Nirvana", "Queen", "Pink Floyd", "The Beatles"]
    }
    
    for pattern, replacements in tag_fix_mappings.items():
        if re.search(pattern, tag):
            new_string = random.choice(replacements)
                
    if tag in ["<data_extract>", "<data_excerpt>", "<link>", "<line_break>", "<tab>", "<table>", "<image>", "<images>", "<nl>", "<new>", "<figure>", "<susiness>"]:
        new_string = " "
    elif not new_string: 
        new_string = re.sub(r"[<>\d]", "", tag)
    
    return new_string

def normalize_spacing(tokens:List[str]) -> str:
    """Ensures a document's spacing is consistent"""
    return " ".join(tokens)

def fix_BOS_cutoffs(text:str) -> List[str]:
    """
    Detects beginning of string cutoff (if applicable) and fixes it. 
    String is split into a list first instead of just using 
    str.startswith() to avoid potential false positives
    
    These mappings are also inferred through context clues from the raw data
    
    :param text: string to potentially fix
    :returns: list of tokens with potentially fixed string beginning
    """
    tokens = text.split()               
    fix_mapping = {
        "r":"Dear",
        "nk":"Thank",
        "nks":"Thanks",
        "d":"Good",
        "lo":"Hello",
        "lo,":"Hello",
        "t's":"It's",
        "t`s":"It's",
        "t":"It",
        "py":"Happy",
        "ning":'Morning',
        "ry":"Sorry",
        "y>":"",
        "nl>":""}  
    try:
        tokens[0] = fix_mapping[tokens[0]]
    except KeyError:pass  
    return tokens

def fix_html_tags(text:str):
    """Replaces any leftover HTML tags"""
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "and")

# def remove_parenthesis(text:str):
#     """Removes parenthesis because of an odd bug involving replace_tag"""
#     return re.sub(r"\(.*\)", "", text)  

def get_tags(text:str) -> List[str]:
    """Gets list of redaction tags from a text document"""
    document_tags = re.findall(r"<(.*?)>", text)
    return list(map(lambda x: f"<{x}>", document_tags))
    
def apply_all_fixes(document:str) -> str:
    """
    Applies the following fixes:
    1. Fixes HTML tags
    2. Replace redaction tags
    3. Fixes beginning of string wacky cutoffs
    4. Normalizes spacing
    
    :param document: text document to fix
    :returns: fixed text document
    """
    document = fix_html_tags(document)
    for tag in get_tags(document):
        try:
            document = re.sub(tag, replace_tag(tag), document, count=1)  
        except re.error:
            pass # small number of tags in the raw data are very messed up and crash re, so they're skipped.
    
    document_tokens = fix_BOS_cutoffs(document)
    normalized_document = normalize_spacing(document_tokens)
    return normalized_document


def iter_raw_data(pairs_path:str, truths_path:str) -> Dict:
    """Generator that yields each raw document pair and raw truth pair json object, which are 1-1 corresponding"""
    with jsonlines.open(pairs_path) as pairs_file, jsonlines.open(truths_path) as truths_file:
        for pair, truth in zip(pairs_file, truths_file):
            yield pair, truth
    
def make_author_to_document_mappings(raw_pairs_path:str, raw_truths_path:str)  -> Dict[str,List]: 
    """Maps each unique author id to a list of JSON objects and applies the text preprocessing steps to the raw text"""
    seen_docs = []
    author_mappings = defaultdict(lambda:[])
    
    for doc_pair, truth_pair in iter_raw_data(raw_pairs_path, raw_truths_path):
        
        author_1, author_2 = truth_pair["authors"]
        doc_1, doc_2  = doc_pair["pair"]
        discourse_1, discourse_2 = doc_pair["discourse_types"]
        
        if doc_1 not in seen_docs:
            obj_1 = {"author_id":author_1, "discourse_type":discourse_1, "raw_text":doc_1, "fixed_text":apply_all_fixes(doc_1)}
            seen_docs.append(doc_1)
            author_mappings[author_1].append(obj_1)
        
        if doc_2 not in seen_docs:
            obj_2 = {"author_id":author_2, "discourse_type":discourse_2, "raw_text":doc_2, "fixed_text":apply_all_fixes(doc_2)}
            author_mappings[author_2].append(obj_2)  
            seen_docs.append(doc_2)
            
    return author_mappings

def make_preprocessed_author_pairs(raw_pairs_path:str, verbose=False) -> List[Dict]:
    """Applies the preprocessing steps to raw author pairs data, but preserves the pairs format of the original data"""
    processesed_author_pairs = []
    with jsonlines.open(raw_pairs_path) as pairs_file:
        for doc_pair in pairs_file:
            if verbose: print(f"Processing: {doc_pair['id'], doc_pair['discourse_types'], (doc_pair['pair'][0][0:10],doc_pair['pair'][1][0:10])}")
            doc_pair["pair"] = list(map(lambda x: apply_all_fixes(x), doc_pair["pair"]))
            processesed_author_pairs.append(doc_pair)
    return processesed_author_pairs

# TODO (at some point): re-add my train/dev/test split code, since I apparently removed it (for some reason 🤦‍♂️)
#! all it is: first five docs from each author go to test, next five to dev, the rest go to train
                   
def main(): 

    os.chdir("../")
    
    # save as author to list of documents mappings
    author_to_docs = make_author_to_document_mappings("pan22/raw/pairs.jsonl", "pan22/raw/truth.jsonl")    
    with open("pan22/preprocessed/author_doc_mappings.json", "w") as fout:
         json.dump(author_to_docs, fout, indent=2, ensure_ascii=False)
    
    
    # save as preprocessed pairs
    fixed_author_pairs = make_preprocessed_author_pairs("pan22/raw/pairs.jsonl", verbose=False)
    with jsonlines.open("pan22/preprocessed/pairs_preprocessed.jsonl", "w") as fout:
        for entry in fixed_author_pairs:
            fout.write(entry)

if __name__ == "__main__":
    main()
