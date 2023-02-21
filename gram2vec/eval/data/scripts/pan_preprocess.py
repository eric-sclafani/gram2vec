#!/usr/bin/env python3
import jsonlines
import re
from dataclasses import dataclass
from nltk.corpus import names
import random
import re
import os

random.seed(80085)


def replace_tag(tag:str) -> str:
    """
    Given a tag, either deletes it, or replaces it with semantic information corresponding to that tag.
    
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

def normalize_spacing(tokens:list[str]) -> str:
    """Ensures a document's spacing is correct and consistent"""
    return " ".join(tokens)

def fix_BOS_cutoffs(text:str) -> list[str]:
    """
    Detects beginning of string cutoff (if applicable) and fixes it. 
    String is split into a list first to avoid potential false positives
    
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
    except:pass  
    return tokens

def fix_html_tags(text:str):
    """Replaces any leftover HTML tags"""
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "and")

def remove_parenthesis(text):
    """Removes parenthesis because of an odd bug involving replace_tags"""
    return re.sub(r"\(.*\)", "", text)  

def get_tags(text:str):
    """Gets list of redaction tags from a text document"""
    document_tags = re.findall(r"<(.*?)>", text)
    return list(map(lambda x: f"<{x}>", document_tags))
    
def apply_all_fixes(document:str) -> str:
    """
    Applies the following fixes:
    1. Fixes HTML tags
    2. Removes parenthesis (causing bug thats currently being looked at)
    3. Replace redaction tags
    4. Fixes beginning of string wacky cutoffs
    5. Normalizes spacing
    
    :param document: text document to fix
    :returns: fixed text document
    """
    document = fix_html_tags(document)
    document = remove_parenthesis(document) #! experiment with this
    for tag in get_tags(document):
        document = re.sub(tag, replace_tag(tag), document, count=1)  
    
    document_tokens = fix_BOS_cutoffs(document)
    normalized_document = normalize_spacing(document_tokens)
    return normalized_document


def iter_raw_data(pairs_path:str, truths_path:str) -> tuple[list]:
    """Iterator for the raw jsonl files"""
    with jsonlines.open(pairs_path) as pairs_file, jsonlines.open(truths_path) as truths_file:
        for pair, truth in zip(pairs_file, truths_file):
            yield pair, truth
            
     
    

             
def main(): 

    os.chdir("../")
    iter_raw_data("pan22/raw/pairs.jsonl", "pan22/raw/truth.jsonl")

    # print("Loading raw data...")
    # id_pairs, text_pairs = load_raw_data("pan/raw/pairs.jsonl", "pan/raw/truth.jsonl")
    # print("Done!")
    # print("Sorting and fixing data...")
    # sorted_authors = sort_data(id_pairs, text_pairs)
    # fixed_sorted_authors = fix_data(sorted_authors)
    # print("Done!")
    # print("Saving sorted datasets...")
    # utils.save_json(sorted_authors, "pan/preprocessed/sorted_authors.json")
    # print("Done!")
    # print("Saving preprocessed datasets...")
    # utils.save_json(fixed_sorted_authors, "pan/preprocessed/fixed_sorted_author.json")
    # print("Done!")


    
    
    
if __name__ == "__main__":
    main()
