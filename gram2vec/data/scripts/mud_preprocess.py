import ijson
import re
import numpy as np
import pandas as pd
import os
from typing import List

def clean_doc(doc:str):
    """Performs simple text preprocessing"""
    doc = re.sub(r"\(?http[s]?://\S+", "", doc)
    doc = doc.replace("&gt;", ">").replace("&lt;", "<").replace("\'", "'").replace("&amp;", "&").replace("\n", "")
    return doc

def iter_raw_mud(raw_path:str):
    """
    Generator for the raw MUD data. Iterates over each 
    
    ijson.parse() opens the data as a file stream instead of loading everything into memory
    """
    raw_data = ijson.parse(open(raw_path), multiple_values=True)
    for prefix, _, value in raw_data:
        yield prefix, value
        
def get_author_avg_token_cnt(documents:List[str]) -> float:
    return np.mean([len(doc.split()) for doc in documents])
    
def extract_authors(raw_path:str, desired_authors:int, doc_count:range, tok_threshold:int, verbose=False) -> dict:
    """
    Extracts authors with a certain amount of documents and average tokens
    
    :param raw_path: path to raw MUD data
    :param desired_authors: number of authors to extract
    :param doc_count: range of document frequency to look for
    :param tok_threshold: author token count threshold
    """
    seen_authors = 0
    extracted_data, posts = {}, []
    for prefix, value in iter_raw_mud(raw_path):
        
        if prefix == "syms.item":
            posts.append(clean_doc(value))
            
        if prefix.startswith("author_id"):
            if len(posts) in doc_count and get_author_avg_token_cnt(posts) >= tok_threshold:
                extracted_data[value] = posts
                posts = []
                seen_authors += 1
                if verbose:
                    print(f"Author extracted: {value}")
            
        if seen_authors == desired_authors:
            return extracted_data 
        
    raise ValueError(f"Insufficient # of authors found: {seen_authors} != {desired_authors}")

def main():
    
    os.chdir("../")
    MUD_PATH = "mud/raw_all/data.jsonl"

    d = extract_authors(MUD_PATH, 
                        desired_authors = 50, 
                        doc_count=range(1000,3000),
                        tok_threshold=250,
                        verbose=True)
    


if __name__ == "__main__":
    main()
