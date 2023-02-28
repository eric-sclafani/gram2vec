#! usr/bin/env python3

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
import os
from nltk.tokenize import word_tokenize


@dataclass
class Author:
    """
    Stores author information in an easy to work with format
    
    :param author_id: unique author id
    :param fixed_texts: list of author documents with regex fixes
    :param raw_texts: list of author documents without regex fixes
    :param discourse_types: list of discourse types
    
    Note: fixed_docs, raw_docs, and discourse_types are all 1 - 1 corresponding
    """
    author_id:str
    fixed_texts:list[str]
    raw_texts:list[str]
    discourse_types:list[str]

def load_preproccessed_json(path:str) -> dict[str, list[dict]]:
    with open(path, "r") as fin:
        data = json.load(fin)
        return data

def _extract_from_dict(author_entry:dict, to_extract:str) -> list[str]:
    return [entry[to_extract] for entry in author_entry]
    

def create_author_list(preprocessed_data:dict[str, list[dict]]) -> list[Author]:
    """
    Converts the preprocessed_data.json data into a list of Author objects
    """
    authors = []
    for author_id in preprocessed_data.keys():
        author_entry = preprocessed_data[author_id]
        fixed_texts = _extract_from_dict(author_entry,"fixed_text")
        raw_texts = _extract_from_dict(author_entry,"raw_text")
        discourse_types = _extract_from_dict(author_entry,"discourse_type")
            
        authors.append(Author(author_id, fixed_texts, raw_texts, discourse_types))
        
    return authors
    
def get_total_doc_count(authors:list[Author]) -> int:
    return sum([len(author.fixed_texts) for author in authors])

def get_total_author_count(authors:list[Author]) -> int:
    return sum(1 for _ in authors)

def get_doc_token_stats(authors:list[Author]):

    pass
    


# doc_token_counts = get_document_token_counts(data) 
# print("Mean/std tokens per document")      
# print(np.mean(doc_token_counts))
# print(np.std(doc_token_counts))


def get_author_token_counts(data:dict[str, list[str]]) -> list[int]:
    
    author_to_token_counts = {}
    for author_id in data.keys():
        token_counts = []
        for doc in data[author_id]:
            doc_tokens = doc.split()
            token_counts.append(len(doc_tokens))
            
        author_to_token_counts[author_id] = sum(token_counts)
            
    return list(author_to_token_counts.values())


# author_token_counts = get_author_token_counts(data)

# print("\nMean/std tokens per author")
# print(np.mean(author_token_counts))
# print(np.std(author_token_counts))


def get_num_docs_per_author(data:dict[str, list[str]]) -> list[int]:
    author_to_doc_freq = {}
    for author_id in data.keys():
        author_to_doc_freq[author_id] = len(data[author_id])
    return list(author_to_doc_freq.values())

# docs_per_author = get_num_docs_per_author(data)

# print("\nMean/std document frequency per author")
# print(np.mean(docs_per_author))
# print(np.std(docs_per_author))


# emails = [entry.fixed_text for entry in all_entries if entry.discourse_type == "email"]
# txt_msgs = [entry.fixed_text for entry in all_entries if entry.discourse_type == "text_message"]
# essays = [entry.fixed_text for entry in all_entries if entry.discourse_type == "essay"]
# memos = [entry.fixed_text for entry in all_entries if entry.discourse_type == "memo"]

def get_token_counts(documents:list[str]) -> list[int]:
    return [len(doc.split()) for doc in documents]

# print("Email avg tokens: ", np.mean(get_token_counts(emails)))
# print("Txt msgs avg tokens: ", np.mean(get_token_counts(txt_msgs)))
# print("Essays avg tokens: ", np.mean(get_token_counts(essays)))
# print("Memo avg tokens: ", np.mean(get_token_counts(memos)))


def main():
    
    os.chdir("../")
    data = load_preproccessed_json("pan22/preprocessed/preprocessed_data.json")
    all_authors = create_author_list(data)
    
    print(f"Total # of documents: {get_total_doc_count(all_authors)}")
    print(f"Total # of authors: {get_total_author_count(all_authors)}")
    


    




if __name__ == "__main__":
    main()