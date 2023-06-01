#!/usr/bin/env python3

from collections import Counter
from typing import Tuple, List
from load_spacy import Doc

def _sum_counters(counters:List[Counter]) -> Counter:
    """Adds a list of Counter objects into one"""
    return sum(counters, Counter())

def _count_pos_bigrams(doc:Doc) -> Counter:
    """Counts the POS bigrams in a spacy doc"""
    return Counter(doc._.pos_bigrams)

def _count_mixed_bigrams(doc:Doc) -> Counter:
    """Counts the mixed bigrams in a spacy doc"""
    return Counter(doc._.mixed_bigrams)
        
def _get_most_common(documents, n, count_func) -> Tuple[str]:
    """Generates n most common elements according to count_function"""
    counters = []
    for document in documents:
        counter = count_func(document)
        counters.append(counter)
        
    n_most_common = dict(_sum_counters(counters).most_common(n))
    return tuple(n_most_common.keys())

def _save_vocab_to_txt_file(vocab:Tuple, path:str) -> None:
    """Writes vocab to a txt file"""
    print(f"Saving to {path}")
    with open (path, "w") as fout:
        for entry in vocab:
            fout.write(f"{entry}\n")
            
def generate_non_static_vocab(documents:List[Doc], n=50) -> None: 
    """Given a collection of documents, gets their most common non-static vocab items and saves them to disk"""
    vocabs = {
        "pos_bigrams": _count_pos_bigrams,
        "mixed_bigrams":_count_mixed_bigrams
    }
    for name, count_func in vocabs.items():
        print(f"Generating vocab for {name}...")
        vocab = _get_most_common(documents, n, count_func)
        _save_vocab_to_txt_file(vocab, f"vocab/non_static/{name}.txt")