#!/usr/bin/env python3

import argparse
from collections import Counter
from dataclasses import dataclass
from typing import Tuple, Dict, List
import os
import shutil
import spacy
from spacy.tokens import Doc


@dataclass
class Vocab:
    name:str
    features:Tuple[str]


def combine_counters(counters:List[Counter]) -> Counter:
    """Adds a list of Counter objects into one"""
    return sum(counters, Counter())


def count_pos_bigrams(doc):
    """Counter function: counts the POS bigrams in a doc"""
    counter = Counter(vectorizer.get_bigrams_with_boundary_syms(doc, doc.pos_tags))
    return counter

def count_mixed_bigrams(doc:Document):
    """Counter function: counts the mixed bigrams in a doc"""
    return Counter(vectorizer.bigrams(vectorizer.replace_openclass(doc.tokens, doc.pos_tags)))
        

def generate_most_common(documents:List[Document], n:int, count_function) -> Tuple[str]:
    """Generates n most common elements according to count_function"""
    counters = []
    for document in documents:
        counter = count_function(document)
        counters.append(counter)
        
    n_most_common = dict(combine_counters(counters).most_common(n))
    return tuple(n_most_common.keys())

def save_vocab_to_txt_file(vocab:Tuple, path:str):
    """Writes vocab to a txt file for debugging purposes only"""
    with open (path, "w") as fout:
        for entry in vocab:
            fout.write(f"{entry}\n")
            
def save_vocab(dataset_name:str, vocab:Tuple[str]):
    """
    Saves non-static vocabs as both a pickle and text file.
    The text file is purely for debugging purposes (only for non-static vocabs)
    """
    vocab_name = vocab.name
    vocab_features = vocab.features
    path = f"vocab/non_static/{dataset_name}/{vocab_name}/"
    
    if os.path.exists(path):
        shutil.rmtree(path)
    
    os.makedirs(path)
    save_vocab_to_txt_file(vocab_features, f"{path}/{vocab_name}.txt")

    

def main():
    
    nlp = nlp = spacy.load("en_core_web_md", disable=["ner", "lemmatizer"])
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-train",
                        "--train_path",
                        type=str,
                        help="Path to train data",
                        default="eval/pan22_splits/knn/train.json")
    
    args = parser.parse_args()
    train_path = args.train_path
    
    print("Retrieving all training documents...")
    dataset_name = get_dataset_name(train_path)
    all_documents = get_all_documents(train_path, nlp)
    print("Done!")
    
    print("Generating non-static vocabularies...")

    # any new non-static vocabs can be added here
    POS_BIGRAMS   = Vocab(name="pos_bigrams", features=generate_most_common(all_documents, 50, count_pos_bigrams))
    MIXED_BIGRAMS = Vocab(name="mixed_bigrams", features=generate_most_common(all_documents, 50, count_mixed_bigrams))
    
    VOCABS = [POS_BIGRAMS, MIXED_BIGRAMS]
    print("Done!")
    
    for vocab in VOCABS:
        print(f"Saving vocabulary '{vocab.name}'...")
        save_vocab(dataset_name, vocab)
    print("Done!") 
    
if __name__ == "__main__":
    main()