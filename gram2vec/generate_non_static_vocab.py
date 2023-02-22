#!/usr/bin/env python3

import argparse
from collections import Counter
from dataclasses import dataclass
import os
import shutil
import spacy
import pickle
import json
import jsonlines
from pathlib import Path

# project imports
import featurizers as feats
from featurizers import Document

@dataclass
class Vocab:
    name:str
    features:tuple[str]


def iter_author_jsonls(author_files_dir:str) -> str:
    """Yields each {author_id}.jsonl from a given dir"""
    for author_file in Path(author_files_dir).glob("*.jsonl"):
        yield author_file
        
def get_dataset_name(train_path:str) -> str:
    """
    Gets the dataset name from training data path which is needed to generate paths
    NOTE: This function needs to be manually updated when new datasets are used.
    """
    if "pan" in train_path:
        dataset_name = "pan"
    else:
        raise ValueError(f"Dataset name unrecognized in path: {train_path}")
    return dataset_name 

def get_all_documents_from_data(train_path:str, nlp) -> list[Document]:
    """Retrieves all training documents and aggregates them"""
    
    documents = []
    for file in iter_author_jsonls(train_path):
        with jsonlines.open(file) as author_entries:
            for entry in author_entries:
                doc = entry["fixed_text"]
                document = feats.make_document(doc, nlp)
                documents.append(document)
                
    return documents  

def count_pos_bigrams(doc:Document):
    counter = Counter(feats.get_bigrams_with_boundary_syms(doc, doc.pos_tags))
    return counter

def count_mixed_bigrams(doc:Document):
    return Counter(feats.bigrams(feats.replace_openclass(doc.tokens, doc.pos_tags)))
        
def combine_counters(counters:list[Counter]) -> Counter:
    """Adds a list of Counter objects into one"""
    return sum(counters, Counter())

def generate_most_common(documents:list[Document], n:int, count_function) -> tuple[str]:
    """Generates n most common elements according to count_function"""
    counters = []
    for document in documents:
        counter = count_function(document)
        counters.append(counter)
        
    n_most_common = dict(combine_counters(counters).most_common(n))
    return tuple(n_most_common.keys())

def save_vocab_to_pickle(vocab:tuple, path:str):
    """Writes vocab to pickle to be used by featurizers"""
    with open (path, "ab") as fout:
        pickle.dump(vocab, fout)

def save_vocab_to_txt_file(vocab:tuple, path:str):
    """Writes vocab to a txt file for debugging purposes only"""
    with open (path, "w") as fout:
        for entry in vocab:
            fout.write(f"{entry}\n")
            
def save_vocab(dataset_name:str, vocab:tuple[str]):
    """
    Saves non-static vocabs as both a pickle and text file.
    The text file is purely for debugging purposes
    """
    vocab_name = vocab.name
    vocab_features = vocab.features
    path = f"vocab/non_static/{vocab_name}/{dataset_name}/"
    
    if os.path.exists(path):
        shutil.rmtree(path)
    
    os.makedirs(path)
    save_vocab_to_txt_file(vocab_features, f"{path}/{vocab_name}.txt")
    save_vocab_to_pickle(vocab_features, f"{path}/{vocab_name}.pkl")
    

def main():
    
    nlp = nlp = spacy.load("en_core_web_md", disable=["ner", "lemmatizer"])
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-train",
                        "--train_path",
                        type=str,
                        help="Path to train data",
                        default="data/pan22/splits/knn/train/")
    
    args = parser.parse_args()
    train_path = args.train_path
    
    print("Retrieving all training documents...")
    dataset_name = get_dataset_name(train_path)
    all_documents = get_all_documents_from_data(train_path, nlp)
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