#!/usr/bin/env python3

import argparse
from collections import Counter
from dataclasses import dataclass
import featurizers as feats
from featurizers import Document
import os
import utils

@dataclass
class Vocab:
    name:str
    features:tuple[str]



# This function will likely change when integrated into Delip's system
def check_for_valid_format(train_path:str) -> bool:
    """
    Validates training set JSON for the following format:
    {
        author_id : [doc1, doc2,...doc_n],
        author_id : [...],
    }
    WHERE:
        type(author_id) = str
        type([doc1, doc2,...doc_n]) = array[str]
    """
    try:
        data = utils.load_json(train_path)
        assert all(isinstance(author_id, str) for author_id in data.keys()),\
        "Each author id must be a string"
        assert all(isinstance(author_docs, list) for author_docs in data.values()),\
        "Each collection of documents must be an array"
        assert all(isinstance(author_doc, str) for author_docs in data.values() for author_doc in author_docs),\
        "Each document must be a string"
    except:
        raise Exception("Data format incorrect :(. Check documentation for expected format.")
    return True

def get_all_documents_from_data(train_path:str, nlp) -> list[Document]:
    """Retrieves all training documents in training data"""
    check_for_valid_format(train_path)
    data = utils.load_json(train_path)
    documents = []
    for author_docs in data.values():
        for doc in author_docs:
            document = feats.make_document(doc, nlp)
            documents.append(document)
    return documents  

def save_vocab_to_pickle(vocab:tuple, path:str):
    """Writes vocab to pickle to be used by featurizers"""
    utils.save_pkl(vocab, path)

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
    
    os.makedirs(path)
    save_vocab_to_txt_file(vocab_features, f"{path}/{vocab_name}.txt")
    save_vocab_to_pickle(vocab_features, f"{path}/{vocab_name}.pkl")
        
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
 
def main():
    
    nlp = utils.load_spacy("en_core_web_md")
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-train",
                        "--train_path",
                        type=str,
                        help="Path to train data",
                        default="data/pan/train_dev_test/train.json")
    
    args = parser.parse_args()
    train_path = args.train_path
    
    print("Retrieving all training documents...")
    dataset_name = utils.get_dataset_name(train_path)
    all_documents = get_all_documents_from_data(train_path, nlp)
    print("Done!")
    
    print("Generating non-static vocabularies...")

    # any new non-static vocabs can be added here
    POS_BIGRAMS   = Vocab(name="pos_bigrams", features=generate_most_common(all_documents, 50, feats.count_pos_bigrams))
    MIXED_BIGRAMS = Vocab(name="mixed_bigrams", features=generate_most_common(all_documents, 50, feats.count_mixed_bigrams))
    
    VOCABS = [POS_BIGRAMS, MIXED_BIGRAMS]
    
    print("Done!")
        
    for vocab in VOCABS:
        print(f"Saving vocabulary '{vocab.name}'...")
        save_vocab(dataset_name, vocab)
    print("Done!") 
    
if __name__ == "__main__":
    main()