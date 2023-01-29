#!/usr/bin/env python3

import argparse
from collections import Counter
import featurizers as feats
from featurizers import Document
import os
import utils
from pathlib import Path


# ~~~ Helper functions ~~~

def get_dataset_name(train_path:str) -> str:
    """
    Gets the dataset name from training data path.  
    Needed to generate path for vocab per dataset
    NOTE: This function needs to be manually updated when new datasets are used.
    """
    if "pan" in train_path:
        dataset_name = "pan"
    elif "mud" in train_path:
        dataset_name = "mud"
    # add other dataset names here following the same condition-checking format
    else:
        raise ValueError(f"Dataset name unrecognized in path: {train_path}")
    return dataset_name

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
        documents.extend(list(map(lambda doc: feats.make_document(doc,nlp), author_docs)))
    return documents  


def write_dataset_directory(dataset_name:str):
    pass


def write_vocab_to_pickle():
    pass


def write_vocab_to_txt_file():
    pass

def combine_counters(counters:list[Counter]) -> Counter:
    """Adds a list of Counter objects into one"""
    return sum(counters, Counter())

# ~~~ STATIC VOCABULARIES ~~~
# Static: non-changing sets of elements to count
POS_TAGS   = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE")
PUNC_MARKS = (".", ",", ":", ";", "\'", "\"", "?", "!", "`", "*", "&", "_", "-", "%", "(", ")", "–", "‘", "’")
LETTERS    = ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "à", "è", "ì", "ò", "ù", "á", "é", "í", "ó", "ú", "ý")
EMOJIS     = ("😅", "😂", "😊", "❤️", "😭", "👍", "👌", "😍", "💕", "🥰")
DEP_LABELS = ('ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp', 'compound', 'conj', 'csubj', 'csubjpass', 'dative', 'dep', 'det', 'dobj', 'expl', 'intj', 'mark', 'meta', 'neg', 'nmod', 'npadvmod', 'nsubj', 'nsubjpass', 'nummod', 'oprd', 'parataxis', 'pcomp', 'pobj', 'poss', 'preconj', 'predet', 'prep', 'prt', 'punct', 'quantmod', 'relcl', 'xcomp')

# ~~~ NON-STATIC VOCABULARIES ~~~
# Non-static: sets of elements that change depending on the dataset 
# (generated under 'main' function)

def generate_most_common(documents:list[Document], n:int, count_function) -> tuple[str]:
    """Generates n most common elements according to count_function"""
    counters = []
    for document in documents:
        counter = count_function(document)
        counters.append(counter)
        
    n_most_common = dict(combine_counters(counters).most_common(n))
    return tuple(n_most_common.keys())



def _generate_vocab(self, data_path):
    
    counters = {
        "pos_bigrams"   : [],
        "mixed_bigrams" : []
    } 
    
    all_text_docs = [entry for id in data.keys() for entry in data[id]]
    for feature, counter_list in counters.items():
        out_path = f"vocab/{dataset}_{feature}_vocab.pkl"
        
        if not os.path.exists(out_path):
            for text in all_text_docs:
                doc = self.nlp(text)
                
                pos_counts = get_pos_bigrams(doc)
                counters["pos_bigrams"].append(pos_counts)
            
                mixed_bigrams = get_mixed_bigrams(doc)
                counters["mixed_bigrams"].append(mixed_bigrams)
    
            # this line condenses all the counters into one dict, getting the 50 most common elements
            most_common = dict(sum(counter_list, Counter()).most_common(50)) # most common returns list of tuples, gets converted back to dict
            utils.save_pkl(set(most_common.keys()), out_path)
    
    
    

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
    
    dataset_name = get_dataset_name(train_path)
    all_documents = get_all_documents_from_data(train_path, nlp)
    
    POS_BIGRAMS = generate_most_common(all_documents, 50, feats.count_pos_bigrams)
    import ipdb;ipdb.set_trace()
    

 
   
if __name__ == "__main__":
    main()