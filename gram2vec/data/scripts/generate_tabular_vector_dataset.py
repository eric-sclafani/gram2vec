
import json
from dataclasses import dataclass
from gram2vec.featurizers import GrammarVectorizer
import os
import pandas as pd

@dataclass
class AuthorEntry:
    author_id:str
    discourse_type:str
    fixed_text:str
    
def load_data(data_path:str) -> dict[str, list[dict]]:
    """Loads in a JSON consisting of author_ids mapped to lists of dict entries as a dict"""
    with open(data_path) as fin:
        data = json.load(fin)
    return data


def get_all_entries(data_path:str) -> list[AuthorEntry]:
    """Extracts and aggregates author file entries as AuthorEntry objects into one list"""
    all_entries = []
    for author_entries in load_data(data_path).values():
        for author_dict in author_entries:
            author_id = author_dict["author_id"]
            fixed_text = author_dict["fixed_text"]
            discourse_type = author_dict["discourse_type"]
            all_entries.append(AuthorEntry(author_id, discourse_type, fixed_text))
    return all_entries


def get_vocab(path:str) -> list[str]:
    """Retrieves a featurizer vocabulary stored in a given path"""
    with open(path, "r") as fin:
        return fin.read().strip().split("\n")

def fix_feature_names(all_features:list[str]) -> list[str]:
    """
    Raw feature labels contain DIFFERENT features with the SAME name, so they need to be 
    differentiated when generating a tabular dataset. Conditionals are hard-coded following
    the concatenation order of the vectors (see featurizers.py)
    
    :param all_features: list of feature names containing duplicates
    :returns: list of feature names with renamed features
    """
    seen_feats = []
    seen_i, seen_a, seen_X = False, False, False
    
    for feature in all_features:
        if feature == "i": 
            feature = "i (func_word)" if  not seen_i else "i (letter)"
            seen_i = True
            
        elif feature == "a":
            feature = "a (func_word)" if not seen_a else "a (letter)"
            seen_a = True
        
        elif feature == "X":
            feature = "X (pos_unigram)" if not seen_X else "X (letter)"
            seen_X = True
            
        seen_feats.append(feature)  
    return seen_feats


def main():
    #TODO: add ability to generate tabular data set for any supplied dataset
    
    os.chdir("../pan22/preprocessed/")
    
    all_entries = get_all_entries("author_doc_mappings.json")
    documents = [entry.fixed_text for entry in all_entries]
    authors = [entry.author_id for entry in all_entries]
    discourse_types = [entry.discourse_type for entry in all_entries]
    
    os.chdir("../../../")
    g2v = GrammarVectorizer()
    feature_vectors = g2v.vectorize_episode(documents)
    
    pos_unigrams  = get_vocab("vocab/static/pos_unigrams.txt")
    pos_bigrams   = get_vocab("vocab/non_static/pan/pos_bigrams/pos_bigrams.txt")
    func_words    = get_vocab("vocab/static/function_words.txt")
    punc          = get_vocab("vocab/static/punc_marks.txt")
    letters       = get_vocab("vocab/static/letters.txt")
    common_emojis = get_vocab("vocab/static/common_emojis.txt")
    doc_stats     = ["short_words", "large_words", "word_len_avg", "word_len_std", "sent_len_avg", "sent_len_std", "hapaxes"]
    deps          = get_vocab("vocab/static/dep_labels.txt")
    mixed_bigrams = get_vocab("vocab/non_static/pan/mixed_bigrams/mixed_bigrams.txt")
    
    all_features = pos_unigrams + pos_bigrams + func_words + punc + letters + common_emojis + doc_stats + deps + mixed_bigrams
    
    fixed_feature_names = fix_feature_names(all_features)
    
    df = pd.DataFrame(feature_vectors)
    df.columns = fixed_feature_names

    df.insert(0, "author_id", authors)
    df.insert(1, "discourse_type", discourse_types)
    
    os.chdir("data/scripts")
    df.to_csv("pan22_features.csv", index=None)


if __name__ == "__main__":
    main()