
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


def convert_feature_name(feature:str, seen_i, seen_a, seen_X) -> str:
    """
    Hard coded way of making certain conflicting feature names unique.
    Needs to be done to ensure each feature in data viz dataset is unique
    """
    if feature == "i": 
        feature = "i (func_word)" if not seen_i else "i (letter)"
            
    elif feature == "a":
        feature = "a (func_word)" if not seen_a else "a (letter)"
        
    elif feature == "X":
        feature = "X (pos_unigram)" if not seen_X else "X (letter)"
        
    return feature
        

def make_feature_to_counts_map(all_features:list[str]) -> dict[str,list]:
    """
    Maps each feature to an empty list. Accounts for DIFFERENT features with the SAME label
    
    i.e. some distinct features have the same labels ("i", "a", "X"), so for data visualization purposes,
    they need to be renamed to be distinct. This DOES NOT affect the vectors in any way. 
    
    The conditionals here follow the same concatenation order as all_features
    """
    seen_i, seen_a, seen_X = False, False, False
    count_dict = {}
    for feature in all_features:
        if feature == "i":
            feature = convert_feature_name(feature, seen_i, seen_a, seen_X)
            seen_i = True
            
        if feature == "a":
            feature = convert_feature_name(feature, seen_i, seen_a, seen_X)
            seen_a = True
            
        if feature == "X":
            feature = convert_feature_name(feature, seen_i, seen_a, seen_X)
            seen_X = True
            
        count_dict[feature] = []
    return count_dict

            
def populate_feature_to_counts_map(all_features:list[str], feature_vectors:list) -> dict[str,list[int]]:
    """
    Populates the feature_to_count dict. Accounts for DIFFERENT features with the SAME label
    
    For every feature's count_dict, append the feature name's count number to 
    corresponding list in feats_to_counts
    """
    feats_to_counts = make_feature_to_counts_map(all_features)
    seen_i, seen_a, seen_X = False, False, False
    
    for feature in feature_vectors:
        for count_dict in feature.count_map.values():
            for feat_name, count in count_dict.items():
                
                if feat_name == "i":
                    feat_name = convert_feature_name(feat_name, seen_i, seen_a, seen_X)
                    seen_i = True
                    
                if feat_name == "a":
                    feat_name = convert_feature_name(feat_name, seen_i, seen_a, seen_X)
                    seen_a = True
                    
                if feat_name == "X":
                    feat_name = convert_feature_name(feat_name, seen_i, seen_a, seen_X)
                    seen_X = True
                
                if feat_name != "embedding_vector": 
                    feats_to_counts[str(feat_name)].append(count)
        seen_i, seen_a, seen_X = False, False, False # reset flags for every count_dict
            
    return feats_to_counts


def main():
    #TODO: add ability to generate tabular data set for any supplied dataset
    
    os.chdir("../pan22/preprocessed/")
    
    all_entries = get_all_entries("author_doc_mappings.json")
    documents = [entry.fixed_text for entry in all_entries]
    authors = [entry.author_id for entry in all_entries]
    discourse_types = [entry.discourse_type for entry in all_entries]
    
    os.chdir("../../../")
    g2v = GrammarVectorizer()
    feature_vectors = g2v.vectorize_episode(documents, return_obj=True)
    
    pos_unigrams  = get_vocab("vocab/static/pos_unigrams.txt")
    pos_bigrams   = get_vocab("vocab/non_static/pos_bigrams/pan/pos_bigrams.txt")
    func_words    = get_vocab("vocab/static/function_words.txt")
    punc          = get_vocab("vocab/static/punc_marks.txt")
    letters       = get_vocab("vocab/static/letters.txt")
    common_emojis = get_vocab("vocab/static/common_emojis.txt")
    doc_stats     = ["short_words", "large_words", "word_len_avg", "word_len_std", "sent_len_avg", "sent_len_std", "hapaxes"]
    deps          = get_vocab("vocab/static/dep_labels.txt")
    mixed_bigrams = get_vocab("vocab/non_static/mixed_bigrams/pan/mixed_bigrams.txt")
    
    all_features = pos_unigrams + pos_bigrams + func_words + punc + letters + common_emojis + doc_stats + deps + mixed_bigrams
    
    features_to_count_lists = populate_feature_to_counts_map(all_features, feature_vectors)
    df = pd.DataFrame(features_to_count_lists)
    df.insert(0, "author_id", authors)
    df.insert(1, "discourse_type", discourse_types)
    df.to_csv("pan22_features.csv", index=None)


if __name__ == "__main__":
    main()