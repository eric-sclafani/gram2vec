
from collections import Counter
import demoji
import numpy as np 
import pandas as pd
from pathlib import Path
import json
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Callable, Generator

from load_spacy import nlp, Doc
from generate_non_static_vocab import generate_non_static_vocab

# ~~~ Vocab ~~~

def _load_from_txt(path:str) -> Tuple[str]:
    with open (path, "r") as fin:
        return tuple(map(lambda x: x.strip("\n"), fin.readlines()))
    
def vocab_loader() -> Dict[str, Tuple[str]]:
    """Loads in all feature vocabs. For any new vocabs, add them to this function"""

    return {
        "pos_unigrams": _load_from_txt("vocab/pos_unigrams.txt"),
        "pos_bigrams": _load_from_txt("vocab/pos_bigrams.txt"),
        "func_words": _load_from_txt("vocab/func_words.txt"),
        "punctuation": _load_from_txt("vocab/punctuation.txt"),
        "letters": _load_from_txt("vocab/letters.txt"),
        "emojis":_load_from_txt("vocab/emojis.txt"),
        "dep_labels": _load_from_txt("vocab/dep_labels.txt"),
        "mixed_bigrams":_load_from_txt("vocab/mixed_bigrams.txt"),
        "morph_tags":_load_from_txt("vocab/morph_tags.txt")
    }
    
VOCABS = vocab_loader()  
   
#~~~ Features ~~~

REGISTERD_FEATURES = {}

class Feature:
    """Encapsulates a feature counting function. When the function is called, normalization is applied to the counted features"""
    def __init__(self, func:Callable):
        self.func = func    
        self.name = func.__name__
        
    def __call__(self, doc, vocab):
        counted_features = self.func(doc)
        all_counts = self._include_zero_vocab_counts(counted_features, vocab)
        normalized_counts = self._normalize(all_counts)
        return self._prefix_feature_names(normalized_counts)
    
    @classmethod
    def register(cls, func):
        """Creates a Feature object and registers it to the REGISTERED_FEATURES dict"""
        func = cls(func)
        REGISTERD_FEATURES[func.name] = func
        return func

    def _include_zero_vocab_counts(self, counted_features:Counter, vocab:Tuple[str]) -> pd.Series:
        """Includes the vocabulary items that were not counted in the document (to ensure the same size vector for all documents)"""
        count_dict = {}
        for feature in vocab:
            if feature in counted_features:
                count = counted_features[feature] 
            else:
                count = 0
            count_dict[feature] = count
        return pd.Series(count_dict)
    
    def _get_sum(self, counts:pd.Series) -> int:
        """Gets sum of counts. Accounts for possible zero counts"""
        return sum(counts) if sum(counts) > 0 else 1

    def _normalize(self, counts:pd.Series) -> pd.Series:
        """Normalizes all counts"""
        return counts / self._get_sum(counts)
    
    def _prefix_feature_names(self, features:pd.Series) -> pd.Series:
        """
        For each individual feature element, prefix the name of the encompassing feature to it 
                                EXAMPLE:  ADJ -> pos_unigrams:ADJ
        """
        return features.add_prefix(f"{self.name}:")
        
@Feature.register
def pos_unigrams(doc) -> Feature:
    return Counter(doc.doc._.pos_tags)
    
@Feature.register
def pos_bigrams(doc) -> Feature:
    return Counter(doc.doc._.pos_bigrams)

@Feature.register
def func_words(doc) -> Feature:
    return Counter([token for token in doc.doc._.tokens if token in VOCABS["func_words"]])
 
@Feature.register
def punctuation(doc) -> Feature:
    vocab = VOCABS["punctuation"]
    return Counter([punc for token in doc.doc._.tokens for punc in token if punc in vocab])

@Feature.register
def letters(doc) -> Feature:
    vocab = VOCABS["letters"]
    return Counter([letter for token in doc.doc._.tokens for letter in token if letter in vocab])

@Feature.register
def dep_labels(doc) -> Feature:
    return Counter([dep for dep in doc.doc._.dep_labels])

@Feature.register
def mixed_bigrams(doc) -> Feature:
    return Counter(doc.doc._.mixed_bigrams)

@Feature.register
def morph_tags(doc) -> Feature:
    return Counter(doc.doc._.morph_tags)


#! MAJOR WIP
# @Feature.register
# def emojis(doc) -> Feature:
    
#     vocab = load_vocab("vocab/static/common_emojis.txt")
#     extract_emojis = demoji.findall_list(doc.text, desc=False)
#     doc_emoji_counts = Counter(filter(lambda x: x in vocab, extract_emojis))
#     all_emoji_counts = add_zero_vocab_counts(vocab, doc_emoji_counts)
    
#     return Feature("Emoji", all_emoji_counts, len(doc.tokens))

#! MAJOR WIP
# @Feature.register
# def document_stats(doc) -> Feature:
#     words = doc.words #!doc.words causing crash
    
#     #! GETTING SPLIT INTO SEPARATE VECTORS
#     #! REPLACE ALL STATISTICAL CALCULATIONS WITH BUILT IN
    
    
#     doc_statistics = {"short_words" : len([1 for word in words if len(word) < 5])/len(words) if len(words) else 0, 
#                       "large_words" : len([1 for word in words if len(word) > 4])/len(words) if len(words) else 0,
#                       "word_len_avg": np.mean([len(word) for word in words]),
#                       "word_len_std": np.std([len(word) for word in words]),
#                       "sent_len_avg": np.mean([len(sent) for sent in doc.sentences]),
#                       "sent_len_std": np.std([len(sent) for sent in doc.sentences]),
#                       "hapaxes"     : len(FreqDist(words).hapaxes())/len(words) if len(words) else 0}
#     return Feature("Document statistic", doc_statistics)


# ~~~ Processing ~~~

@dataclass
class Document:
    """
    Encapsulates the raw text and spacy doc. Needed because emojis must be taken out of the spacy doc before 
    the dependency parse, but the common_emoji feature still needs access to the emojis from the text
    """
    raw:str
    doc:Doc

def config(path="config.json") -> List[str]:
    """Reads in the """
    with open(path) as config_file:
        config = json.load(config_file)
        return [feat for feat, option in config.items() if option == 1]

def get_activated_features() -> List[Feature]:
    """Retrieves the activated features from the register according to the config file"""
    return [REGISTERD_FEATURES[feat_name] for feat_name in config()]
    
def load_jsonlines(path:str) -> pd.DataFrame:
    """Loads 1 or more .jsonl files into a dataframe"""
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    else:
        dfs = [pd.read_json(file, lines=True) for file in Path(path).glob("*.jsonl")]
        return pd.concat(dfs).reset_index().drop(columns=["index"])
    
def remove_emojis(document:str) -> str:
    """Removes emojis from a string and fixes spacing issue caused by emoji removal"""
    new_string = demoji.replace(document, "").split()
    return " ".join(new_string)
 
def process_documents(documents:List[str]) -> List[Document]:
    """Converts all provided documents into Document instances, which encapsulates the raw text and spacy doc"""
    nlp_docs = nlp.pipe([remove_emojis(doc) for doc in documents])
    processed = []
    for raw_text, nlp_doc in zip(documents, nlp_docs):
        processed.append(Document(raw_text, nlp_doc))
    return processed

def get_json_entries(df) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Retrieves the 'fullText', 'authorIDs', and 'documentID' fields from a json-loaded dataframe"""
    try:
        documents = df["fullText"]
        author_ids = df["authorIDs"]
        document_ids = df["documentID"]
    except KeyError:
        raise KeyError("Specified jsonlines file missing one or more fields: 'fullText', 'authorIDs', 'documentID'")
    
    return documents, author_ids, document_ids

def content_embedding(doc:Document) -> pd.Series:
    """Retrieves the spacy document embedding and returns it as a Series object"""
    return pd.Series(doc.doc.vector).add_prefix("Embedding dim: ")
    

def apply_features(doc:Document, include_content_embedding:bool) -> pd.Series:
    
    features = []
    for feature in get_activated_features():
        vocab = VOCABS[feature.name]
        features.append(feature(doc, vocab))
    
    if include_content_embedding:
        features.append(content_embedding(doc))
    return pd.concat(features, axis=0)

def apply_features_to_docs(docs:List[Document], include_content_embedding:bool) -> pd.DataFrame:
    
    feature_vectors = []
    for doc in docs:
        vector = apply_features(doc, include_content_embedding)
        feature_vectors.append(vector)
    return pd.concat(feature_vectors, axis=1).T
    
def run_non_static_vocab(documents:List[Document]) -> None:
    print("Gram2Vec: refresh_vocab parameter set to True. Running the non-static vocab generation")
    generate_non_static_vocab(documents)


def from_jsonlines(
    path:str, 
    refresh_vocab=False, 
    include_content_embedding=False,
    verbose=False
    ):
    
    df = load_jsonlines(path)
    documents, author_ids, document_ids = get_json_entries(df)
    documents = process_documents(documents)
    
    if refresh_vocab:
        run_non_static_vocab(documents)
    
    if include_content_embedding:
        print("Gram2Vec: Including spaCy document embedding (WARNING: embedding should only be used for experiments, not attribution)")
    vector_df = apply_features_to_docs(documents, include_content_embedding)
    vector_df.insert(0, "authorIDs", author_ids)
    vector_df.insert(1, "documentID", document_ids)
    return vector_df
    
def from_document_list(
    documents:List[str], 
    refresh_vocab=False, 
    include_content_embedding=False,
    verbose=False
    ):
    
    pass 


    
if __name__ == "__main__":

    df = from_jsonlines("data/pan22/preprocessed/pan22_preprocessed.jsonl", 
                        #refresh_vocab=True,
                        verbose=True,
                        include_content_embedding=True)
