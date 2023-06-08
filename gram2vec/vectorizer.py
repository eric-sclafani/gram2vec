
from collections import Counter
import demoji
import time
import pandas as pd
from pathlib import Path
import json
from dataclasses import dataclass
from typing import Tuple, List, Dict, Callable, Generator

from load_spacy import nlp, Doc
from generate_non_static_vocab import generate_non_static_vocab

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"The function {func.__name__} took {execution_time:.6f} seconds to run.")
        return result
    return wrapper

# ~~~ Vocab ~~~

def load_from_txt(path:str) -> Tuple[str]:
    with open (path, "r") as fin:
        return tuple(map(lambda x: x.strip("\n"), fin.readlines()))
    
def vocab_loader() -> Dict[str, Tuple[str]]:
    """Loads in all feature vocabs. For any new vocabs, add them to this function"""

    return {
        "pos_unigrams": load_from_txt("vocab/pos_unigrams.txt"),
        "pos_bigrams": load_from_txt("vocab/pos_bigrams.txt"),
        "func_words": load_from_txt("vocab/func_words.txt"),
        "punctuation": load_from_txt("vocab/punctuation.txt"),
        "letters": load_from_txt("vocab/letters.txt"),
        "emojis":load_from_txt("vocab/emojis.txt"),
        "dep_labels": load_from_txt("vocab/dep_labels.txt"),
        "mixed_bigrams":load_from_txt("vocab/mixed_bigrams.txt"),
        "morph_tags":load_from_txt("vocab/morph_tags.txt")
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
        For each low level feature, prefix the name of the high level feature to it 
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
    vocab = VOCABS["func_words"]
    return Counter([token for token in doc.doc._.tokens if token in vocab])
 
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

@Feature.register
def emojis(doc) -> Feature:
    vocab = VOCABS["emojis"]
    extracted_emojis = demoji.findall_list(doc.raw, desc=False)
    doc_emoji_counts = [emoji for emoji in extracted_emojis if emoji in vocab]
    return Counter(doc_emoji_counts)


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
    """Applies all feature extractors to a given document, optionally adding the spaCy emedding vector"""
    features = []
    for feature in get_activated_features():
        vocab = VOCABS[feature.name]
        features.append(feature(doc, vocab))
    
    if include_content_embedding:
        features.append(content_embedding(doc))
    return pd.concat(features, axis=0)

def apply_features_to_docs(docs:List[Document], include_content_embedding:bool) -> pd.DataFrame:
    """Applies the feature extractors to all documents and creates a style vector matrix"""
    feature_vectors = []
    for doc in docs:
        vector = apply_features(doc, include_content_embedding)
        feature_vectors.append(vector)
    return pd.concat(feature_vectors, axis=1).T

def from_jsonlines(path:str, include_content_embedding=False) -> pd.DataFrame:
    """
    Given a path to either a jsonlines file OR directory of jsonlines files, creates a stylistic feature 
    vector matrix. Document IDs and author IDs are included, retrieved from the provided jsonlines file(s)\n
    Args:
    -----
        path (str):  
            path to a jsonlines file OR directory of jsonlines files
        include_content_embedding (bool): 
            option to include the word2vec document embedding\n
    Returns:
    -------
        pd.DataFrame: dataframe where each row is a document and column is a low level feature
    """
    df = load_jsonlines(path)
    documents, author_ids, document_ids = get_json_entries(df)
    documents = process_documents(documents)
     
    if include_content_embedding:
        print("Gram2Vec: 'include_content_embedding' flag set to True. Including document word2vec embedding...")
        print("Gram2Vec: (WARNING) embedding should only be used for experiments, not attribution")
    vector_df = apply_features_to_docs(documents, include_content_embedding)
    vector_df.insert(0, "authorIDs", author_ids)
    vector_df.insert(1, "documentID", document_ids)
    return vector_df
    
def from_document_list(documents:List[str], include_content_embedding=False) -> pd.DataFrame:
    """
    Given a list of documents, creates a stylistic feature vector matrix. Document IDs and author IDs are NOT included\n
    Args:
    -----
        documents(list):
            list of text documents to be converted into a mtrix
        include_content_embedding(bool):
            option to include the word2vec document embedding\n
    Returns:
    --------
        pd.DataFrame: dataframe where each row is a document and column is a low level feature
    """
    documents = process_documents(documents)
    vector_df = apply_features_to_docs(documents, include_content_embedding)
    return vector_df