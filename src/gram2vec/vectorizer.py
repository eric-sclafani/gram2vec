
from collections import Counter
import demoji
import time
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Callable, Optional, Iterable

from ._load_spacy import nlp, Doc
from ._load_vocab import vocab

def measure_time(func):
    """Debugging function for measuring function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds.")
        return result
    return wrapper

#~~~ Features ~~~

REGISTERD_FEATURES = {}

@dataclass
class Document:
    """
    Encapsulates the raw text and spacy doc. Needed because emojis must be taken out of the spacy doc before 
    the dependency parse, but the emojis feature still needs access to the emojis from the text
    """
    raw:str
    doc:Doc

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
        """Normalizes each count by the sum of counts for that feature"""
        return counts / self._get_sum(counts)
    
    def _prefix_feature_names(self, features:pd.Series) -> pd.Series:
        """
        For each low level feature, prefix the name of the high level feature to it 
                                EXAMPLE:  ADJ -> pos_unigrams:ADJ
        """
        return features.add_prefix(f"{self.name}:")
        
@Feature.register
def pos_unigrams(text:Document) -> Feature:
    return Counter(text.doc._.pos_tags)
    
@Feature.register
def pos_bigrams(text:Document) -> Feature:
    return Counter(text.doc._.pos_bigrams)

@Feature.register
def func_words(text:Document) -> Feature:
    return Counter(text.doc._.func_words)
 
@Feature.register
def punctuation(text:Document) -> Feature:
    return Counter(text.doc._.punctuation)

@Feature.register
def letters(text:Document) -> Feature:
    return Counter(text.doc._.letters)

@Feature.register
def dep_labels(text:Document) -> Feature:
    return Counter(text.doc._.dep_labels)

@Feature.register
def morph_tags(text:Document) -> Feature:
    return Counter(text.doc._.morph_tags)

@Feature.register
def sentences(text:Document) -> Feature:
    return Counter(text.doc._.sentences)

# emojis must get removed before processed through spaCy,
# so spaCy extensions cannot be used here unfortunately
@Feature.register
def emojis(text:Document) -> Feature:
    emojis_vocab = vocab.get("emojis")
    extracted_emojis = demoji.findall_list(text.raw, desc=False)
    return Counter([emoji for emoji in extracted_emojis if emoji in emojis_vocab])

# ~~~ Processing ~~~


    
def get_activated_features(config:Optional[Dict]) -> List[Feature]:
    """Retrieves activated features from register according to a given config. Falls back to default config if none is provided"""
    if config is None:
        default_config = {
            "pos_unigrams":1,
            "pos_bigrams":1,
            "func_words":1,
            "punctuation":1,
            "letters":1,
            "emojis":1,
            "dep_labels":1,
            "morph_tags":1,
            "sentences":1
            }
        config = default_config
    return [REGISTERD_FEATURES[feat_name] for feat_name, num in config.items() if num == 1]

def load_jsonlines(path:str) -> pd.DataFrame:
    """Loads 1 or more .jsonl files into a dataframe"""
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    else:
        dfs = [pd.read_json(file, lines=True) for file in Path(path).glob("*.jsonl")]
        return pd.concat(dfs).reset_index(drop=True)
    
def _remove_emojis(document:str) -> str:
    """Removes emojis from a string and fixes spacing issue caused by emoji removal"""
    new_string = demoji.replace(document, "").split()
    return " ".join(new_string)

def _process_documents(documents:Iterable[str]) -> List[Document]:
    """Converts all provided documents into Document instances, which encapsulates the raw text and spacy doc"""
    nlp_docs = nlp.pipe([_remove_emojis(doc) for doc in documents])
    processed = []
    for raw_text, nlp_doc in zip(documents, nlp_docs):
        processed.append(Document(raw_text, nlp_doc))
    return processed

def _get_json_entries(df) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Retrieves the 'fullText', 'authorIDs', and 'documentID' fields from a json-loaded dataframe"""
    try:
        documents = df["fullText"]
        author_ids = df["authorIDs"]
        document_ids = df["documentID"]
    except KeyError:
        raise KeyError("Specified jsonline(s) file missing one or more fields: 'fullText', 'authorIDs', 'documentID'")
    
    return documents, author_ids, document_ids

def _content_embedding(doc:Document) -> pd.Series:
    """Retrieves the spacy document embedding and returns it as a Series object"""
    return pd.Series(doc.doc.vector).add_prefix("Embedding dim: ")
    
def _apply_features(doc:Document, config:Optional[Dict], include_content_embedding:bool) -> pd.Series:
    """Applies all feature extractors to a given document, optionally adding the spaCy emedding vector"""
    features = []
    for feature in get_activated_features(config):
        feature_vocab = vocab.get(feature.name)
        extraction = feature(doc, feature_vocab)
        features.append(extraction)
        
    if include_content_embedding:
        features.append(_content_embedding(doc))
    return pd.concat(features, axis=0)

def _apply_features_to_docs(docs:List[Document],
                            config:Optional[Dict], 
                            include_content_embedding:bool) -> pd.DataFrame:
    """Applies the feature extractors to all documents and creates a style vector matrix"""
    feature_vectors = []
    for doc in docs:
        vector = _apply_features(doc, config, include_content_embedding)
        feature_vectors.append(vector)
    return pd.concat(feature_vectors, axis=1).T

def from_jsonlines(path:str, 
                   config:Optional[Dict]=None, 
                   include_content_embedding=False) -> pd.DataFrame:
    """
    Given a path to either a jsonlines file OR directory of jsonlines files, creates a stylistic feature 
    vector matrix. Document IDs and author IDs are included, retrieved from the provided jsonlines file(s)\n
    Args:
    -----
        path (str): 
            path to a jsonlines file OR directory of jsonlines files
        config(Dict | None): 
            Feature activation configuration. Uses a default if none is provided
        include_content_embedding (bool):
            option to include the word2vec document embedding\n
    Returns:
    -------
        pd.DataFrame: dataframe where rows are documents and columns are low level features
    """
    df = load_jsonlines(path)
    documents, author_ids, document_ids = _get_json_entries(df)
    documents = _process_documents(documents)
    if include_content_embedding:
        print("Gram2Vec: 'include_content_embedding' flag set to True. Including document word2vec embedding...")
        print("Gram2Vec: (WARNING) embedding should only be used for experiments, not attribution")
    vector_df = _apply_features_to_docs(documents, config, include_content_embedding)
    vector_df.insert(0, "authorIDs", author_ids)
    vector_df.set_index(document_ids, inplace=True)
    return vector_df
    
def from_documents(documents:Iterable[str], 
                   config:Optional[Dict]=None, 
                   include_content_embedding=False) -> pd.DataFrame:
    """
    Given an iterable of documents, creates a stylistic feature vector matrix. Document IDs and author IDs are NOT included\n
    Args:
    -----
        documents(Iterable):
            iterable of strings to be converted into a matrix
        config(Dict | None): 
            Feature activation configuration. Uses a default if none is provided
        include_content_embedding (bool):
            option to include the word2vec document embedding\n
    Returns:
    --------
        pd.DataFrame: dataframe where each row is a document and column is a low level feature
    """
    documents = _process_documents(documents)
    if include_content_embedding:
        print("Gram2Vec: 'include_content_embedding' flag set to True. Including document word2vec embedding...")
        print("Gram2Vec: (WARNING) embedding should only be used for experiments, not attribution")
    vector_df = _apply_features_to_docs(documents, config, include_content_embedding)
    return vector_df