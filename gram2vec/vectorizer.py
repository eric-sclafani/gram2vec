
from collections import Counter
from copy import copy
from dataclasses import dataclass
import demoji
from nltk import bigrams
from nltk import FreqDist
import numpy as np 
import pandas as pd
import os
import spacy
from spacy.tokens import Doc
from typing import Optional, Tuple, List, Dict, Callable
import pickle

# ~~~ Spacy stuff ~~~

nlp = spacy.load("en_core_web_md", disable=["ner", "lemmatizer"]) #! test time and performance between small and medium models

def set_spacy_extension(name:str, count_function:Callable) -> None:
    """Creates spacy extensions to easily access certain information"""
    if not Doc.has_extension(name):
        Doc.set_extension(name, getter=count_function)
   
# add more extensions here as needed     
set_spacy_extension("tokens", lambda doc: [token.text for token in doc])        
set_spacy_extension("words", lambda doc: [token.text for token in doc if not token.is_punct])
set_spacy_extension("pos_tags", lambda doc: [token.pos_ for token in doc])
set_spacy_extension("dep_labels", lambda doc: [token.dep_ for token in doc])
set_spacy_extension("morph_tags", lambda doc: [morph for token in doc for morph in token.morph if morph != ""])
        

#! make a vocab class
def load_vocab(path:str, type="static") -> Tuple[str]:
    """
    Loads in a vocabulary file as a tuple of strings
    
    Params
    -----
        path (str): path of vocab file
        type (str, default="static"): determines what type of vocab is to be loaded
        
    Note
    ----
        static: dataset-agnostic vocabulary; it looks for the same exact elements for any dataset
        non-static: vocabulary generated from a dataset, which is stored as a pickle file
    
    Returns
    -------
        tuple[str]: tuple of vocabulary elements
    """
    
    if type == "static":
        assert path.endswith(".txt")
        with open (path, "r") as fin:
            return tuple(map(lambda x: x.strip("\n"), fin.readlines()))
        
    elif type == "non_static":
        assert path.endswith(".pkl")
        with open (path, "rb") as fin:
            return pickle.load(fin)
    else:
        raise ValueError(f"Vocab type '{type}' doesn't exist. Check your vocab call")

# ~~~ Helper functions ~~~

#pos bigrams
def get_sentence_spans(doc) -> List[Tuple[int,int]]:
    """Gets each start and end index of all sentences in a document"""
    return [(sent.start, sent.end) for sent in doc.sentences]

#pos bigrams 
def insert_sentence_boundaries(spans:List[Tuple[int,int]], 
                               tokens:List[str]
                               ) -> List[str]:
    """Inserts sentence boundaries into a list of tokens"""
    new_tokens = []
    
    for i, item in enumerate(tokens):
        for start, end in spans:
            if i == start:
                new_tokens.append("BOS")
            elif i == end:
                new_tokens.append("EOS")    
        new_tokens.append(item)
    new_tokens.append("EOS")  
    return new_tokens

# mixed bigrams
def get_bigrams_with_boundary_syms(doc, tokens:List[str]):
    """Gets the bigrams from given list of tokens, including sentence boundaries"""
    sent_spans = get_sentence_spans(doc)
    tokens_with_boundary_syms = insert_sentence_boundaries(sent_spans, tokens)
    token_bigrams = bigrams(tokens_with_boundary_syms)
    return list(filter(lambda x: x != ("EOS","BOS"), token_bigrams))


# mixed bigrams
def remove_openclass_bigrams(tokens:List[str], OPEN_CLASS:List[str]) -> List[str]:
    """Removes (OPEN_CLASS, OPEN_CLASS) bigrams that inadvertently get created in replace_openclass """
    filtered = []
    for pair in bigrams(tokens):
        if pair[0] not in OPEN_CLASS and pair[1] in OPEN_CLASS:
            filtered.append(pair[0])
            
        if pair[0] in OPEN_CLASS and pair[1] not in OPEN_CLASS:
            filtered.append(pair[0])
    return filtered
    
# mixed bigrams
def replace_openclass(tokens:List[str], pos:List[str]) -> List[str]:
    """Replaces all open class tokens with corresponding POS tags"""
    OPEN_CLASS = ["ADJ", "ADV", "NOUN", "VERB", "INTJ"]
    tokens_replaced = copy(tokens)
    for i in range(len(tokens_replaced)):
        if pos[i] in OPEN_CLASS:
            tokens_replaced[i] = pos[i]

    return remove_openclass_bigrams(tokens_replaced, OPEN_CLASS)

#~~~ Features ~~~


def add_zero_vocab_counts(vocab, counted_doc_features:Counter) -> Dict[str, int]:
    """
    Combines vocab and counted_doc_features into one dictionary such that
    any feature in vocab counted 0 times in counted_doc_features is preserved in the feature vector
       
    Params
    -------
        vocab (Vocab): vocabulary of elements to look for in documents
        counted_doc_features (Counter): features counted from document
        
    Example
    -------
            >> vocab = ("a", "b", "c", "d")
            
            >> counted_doc_features = Counter({"a":5, "c":2})
            
            >> add_zero_vocab_counts(vocab, counted_doc_features)
            
                '{"a": 5, "b" : 0, "c" : 2, "d" : 0}'
                
    Returns
    -------
        Dict[str,int]: counts of every element in vocab with 0 counts preserved
    """
    count_dict = {}
    for feature in vocab:
        if feature in counted_doc_features:
            count = counted_doc_features[feature] 
        else:
            count = 0
        count_dict[feature] = count
    return count_dict

def sum_of_counts(counts:Dict) -> int:
    """Sums the counts of a count dictionary. Returns 1 if counts sum to 0"""
    count_sum = sum(counts.values())
    return count_sum if count_sum > 0 else 1
              


@dataclass
class Feature:
    """
    This class represents the output of each featurizer. Gives access to both the counts themselves and vector
    
    Params
    ------
        featurizer_name (str): featurizer name to prepend to features in self.get_feature_names()
        feature_counts (Dict[str,int]): dictionary of counts
        normalize_by (int): option to normalize. Defaults to 1
    """
    featurizer_name:str
    feature_counts:Dict[str,int]
    normalize_by:int = 1
    
    def counts_to_vector(self) -> np.ndarray:
        """Converts a dictionary of counts into a numpy array and normalizes"""
        counts = list(self.feature_counts.values())
        return np.array(counts).flatten() / self.normalize_by
    
    def get_feature_names(self) -> List[str]:
        """Prepends the feature type to each individual feature"""
        return [f"{self.featurizer_name}: {feat}" for feat in self.feature_counts.keys()]
    
        
    
    
def pos_unigrams(doc) -> Feature:
    
    vocab = load_vocab("vocab/static/pos_unigrams.txt")
    doc_pos_tag_counts = Counter(doc.pos_tags)
    all_pos_tag_counts = add_zero_vocab_counts(vocab, doc_pos_tag_counts)
    
    return Feature("POS Unigram", all_pos_tag_counts, len(doc.pos_tags))

def pos_bigrams(doc) -> Feature:
    
    vocab = load_vocab("vocab/non_static/pan/pos_bigrams/pos_bigrams.pkl", type="non_static")
    doc_pos_bigram_counts = Counter(get_bigrams_with_boundary_syms(doc, doc.pos_tags))
    all_pos_bigram_counts = add_zero_vocab_counts(vocab, doc_pos_bigram_counts)
    
    return Feature("POS Bigram", all_pos_bigram_counts, sum_of_counts(doc_pos_bigram_counts))

def func_words(doc) -> Feature:
    
    vocab = load_vocab("vocab/static/function_words.txt")
    doc_func_word_counts = Counter([token for token in doc.tokens if token in vocab])
    all_func_word_counts = add_zero_vocab_counts(vocab, doc_func_word_counts)
    
    return Feature("Function word", all_func_word_counts, sum_of_counts(doc_func_word_counts))

def punc(doc) -> Feature:
    
    vocab = load_vocab("vocab/static/punc_marks.txt")
    doc_punc_counts = Counter([punc for token in doc.tokens for punc in token if punc in vocab])
    all_punc_counts = add_zero_vocab_counts(vocab, doc_punc_counts)
    
    return Feature("Punctuation", all_punc_counts, sum_of_counts(doc_punc_counts))

def letters(doc) -> Feature:
    
    vocab = load_vocab("vocab/static/letters.txt")
    doc_letter_counts = Counter([letter for token in doc.tokens for letter in token if letter in vocab])
    all_letter_counts = add_zero_vocab_counts(vocab, doc_letter_counts)
    
    return Feature("Letter", all_letter_counts, sum_of_counts(doc_letter_counts))

def common_emojis(doc) -> Feature:
    
    vocab = load_vocab("vocab/static/common_emojis.txt")
    extract_emojis = demoji.findall_list(doc.text, desc=False)
    doc_emoji_counts = Counter(filter(lambda x: x in vocab, extract_emojis))
    all_emoji_counts = add_zero_vocab_counts(vocab, doc_emoji_counts)
    
    return Feature("Emoji", all_emoji_counts, len(doc.tokens))

def embedding_vector(doc) -> Feature:
    """spaCy word2vec (or glove?) document embedding"""
    embedding = {"embedding_vector" : doc.spacy_doc.vector}
    return Feature(None, embedding)

#! doc.words causing crash
def document_stats(doc) -> Feature:
    words = doc.words
    doc_statistics = {"short_words" : len([1 for word in words if len(word) < 5])/len(words) if len(words) else 0, 
                      "large_words" : len([1 for word in words if len(word) > 4])/len(words) if len(words) else 0,
                      "word_len_avg": np.mean([len(word) for word in words]),
                      "word_len_std": np.std([len(word) for word in words]),
                      "sent_len_avg": np.mean([len(sent) for sent in doc.sentences]),
                      "sent_len_std": np.std([len(sent) for sent in doc.sentences]),
                      "hapaxes"     : len(FreqDist(words).hapaxes())/len(words) if len(words) else 0}
    return Feature("Document statistic", doc_statistics)

def dep_labels(doc) -> Feature:

    vocab = load_vocab("vocab/static/dep_labels.txt")
    doc_dep_labels = Counter([dep for dep in doc.dep_labels])
    all_dep_labels = add_zero_vocab_counts(vocab, doc_dep_labels)
    
    return Feature("Dependency label", all_dep_labels, sum_of_counts(doc_dep_labels))

def mixed_bigrams(doc) -> Feature:
    
    vocab = load_vocab("vocab/non_static/pan/mixed_bigrams/mixed_bigrams.pkl", type="non_static")
    doc_mixed_bigrams = Counter(bigrams(replace_openclass(doc.tokens, doc.pos_tags)))
    all_mixed_bigrams = add_zero_vocab_counts(vocab, doc_mixed_bigrams)
    
    return Feature("Mixed Bigram", all_mixed_bigrams, sum_of_counts(doc_mixed_bigrams))

def morph_tags(doc) -> Feature:
    
    vocab = load_vocab("vocab/static/morph_tags.txt")
    doc_morph_tags = Counter(parse_morph([token.morph for token in doc.spacy_doc]))
    all_morph_tags = add_zero_vocab_counts(vocab, doc_morph_tags)
    
    return Feature("Morphology tag", all_morph_tags, sum_of_counts(doc_morph_tags))

# ~~~ Featurizers end ~~~


     
class GrammarVectorizer:
    
    
    register = (pos_unigrams,
                pos_bigrams,
                func_words,
                punc,
                letters,
                common_emojis,
                embedding_vector,
                document_stats,
                dep_labels,
                mixed_bigrams,
                morph_tags)
    
    def __init__(self, config:Dict[str,int]=None):
        
        
        self._config = self._process_config(config)
        os.system("./clear_logs.sh")
        
    def get_config(self) -> List[str]:
        """Retrieves the names of all activated features"""
        return [feat.__name__ for feat in self._config]
        
    def _process_config(self, passed_config: Optional[Dict]) -> List:
        """Reads which features to activate and returns a list of featurizer functions"""
        current_config = DEFAULT_CONFIG if not passed_config else passed_config
        activated_feats = []
        for feat in self.register:
            try:
                if current_config[feat.__name__] == 1:
                    activated_feats.append(feat)
            except KeyError:
                raise KeyError(f"Feature '{feat.__name__}' does not exist in given configuration")
        return activated_feats
     
    def _apply_featurizers(self, document:str) -> List[Feature]:
        """Applies featurizers to a document and returns a list of features for a single document"""
        doc = make_document(document, self.nlp)
        features = []
        for featurizer in self._config:
            feature = featurizer(doc)
            counts = feature.feature_counts
            vector = feature.counts_to_vector()

            features.append(feature)
            feature_logger(featurizer.__name__, f"{counts}\n{vector}\n\n") 

        return features

    def _concat_vectors(self, features:List[Feature]) -> np.ndarray:
        """Concatenates a list of vectorized feature counts"""
        return np.concatenate([feat.counts_to_vector() for feat in features])
    
    def _get_all_feature_names(self, features:List[Feature]) -> List[str]:
        """Gets all feature names from a list of document features"""
        feature_names = []
        for feat in features:
            feature_names.extend(feat.get_feature_names())
        return feature_names
        
    def create_vector_df(self, documents:List[str]) -> pd.DataFrame:
        """Applies featurizers to all documents and stores the resulting matrix as a dataframe"""
        all_vectors = []
        feature_names = None
        for document in documents:
            doc_features = self._apply_featurizers(document)
            all_vectors.append(self._concat_vectors(doc_features))
            
            if feature_names is None:
                # hacky way of doing this, but it works well
                feature_names = self._get_all_feature_names(doc_features)  
             
        df = pd.DataFrame(np.vstack(all_vectors), columns=feature_names)
        return df
    
    
    @classmethod
    def from_jsonlines(cls, path:str):
        """Generate a dataframe given a jsonlines file containing authorIDs and documentID fields"""
        pass
    
    @classmethod
    def from_list(cls, documents:List[str]):
        """Generate a dataframe given a list of documents."""
        
        documents = map(lambda x: demoji.replace(x, ""))
        docs = cls.nlp.pipe(documents)
        
        for i, doc in enumerate(docs):
            current_doc = UniversalDocument(documents[i], doc)


