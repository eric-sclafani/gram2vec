#!/usr/bin/env python3

import spacy
import toml
import numpy as np
from nltk import bigrams
import os
import logging
from dataclasses import dataclass
import demoji
from collections import Counter

# project import
import utils

np.seterr(invalid="ignore")

# ~~~ Logging ~~~

def feature_logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    Credits: https://stackoverflow.com/questions/54591352/python-logging-new-log-file-each-loop-iteration
    """
    
    if not os.path.exists("logs"):
        os.mkdir("logs")
        
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = ("%(message)s")
    log_format = logging.Formatter(format_string)
    
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger
    


# ~~~ Helper functions ~~~

def get_counts(sample_space:list, features:list) -> list[int]:
    """
    Counts the frequency of items in 'sample_space' that occur in 'features'.
    When 'feat_dict' and 'count_doc_features' are merged, the 0 counts in 'feat_dict' 
    get overwritten by the counts in 'doc_features'. When features are not found in 'doc_features', 
    the 0 count in 'feat_dict' is preserved, indicating that the feature is absent in the current document
    
    Params:
        sample_space(list) = list of set features to count. Each feature is initially mapped to 0
        doc_features(list) = list of features from a document to count. 
    Returns:
        list: list of feature counts
    
    """
    feat_dict = {feat:0 for feat in sample_space}
    doc_features = Counter(features)
    
    count_dict = {}
    for feature in feat_dict.keys():
        if feature in doc_features:
            to_add = doc_features[feature]
        else:
            to_add = feat_dict[feature]
        
        count_dict[feature] = to_add
        
    return list(count_dict.values()), count_dict


def get_pos_bigrams(doc):
    return Counter(bigrams([token.pos_ for token in doc]))


def generate_pos_vocab(path):
    
    data = utils.load_json(path)
    nlp = utils.load_spacy("en_core_web_md")
    bigram_counters = []
    
    all_text_docs = [entry for id in data.keys() for entry in data[id]]
    for text in all_text_docs:
        doc = nlp(text)
        bigram_counters.append(get_pos_bigrams(doc))
    
    # this line adds all the counters into one dict, getting the 50 most common pos bigrams
    pos_bigrams = dict(sum(bigram_counters, Counter()).most_common(50))
        
    utils.save_pkl(list(pos_bigrams.keys()),"resources/pan_pos_vocab.pkl")
    
    
def docify(text:str, nlp):
    """Converts a text into a Document object"""
    text_demojified = demoji.replace(text, "") # dep parser hates emojis 
    doc = nlp(text_demojified)
    return Document.from_nlp(doc, text)
    

# ~~~ Featurizers ~~~

def pos_unigrams(document) -> np.ndarray:  
    
    tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
    counts, doc_features = get_counts(tags, document.pos_tags)
    result = np.array(counts) / len(document.pos_tags)
    assert len(tags) == len(counts)
    
    return result, doc_features

def pos_bigrams(document):

    if not os.path.exists("resources/pan_pos_vocab.pkl"):
        generate_pos_vocab("data/pan/preprocessed/fixed_sorted_author.json")
    
    vocab = utils.load_pkl("resources/pan_pos_vocab.pkl")
    doc_pos_bigrams = get_pos_bigrams(document.doc)
    counts, doc_features = get_counts(vocab, doc_pos_bigrams)
    result = np.array(counts) / len(document.pos_tags)
    assert len(vocab) == len(counts)
    
    return result, doc_features


def func_words(document) -> np.ndarray:  
    
    # modified NLTK stopwords set
    with open ("resources/function_words.txt", "r") as fin:
        function_words = set(map(lambda x: x.strip("\n"), fin.readlines()))

    doc_func_words = [token for token in document.tokens if token in function_words]
    counts, doc_features = get_counts(function_words, doc_func_words)
    result = np.array(counts) / len(document.tokens)
    assert len(function_words) == len(counts)
    
    return result, doc_features


def punc(document) -> np.ndarray:
    
    punc_marks = [".", ",", ":", ";", "\'", "\"", "?", "!", "`", "*", "&", "_", "-", "%", ":(", ":)", "...", "..", "(", ")", ":))", "–", "‘", "’", ";)"]
    doc_punc_marks = [token.text for token in document.doc if token.text in punc_marks]
    counts, doc_features = get_counts(punc_marks, doc_punc_marks)
    result = np.array(counts) / len(document.tokens) 
    assert len(punc_marks) == len(counts)
    
    return result, doc_features


def letters(document) -> np.ndarray: 

    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
               "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    doc_letters = [letter for token in document.doc 
                          for letter in token.text 
                          if letter in letters]
    counts, doc_features = get_counts(letters, doc_letters)
    result = np.array(counts) / len(doc_letters)
    assert len(letters) == len(counts)
    
    return result, doc_features



    


# incomplete
def mixed_ngrams(n, document):
    pass



# incomplete
def count_emojis(document):
    emojis = demoji.findall_list(document.text, desc=False)
    return len(emojis) / len(document.text)



# incomplete
def num_words_per_sent(document):
    words_per_sent = np.array([len(sent) for sent in document.doc.sents]) / len(document.tokens) 
    avg = np.mean(words_per_sent)
    std = np.std(words_per_sent)
    return avg, std
    
# incomplete
def num_short_words(document):
    pass

# incomplete
def absolute_sent_length(document):
    pass

def avg_word_length(document):
    pass

#? character ngrams?

#? pos subsequences?


# ~~~ Featurizers end ~~~


@dataclass
class Document:
    doc          :spacy.tokens.doc.Doc
    tokens       :list[str]
    pos_tags     :list[str]
    text         :str
    
    @classmethod
    def from_nlp(cls, doc, text):
        tokens       = [token.text for token in doc]                   
        pos_tags     = [token.pos_ for token in doc]                   
        return cls(doc, tokens, pos_tags, text)
    
class GrammarVectorizer:
    """This constructor houses all featurizers and the means to apply them"""
    
    def __init__(self):
        self.nlp = utils.load_spacy("en_core_web_md")
        
        self.featurizers = {
            "pos_unigrams":pos_unigrams,
            "pos_bigrams" :pos_bigrams,
            "func_words"  :func_words, 
            "punc"        :punc,
            "letters"     :letters,}
        
    def _config(self):
        
        toml_config = toml.load("config.toml")["Featurizers"]
        config = []
        for name, feat in self.featurizers.items():
            try:
                if toml_config[name] == 1:
                    config.append(feat)
            except KeyError:
                raise KeyError(f"Feature '{name}' does not exist in config.toml")
        return config
        
    
    def vectorize(self, text:str) -> np.ndarray:
        """Applies featurizers to an input text. Returns a 1-D array."""
        
        document = docify(text, self.nlp)
        
        vectors = []
        for feat in self._config():
            
            vector, doc_features = feat(document)
            assert not np.isnan(vector).any() 
            vectors.append(vector)
            
            logger = feature_logger(f"logs/{feat.__name__}_log.log")
            logger.debug(f"{doc_features}\n{vector}\n\n")
                    
        return np.concatenate(vectors)
    
    
    
