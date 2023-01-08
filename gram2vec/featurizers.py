#!/usr/bin/env python3

import spacy
import toml
import numpy as np
from nltk import bigrams
from nltk import FreqDist
import os
from dataclasses import dataclass
import demoji
from collections import Counter

# project imports 
import utils

# ~~~ Logging and global variables ~~~

def feature_logger(filename, writable):
    
    if not os.path.exists("logs"): # make log dir if not exists
        os.mkdir("logs")
               
    with open(f"logs/{filename}.log", "a") as fout: 
        fout.write(writable)
        
OPEN_CLASS = ["ADJ", "ADV", "NOUN", "VERB", "INTJ"]

np.seterr(invalid="ignore")
        
    
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

def replace_openclass(tokens, pos):

    for i in range(len(tokens)):
        if pos[i] in OPEN_CLASS:
            tokens[i] = pos[i]
            
    return tokens
        
def get_mixed_bigrams(doc) -> Counter:
    
    tokens = [token.text for token in doc]
    pos    = [token.pos_ for token in doc]
    mixed_bigrams = list(bigrams(replace_openclass(tokens, pos)))
    
    # remove bigrams which are not mixed (i.e. (token,token) and (OPEN CLASS TAG, OPEN CLASS TAG) bigrams)
    for x, y in mixed_bigrams:
        if x in OPEN_CLASS and y in OPEN_CLASS or x not in OPEN_CLASS and y not in OPEN_CLASS:
            mixed_bigrams.remove((x,y))
            
    return Counter(mixed_bigrams)
          
    
    
    
    
def insert_boundaries(sent_spans:list[tuple], tokens:list):
    """
    This function inserts sentence boundaries to a list of tokens 
    according to a list of (START, END) sentence index markers
    
    Works by enumerating the tokens and checking if each position 
    is the start or end of a sentence, inserting the appropriate tag when
    """
    new_tokens = []
    for i, item in enumerate(tokens):
        for start, end in sent_spans:
            if i == start:
                new_tokens.append("BOS")
            elif i == end:
                new_tokens.append("EOS")    
        new_tokens.append(item)
    new_tokens.append("EOS")  
        
    return new_tokens


def get_pos_bigrams(doc) -> Counter:
    
    sent_spans = [(sent.start, sent.end) for sent in doc.sents]
    pos = insert_boundaries(sent_spans, [token.pos_ for token in doc])
    counter = Counter(bigrams(pos))
    try:
        del counter[("EOS","BOS")]
    except: pass
    
    return counter  

    
def generate_pos_vocab(path):
    
    data = utils.load_json(path)
    nlp = utils.load_spacy("en_core_web_md")
    bigram_counters = [] # becomes a list of dicts
    
    all_text_docs = [entry for id in data.keys() for entry in data[id]]
    for text in all_text_docs:
        doc = nlp(text)
        
        
        counts = get_pos_bigrams(doc)
        
        
        bigram_counters.append(counts)
    
    # this line condenses all the counters into one dict, getting the 50 most common bigrams
    common_bigrams = dict(sum(bigram_counters, Counter()).most_common(50))
    
    # saves the 50 most common bigrams as a list to a pickle
    utils.save_pkl(list(common_bigrams.keys()),"resources/pan_pos_vocab.pkl")
    

# ~~~ Featurizers ~~~

def pos_unigrams(document) -> np.ndarray: 
    
    tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
    counts, doc_features = get_counts(tags, document.pos_tags)
    result = np.array(counts) #/ len(document.pos_tags)
    assert len(tags) == len(counts)
    
    return result, doc_features

def pos_bigrams(document): # len = 50

    if not os.path.exists("resources/pan_pos_vocab.pkl"):
        generate_pos_vocab("data/pan/preprocessed/fixed_sorted_author.json")
    
    vocab = utils.load_pkl("resources/pan_pos_vocab.pkl")
    doc_pos_bigrams = get_pos_bigrams(document.doc)
    counts, doc_features = get_counts(vocab, doc_pos_bigrams)
    result = np.array(counts) #/ len(document.pos_tags)
    assert len(vocab) == len(counts)
    
    return result, doc_features


def func_words(document) -> np.ndarray:  # len = 145
    
    # modified NLTK stopwords set
    with open ("resources/function_words.txt", "r") as fin:
        function_words = set(map(lambda x: x.strip("\n"), fin.readlines()))

    doc_func_words = [token for token in document.tokens if token in function_words]
    counts, doc_features = get_counts(function_words, doc_func_words)
    result = np.array(counts) #/ len(document.tokens)
    assert len(function_words) == len(counts)
    
    return result, doc_features


def punc(document) -> np.ndarray:
    
    punc_marks = [".", ",", ":", ";", "\'", "\"", "?", "!", "`", "*", "&", "_", "-", "%", "(", ")", "–", "‘", "’"]
    doc_punc_marks = [punc for token in document.doc 
                           for punc in token.text
                           if punc in punc_marks]
    
    counts, doc_features = get_counts(punc_marks, doc_punc_marks)
    result = np.array(counts) #/ len(document.tokens) 
    assert len(punc_marks) == len(counts)
    
    return result, doc_features


def letters(document) -> np.ndarray: 

    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
               "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
               "à", "è", "ì", "ò", "ù", "á", "é", "í", "ó", "ú", "ý"]
    doc_letters = [letter for token in document.doc 
                          for letter in token.text 
                          if letter in letters]
    
    counts, doc_features = get_counts(letters, doc_letters)
    result = np.array(counts) #/ len(doc_letters)
    assert len(letters) == len(counts)
    
    return result, doc_features


def common_emojis(document):
    
    vocab = ["😅", "😂", "😊", "❤️", "😭", "👍", "👌", "😍", "💕", "🥰"]
    extract_emojis = demoji.findall_list(document.text, desc=False)
    emojis = list(filter(lambda x: x in vocab, extract_emojis))
    
    counts, doc_features = get_counts(vocab, emojis)
    result = np.array(counts) #/ len(document.tokens)
    
    return result, doc_features

def doc_vector(document):
    result = document.doc.vector
    return result, None


def doc_stats(document):
    
    words = document.words

    # num short and large words
    short_words = len([1 for word in words if len(word) < 5])
    large_words = len([1 for word in words if len(word) > 4])
    
    # avg, std word length
    word_lens = [len(word) for word in words] 
    word_len_avg = np.mean(word_lens)
    word_len_std = np.std(word_lens)
    
    # avg, std sentence length
    sent_lens = [len(sent) for sent in document.doc.sents]
    sent_len_avg = np.mean(sent_lens)
    sent_len_std = np.std(sent_lens)
    
    # hapax legomena ratio (vocabulary richness - how many words only occur once)
    fd = FreqDist(words)
    hapax = len(fd.hapaxes())
    
    doc_features = {"short_words": short_words, 
                    "large_words": large_words,
                    "word_len_avg": word_len_avg,
                    "word_len_std": word_len_std,
                    "sent_len_avg": sent_len_avg,
                    "sent_len_std": sent_len_std,
                    "hapaxes": hapax}
    
    array = np.array([short_words, 
                      large_words,
                      word_len_avg,
                      word_len_std,
                      sent_len_avg,
                      sent_len_std,
                      hapax,])
    
    return array, doc_features
    
    
def dep_labels(document):
    
    labels = ['ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp', 'compound', 
              'conj', 'csubj', 'csubjpass', 'dative', 'dep', 'det', 'dobj', 'expl', 'intj', 'mark', 'meta', 'neg', 'nmod', 'npadvmod', 'nsubj', 
              'nsubjpass', 'nummod', 'oprd', 'parataxis', 'pcomp', 'pobj', 'poss', 'preconj', 'predet', 'prep', 'prt', 'punct', 'quantmod', 'relcl', 'xcomp']
    
    document_dep_labels = [token.dep_ for token in document.doc]
    counts, doc_features = get_counts(labels, document_dep_labels)
    result = np.array(counts) / len(document_dep_labels)
    assert len(counts) == len(labels)
    
    return result, doc_features



def mixed_bigrams(document):
    pass



#? pos subsequences?


# ~~~ Featurizers end ~~~


@dataclass
class Document:
    doc      :spacy.tokens.doc.Doc
    tokens   :list[str]
    words    :list[str]   
    pos_tags :list[str]
    text     :str
    
    
    @classmethod
    def from_nlp(cls, doc, text):
        tokens   = [token.text for token in doc]                   
        pos_tags = [token.pos_ for token in doc]    
        words    = [token.text for token in doc if not token.is_punct]              
        return cls(doc, tokens, words, pos_tags, text)
    
class GrammarVectorizer:
    """This constructor houses all featurizers and the means to apply them"""
    
    def __init__(self, logging=False):
        self.nlp = utils.load_spacy("en_core_web_md")
        self.logging = logging
        
        self.featurizers = {
            "pos_unigrams"  :pos_unigrams,
            "pos_bigrams"   :pos_bigrams,
            "func_words"    :func_words, 
            "punc"          :punc,
            "letters"       :letters,
            "common_emojis" :common_emojis,
            "doc_vector"    :doc_vector,
            "doc_stats"     :doc_stats,
            "dep_labels"    :dep_labels}
        
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
        
        
        text_demojified = demoji.replace(text, "") # dep parser hates emojis 
        doc = self.nlp(text_demojified)
        document = Document.from_nlp(doc, text)
        
        vectors = []
        for feat in self._config():
            
            vector, doc_features = feat(document)
            assert not np.isnan(vector).any() 
            vectors.append(vector)
            
            if self.logging:
                feature_logger(f"{feat.__name__}", f"{doc_features}\n{vector}\n\n")
                    
        return np.concatenate(vectors)
    
    
    
    
def main():
    pass




if __name__ == "__main__":
    main()