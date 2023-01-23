#!/usr/bin/env python3

import spacy
import toml
import numpy as np
np.seterr(invalid="ignore")
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
    
    if not os.path.exists("logs"):
        os.mkdir("logs")
               
    with open(f"logs/{filename}.log", "a") as fout: 
        fout.write(writable)
        
OPEN_CLASS = ["ADJ", "ADV", "NOUN", "VERB", "INTJ"]


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
        del counter[("EOS","BOS")] # removes artificial bigram
    except: pass
    
    return counter


def replace_openclass(tokens, pos):

    for i in range(len(tokens)):
        if pos[i] in OPEN_CLASS:
            tokens[i] = pos[i]
    return tokens
        
        
def get_mixed_bigrams(doc) -> Counter:
    
    tokens = [token.text for token in doc]
    pos    = [token.pos_ for token in doc]
    mixed_bigrams = list(bigrams(replace_openclass(tokens, pos)))
    
    return Counter(mixed_bigrams)


def get_pos_subsequences(doc):
    
    pos = [token.pos_ for token in doc]
    
    subseqs = []
    for i in range(len(pos)-1):
        subseqs.extend([(pos[i],pos[n]) for n in range(i+1, len(pos))])   
        
    return Counter(subseqs)  

# ~~~ Featurizers ~~~

def pos_unigrams(document) -> np.ndarray: 
    
    tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
    counts, doc_features = get_counts(tags, document.pos_tags)
    result = np.array(counts) #/ len(document.pos_tags)
    assert len(tags) == len(counts)
    
    return result, doc_features

def pos_bigrams(document) -> np.ndarray : # len = 50

    vocab = utils.load_pkl("vocab/pan_pos_bigrams_vocab.pkl") # path will need to change per dataset 
    doc_pos_bigrams = get_pos_bigrams(document.doc)
    counts, doc_features = get_counts(vocab, doc_pos_bigrams)
    result = np.array(counts) #/ len(document.pos_tags)
    assert len(vocab) == len(counts)
    
    return result, doc_features


def func_words(document) -> np.ndarray:  # len = 145
    
    # modified NLTK stopwords set
    with open ("vocab/function_words.txt", "r") as fin:
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
    
    vocab = utils.load_pkl("vocab/pan_mixed_bigrams_vocab.pkl")
    doc_mixed_bigrams = get_mixed_bigrams(document.doc)
    counts, doc_features = get_counts(vocab, doc_mixed_bigrams)
    result = np.array(counts) 
    assert len(vocab) == len(counts)
    
    return result, doc_features

# Under construction. Keep commented out.
# def pos_subsequences(document):
    
#     vocab = utils.load_pkl("vocab/pan_pos_subsequences_vocab.pkl")
#     doc_pos_subsuences = get_pos_subsequences(document.doc)
#     counts, doc_features = get_counts(vocab, doc_pos_subsuences)
#     result = np.array(counts) 
#     assert len(vocab) == len(counts)

#     return result, doc_features


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
    """This class houses all featurizers"""
    
    def __init__(self, data_path, logging=False):
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
            "dep_labels"    :dep_labels,
            "mixed_bigrams" :mixed_bigrams}
        
        self._generate_vocab(data_path)
        
    def _config(self):
        """Reads 'config.toml' to retrieve which features to apply. 0 = deactivated, 1 = activated"""
        toml_config = toml.load("config.toml")["Features"]
        config = []
        for name, feat in self.featurizers.items():
            try:
                if toml_config[name] == 1:
                    config.append(feat)
            except KeyError:
                raise KeyError(f"Feature '{name}' does not exist in config.toml")
        return config
    
    def _generate_vocab(self, data_path):
        """
        Generates vocab files required by some featurizers. Assumes the following input data format:
                            {
                             author_id : [doc1, doc2,...docn],
                             author_id : [...]
                             }
        """
        data = utils.load_json(data_path)
        dataset = "pan" if "pan" in data_path else "mud" # will need to be changed based on data set
        counters = {
            "pos_bigrams"     : [],
            "pos_subsequences": [],
            "mixed_bigrams"   : []
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
                    
                    pos_subseqs = get_pos_subsequences(doc)
                    counters["pos_subsequences"].append(pos_subseqs)
        
                # this line condenses all the counters into one dict, getting the 50 most common elements
                most_common = dict(sum(counter_list, Counter()).most_common(50)) # most common returns list of tuples, gets converted back to dict
                utils.save_pkl(list(most_common.keys()), out_path)
        
        
    
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