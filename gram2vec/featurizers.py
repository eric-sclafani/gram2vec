
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import demoji
from nltk import bigrams
import numpy as np 
import os
import spacy
import toml
from typing import Iterable

# project imports 
import utils


# ~~~ Logging and type aliases~~~

Counts = list[int]
SentenceSpan = tuple[int,int]

def feature_logger(filename, writable):
    
    if not os.path.exists("logs"):
        os.mkdir("logs")
               
    with open(f"logs/{filename}.log", "a") as fout: 
        fout.write(writable)
    
# ~~~ Static vocabularies ~~~
pos_unigram_vocab    = utils.load_vocab("vocab/static/pos_unigrams.txt")
function_words_vocab = utils.load_vocab("vocab/static/function_words.txt")
dep_labels_vocab     = utils.load_vocab("vocab/static/dep_labels.txt")
punc_marks_vocab     = utils.load_vocab("vocab/static/punc_marks.txt")
letters_vocab        = utils.load_vocab("vocab/static/letters.txt")
common_emojis_vocab  = utils.load_vocab("vocab/static/common_emojis.txt")

# ~~~ Non-static vocabularies ~~~
#NOTE: the path needs to change be manually changed to matched appropriate dataset
pos_bigrams_vocab = utils.load_pkl("vocab/non_static/pos_bigrams/pan/pos_bigrams.pkl")
mixed_bigrams_vocab = utils.load_pkl("vocab/non_static/mixed_bigrams/pan/mixed_bigrams.pkl")

# ~~~ Document representation ~~~

@dataclass
class Document:
    """
    Class representing elements from a spaCy Doc object
        :param raw_text: text before being processed by spaCy
        :param doc: spaCy's document object
        :param tokens = list of tokens
        :param pos_tags: list of pos tags
        :param dep_labels: list of dependency parse labels
        :param sentences: list of spaCy-sentencized sentences
    Note: instances should only be created using the 'make_document' function 
"""
    raw_text   :str
    doc        :spacy.tokens.doc.Doc
    tokens     :list[str]
    pos_tags   :list[str]
    dep_labels :list[str]
    sentences  :list[spacy.tokens.span.Span]
    
    def __repr__(self):
        return f"Document({self.tokens[0:10]}..)"
    
def make_document(text:str, nlp) -> Document:
    """Converts raw text into a Document object"""
    raw_text   = deepcopy(text)
    doc        = nlp(demojify_text(text)) # dep parser hates emojis
    tokens     = [token.text for token in doc]
    pos_tags   = [token.pos_ for token in doc]
    dep_labels = [token.dep_ for token in doc]
    sentences  = list(doc.sents)
    return Document(raw_text, doc, tokens, pos_tags, dep_labels, sentences)
    
# ~~~ Helper functions ~~~

def demojify_text(text:str):
    return demoji.replace(text, "") # dep parser hates emojis 

def get_sentence_spans(doc:Document) -> list[SentenceSpan]:
    return [(sent.start, sent.end) for sent in doc.sentences]
   
def insert_pos_sentence_boundaries(doc:Document) -> list[str]:
    """Inserts sentence boundaries into a list of POS tags"""
    spans = get_sentence_spans(doc)
    new_tokens = []
    
    for i, item in enumerate(doc.pos_tags):
        for start, end in spans:
            if i == start:
                new_tokens.append("BOS")
            elif i == end:
                new_tokens.append("EOS")    
        new_tokens.append(item)
    new_tokens.append("EOS")  
        
    return new_tokens

def replace_openclass(tokens:list[str], pos:list[str]) -> list[str]:
    """Replaces all open class tokens with corresponding POS tags"""
    OPEN_CLASS = ["ADJ", "ADV", "NOUN", "VERB", "INTJ"]
    tokens = deepcopy(tokens)
    for i in range(len(tokens)):
        if pos[i] in OPEN_CLASS:
            tokens[i] = pos[i]
    return tokens

   
# ~~~ Counter functions ~~~
# These functions are used to count certain text elements from documents

def count_pos_unigrams(doc:Document) -> Counter:
    return Counter(doc.pos_tags)

def count_pos_bigrams(doc:Document) -> Counter:
    pos_tags_with_boundaries = insert_pos_sentence_boundaries(doc)
    counter = Counter(bigrams(pos_tags_with_boundaries))
    try:
        del counter[("EOS","BOS")] # removes artificial bigram #! move this to pos sentence boundary function
    except: pass
    
    return counter

def count_func_words(doc:Document) -> Counter:
    return Counter([token for token in doc.tokens if token in function_words_vocab])

def count_punctuation(doc:Document) -> Counter:
    return Counter([punc for token in doc.tokens for punc in token.text if punc in punc_marks_vocab])

def count_letters(doc:Document) -> Counter:
    return Counter([letter for token in doc.tokens for letter in token.text if letter in letters_vocab])

def count_emojis(doc:Document) -> Counter:
    extract_emojis = demoji.findall_list(doc.text, desc=False)
    return Counter(filter(lambda x: x in common_emojis_vocab, extract_emojis))

def count_dep_labels(doc:Document) -> Counter:
    return Counter([dep for dep in doc.dep_labels])


# ~~~ Featurizers ~~~

def pos_unigrams(document) -> np.ndarray: 
    
    
    tags = {"ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"}
    
    doc_pos_tags = [token.pos_ for token in document]   # Document.pos_tags 
    
    counts, doc_features = get_counts(tags, doc_pos_tags)
    
    result = np.array(counts) #/ len(document.pos_tags)
    assert len(tags) == len(counts)
    
    return result, doc_features

def most_common_pos_bigrams(document) -> np.ndarray : # len = 50

    vocab = utils.load_vocab("vocab/pan_pos_bigrams_vocab.pkl") # path will need to change per dataset 
    
    doc_pos_bigrams = get_pos_bigrams(document)
    
    counts, doc_features = get_counts(vocab, doc_pos_bigrams)
    
    result = np.array(counts) #/ len(document.pos_tags)
    assert len(vocab) == len(counts)
    
    return result, doc_features


def func_words(document) -> np.ndarray:
    
    # modified NLTK stopwords set
    #! use pkl file for vocab
    with open ("vocab/function_words.txt", "r") as fin:
        function_words = tuple(map(lambda x: x.strip("\n"), fin.readlines()))

    tokens = [token.text for token in document] 
    doc_func_words = [token for token in tokens if token in function_words]
    
    counts, doc_features = get_counts(function_words, doc_func_words)
    result = np.array(counts) #/ len(document.tokens)
    assert len(function_words) == len(counts)
    
    return result, doc_features


def punc(document) -> np.ndarray:
    
    punc_marks = {".", ",", ":", ";", "\'", "\"", "?", "!", "`", "*", "&", "_", "-", "%", "(", ")", "–", "‘", "’"}
    
    doc_punc_marks = [punc for token in document 
                           for punc in token.text
                           if punc in punc_marks]
    
    counts, doc_features = get_counts(punc_marks, doc_punc_marks)
    
    result = np.array(counts) #/ len(document.tokens) 
    assert len(punc_marks) == len(counts)
    
    return result, doc_features


def letters(document) -> np.ndarray: 

    letters = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
               "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
               "à", "è", "ì", "ò", "ù", "á", "é", "í", "ó", "ú", "ý"}
    
    doc_letters = [letter for token in document
                          for letter in token.text 
                          if letter in letters]
    
    counts, doc_features = get_counts(letters, doc_letters)
    result = np.array(counts) #/ len(doc_letters)
    assert len(letters) == len(counts)
    
    return result, doc_features


def common_emojis(document):
    
    vocab = {"😅", "😂", "😊", "❤️", "😭", "👍", "👌", "😍", "💕", "🥰"}
    
    extract_emojis = demoji.findall_list(document.text, desc=False)
    emojis = list(filter(lambda x: x in vocab, extract_emojis))
    
    counts, doc_features = get_counts(vocab, emojis)
    result = np.array(counts) #/ len(document.tokens)
    
    return result, doc_features

def doc_vector(document):
    result = document.doc.vector
    return result, None
    
    
def dep_labels(document):
    
    labels = {'ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp', 'compound', 
              'conj', 'csubj', 'csubjpass', 'dative', 'dep', 'det', 'dobj', 'expl', 'intj', 'mark', 'meta', 'neg', 'nmod', 'npadvmod', 'nsubj', 
              'nsubjpass', 'nummod', 'oprd', 'parataxis', 'pcomp', 'pobj', 'poss', 'preconj', 'predet', 'prep', 'prt', 'punct', 'quantmod', 'relcl', 'xcomp'}
    
    document_dep_labels = [token.dep_ for token in document]
    
    counts, doc_features = get_counts(labels, document_dep_labels)
    
    result = np.array(counts) / len(document_dep_labels)
    assert len(counts) == len(labels)
    
    return result, doc_features


def mixed_bigrams(document):
    
    vocab = utils.load_pkl("vocab/pan_mixed_bigrams_vocab.pkl")
    
    doc_mixed_bigrams = get_mixed_bigrams(document)
    
    counts, doc_features = get_counts(vocab, doc_mixed_bigrams)
    result = np.array(counts) 
    assert len(vocab) == len(counts)
    
    return result, doc_features


# ~~~ Featurizers end ~~~
   

class FeatureVector:
    """
    Each feature vector object should have access to :
        - all individual feature vectors, as well as the concatenated one

    """
    def __init__(self,):
        pass 
    
    def __add__(self, other):
        pass
    
    


class CountBasedFeaturizer:
    
    def __init__(self, name:str, vocab:tuple, counter):
        self.name = name
        self.vocab = vocab
        self.counter = counter
        
    def __repr__(self):
        return self.name
        
    def _get_all_feature_counts(self, counted_doc_features:Counter) -> dict:
        """
        Combines vocab and counted_document_features into one dictionary such that
        any feature in vocab counted 0 times in counted_document_features is preserved in the feature vector. 
        
        :param document_counts: features counted from document
        :returns: counts of every element in vocab with 0 counts preserved
        :rtype: dict
        
        Example:
                >> self.vocab = ("a", "b", "c", "d")
                
                >> counted_doc_features = Counter({"a":5, "c":2})
                
                >> self._get_all_feature_counts(vocab, counted_doc_features)
                
                    '{"a": 5, "b" : 0, "c" : 2, "d" : 0}'
        """
        count_dict = {}
        for feature in self.vocab:
            if feature in counted_doc_features:
                count = counted_doc_features[feature] 
            else:
                count = 0
            count_dict[feature] = count
        return count_dict
        
    def get_counts(self, document:Document) -> dict:
        """
        Applies counter function to document to get all feature counts
        
        """
        counted_doc_features = self.counter(document)
        return self._get_all_feature_counts(counted_doc_features)

    def vectorize(self, document:Document):
        # normalization here
        counts = self.get_counts(document).values()
        return np.array(counts)
        








 
def count_mixed_bigrams(doc:Document) -> Counter:
    return Counter(bigrams(replace_openclass(doc.tokens, doc.pos_tags)))

mixed_bigrams = CountBasedFeaturizer(
    name = "mixed_bigrams",
    vocab = [("hi", "NOUN")],
    counter=count_mixed_bigrams
    )




def config(path_to_config:str):
    pass



   
   

    
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
        
    
    def vectorize(self, text:str) -> np.ndarray:
        """Applies featurizers to an input text. Returns a 1-D array."""
        
        doc = make_document(text, self.nlp)
        
        vectors = []
        for feat in self._config():
            
            vector, doc_features = feat(doc)
            assert not np.isnan(vector).any() 
            vectors.append(vector)
            
            if self.logging:
                feature_logger(f"{feat.__name__}", f"{doc_features}\n{vector}\n\n")
                    
        return np.concatenate(vectors)
    
    
# debugging
if __name__ == "__main__":
    pass