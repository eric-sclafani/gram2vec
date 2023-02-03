
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import demoji
from nltk import bigrams
from nltk import FreqDist
import numpy as np 
import os
import spacy
import toml
from typing import Union

# project imports 
import utils


# ~~~ Logging and type aliases~~~

SentenceSpan = tuple[int,int]
Vocab = tuple[str]

def feature_logger(filename, writable):
    
    if not os.path.exists("logs"):
        os.mkdir("logs")
               
    with open(f"logs/{filename}.log", "a") as fout: 
        fout.write(writable)
        
# ~~~ Document representation ~~~
@dataclass
class Document:
    """
    This class represents elements from a spaCy Doc object
        :param raw_text: text before being processed by spaCy
        :param spacy_doc: spaCy's document object
        :param tokens = list of tokens
        :param words = list of words (no punc tokens)
        :param pos_tags: list of pos tags
        :param dep_labels: list of dependency parse labels
        :param sentences: list of spaCy-sentencized sentences
    Note: instances should only be created using the 'make_document' function 
"""
    raw_text   :str
    spacy_doc  :spacy.tokens.doc.Doc
    tokens     :list[str]
    words      :list[str]
    pos_tags   :list[str]
    dep_labels :list[str]
    sentences  :list[spacy.tokens.span.Span]
    
    def __repr__(self):
        return f"Document({self.tokens[0:10]}..)"
    
def make_document(text:str, nlp) -> Document:
    """Converts raw text into a Document object"""
    raw_text   = deepcopy(text)
    spacy_doc  = nlp(demojify_text(text)) # dep parser hates emojis
    tokens     = [token.text for token in spacy_doc]
    words      = [token.text for token in spacy_doc if not token.is_punct]
    pos_tags   = [token.pos_ for token in spacy_doc]
    dep_labels = [token.dep_ for token in spacy_doc]
    sentences  = list(spacy_doc.sents)
    return Document(raw_text, spacy_doc, tokens, words, pos_tags, dep_labels, sentences)
 
# ~~~ Helper functions ~~~


def demojify_text(text:str):
    """Strips text of its emojis (used only when making spaCy object, since dep parser seems to hate emojis)"""
    return demoji.replace(text, "")

def get_sentence_spans(doc:Document) -> list[SentenceSpan]:
    """Gets each start and end index of all sentences in a document"""
    return [(sent.start, sent.end) for sent in doc.sentences]
   
def insert_sentence_boundaries(spans:list[SentenceSpan], tokens:list[str]) -> list[str]:
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

def get_bigrams_with_boundary_syms(doc:Document, tokens:list[str]):
    """Gets the bigrams from given list of tokens, including sentence boundaries"""
    sent_spans = get_sentence_spans(doc)
    tokens_with_boundary_syms = insert_sentence_boundaries(sent_spans, tokens)
    token_bigrams = bigrams(tokens_with_boundary_syms)
    return list(filter(lambda x: x != ("EOS","BOS"), token_bigrams))

def replace_openclass(tokens:list[str], pos:list[str]) -> list[str]:
    """Replaces all open class tokens with corresponding POS tags"""
    OPEN_CLASS = ["ADJ", "ADV", "NOUN", "VERB", "INTJ"]
    tokens = deepcopy(tokens)
    for i in range(len(tokens)):
        if pos[i] in OPEN_CLASS:
            tokens[i] = pos[i]
    return tokens

def add_zero_vocab_counts(vocab:Vocab, counted_doc_features:Counter) -> dict:
    
    """
    Combines vocab and counted_document_features into one dictionary such that
    any feature in vocab counted 0 times in counted_document_features is preserved in the feature vector
    
    :param document_counts: features counted from document
    :returns: counts of every element in vocab with 0 counts preserved
    
    Example:
            >> vocab = ("a", "b", "c", "d")
            
            >> counted_doc_features = Counter({"a":5, "c":2})
            
            >> add_zero_vocab_counts(vocab, counted_doc_features)
            
                '{"a": 5, "b" : 0, "c" : 2, "d" : 0}'
    """
    count_dict = {}
    for feature in vocab:
        if feature in counted_doc_features:
            count = counted_doc_features[feature] 
        else:
            count = 0
        count_dict[feature] = count
    return count_dict

def sum_of_counts(counts:dict) -> int:
    """
    Sums the counts of a count dictionary
    Returns 1 if counts sum to 0
    """
    count_sum = sum(counts.values())
    return count_sum if count_sum > 0 else 1
              
# ~~~ FEATURIZERS ~~~

@dataclass
class Feature:
    """
    This class represents the output of each feature extractor
    
    :param feature_counts: dictionary of counts
    :normalize_by: option to normalize. Defaults to 1
    """
    feature_counts:dict
    normalize_by:int = 1
    
    def counts_to_vector(self) -> np.ndarray:
        """Converts a dictionary of counts into a numpy array"""
        counts = list(self.feature_counts.values())
        return np.array(counts).flatten() / self.normalize_by
    
    
def pos_unigrams(doc:Document) -> Feature:
    
    vocab = utils.load_vocab("vocab/static/pos_unigrams.txt")
    doc_pos_tag_counts = Counter(doc.pos_tags)
    all_pos_tag_counts = add_zero_vocab_counts(vocab, doc_pos_tag_counts)
    
    return Feature(all_pos_tag_counts, normalize_by=len(doc.pos_tags))

def pos_bigrams(doc:Document) -> Feature:
    
    vocab = utils.load_pkl("vocab/non_static/pos_bigrams/pan/pos_bigrams.pkl")
    doc_pos_bigram_counts = Counter(get_bigrams_with_boundary_syms(doc, doc.pos_tags))
    all_pos_bigram_counts = add_zero_vocab_counts(vocab, doc_pos_bigram_counts)
    
    return Feature(all_pos_bigram_counts, normalize_by=sum_of_counts(doc_pos_bigram_counts))

def func_words(doc:Document) -> Feature:
    
    vocab = utils.load_vocab("vocab/static/function_words.txt")
    doc_func_word_counts = Counter([token for token in doc.tokens if token in vocab])
    all_func_word_counts = add_zero_vocab_counts(vocab, doc_func_word_counts)
    
    return Feature(all_func_word_counts, normalize_by=sum_of_counts(doc_func_word_counts))

def punc(doc:Document) -> Feature:
    
    vocab = utils.load_vocab("vocab/static/punc_marks.txt")
    doc_punc_counts = Counter([punc for token in doc.tokens for punc in token if punc in vocab])
    all_punc_counts = add_zero_vocab_counts(vocab, doc_punc_counts)
    
    return Feature(all_punc_counts, normalize_by=sum_of_counts(doc_punc_counts))

def letters(doc:Document) -> Feature:
    
    vocab = utils.load_vocab("vocab/static/letters.txt")
    doc_letter_counts = Counter([letter for token in doc.tokens for letter in token if letter in vocab])
    all_letter_counts = add_zero_vocab_counts(vocab, doc_letter_counts)
    
    return Feature(all_letter_counts, normalize_by=sum_of_counts(doc_letter_counts))

def common_emojis(doc:Document) -> Feature:
    
    vocab = utils.load_vocab("vocab/static/common_emojis.txt")
    extract_emojis = demoji.findall_list(doc.raw_text, desc=False)
    doc_emoji_counts = Counter(filter(lambda x: x in vocab, extract_emojis))
    all_emoji_counts = add_zero_vocab_counts(vocab, doc_emoji_counts)
    
    return Feature(all_emoji_counts, normalize_by=len(doc.tokens))

def embedding_vector(doc:Document) -> Feature:
    """spaCy word2vec document embedding"""
    embedding = {"embedding_vector" : doc.spacy_doc.vector}
    return Feature(embedding)

def document_stats(doc:Document) -> Feature:
    words = doc.words
    doc_statistics = {"short_words" : len([1 for word in words if len(word) < 5]), 
                      "large_words" : len([1 for word in words if len(word) > 4]),
                      "word_len_avg": np.mean([len(word) for word in words]),
                      "word_len_std": np.std([len(word) for word in words]),
                      "sent_len_avg": np.mean([len(sent) for sent in doc.sentences]),
                      "sent_len_std": np.std([len(sent) for sent in doc.sentences]),
                      "hapaxes"     : len(FreqDist(words).hapaxes())}
    return Feature(doc_statistics)

def dep_labels(doc:Document) -> Feature:

    vocab = utils.load_vocab("vocab/static/dep_labels.txt")
    doc_dep_labels = Counter([dep for dep in doc.dep_labels])
    all_dep_labels = add_zero_vocab_counts(vocab, doc_dep_labels)
    
    return Feature(all_dep_labels, normalize_by=sum_of_counts(doc_dep_labels))

def mixed_bigrams(doc:Document) -> Feature:
    
    vocab = utils.load_pkl("vocab/non_static/mixed_bigrams/pan/mixed_bigrams.pkl")
    doc_mixed_bigrams = Counter(bigrams(replace_openclass(doc.tokens, doc.pos_tags)))
    all_mixed_bigrams = add_zero_vocab_counts(vocab, doc_mixed_bigrams)
    
    return Feature(all_mixed_bigrams, sum_of_counts(doc_mixed_bigrams))

# ~~~ Featurizers end ~~~

def read_config(register:tuple, path="config.toml") -> list:
    """
    Reads config.toml to see which features to activate
    :param register: tuple of featurizer functions
    """
    toml_config = toml.load(path)["Features"]
    config = []
    for feature in register:
        try:
            if toml_config[feature.__name__] == 1:
                config.append(feature)
        except KeyError:
            raise KeyError(f"Feature '{feature.__name__}' does not exist in config.toml")
    return config


class DocumentVector:
    """
    This class represents a DocumentVector, which contains each individual 
    feature vector, as well as the concatenated one. 
    """
    def __init__(self, doc:Document):
        self.doc = doc
        self._vector_map : dict[str, np.ndarray] = {} 
    
    @property
    def vector(self) -> np.ndarray:
        """Concatenates all feature vectors into one larger 1D vector"""
        return np.concatenate(list(self._vector_map.values()))
    
    def add_feature(self, feature_name:str, vector:np.ndarray):
        """Adds a feature mapped to that feature's vector to self._vector_map"""
        if feature_name not in self._vector_map:
            self._vector_map[feature_name] = vector
        else:
            raise Exception(f"Feature {feature_name} already in this instance")
        
    def get_vector_map(self) -> dict:
        return self._vector_map()
        
    def get_vector_by_feature(self, feature_name:str) -> np.ndarray:
        """
        Accesses an individual feature vector by name
        :param feature_name: name of feature to get vector from
        :returns: vector for specified feature
        """
        if feature_name in self._features:
            return self._vector_map[feature_name]
        else:
            raise KeyError(f"Feature '{feature_name} not in current configuration: See config.toml'")


class GrammarVectorizer:
    """This class houses all featurizers"""
    
    def __init__(self):
        self.nlp = utils.load_spacy("en_core_web_md")
        self.register = (pos_unigrams,
                         pos_bigrams,
                         func_words,
                         punc,
                         letters,
                         common_emojis,
                         embedding_vector,
                         document_stats,
                         dep_labels,
                         mixed_bigrams)
        
        self.config = read_config(self.register)

    def vectorize(self, text:str, return_vector=True) -> Union[DocumentVector, np.ndarray]:
        """
        Applies featurizers to an input text and returns with either a numpy array
        or DocumentVector object depending on the return_vector flag
        
        :param text: string to be vectorized
        :param return_vector: Defaults to True. Option to return numpy array instead of DocumentVector object
        """
        doc = make_document(text, self.nlp)
        document_vector = DocumentVector(doc)
        for featurizer in self.config:
            
            feature = featurizer(doc)
            feature_vector = feature.counts_to_vector()
            feature_counts = feature.feature_counts
            feature_logger(featurizer.__name__, f"{feature_counts}\n{feature_vector}\n\n") 
        
            try:
                assert not np.isnan(feature_vector).any()
            except AssertionError:
                import ipdb;ipdb.set_trace()
                
            document_vector.add_feature(featurizer.__name__, feature_vector)
        
        if return_vector:
            return document_vector.vector
        else:
            return document_vector