
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import demoji
from nltk import bigrams
import numpy as np 
import os
import spacy
import toml

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
    
# ~~~ Static vocabularies ~~~
pos_unigram_vocab    :Vocab = utils.load_vocab("vocab/static/pos_unigrams.txt")
function_words_vocab :Vocab = utils.load_vocab("vocab/static/function_words.txt")
dep_labels_vocab     :Vocab = utils.load_vocab("vocab/static/dep_labels.txt")
punc_marks_vocab     :Vocab = utils.load_vocab("vocab/static/punc_marks.txt")
letters_vocab        :Vocab = utils.load_vocab("vocab/static/letters.txt")
common_emojis_vocab  :Vocab = utils.load_vocab("vocab/static/common_emojis.txt")

# ~~~ Non-static vocabularies ~~~
#NOTE: the path needs to change be manually changed to match appropriate dataset
pos_bigrams_vocab   :Vocab = utils.load_pkl("vocab/non_static/pos_bigrams/pan/pos_bigrams.pkl")
mixed_bigrams_vocab :Vocab = utils.load_pkl("vocab/non_static/mixed_bigrams/pan/mixed_bigrams.pkl")


# ~~~ Document and CountBasedFeaturizer representations
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

class CountBasedFeaturizer:
    """
    Class for representing frequency-based feature extractors
    
    :param name: name of featurizer. Has to be the same name as in config.toml
    :param vocab: tuple of elements to look for in a document
    :param counter: function used to count elements
    """
    def __init__(self, name:str, vocab:tuple[str], counter):
        self.name = name
        self.vocab = vocab
        self.counter = counter
        
    def __repr__(self):
        return self.name
        
    def _add_zero_vocab_counts(self, counted_doc_features:Counter) -> dict:
        """
        Combines self.vocab and counted_document_features into one dictionary such that
        any feature in vocab counted 0 times in counted_document_features is preserved in the feature vector. 
        
        :param document_counts: features counted from document
        :returns: counts of every element in vocab with 0 counts preserved
        
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
        
    def get_all_feature_counts(self, document:Document) -> dict[str,int]:
        """
        Applies counter function to get document feature counts and 
        combines the result with 0 count vocab entries
        
        :param document: document to extract counts from
        :returns: dictionary of counts
        """
        counted_doc_features = self.counter(document)
        return self._add_zero_vocab_counts(counted_doc_features)

    def vectorize(self, document:Document) -> np.ndarray:
        """Converts the feature counts into a numpy array"""
        counts = self.get_all_feature_counts(document).values()
        return np.array(counts)
    
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
        del counter[("EOS","BOS")] # removes artificial bigram
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

def count_mixed_bigrams(doc:Document) -> Counter:
    return Counter(bigrams(replace_openclass(doc.tokens, doc.pos_tags)))

# ~~~ Featurizers ~~~

pos_unigrams = CountBasedFeaturizer(
    name="pos_unigrams",
    vocab=pos_unigram_vocab,
    counter=count_pos_unigrams
)

pos_bigrams = CountBasedFeaturizer(
    name="pos_bigrams",
    vocab=pos_bigrams_vocab,
    counter=count_pos_bigrams
)

func_words = CountBasedFeaturizer(
    name="func_words",
    vocab=function_words_vocab,
    counter=count_func_words
)

punc = CountBasedFeaturizer(
    name="punc",
    vocab=punc_marks_vocab,
    counter=count_punctuation
)

letters = CountBasedFeaturizer(
    name="letters",
    vocab=letters_vocab,
    counter=count_letters
)

common_emojis = CountBasedFeaturizer(
    name="common_emojis",
    vocab=common_emojis_vocab,
    counter=count_emojis
)

dep_labels = CountBasedFeaturizer(
    name="dep_labels",
    vocab=dep_labels_vocab,
    counter=count_dep_labels
)

mixed_bigrams = CountBasedFeaturizer(
    name = "mixed_bigrams",
    vocab = mixed_bigrams_vocab,
    counter=count_mixed_bigrams
    )

# ~~~ Featurizers end ~~~

def read_config(register:tuple[CountBasedFeaturizer], path="config.toml") -> list[CountBasedFeaturizer]:
    """
    Reads config.toml to see which features to activate
    :param register: tuple of featurizers 
    """
    toml_config = toml.load(path)["Features"]
    config = []
    for feature in register:
        try:
            if toml_config[feature.name] == 1:
                config.append(feature)
        except KeyError:
            raise KeyError(f"Feature '{feature}' does not exist in config.toml")
    return config


class DocumentVector:
    """
    This class represents a DocumentVector, which contains each individual 
    feature vector, as well as the concatenated one. 
    """
    def __init__(self, doc:Document):
        self.doc = doc
        self._vector_map :dict[CountBasedFeaturizer, np.ndarray] = {} 
        
    def vector(self) -> np.ndarray:
        """Concatenates all feature vectors into one larger 1D vector"""
        vectors = [vector for vector in self._vector_map.values()]
        return np.concatenate(vectors)
    
    def add_feature(self, feature:CountBasedFeaturizer, vector:np.ndarray):
        """Adds a feature mapped to that feature's vector to self._vector_map"""
        if feature.name not in self._vector_map:
            self._vector_map[feature.name] = vector
        else:
            raise Exception(f"Feature {feature} already in this instance")
        
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
    
    def __init__(self, logging=False):
        self.nlp = utils.load_spacy("en_core_web_md")
        self.logging = logging
        self.register = (pos_unigrams,
                         pos_bigrams,
                         func_words,
                         punc,
                         letters,
                         common_emojis,
                         dep_labels,
                         mixed_bigrams)
        
        self.config = read_config(self.register)

    def apply_features(self, text:str) -> DocumentVector:
        """Applies featurizers to an input text and returns a DocumentVector"""
        
        doc = make_document(text, self.nlp)
        document_vector = DocumentVector(doc)
        for feature in self.config:
            
            feature_counts = feature.get_all_feature_counts(doc)
            feature_vector = feature.vectorize(doc)
            
            if self.logging:
                feature_logger(feature.name, f"{feature_counts}\n{feature_vector}\n\n")
            
            assert not np.isnan(feature_vector).any()
            document_vector.add_feature(feature, feature_vector)
            
        return document_vector
   
                   
# write a function for apply grammarvectorizer to list of documents            
                    
    
    
# debugging
if __name__ == "__main__":
    pass