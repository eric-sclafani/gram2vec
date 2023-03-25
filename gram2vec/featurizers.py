
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import demoji
from nltk import bigrams
from nltk import FreqDist
import numpy as np 
import os
import spacy
from typing import Union, Optional
import pickle

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Logging, type aliases ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SentenceSpan = tuple[int,int]
Vocab = tuple[str]

def feature_logger(filename, writable):
    """Custom logging function. Was having issues with standard logging library"""
    if not os.path.exists("logs"):
        os.mkdir("logs")
               
    with open(f"logs/{filename}.log", "a") as fout: 
        fout.write(writable)
        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Helpers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@dataclass
class Document:
    """
    This class represents elements from a spaCy Doc object
        :param text: text before being processed by spaCy
        :param spacy_doc: spaCy's document object
        :param tokens: list of tokens
        :param words: list of words (no punc tokens)
        :param pos_tags: list of pos tags
        :param dep_labels: list of dependency parse labels
        :param sentences: list of spaCy-sentencized sentences
    Note: instances should only be created using the 'make_document' function 
"""
    text       :str
    spacy_doc  :spacy.tokens.doc.Doc
    tokens     :list[str]
    words      :list[str]
    pos_tags   :list[str]
    dep_labels :list[str]
    sentences  :list[spacy.tokens.span.Span]
    
def make_document(text:str, nlp) -> Document:
    """Converts raw text into a Document object"""
    text   = deepcopy(text)
    spacy_doc  = nlp(demojify_text(text)) # dep parser hates emojis
    tokens     = [token.text for token in spacy_doc]
    words      = [token.text for token in spacy_doc if not token.is_punct]
    pos_tags   = [token.pos_ for token in spacy_doc]
    dep_labels = [token.dep_ for token in spacy_doc]
    sentences  = list(spacy_doc.sents)
    return Document(text, spacy_doc, tokens, words, pos_tags, dep_labels, sentences)

def load_vocab(path:str, type="static") -> tuple[str]:
    """Loads in a vocabulary file as a tuple of strings. This vocabulary file"""
    
    if type == "static":
        assert path.endswith(".txt")
        with open (path, "r") as fin:
            return tuple(map(lambda x: x.strip("\n"), fin.readlines()))
        
    elif type == "non_static":
        assert path.endswith(".pkl")
        with open (path, "rb") as fin:
            return pickle.load(fin)


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


def remove_openclass_bigrams(tokens:list[str], OPEN_CLASS:list[str]) -> list[str]:
    """Removes (OPEN_CLASS, OPEN_CLASS) bigrams that inadvertently get created in replace_openclass """
    filtered = []
    for pair in bigrams(tokens):
        if pair[0] not in OPEN_CLASS and pair[1] in OPEN_CLASS:
            filtered.append(pair[0])
            
        if pair[0] in OPEN_CLASS and pair[1] not in OPEN_CLASS:
            filtered.append(pair[0])
    return filtered
    
def replace_openclass(tokens:list[str], pos:list[str]) -> list[str]:
    """Replaces all open class tokens with corresponding POS tags"""
    OPEN_CLASS = ["ADJ", "ADV", "NOUN", "VERB", "INTJ"]
    tokens_replaced = deepcopy(tokens)
    for i in range(len(tokens_replaced)):
        if pos[i] in OPEN_CLASS:
            tokens_replaced[i] = pos[i]

    return remove_openclass_bigrams(tokens_replaced, OPEN_CLASS)

def add_zero_vocab_counts(vocab:Vocab, counted_doc_features:Counter) -> dict:
    
    """
    Combines vocab and counted_doc_features into one dictionary such that
    any feature in vocab counted 0 times in counted_doc_features is preserved in the feature vector
        Example:
            >> vocab = ("a", "b", "c", "d")
            
            >> counted_doc_features = Counter({"a":5, "c":2})
            
            >> add_zero_vocab_counts(vocab, counted_doc_features)
            
                '{"a": 5, "b" : 0, "c" : 2, "d" : 0}'
    
    :param vocab: vocabulary of elements to look for in documentd
    :param counted_doc_features: features counted from document
    :returns: counts of every element in vocab with 0 counts preserved
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
    """Sums the counts of a count dictionary. Returns 1 if counts sum to 0"""
    count_sum = sum(counts.values())
    return count_sum if count_sum > 0 else 1
              
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FEATURIZERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@dataclass
class Feature:
    """
    This class represents the output of each feature extractor
    
    :param feature_counts: dictionary of counts
    :param normalize_by: option to normalize. Defaults to 1
    """
    feature_counts:dict
    normalize_by:int = 1
    
    def counts_to_vector(self) -> np.ndarray:
        """Converts a dictionary of counts into a numpy array"""
        counts = list(self.feature_counts.values())
        return np.array(counts).flatten() / self.normalize_by
    
    
def pos_unigrams(doc:Document) -> Feature:
    
    vocab = load_vocab("vocab/static/pos_unigrams.txt")
    doc_pos_tag_counts = Counter(doc.pos_tags)
    all_pos_tag_counts = add_zero_vocab_counts(vocab, doc_pos_tag_counts)
    
    return Feature(all_pos_tag_counts, len(doc.pos_tags))

def pos_bigrams(doc:Document) -> Feature:
    
    vocab = load_vocab("vocab/non_static/pan/pos_bigrams/pos_bigrams.pkl", type="non_static")
    doc_pos_bigram_counts = Counter(get_bigrams_with_boundary_syms(doc, doc.pos_tags))
    all_pos_bigram_counts = add_zero_vocab_counts(vocab, doc_pos_bigram_counts)
    
    return Feature(all_pos_bigram_counts, sum_of_counts(doc_pos_bigram_counts))

def func_words(doc:Document) -> Feature:
    
    vocab = load_vocab("vocab/static/function_words.txt")
    doc_func_word_counts = Counter([token for token in doc.tokens if token in vocab])
    all_func_word_counts = add_zero_vocab_counts(vocab, doc_func_word_counts)
    
    return Feature(all_func_word_counts, sum_of_counts(doc_func_word_counts))

def punc(doc:Document) -> Feature:
    
    vocab = load_vocab("vocab/static/punc_marks.txt")
    doc_punc_counts = Counter([punc for token in doc.tokens for punc in token if punc in vocab])
    all_punc_counts = add_zero_vocab_counts(vocab, doc_punc_counts)
    
    return Feature(all_punc_counts, sum_of_counts(doc_punc_counts))

def letters(doc:Document) -> Feature:
    
    vocab = load_vocab("vocab/static/letters.txt")
    doc_letter_counts = Counter([letter for token in doc.tokens for letter in token if letter in vocab])
    all_letter_counts = add_zero_vocab_counts(vocab, doc_letter_counts)
    
    return Feature(all_letter_counts, sum_of_counts(doc_letter_counts))

def common_emojis(doc:Document) -> Feature:
    
    vocab = load_vocab("vocab/static/common_emojis.txt")
    extract_emojis = demoji.findall_list(doc.text, desc=False)
    doc_emoji_counts = Counter(filter(lambda x: x in vocab, extract_emojis))
    all_emoji_counts = add_zero_vocab_counts(vocab, doc_emoji_counts)
    
    return Feature(all_emoji_counts, len(doc.tokens))

def embedding_vector(doc:Document) -> Feature:
    """spaCy word2vec (or glove?) document embedding"""
    embedding = {"embedding_vector" : doc.spacy_doc.vector}
    return Feature(embedding)

def document_stats(doc:Document) -> Feature:
    words = doc.words
    doc_statistics = {"short_words" : len([1 for word in words if len(word) < 5])/len(words), 
                      "large_words" : len([1 for word in words if len(word) > 4])/len(words),
                      "word_len_avg": np.mean([len(word) for word in words]),
                      "word_len_std": np.std([len(word) for word in words]),
                      "sent_len_avg": np.mean([len(sent) for sent in doc.sentences]),
                      "sent_len_std": np.std([len(sent) for sent in doc.sentences]),
                      "hapaxes"     : len(FreqDist(words).hapaxes())/len(words)}
    return Feature(doc_statistics)

def dep_labels(doc:Document) -> Feature:

    vocab = load_vocab("vocab/static/dep_labels.txt")
    doc_dep_labels = Counter([dep for dep in doc.dep_labels])
    all_dep_labels = add_zero_vocab_counts(vocab, doc_dep_labels)
    
    return Feature(all_dep_labels, sum_of_counts(doc_dep_labels))

def mixed_bigrams(doc:Document) -> Feature:
    
    vocab = load_vocab("vocab/non_static/pan/mixed_bigrams/mixed_bigrams.pkl", type="non_static")
    doc_mixed_bigrams = Counter(bigrams(replace_openclass(doc.tokens, doc.pos_tags)))
    all_mixed_bigrams = add_zero_vocab_counts(vocab, doc_mixed_bigrams)
    
    return Feature(all_mixed_bigrams, sum_of_counts(doc_mixed_bigrams))





# ~~~ FEATURIZERS END ~~~

DEFAULT_CONFIG = {
    "pos_unigrams":1,
    "pos_bigrams":1,
    "func_words":1,
    "punc":1,
    "letters":1,
    "common_emojis":1,
    "embedding_vector":0,
    "document_stats":1,
    "dep_labels":1,
    "mixed_bigrams":1,
}  

class FeatureVector:
    """
    This class provides access to the large concatenated feature vector,
    as well as each individual smaller one. Additionally, provides access to
    the exact counts each featurizer extracted from the given document
    """
    def __init__(self, doc:Document):
        self.doc = doc
        self.vector_map : dict[str, np.ndarray] = {} 
        self.count_map  : dict[str, dict] = {} 
    
    @property
    def vector(self) -> np.ndarray:
        """
        Concatenates all individual feature vectors into one
        :returns: 1D numpy array
        """
        return np.concatenate(list(self.vector_map.values()))
    
    def get_vector_by_feature(self, feature_name:str) -> np.ndarray:
        """
        Retrieves an individual feature vector by name from self.vector_map
        :param feature_name: name of the featurizer
        :returns: the vector created by the 'feature_name' featurizer 
        :raises KeyError: if feature_name is not in current configuration
        """
        if feature_name in self.vector_map:
            return self.vector_map[feature_name]
        else:
            raise KeyError(f"Feature '{feature_name}' not in current configuration")
        
    def get_counts_by_feature(self, feature_name:str) -> dict[str, int]:
        """
        Retrieves an individual feature count dict by name from self.count_map
        :param feature_name: name of the featurizer
        :returns: the count dict created by the 'feature_name' featurizer 
        :raises KeyError: if feature_name is not in current configuration
        """
        if feature_name in self.count_map:
            return self.count_map[feature_name]
        else:
            raise KeyError(f"Feature '{feature_name} not in current configuration: See config.toml'")
    
    def _update_vector_map(self, feature_name:str, vector:np.ndarray):
        """Adds a feature mapped to that feature's vector to self.vector_map"""
        if feature_name not in self.vector_map:
            self.vector_map[feature_name] = vector
        else:
            raise Exception(f"Feature {feature_name} already in this instance")
        
    def _update_count_map(self, feature_name:str, counts:dict[str, int]):
        """Adds a feature mapped to that feature's count dict to self.count_map"""
        if feature_name not in self.count_map:
            self.count_map[feature_name] = counts
        else:
            raise Exception(f"Feature {feature_name} already in this instance")
        
class GrammarVectorizer:
    """
    Houses all featurizers to apply to a text document.
    
    Initializes the spaCy nlp object and activated featurizers with each instance
    """
    
    def __init__(self, config=None):
        self.nlp = spacy.load("en_core_web_md", disable=["ner", "lemmatizer"])
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
        
        self._config = self._process_config(config)
        os.system("./clear_logs.sh")
        
    def get_config(self) -> list[str]:
        """Retrieves the names of all activated features"""
        return [feat.__name__ for feat in self._config]
        
    def _process_config(self, passed_config: Optional[dict]) -> list:
        """
        Reads which features to activate and returns a list of featurizer functions
        :param passed_config: User provided configuration dictionary. Can be None.
        :returns: list of activated featurizers
        """
        current_config = DEFAULT_CONFIG if not passed_config else passed_config
        activated_feats = []
        for feat in self.register:
            try:
                if current_config[feat.__name__] == 1:
                    activated_feats.append(feat)
            except KeyError:
                raise KeyError(f"Feature '{feat.__name__}' does not exist in given configuration")
        return activated_feats

    def vectorize_document(self, document:str, return_obj=False) -> Union[np.ndarray, FeatureVector]:
        """
        Applies featurizers to a document and returns either a numpy array
        or FeatureVector object depending on the return_obj flag
        
        :param document: string to be vectorized
        :param return_obj: Defaults to False. Option to return FeatureVector object instead of a numpy array 
        :returns: a 1-D feature vector or FeatureVector object
        """
        doc = make_document(document, self.nlp)
        feature_vector = FeatureVector(doc)
        for featurizer in self._config:
            
            feature = featurizer(doc)
            counts = feature.feature_counts
            vector = feature.counts_to_vector()

            feature_vector._update_vector_map(featurizer.__name__, vector)
            feature_vector._update_count_map(featurizer.__name__, counts)
            feature_logger(featurizer.__name__, f"{counts}\n{vector}\n\n") 

        if not return_obj:
            return feature_vector.vector
        else:
            return feature_vector
        
    def vectorize_episode(self, documents:list[str], return_obj=False) -> Union[np.ndarray, list[FeatureVector]]:
        """
        Applies featurizers to a list of documents and returns either a numpy matrix
        or FeatureVector object depending on the return_obj flag
        
        :param documents: list of documents to be vectorized
        :param return_obj: Defaults to False. Option to return FeatureVector object instead of a numpy matrix
        :returns: a 2-D matrix of feature vectors or list of FeatureVector objects
        """
        all_vectors = []
        for document in documents:
            grammar_vector = self.vectorize_document(document, return_obj)
            all_vectors.append(grammar_vector)
            
        if not return_obj:
            return np.stack(all_vectors)
        else:
            return all_vectors