
import spacy
import toml
import utils
import numpy as np
import os
from dataclasses import dataclass
import demoji
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

np.seterr(invalid="ignore")

# ~~~ Helper functions ~~~


def get_counts(features:list, doc_to_count:list) -> list[int]:
    """
    Counts the occurrences of 'features' in 'doc_to_count'
    Merges two dictionaries so that counts of 0 are preserved
    
    """
    count_dict = {feat:0 for feat in features}
    counter = Counter(doc_to_count)
    count_dict.update(counter)
    
    return list(count_dict.values())

def generate_pos_vocab(path):
    
    data = utils.load_json(path)
    nlp = utils.load_spacy("en_core_web_md")
    
    vocab_vectorizer = CountVectorizer(analyzer="word", ngram_range=(2,2), max_features=50)
    all_tokens = utils.load_txt("resources/all_pos_tags.txt")
    vocab_vectorizer.fit([all_tokens])
    vocab = vocab_vectorizer.vocabulary_
    utils.save_pkl(vocab, "resources/pan_pos_vocab.pkl")
    
    
    
    
def generate_funcwords_vocab():
    pass
        

# ~~~ Featurizers ~~~

def pos_unigrams(document) -> np.ndarray:  
    
    tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
    counts = get_counts(tags, document.pos_tags)
    return np.array(counts) / len(document.pos_tags)


def func_words(document) -> np.ndarray:  
    
    # modified NLTK stopwords file #! make this generate automatically
    with open ("resources/function_words.txt", "r") as fin:
        function_words = list(map(lambda x: x.strip("\n"), fin.readlines()))

    doc_func_words = [token for token in document.tokens if token in function_words]
    counts = get_counts(function_words, doc_func_words)
    
    return np.array(counts) / len(document.tokens)


def punc(document) -> np.ndarray:
    
    punc_marks = [".", ",", ":", ";", "\'", "\"", "?", "!", "`", "*", "&", "_", "-", "%", ":(", ":)", "...", "..", "(", ")", ":))", "–", "‘", "’", ";)"]
    doc_punc_marks = [token.text for token in document.doc if token.text in punc_marks]
    counts = get_counts(punc_marks, doc_punc_marks)
    
    return np.array(counts) / len(document.tokens) 


def letters(document) -> np.ndarray: 

    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
               "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    doc_letters = [letter for token in document.doc 
                   for letter in token.text if letter in letters]
    
    counts = get_counts(letters, doc_letters)
    
    return np.array(counts) / len(doc_letters)


def pos_bigrams(document):
    
    # build the vocab. This enforces the ngram_vectorizer to count the same ngrams for all docs
    
    if not os.path.exists("resources/pan_pos_vocab.pkl"):
        vocab = generate_pos_vocab("data/pan/preprocessed/fixed_sorted_authors.json")
    else:
        vocab = utils.load_pkl("resources/pan_pos_vocab.pkl")
        
    ngram_vectorizer = CountVectorizer(analyzer="word", ngram_range=(2,2), max_features=50, vocabulary=vocab)
    pos_ngrams = ngram_vectorizer.fit_transform([" ".join(document.pos_tags)])

    return (pos_ngrams.toarray().flatten()) / len(document.pos_tags) # normalize by # of pos tags in current document
    


# incomplete
def mixed_ngrams(n, document):
    
    ngrams = lambda n, items: [tuple(items[i:i+n]) for i in range(len(items)-1)]
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
    
    def __init__(self):
        self.nlp = utils.load_spacy("en_core_web_md")
        
        self.featurizers = [
            pos_unigrams,
            pos_bigrams,
            func_words, 
            punc,
            letters,
        ]
        
    def config(self):
        
        toml_config = toml.load("config.toml")["Featurizers"]
        config = []
        for feat in self.featurizers:
            try:
                if toml_config[feat.__name__] == 1:
                    config.append(feat)
            except KeyError:
                raise KeyError(f"Feature '{feat.__name__}' does not exist in config.toml")
        return config
        
    
    def vectorize(self, text:str) -> np.ndarray:
        """Applies featurizers to an input text. Returns a 1-D array."""
        
        text_demojified = demoji.replace(text, "") # dep parser hates emojis 
        doc = self.nlp(text_demojified)
        document = Document.from_nlp(doc, text)
        
        vectors = []
        for feat in self.featurizers:
            if feat in self.config():
                vector = feat(document)
                assert not np.isnan(vector).any() 
                vectors.append(vector)
                    
        return np.concatenate(vectors)
    
    
    
    

# testing code 
def main():
    
    g2v = GrammarVectorizer()
    g2v.vectorize("this is a string. I am happy that this is a string")
    
       


if __name__ == "__main__":
    main()
    