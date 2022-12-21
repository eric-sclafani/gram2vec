
import spacy
import utils
import numpy as np
from dataclasses import dataclass
import demoji
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

np.seterr(invalid="ignore")

def make_counter(document, query:list):
    pass



def pos_tags(document) -> np.ndarray:  
    tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
    tag_dict = {tag:0 for tag in tags}
    
    tag_cnt = Counter(document.pos_tags)
    tag_dict.update(tag_cnt)
    counts = list(tag_dict.values())
    
    assert len(counts) == len(tag_dict)
    return np.array(counts) / len(document.pos_tags)


    
    
def func_words(document) -> np.ndarray:  
    # modified NLTK stopwords file
    
    #! make this func generate function_words.txt if not exists
    with open ("resources/function_words.txt", "r") as fin:
        function_words = list(map(lambda x: x.strip("\n"), fin.readlines()))

    func_dict = {word:0 for word in function_words}
    
    func_cnt = Counter([token for token in document.tokens if token in function_words])
    func_dict.update(func_cnt)
    counts = list(func_dict.values())
    
    assert len(counts) == len(func_dict)
    return np.array(counts) / len(document.tokens)


    
def punc(document) -> np.ndarray:
    
    punc_marks = [".", ",", ":", ";", "\'", "\"", "?", "!", "`", "*", "&", "_", "-", "%", ":(", ":)", "...", "..", "(", ")", ":))", "–", "‘", "’", ";)"]
    punc_dict = {punc:0 for punc in punc_marks}
    
    punc_cnt = Counter([token.text for token in document.doc if token.text in punc_marks])
    punc_dict.update(punc_cnt)
    counts = list(punc_dict.values())
    
    assert len(counts) == len(punc_dict)
    return np.array(counts) / len(document.tokens) 


  
def letters(document) -> np.ndarray: 

    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
               "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    letter_dict = {letter:0 for letter in letters}
    doc_letters = [letter for token in document.doc for letter in token.text if letter in letters]
    
    letter_cnt = Counter(doc_letters)
    letter_dict.update(letter_cnt)
    counts = list(letter_dict.values())
    
    assert len(counts) == len(letter_dict)
    return np.array(counts) / len(doc_letters)



def pos_bigrams(document):
    
    # build the vocab. This enforces the ngram_vectorizer to count the same ngrams for all docs
    
    #! make this func create the txt file if it does not exist
    #! look into hashing vectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html
    vocab_vectorizer = CountVectorizer(analyzer="word", ngram_range=(2,2), max_features=50)
    all_tokens = utils.load_txt("resources/all_pos_tags.txt")
    vocab_vectorizer.fit([all_tokens])
    vocab = vocab_vectorizer.vocabulary_
    
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




def parse_config():
    pass


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
        self.featurizers = {
            1 : pos_tags,
            2 : func_words,
            3 : punc,
            4 : letters,
            5 : pos_bigrams,
        }
    
    def vectorize(self, text:str, config:list[int]=None) -> np.ndarray:
        """Applies featurizers to an input text, optionally according to a configuration"""
        
        text_demojified = demoji.replace(text, "") # dep parser hates emojis 
        doc = self.nlp(text_demojified)
        document = Document.from_nlp(doc, text)
        
        vectors = []
        for feat_id, featurizer in self.featurizers.items():
            try:
                if feat_id in config: 
                    vector = featurizer(document)
                    assert not np.isnan(vector).any()
                    vectors.append(vector)
                    
            except TypeError:
                #! apply sanity checking here
                vector = featurizer(document)
                vectors.append(featurizer(document))
                assert not np.isnan(vector).any()
                
        return np.concatenate(vectors)
    
    
# featurizer testing grounds
def main():
    pass

if __name__ == "__main__":
    main()