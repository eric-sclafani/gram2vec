from gram2vec.featurizers import GrammarVectorizer, docify, get_counts
from pytest import approx
import numpy as np

g2v = GrammarVectorizer()
nlp = g2v.nlp

    


def test_pos_unigrams():
    tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
    document = docify("The dog ate the food", nlp)
    pos_unigrams = g2v.featurizers["pos_unigrams"](document)
    
    counts = get_counts(tags, document.pos_tags)
    print(dict(zip(tags, counts)))
    
    
    