

"""
This file is for code testing
"""


from gram2vec import vectorizer
from gram2vec.verbalizer import Verbalizer


pan22_vectors = vectorizer.from_jsonlines("../data/pan22/preprocessed/")

verb = Verbalizer(pan22_vectors)

print(verb.verbalize_author("en_110"))