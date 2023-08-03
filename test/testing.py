

"""
This file is for code testing
"""


from gram2vec.verbalizer import Verbalizer
from gram2vec import vectorizer
import numpy as np

config = {
    "pos_unigrams":1,
    "pos_bigrams":1,
    "func_words":1,
    "punctuation":1,
    "letters":1,
    "emojis":1,
    "dep_labels":1,
    "morph_tags":1,
    "sentences":1
    }

df = vectorizer.from_jsonlines("../data/pan22/preprocessed/", config=config)
test_vector = df.select_dtypes(include=np.number).iloc[-1]

verbalizer = Verbalizer(df)

a = verbalizer.verbalize_author_id("en_112")

d = verbalizer.verbalize_document_vector(test_vector)


print(d)