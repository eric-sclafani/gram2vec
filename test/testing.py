

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

verb = Verbalizer(df)

a = verb.verbalize_author("en_112")
print(a)
# feed in vector
d = verb.verbalize_document(test_vector)

print(d)