

"""
This file is for code testing
"""


from gram2vec.verbalizer import Verbalizer
from gram2vec import vectorizer
import numpy as np

df = vectorizer.from_jsonlines("../data/pan22/preprocessed/")
test_vector = df.select_dtypes(include=np.number).iloc[-1]

verb = Verbalizer("../data/pan22/preprocessed/")

# a = verb.verbalize_author("en_110")


# feed in vector
d = verb.verbalize_document(test_vector)

print(d)