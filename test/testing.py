

"""
This file is for code testing
"""


from gram2vec.verbalizer import Verbalizer


verb = Verbalizer("../data/pan22/preprocessed/")

print(verb.verbalize_author("en_110"))