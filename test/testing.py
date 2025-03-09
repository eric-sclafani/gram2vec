

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
    "sentences":1,
    "named_entities":1,
    "NEs_person":1,
    "NEs_location_loc":1,
    "NEs_location_gpe":1,
    "NEs_organization":1,
    "NEs_date":1
    }

example_sentences = [
    "The quick brown 😂 fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "Peter Piper picked a peck of pickled peppers.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "The early bird catches the worm.",
    "A picture is worth a thousand words.",
    "When in Rome, do as the Romans do.",
    "I like you",
    'It is what it is',
    'This chocolate is the best chocolate I have ever had!🥳',
    "My cat is smaller than my dog..",
    "This city is more beautiful than that city.",
    "Maria lives in Mexico : )",
    "Are you working at Stony Brook University???",
    "Google is a big company; it has many employees.",
    "Yay!!"
]

features = vectorizer.from_documents(example_sentences)
features.to_csv("features.csv", index=False)

# df = vectorizer.from_jsonlines("data/pan22/preprocessed/", config=config)
# test_vector = df.select_dtypes(include=np.number).iloc[111]

# verbalizer = Verbalizer(df)
# a = verbalizer.verbalize_author_id("en_112")

# d = verbalizer.verbalize_document_vector(test_vector)


# import ipdb;ipdb.set_trace()