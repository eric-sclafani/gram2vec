"""
This file is for code testing
"""

from multiprocessing import freeze_support
from gram2vec.verbalizer import Verbalizer
from gram2vec import vectorizer
from gram2vec.feature_locator import find_feature_spans
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
    "When in Rome, do as the Romans do."
]

def main():
    # Extract features from example sentences
    features = vectorizer.from_documents(example_sentences)
    features.to_csv("features.csv", index=False)

    # Find all occurrences of a specific feature in a text
    example_text = "The quick brown 😂 fox jumps over the lazy dog."
    # example_text = "<PERSON> passed away at <DATE_TIME> on <DATE_TIME>, in her home in <LOCATION>. <PERSON> was a doctor, who specialized in pediatric care."
    print(f"\nAnalyzing text: \"{example_text}\"")

    # Find all nouns
    noun_spans = find_feature_spans(example_text, "pos_unigrams:NOUN")
    print("\nNouns found:")
    for span in noun_spans:
        print(f"  - '{span.text}' at positions {span.start_char}:{span.end_char}")

    # Find all determiners
    det_spans = find_feature_spans(example_text, "pos_unigrams:DET")
    print("\nDeterminers found:")
    for span in det_spans:
        print(f"  - '{span.text}' at positions {span.start_char}:{span.end_char}")

    # Find all function words
    the_spans = find_feature_spans(example_text, "func_words:the")
    print("\nFunction word 'the' found:")
    for span in the_spans:
        print(f"  - '{span.text}' at positions {span.start_char}:{span.end_char}")

    # Find emojis
    emoji_spans = find_feature_spans(example_text, "emojis:😂")
    print("\nEmojis found:")
    for span in emoji_spans:
        print(f"  - '{span.text}' at positions {span.start_char}:{span.end_char}")

    # Find punctuation
    punct_spans = find_feature_spans(example_text, "punctuation:.")
    print("\nPunctuation '.' found:")
    for span in punct_spans:
        print(f"  - '{span.text}' at positions {span.start_char}:{span.end_char}")

    # Find noun->verb sequences
    spans = find_feature_spans(example_text, "pos_bigrams:NOUN VERB")
    print("\nNoun->verb sequences found:")
    for span in spans:
        print(f"  - '{span.text}' at positions {span.start_char}:{span.end_char}")

    # Find sentences starting with adjectives
    spans = find_feature_spans(example_text, "pos_bigrams:BOS ADJ")
    print("\nSentences starting with adjectives found:")
    for span in spans:
        print(f"  - '{span.text}' at positions {span.start_char}:{span.end_char}")

    # Find sentences ending with punctuation
    spans = find_feature_spans(example_text, "pos_bigrams:PUNCT EOS")
    print("\nSentences ending with punctuation found:")
    for span in spans:
        print(f"  - '{span.text}' at positions {span.start_char}:{span.end_char}")

if __name__ == '__main__':
    freeze_support()  # Required for Windows
    main()

# df = vectorizer.from_jsonlines("data/pan22/preprocessed/", config=config)
# test_vector = df.select_dtypes(include=np.number).iloc[111]

# verbalizer = Verbalizer(df)
# a = verbalizer.verbalize_author_id("en_112")

# d = verbalizer.verbalize_document_vector(test_vector)


# import ipdb;ipdb.set_trace()