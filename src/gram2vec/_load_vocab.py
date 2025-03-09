import os
from typing import Tuple
from . import matcher


class Vocab:
    """
    This class encapsulates feature vocabularies and allows for different ways to add vocab
    """

    def __init__(self):
        self._vocabs = {}

    def _load_from_txt(self, path: str) -> Tuple[str]:
        """Loads a .txt file delimited by newlines"""
        with open(path, "r", encoding="utf-8") as fin:
            return [line.strip("\n") for line in fin.readlines()]

    def _get_user_vocab_path(self):
        """Gets the user's path to the vocabulary files"""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        vocab_path = os.path.join(dir_path, "vocab/")
        return vocab_path

    def get(self, vocab_name: str) -> Tuple[str]:
        """Fetches the desired vocab from the registered vocabularies"""
        if vocab_name in self._vocabs:
            return self._vocabs[vocab_name]
        else:
            raise KeyError(f"Requested vocab '{vocab_name}' not found")

    def add_from_path(self, vocab_name: str) -> None:
        """Adds vocab from a newline delimited txt file given a vocab name (the file must already exist)"""
        path = f"{self._get_user_vocab_path()}/{vocab_name}.txt"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocab path '{path}' does not exist")

        items = self._load_from_txt(path)


        self._vocabs[vocab_name] = items

    def add_items(self, vocab_name: str, items: Tuple[str]) -> None:
        """Adds vocab directly given a vocab name and tuple of items to count"""
        self._vocabs[vocab_name] = items


vocab = Vocab()

# ~~~ Path loaded vocabs ~~~
vocab.add_from_path("pos_unigrams")
vocab.add_from_path("pos_bigrams")
vocab.add_from_path("dep_labels")
vocab.add_from_path("emojis")
vocab.add_from_path("func_words")
vocab.add_from_path("morph_tags")
vocab.add_from_path("punctuation")
vocab.add_from_path("letters")
vocab.add_from_path("named_entities")
vocab.add_from_path("token_tags")

# ~~~ Non-path loaded vocabs ~~~

language = os.environ.get("LANGUAGE", "en")
if language == "ru":
    matcher = matcher.SyntaxRegexMatcher(language="ru")
elif language == "en":
    matcher = matcher.SyntaxRegexMatcher(language="en")

vocab.add_items("sentences", tuple(matcher.patterns.keys()))

# Example of filtering logic
def filter_entities(vocab_name, label):
    return [item for item in vocab.get(vocab_name) if item == label]

def filter_entities_excluding(vocab_name, exclude_label):
    return [item for item in vocab.get(vocab_name) if item != exclude_label]

# Register filtered vocabularies
vocab.add_items("NEs_person", tuple(filter_entities("named_entities", "PERSON")))
vocab.add_items("NEs_location_loc", tuple(filter_entities("named_entities", "LOC")))
vocab.add_items("NEs_location_gpe", tuple(filter_entities("named_entities", "GPE")))
vocab.add_items("NEs_organization", tuple(filter_entities("named_entities", "ORG")))
vocab.add_items("NEs_date", tuple(filter_entities("named_entities", "DATE")))
vocab.add_items("NEs_except_date", tuple(filter_entities_excluding("named_entities", "DATE")))
vocab.add_items("punct_periods", tuple(filter_entities("punctuation", ".")))
vocab.add_items("punct_commas", tuple(filter_entities("punctuation", ",")))
vocab.add_items("punct_colons", tuple(filter_entities("punctuation", ":")))
vocab.add_items("punct_semicolons", tuple(filter_entities("punctuation", ";")))
vocab.add_items("punct_exclamations", tuple(filter_entities("punctuation", "!")))
vocab.add_items("punct_questions", tuple(filter_entities("punctuation", "?")))
vocab.add_items("pos_verbs", tuple(filter_entities("pos_unigrams", "VERB")))
vocab.add_items("pos_adjectives", tuple(filter_entities("pos_unigrams", "ADJ")))
vocab.add_items("pos_adverbs", tuple(filter_entities("pos_unigrams", "ADV")))
vocab.add_items("pos_proper_nouns", tuple(filter_entities("pos_unigrams", "PROPN")))
vocab.add_items("pos_adpositions", tuple(filter_entities("pos_unigrams", "ADP")))
vocab.add_items("token_VB", tuple(filter_entities("token_tags", "VB")))
vocab.add_items("token_VBD", tuple(filter_entities("token_tags", "VBD")))
vocab.add_items("token_VBG", tuple(filter_entities("token_tags", "VBG")))
vocab.add_items("token_VBN", tuple(filter_entities("token_tags", "VBN")))
vocab.add_items("token_VBP", tuple(filter_entities("token_tags", "VBP")))
vocab.add_items("token_VBZ", tuple(filter_entities("token_tags", "VBZ")))
vocab.add_items("token_EX", tuple(filter_entities("token_tags", "EX")))
vocab.add_items("token_FW", tuple(filter_entities("token_tags", "FW")))
vocab.add_items("token_PRP", tuple(filter_entities("token_tags", "PRP")))


