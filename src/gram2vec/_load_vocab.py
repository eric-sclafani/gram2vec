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

    def add_from_path(self, vocab_name: str, language: str = None) -> None:
        """
        Adds vocab from a newline delimited txt file given a vocab name (the file must already exist)
        If language is specified, will first try to load the language-specific file (e.g., russian_dep_labels.txt),
        and fall back to the default file if not found.
        """
        base_path = self._get_user_vocab_path()
        
        # print("language: ", language)
        # Try language-specific file first if language is specified
        if language and language != "en":
            lang_path = f"{base_path}/{language}_{vocab_name}.txt"
            if os.path.exists(lang_path):
                items = self._load_from_txt(lang_path)
                self._vocabs[vocab_name] = items
                return
        
        # Fall back to default file
        default_path = f"{base_path}/{vocab_name}.txt"
        # print("default_path: ", default_path)
        if not os.path.exists(default_path):
            raise FileNotFoundError(f"Vocab path '{default_path}' does not exist")

        items = self._load_from_txt(default_path)
        self._vocabs[vocab_name] = items

    def add_items(self, vocab_name: str, items: Tuple[str]) -> None:
        """Adds vocab directly given a vocab name and tuple of items to count"""
        self._vocabs[vocab_name] = items


vocab = Vocab()

# Get the language from environment variables, default to English
language = os.environ.get("LANGUAGE", "en")

# ~~~ Path loaded vocabs ~~~
vocab.add_from_path("pos_unigrams", language)
vocab.add_from_path("pos_bigrams", language)
vocab.add_from_path("dep_labels", language)
vocab.add_from_path("emojis", language)
vocab.add_from_path("func_words", language)
vocab.add_from_path("morph_tags", language)
vocab.add_from_path("punctuation", language)
vocab.add_from_path("letters", language)

# ~~~ Non-path loaded vocabs ~~~

if language == "ru":
    sentence_matcher = matcher.SyntaxRegexMatcher(language="ru")
elif language == "en":
    sentence_matcher = matcher.SyntaxRegexMatcher(language="en")

vocab.add_items("sentences", tuple(sentence_matcher.patterns.keys()))