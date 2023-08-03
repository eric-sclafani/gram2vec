import os
from typing import Tuple


class Vocab:
    """
    This class encapsulates feature vocabularies and allows for different ways to add vocab
    """
    
    def __init__(self):
        self._vocabs = {}
        
    def _load_from_txt(self, path:str) -> Tuple[str]:
        """Loads a .txt file delimited by newlines"""
        with open (path, "r") as fin:
            return [line.strip("\n") for line in fin.readlines()]

    def _get_user_vocab_path(self):
        """Gets the user's path to the vocabulary files"""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        vocab_path = os.path.join(dir_path, "vocab/")
        return vocab_path
        
    def get(self, vocab_name:str) -> Tuple[str]:
        """Fetches the desired vocab from the registered vocabularies"""
        if vocab_name in self._vocabs:
            return self._vocabs[vocab_name]
        else:
            raise KeyError(f"Requested vocab '{vocab_name}' not found")
        
    def add_from_path(self, vocab_name:str) -> None:
        """Adds vocab from a newline delimited txt file given a vocab name (the file must already exist)"""
        path = f"{self._get_user_vocab_path()}/{vocab_name}.txt"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocab path '{path}' does not exist")
        
        items = self._load_from_txt(path)
        self._vocabs[vocab_name] = items
            
    def add_items(self, vocab_name:str, items:Tuple[str]) -> None:
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

# ~~~ Non-path loaded vocabs ~~~
from srm import SyntaxRegexMatcher
matcher = SyntaxRegexMatcher()
vocab.add_items("sentences", tuple(matcher.patterns.keys()))