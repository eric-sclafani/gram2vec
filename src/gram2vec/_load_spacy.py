import spacy
from spacy.tokens import Doc
from sys import stderr
from typing import Callable, List, Tuple, Iterable
import os

from ._load_vocab import vocab
from . import matcher
# ~~~ Type aliases ~~~

SentenceSpan = Tuple[int,int]
Bigram = Tuple[str,str]

language = os.environ.get("LANGUAGE", "en")
if language == "ru":
    matcher = matcher.SyntaxRegexMatcher(language="ru")
elif language == "en":
    matcher = matcher.SyntaxRegexMatcher(language="en")

# ~~~ Getters ~~~

def get_tokens(doc):
    return [token.text for token in doc]

def get_pos_tags(doc):
    return [token.pos_ for token in doc]

def get_dep_labels(doc):
    return [token.dep_ for token in doc]

def get_morph_tags(doc):
    return [morph for token in doc for morph in token.morph if morph != ""]

def get_pos_bigrams(doc) -> List[Bigram]:

    def get_sentence_spans(doc) -> List[SentenceSpan]:
        """Gets each start and end index of all sentences in a document"""
        return [(sent.start, sent.end) for sent in doc.sents]

    def insert_sentence_boundaries(spans:List[SentenceSpan]) -> List[str]:
        """Marks sentence boundaries with symbols BOS (beginning of sentence) & EOS (end of sentence)"""
        new_tokens = []
        for i, pos in enumerate(doc._.pos_tags):
            for sent_start, sent_end in spans:
                if i == sent_start:
                    new_tokens.append("BOS")
                elif i == sent_end:
                    new_tokens.append("EOS")
            new_tokens.append(pos)
        new_tokens.append("EOS")
        return new_tokens

    def convert_bigrams_to_strings(bigrams) -> List[str]:
        """Converts bigrams into a list of bigram strings"""
        return [" ".join(bigram) for bigram in bigrams]

    def bigrams(iter:Iterable) -> List[Tuple]:
        return [tuple(iter[i:i+2]) for i in range(len(iter)-1)]

    sent_spans = get_sentence_spans(doc)
    pos_tags_with_boundary_syms = insert_sentence_boundaries(sent_spans)
    pos_bigrams = bigrams(pos_tags_with_boundary_syms)
    return convert_bigrams_to_strings(pos_bigrams)

def get_sentences(doc):
    sentence_matches = matcher.match_document(doc)
    return [match.pattern_name for match in sentence_matches]

def get_func_words(doc):
    func_words_vocab = vocab.get("func_words")
    return [token for token in doc.doc._.tokens if token in func_words_vocab]

def get_punctuation(doc):
    punctuation_vocab = vocab.get("punctuation")
    return [punc for token in doc.doc._.tokens for punc in token if punc in punctuation_vocab]

def get_punct_periods(doc):
    punct_periods_vocab = vocab.get("punct_periods")
    return [punc for token in doc.doc._.tokens for punc in token if punc in punct_periods_vocab]

def get_punct_commas(doc):
    punct_commas_vocab = vocab.get("punct_commas")
    return [punc for token in doc.doc._.tokens for punc in token if punc in punct_commas_vocab]

def get_punct_colons(doc):
    punct_colons_vocab = vocab.get("punct_colons")
    return [punc for token in doc.doc._.tokens for punc in token if punc in punct_colons_vocab]

def get_punct_semicolons(doc):
    punct_semicolons_vocab = vocab.get("punct_semicolons")
    return [punc for token in doc.doc._.tokens for punc in token if punc in punct_semicolons_vocab]

def get_punct_exclamations(doc):
    punct_exclamations_vocab = vocab.get("punct_exclamations")
    return [punc for token in doc.doc._.tokens for punc in token if punc in punct_exclamations_vocab]

def get_punct_questions(doc):
    punct_questions_vocab = vocab.get("punct_questions")
    return [punc for token in doc.doc._.tokens for punc in token if punc in punct_questions_vocab]

def get_letters(doc):
    letters_vocab = vocab.get("letters")
    return [letter for token in doc.doc._.tokens for letter in token if letter in letters_vocab]


# Add more extensions here as needed!
# Extension syntax: (extension name, getter function that returns a list)

helper_extensions = {
    ("tokens", get_tokens),
}

feature_extensions = {
    ("pos_tags", get_pos_tags),
    ("dep_labels", get_dep_labels),
    ("morph_tags", get_morph_tags),
    ("pos_bigrams", get_pos_bigrams),
    ("sentences", get_sentences),
    ("func_words", get_func_words),
    ("punctuation", get_punctuation),
    ("punct_periods", get_punct_periods),   
    ("punct_commas", get_punct_commas),
    ("punct_colons", get_punct_colons),
    ("punct_semicolons", get_punct_semicolons),
    ("punct_exclamations", get_punct_exclamations),
    ("punct_questions", get_punct_questions),
    ("letters", get_letters),
}

def set_spacy_extension(name:str, function:Callable) -> None:
    """Creates spacy extensions to easily access certain information"""
    if not Doc.has_extension(name):
        Doc.set_extension(name, getter=function)

model = os.environ.get("SPACY_MODEL", "en_core_web_lg")
try:
    nlp = spacy.load(model)
except OSError:
    print(f"Downloading spaCy language model '{model}' (this will only happen once)", file=stderr)
    from spacy.cli import download
    download(model)

nlp = spacy.load(model)
nlp.max_length = 10000000 
print(f"Gram2Vec: Using '{model}'")

for name, function in helper_extensions:
    set_spacy_extension(name, function)

for name, function in feature_extensions:
    set_spacy_extension(name, function)
