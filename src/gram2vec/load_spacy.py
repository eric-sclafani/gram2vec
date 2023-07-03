import spacy
from spacy.tokens import Doc
from sys import stderr
from typing import Callable, List, Tuple, Iterable
from nltk import bigrams
from srm import SentenceRegexMatcher

# ~~~ Type aliases ~~~

SentenceSpan = Tuple[int,int]
Bigram = Tuple[str,str]

# ~~ Regex matcher ~~~

matcher = SentenceRegexMatcher()

# ~~~ Helper funcs ~~~

def convert_bigrams_to_strings(bigrams) -> List[str]:
    """Converts nltk bigrams into a list of bigram strings"""
    return [" ".join(bigram) for bigram in bigrams]

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
    
    sent_spans = get_sentence_spans(doc)
    pos_tags_with_boundary_syms = insert_sentence_boundaries(sent_spans)
    pos_bigrams = bigrams(pos_tags_with_boundary_syms)
    
    return convert_bigrams_to_strings(pos_bigrams)
    
def get_mixed_bigrams(doc):
    
    def replace_openclass(doc, open_class:List[str]) -> List[str]:
        """Replaces all open class tokens with corresponding POS tags and returns a new list"""
        tokens_with_replacements = doc._.tokens
        for i, _ in enumerate(tokens_with_replacements):
            if doc._.pos_tags[i] in open_class:
                tokens_with_replacements[i] = doc._.pos_tags[i]
        return tokens_with_replacements

    def remove_illicit_bigrams(bigrams, open_class:List[str]) -> List[Bigram]:
        """Ensures that only (OPEN_CLASS, CLOSED_CLASS) and (CLOSED_CLASS, OPEN_CLASS) bigrams are included in mixed bigrams"""
        filtered = []
        for pair in bigrams:
            if pair[0] not in open_class and pair[1] in open_class or \
               pair[0] in open_class and pair[1] not in open_class:
                filtered.append(pair)
        return filtered
        
    OPEN_CLASS = ["ADJ", "ADV", "NOUN", "VERB", "INTJ", "PROPN"]
    tokens_with_replacements = replace_openclass(doc, OPEN_CLASS)
    mixed_bigrams = bigrams(tokens_with_replacements)
    mixed_bigrams = remove_illicit_bigrams(mixed_bigrams, OPEN_CLASS)
    return convert_bigrams_to_strings(mixed_bigrams)

def get_syntactic_patterns(doc):
    sentence_matches = matcher.match_document(doc)
    return [match.pattern_name for match in sentence_matches]
    
    
# Add more extensions here as needed!
# Extension syntax: (extension_name, getter function that returns a list)
custom_extensions = {
    ("tokens", get_tokens),
    ("pos_tags", get_pos_tags),
    ("dep_labels", get_dep_labels),
    ("morph_tags", get_morph_tags),
    ("pos_bigrams", get_pos_bigrams),
    ("mixed_bigrams", get_mixed_bigrams),
    ("syntactic_patterns", get_syntactic_patterns)
}

def set_spacy_extension(name:str, function:Callable) -> None:
    """Creates spacy extensions to easily access certain information"""
    if not Doc.has_extension(name):
        Doc.set_extension(name, getter=function)
 
try:
    nlp = spacy.load("en_core_web_md", exclude=["ner"])
except OSError:
    print("Downloading spaCy language model 'en_core_web_md' (this will only happen once)", file=stderr)
    from spacy.cli import download
    download("en_core_web_md")
    
nlp = spacy.load("en_core_web_md", exclude=["ner"])
for name, function in custom_extensions:
    set_spacy_extension(name, function)