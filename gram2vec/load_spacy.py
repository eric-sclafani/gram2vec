import spacy
from spacy.tokens import Doc
from typing import Callable, List, Tuple, Generator
from nltk import bigrams

# ~~~ Type aliases ~~~
SentenceSpan = Tuple[int,int]
Bigram = Tuple[str,str]

def get_tokens(doc):
    return [token.text for token in doc]

def get_words(doc):
    return [token.text for token in doc if not token.is_punct]

def get_pos_tags(doc):
    return [token.pos_ for token in doc]

def get_dep_labels(doc):
    return [token.dep_ for token in doc]

def get_morph_tags(doc):
    return [morph for token in doc for morph in token.morph if morph != ""]

def get_sentences(doc):
    return list(doc.sents)

def get_pos_bigrams(doc:Doc) -> List[Bigram]:

    def get_sentence_spans(doc:Doc) -> List[SentenceSpan]:
        """Gets each start and end index of all sentences in a document"""
        return [(sent.start, sent.end) for sent in doc._.sentences]

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
    
    return list(pos_bigrams)
    
def get_mixed_bigrams(doc):
    
    def replace_openclass(doc:Doc, open_class:List[str]) -> List[str]:
        """Replaces all open class tokens with corresponding POS tags and returns a new list"""
        tokens_with_replacements = doc._.tokens
        for i, _ in enumerate(tokens_with_replacements):
            if doc._.pos_tags[i] in open_class:
                tokens_with_replacements[i] = doc._.pos_tags[i]
        return tokens_with_replacements

    def remove_openclass_bigrams(bigrams, open_class:List[str]) -> List[Bigram]:
        """Filters out (OPEN_CLASS, OPEN_CLASS) and (CLOSED_CLASS, CLOSED_CLASS) bigrams that inadvertently get created in replace_openclass"""
        filtered = []
        for pair in bigrams:
            if pair[0] not in open_class and pair[1] in open_class or \
               pair[0] in open_class and pair[1] not in open_class:
                filtered.append(pair)
        return filtered
        
    OPEN_CLASS = ["ADJ", "ADV", "NOUN", "VERB", "INTJ"]
    tokens_with_replacements = replace_openclass(doc, OPEN_CLASS)
    mixed_bigrams = bigrams(tokens_with_replacements)
    mixed_bigrams = remove_openclass_bigrams(mixed_bigrams, OPEN_CLASS)
    return mixed_bigrams
    





def set_spacy_extension(name:str, function:Callable) -> None:
    """Creates spacy extensions to easily access certain information"""
    if not Doc.has_extension(name):
        Doc.set_extension(name, getter=function)
   
   

   
# Add more extensions here as needed!
# Extension syntax: (extension_name, function that returns a list)
custom_extensions = {
    ("tokens", get_tokens),
    ("words", get_words),
    ("pos_tags", get_pos_tags),
    ("dep_labels", get_dep_labels),
    ("morph_tags", get_morph_tags),
    ("sentences", get_sentences),
    ("pos_bigrams", get_pos_bigrams),
    ("mixed_bigrams", get_mixed_bigrams)
}
nlp = spacy.load("en_core_web_md", exclude=["ner", "lemmatizer"])
spacy.prefer_gpu()

for name, function in custom_extensions:
    set_spacy_extension(name, function)
    
    
    
if __name__ == "__main__":
    test_docs = [
        "It was Jane's car that got stolen last night. She is odd.",
        "I ate the pineapple"
        ]

    doc = nlp(test_docs[1])
    
    print(doc._.mixed_bigrams)
    

