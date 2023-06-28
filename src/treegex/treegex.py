import spacy
from dataclasses import dataclass
from typing import Tuple
from sys import stderr
import re


#! will get replaced when uploaded to Pypi
def load_spacy(model="en_core_web_md"):
    try:
        nlp = spacy.load(model, exclude=["ner"])
    except OSError:
        print(f"Downloading spaCy language model '{model}' (don't worry, this will only happen once)", file=stderr)
        from spacy.cli import download
        download(model)
        nlp = spacy.load(model, exclude=["ner"])
    return nlp 

nlp = load_spacy()  


def save_test_sentences(sentences, fname):
    with open(f"{fname}.txt", "w") as fout:
        for test in sentences:
            doc = nlp(test.text)
            for sent in doc.sents:
                fout.write(f"{test.truth}\n{test.text}\n{tree_to_string(sent)}\n\n")


@dataclass
class Treegex:    
    name:str
    pattern:str
    
@dataclass
class Match:
    pattern_name:str
    captured_tree_string:str
    sentence:str
    
    def __repr__(self) -> str:
        return f"{self.pattern_name} : {self.sentence}"

def load_patterns() -> Tuple[Treegex]:
    return tuple([Treegex(name, pattern) for name, pattern in PATTERNS.items()])
        
def get_num_non_terminals(sentence) -> int:
    return sum([1 for token in sentence if list(token.children)])

def add_ending_parenthesis(sentence, result:str) -> str:
    return f"{result}{ ')' * get_num_non_terminals(sentence)}"

def tree_to_string(sentence):
    stack = [sentence.root]
    result = ""
    
    while stack:
        token = stack.pop()
        result += f"({token.text}-{token.lemma_}-{token.tag_}-{token.dep_}" 
        
        for child in reversed(list(token.children)):
            stack.append(child)
        
        if not list(token.children):
            result += ")"
    return add_ending_parenthesis(sentence, result)


def findall(document:str):
    
    doc = nlp(document)
    patterns = load_patterns()
    matches = []
    for sent in doc.sents:
        tree_string = tree_to_string(sent)
        for pattern in patterns:
            match = re.findall(pattern.pattern, tree_string)
            matches.extend([Match(pattern.name, tree_string, sent.text) for _ in match])
    return matches
    
def main():
    

    
                    
                    
    test_sentence = """It wasn’t the most obvious problem that intrigued me, it was the subtle issue of responsibility. All the girl does is complain about everything. There's this orphan kid that I'm trying to adopt. What I want is some peace and quiet! If I ate the chicken, it's because I like meat. The windows have been cleaned."""
    matches = findall(test_sentence)
    for match in matches:
        print(match)
    

if __name__ == "__main__":
    main()