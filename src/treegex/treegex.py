import spacy
from dataclasses import dataclass
from typing import Tuple
from sys import stderr
import re

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

PATTERNS = {
    "it-cleft": r"\([^-]*-be-[^-]*-ROOT.*\([iI]t-it-PRP-nsubj\).*\([^-]*-[^-]*-NN[^-]*-attr.*\([^-]*-[^-]*-VB[^-]*-relcl"
}

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



def search_tree_with_pattern(tree_string:str, regex:re.Pattern) -> bool:
    matches = regex.findall(tree_string)



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
    
    d = "It was Jane's car that got stolen last night. It was my truck that got stolen yesterday."
    matches = findall(d)
    print(matches)


if __name__ == "__main__":
    main()