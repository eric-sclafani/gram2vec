import spacy
from spacy.language import Doc
from spacy.tokens import Span
from dataclasses import dataclass
from typing import Tuple, Iterable
from sys import stderr
import re

from patterns import PATTERNS

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
        

def linearlize_tree(sentence:Span) -> str:
    """
    Converts a spaCy-produced dependency parse into a linear tree string while preserving dependency relations
    """
    
    def get_NT_count(sentence) -> int:
        """Returns the number of non-terminal nodes in a dep tree"""
        return sum([1 for token in sentence if list(token.children)])

    def ending_parenthesis(n:int) -> str:
        """Returns the appropriate amount of parenthesis to add to linearlized tree"""
        return f"{')' * n}"
    
    def parse_dependency_parse(sentence):
        """Processes a dependency parse in a bottom-up fashion"""
        stack = [sentence.root]
        result = ""
        while stack:
            token = stack.pop()
            result += f"({token.text}-{token.lemma_}-{token.tag_}-{token.dep_}" 
            
            for child in reversed(list(token.children)):
                stack.append(child)
            
            if not list(token.children):
                result += ")"
        return result
    
    parse = parse_dependency_parse(sentence)
    nt_count = get_NT_count(sentence)
    return f"{parse}{ending_parenthesis(nt_count)}"





def find_treegex_matches(document:str):
    
    doc = nlp(document)
    patterns = load_patterns()
    matches = []
    for sent in doc.sents:
        tree_string = linearlize_tree(sent)
        for pattern in patterns:
            match = re.findall(pattern.pattern, tree_string)
            matches.extend([Match(pattern.name, tree_string, sent.text) for _ in match])
    return matches


 
 
 
 

def findall(documents:Iterable[str|Doc]):
    pass
    

    
        
        
        
   
   
 
def main():

    doc = nlp("this is a string")
    
    print(linearlize_tree(list(doc.sents)[0]))    
    

if __name__ == "__main__":
    main()