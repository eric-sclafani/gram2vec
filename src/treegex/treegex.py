import spacy
from spacy.language import Doc
from spacy.tokens import Span
from dataclasses import dataclass
from typing import Tuple, Iterable, Dict
from sys import stderr
import re

PATTERNS = {
    
}




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

    
@dataclass
class Match:
    pattern_name:str
    captured_tree_string:str
    sentence:str
    
    def __repr__(self) -> str:
        return f"{self.pattern_name} : {self.sentence}"

        

def linearlize_tree(sentence:Span) -> str:
    """Converts a spaCy dependency-parsed sentence into a linear tree string while preserving dependency relations"""
    
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


class TreegexPatternMatcher:
    
    def __init__(self):
        self.patterns = {
            "it-cleft": r"\([^-]*-be-[^-]*-ROOT.*\([iI]t-it-PRP-nsubj\).*\([^-]*-[^-]*-NN[^-]*-attr.*\([^-]*-[^-]*-VB[^-]*-relcl",
            "psuedo-cleft": r"\([^-]*-be-[^-]*-ROOT.*\([^-]*-[^-]*-(WP|WRB)-(dobj|advmod)",
            "all-cleft" : r"(\([^-]*-be-[^-]*-[^-]*\([^-]*-all-(P)?DT-[^-]*.*)|(\([^-]*-all-(P)?DT-[^-]*.*\([^-]*-be-VB[^-]*-[^-]*)",
            "there-cleft": r"\([^-]*-be-[^-]*-[^-]*.*\([^-]*-there-EX-expl.*\([^-]*-[^-]*-[^-]*-attr",
            "if-because-cleft" : r"\([^-]*-be-[^-]*-ROOT\([^-]*-[^-]*-[^-]*-advcl\([^-*]*-if-IN-mark",
            "passive" : r"\([^-]*-[^-]*-(NN[^-]*|PRP)-nsubjpass.*\([^-]*-be-[^-]*-auxpass"
        }
    
    def add_patterns(self, patterns:Dict[str,str]) -> None:
        """Updates the default patterns dictionary with a user supplied dictionary of {pattern_name:regex} pairs"""
        self.patterns.update(patterns)
        
    
    def _find_treegex_matches(self, doc:Doc):
        """Iterates through a document's sentences, applying every regex to each sentence"""
        matches = []
        for sent in doc.sents:
            tree_string = linearlize_tree(sent)
            for name, pattern in self.patterns.items():
                match = re.findall(pattern, tree_string)
                matches.extend([Match(name, tree_string, sent.text) for _ in match])
        return matches  
            
        
    def match_spacy_docs(self):
        pass
    
    def match_strings(self):
        pass
    

        
    
        






 

def main():

    doc = nlp("this is a string")
    
    print(linearlize_tree(list(doc.sents)[0]))    
    

if __name__ == "__main__":
    main()