import spacy
from spacy.language import Doc
from spacy.tokens import Span
from dataclasses import dataclass
from typing import Dict, Tuple
from collections.abc import Container
import re


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

@dataclass
class Match:
    pattern_name:str
    matched:str
    sentence:str
    
    def __repr__(self) -> str:
        return f"{self.pattern_name} : {self.sentence}"

class TreegexPatternMatcher:
    
    def __init__(self):
        self.patterns = {
            "it-cleft": r"\([^-]*-be-[^-]*-ROOT.*\([iI]t-it-PRP-nsubj\).*\([^-]*-[^-]*-NN[^-]*-attr.*\([^-]*-[^-]*-VB[^-]*-(relcl|advcl)",
            "pseudo-cleft": r"\([^-]*-be-[^-]*-ROOT.*\([^-]*-[^-]*-(WP|WRB)-(dobj|advmod)",
            "all-cleft" : r"(\([^-]*-be-[^-]*-[^-]*\([^-]*-all-(P)?DT-[^-]*.*)|(\([^-]*-all-(P)?DT-[^-]*.*\([^-]*-be-VB[^-]*-[^-]*)",
            "there-cleft": r"\([^-]*-be-[^-]*-[^-]*.*\([^-]*-there-EX-expl.*\([^-]*-[^-]*-[^-]*-attr",
            "if-because-cleft" : r"\([^-]*-be-[^-]*-ROOT.*\([^-]*-[^-]*-[^-]*-advcl\([^-*]*-if-IN-mark",
            "passive" : r"\([^-]*-[^-]*-(NN[^-]*|PRP)-nsubjpass.*\([^-]*-be-[^-]*-auxpass"
        }
        
    def _find_treegex_matches(self, doc:Doc) -> Tuple[Match]:
        """Iterates through a document's sentences, applying every regex to each sentence"""
        matches = []
        for sent in doc.sents:
            tree_string = linearlize_tree(sent)
            for name, pattern in self.patterns.items():
                match = re.search(pattern, tree_string)
                if match:
                    matches.append(Match(name, match.group(), sent.text))
        return tuple(matches)
    
    def _has_correct_type(self, documents:Doc|Container[Doc]) -> bool:
        """Messy way to ensure correct types (if there's a cleaner way, please lmk)"""
        try:
            if not isinstance(documents, Doc) and not isinstance(documents[0], Doc):
                return False
        except IndexError:
            return False
        return True
    
    
    def add_patterns(self, patterns:Dict[str,str]) -> None:
        """Updates the default patterns dictionary with a user supplied dictionary of {pattern_name:regex} pairs"""
        self.patterns.update(patterns)
            
    def match_documents(self, documents:Doc|Container[Doc]):
        
        if not self._has_correct_type(documents):
            raise TypeError("'documents' arg must be a spaCy doc or container of spaCy docs")
            

    

def main():

    nlp = spacy.load("en_core_web_md")
    doc1 = nlp("It was the dog that John bought and if he bought the dog, it is because he likes animals. ")  
    doc2 = nlp("It was the dog that John bought.")
    doc3 = nlp("It was the dog the man adopted and it was the cat the woman adopted.")
    treegex = TreegexPatternMatcher()
    
    

    
    
    

    

if __name__ == "__main__":
    main()