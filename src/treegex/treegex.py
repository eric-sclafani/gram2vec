import spacy
from spacy.language import Doc
from spacy.tokens import Span
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, List
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

    def add_patterns(self, patterns:Dict[str,str]) -> None:
        """Updates the default patterns dictionary with a user supplied dictionary of {pattern_name:regex} pairs"""
        self.patterns.update(patterns)
        
    def remove_patterns(self, to_remove:Iterable[str]) -> None:
        """Given an iterable of pattern names, removes those patterns from the registered pattens list"""
        for pattern_name in to_remove:
            try:
                del self.patterns[pattern_name]
            except KeyError:
                raise KeyError(f"Pattern '{pattern_name}' not in registered patterns.")
            
            
    def match_document(self, document:Doc) -> Tuple[Match]:
        """
        Applies all registered treegexes to one spaCy-generated document
        
        Args
        ----
            - document - a single spaCy document\n
        Returns
        -------
            - a tuple of sentence matches for a single document
        """
        return self._find_treegex_matches(document)

    def match_documents(self, documents:Iterable[Doc]) -> List[Tuple[Match]]:
        """
        Applies all registered treegexes to a collection of spaCy-generated documents
        
        Args
        ----
            - documents - iterable of spacy documents\n
        Returns
        -------
            - A list of tuples such that each tuple contains one document's sentence matches
        """
        all_matches = []
        for document in documents:
            all_matches.append(self._find_treegex_matches(document))
        return all_matches

            
            

    

def main():

    DOCS = [
        "It was the dog that John bought and if he bought the dog, it is because he likes animals",
        "It was the dog that John bought.",
        "It was the dog the man adopted and it was the cat the woman adopted."
    ]
    nlp = spacy.load("en_core_web_md")
    docs = nlp.pipe(DOCS)
    treegex = TreegexPatternMatcher()
    for match in treegex.match_documents(docs):
        print(match)
    

    
    
    

    

if __name__ == "__main__":
    main()