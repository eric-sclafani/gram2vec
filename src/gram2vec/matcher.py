from spacy.language import Doc
from spacy.tokens import Span
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, List
import re

@dataclass
class Match:
    pattern_name:str
    matched:str
    sentence:str
    
    def __repr__(self) -> str:
        return f"{self.pattern_name} : {self.sentence}"

class SyntaxRegexMatcher:
    """
    This class encapsulates the sentence regex patterns and methods to apply them to target documents
    """
    def __init__(self, language:str):
        if language == "en":
            self.patterns = {
                "it-cleft": r"\([^-]*-be-[^-]*-[^-]*.*\([iI]t-it-PRP-nsubj\).*\([^-]*-[^-]*-NN[^-]*-attr.*\([^-]*-[^-]*-VB[^-]*-(relcl|advcl)",
                "pseudo-cleft": r"\([^-]*-be-[^-]*-[^-]*.*\([^-]*-[^-]*-(WP|WRB)-(dobj|advmod)",
                "all-cleft" : r"(\([^-]*-be-[^-]*-[^-]*.*\([^-]*-all-(P)?DT)|(\([^-]*-all-(P)?DT-[^-]*.*\([^-]*-be-[^-]*)",
                "there-cleft": r"\([^-]*-be-[^-]*-[^-]*.*\([^-]*-there-EX-expl.*\([^-]*-[^-]*-[^-]*-attr.*\([^-]*-[^-]*-[^-]*-(relcl|acl)",
                "if-because-cleft" : r"\([^-]*-be-[^-]*-[^-]*.*\([^-]*-[^-]*-[^-]*-advcl\([^-*]*-if-IN-mark",
                "passive" : r"\([^-]*-[^-]*-(NN[^-]*|PRP|WDT)-nsubjpass.*\([^-]*-be-[^-]*-auxpass",
                "subj-relcl" : r"\([^-]*-[^-]*-[^-]*-relcl.*\([^-]*-[^-]*-(WP|WDT)-nsubj",
                "obj-relcl" : r"\([^-]*-[^-]*-NN[^-]*-(nsubj|attr).*\([^-]*-[^-]*-[^-]*-(relcl|ccomp).*\([^-]*-[^-]*-(WP|WDT|IN)-(pobj|dobj)",
                "tag-question" : r"\([^-]*-(do|be|could|can|have)-[^-]*-ROOT.*\(\?-\?-\.-punct",
                "coordinate-clause" : r"\([^-]*-[^-]*-CC-cc\).*\([^-]*-[^-]*-(VB[^-]*|JJ)-conj.*\([^-]*-[^-]*-[^-]*-nsubj"
            }
            
        elif language == "ru":
            self.patterns = {
                "passive_rus" : r"\([^-]*-[^-]*-(NOUN[^-]*|PRON)-nsubj:pass.*\)",
                "parataxis_rus": r"\([^-]*-[^-]*-[^-]*-parataxis.*\)",
                "participle_gerund_rus": r"\([^-]*-[^-]*-(VERB-acl(?!:relcl)|ADJ-amod|VERB-advcl).*?\)",
                "conj_rus": r"\([^()]*-[^-]*-[^-]*-conj[^()]*\)",
                "nested_structure_rus": r"\([^-]*-[^-]*-(VERB-acl:relcl).*?\)",
                "one_word_sent_rus": r"\([^-]*-[^-]*-(NOUN|VERB)-ROOT\(.*?-PUNCT-punct\)\)"
            }

        self.language = language

    def print_patterns(self) -> None:
        for pattern_name, pattern in self.patterns.items():
            print(f"{pattern_name} : {pattern}\n")

        
    def _find_treegex_matches(self, doc:Doc) -> Tuple[Match]:
        """Iterates through a document's sentences, applying every regex to each sentence"""
        matches = []
        for sent in doc.sents:
            tree_string = self.linearize_tree(sent)
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
        Applies all registered patterns to one spaCy-generated document
        
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
        Applies all registered patterns to a collection of spaCy-generated documents
        
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
    
    def linearize_tree(self, sentence:Span) -> str:
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
                if self.language == "en":
                    result += f"({token.text}-{token.lemma_}-{token.tag_}-{token.dep_}" 
                elif self.language == "ru":
                    result += f"({token.text}-{token.lemma_}-{token.tag_}-{token.dep_}${token.morph}" 
                
                for child in reversed(list(token.children)):
                    stack.append(child)
                
                if not list(token.children):
                    result += ")"
            return result
        
        parse = parse_dependency_parse(sentence)
        nt_count = get_NT_count(sentence)
        return f"{parse}{ending_parenthesis(nt_count)}"