import spacy
from dataclasses import dataclass
from typing import Tuple
import re
import os


# def get_user_path():
#     """Gets the user's path"""
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     vocab_path = os.path.join(dir_path, "vocab/")
#     return vocab_path

nlp = spacy.load("en_core_web_md", exclude=["ner"])

@dataclass
class Treegex:    
    name:str
    pattern:re.Pattern
    

    


def load_patterns(path="patterns.txt") -> Tuple[Treegex]:
    with open (path, "r") as fin:
        patterns = [line.strip("\n").split(":") for line in fin.readlines()]
        return patterns
        
        


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



def main():
    
    from patterns import PATTERNS
    print(PATTERNS["it-cleft"])


if __name__ == "__main__":
    main()