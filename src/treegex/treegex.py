import spacy
from spacy import displacy
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
    "it-cleft": r"\([^-]*-be-[^-]*-ROOT.*\([iI]t-it-PRP-nsubj\).*\([^-]*-[^-]*-NN[^-]*-attr.*\([^-]*-[^-]*-VB[^-]*-relcl",
    "wh-cleft": r"\([^-]*-be-[^-]*-ROOT.*\([^-]*-[^-]*-W[^-]*-(dobj|advmod)"
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
    
    @dataclass
    class TestSentence:
        truth:str
        text:str
    
    test_sents = [
        TestSentence("TRUE", "All Jimmy wants for Christmas is a brand new bicycle."),
        TestSentence("TRUE", "All the girl does is complain about everything"),
        TestSentence("TRUE", "Was all she wanted a good job?"),
        TestSentence("TRUE", "Was all Jeff saw blue and yellow?"),
        TestSentence("TRUE", "All the dog in the tree knew was that the bone was on the grass."),
        TestSentence("TRUE", "While riding her bike, all Sarah though about was seeing her friends at the ball game."),
        TestSentence("FALSE", "I want all of these shirts"),
        TestSentence("FALSE", "The boxes were all filled with potatoes"),
        TestSentence("FALSE", "In all of this time, I have never seen a goose."),
    ]
    
    with open("temp.txt", "w") as fout:
        for test in test_sents:
            doc = nlp(test.text)
            for sent in doc.sents:
                fout.write(f"{test.truth}\n{test.text}\n{tree_to_string(sent)}\n\n")

    # matches = findall(" ".join(bad_sents))
    # for match in matches:
    #     print(match.captured_tree_string)




if __name__ == "__main__":
    main()