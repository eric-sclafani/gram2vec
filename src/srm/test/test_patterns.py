import spacy
import json
from collections import defaultdict
from dataclasses import dataclass
from srm import linearize_tree

@dataclass
class PatternTest:
    truth:str
    sentence:str
    
    def __repr__(self):
        return f"{self.truth} : {self.sentence}"
    
def load_pattern_tests(path="pattern_tests.json"):
    pattern_tests = defaultdict(list)
    with open(path) as reader:
        for obj in json.load(reader):
            current_pattern = obj["name"]
            for test_sentence in obj["tests"]:
                pattern_tests[current_pattern].append(
                    PatternTest(test_sentence[0], test_sentence[1]))
    return pattern_tests

def get_truths(tests):
    return [test.truth for test in tests]

def get_sentences(tests):
    return [test.sentence for test in tests]

def apply_linearizer(docs):
    return [linearize_tree(sent) for doc in docs for sent in doc.sents]

def write_to_file(path,
                  truths, 
                  sentences, 
                  linear_trees
                  ) -> None:
    with open(path, "w") as writer:
        for truth, sentence, linear_tree in zip(truths, sentences, linear_trees):
            writer.write(f"{truth}\n{sentence}\n{linear_tree}\n\n")
               
def main():
    
    nlp = spacy.load("en_core_web_lg")
    pattern_tests = load_pattern_tests()
    
    for pattern_name, tests in pattern_tests.items():
        path = f"pattern_test_outputs/{pattern_name}.txt"
        truths = get_truths(tests)
        sentences = get_sentences(tests)
        docs = nlp.pipe(sentences)
        linear_trees = apply_linearizer(docs)
        write_to_file(path,
                      truths,
                      sentences,
                      linear_trees)


if __name__ == "__main__":
    main()