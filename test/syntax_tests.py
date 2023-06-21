
# this file is just for experimenting with the dependency tree regex matching

import spacy
from spacy import displacy
from dataclasses import dataclass
import re


@dataclass
class TestSentence:
    text:str
    truth:bool
    
    def __repr__(self):
        return self.text

nlp = spacy.load("en_core_web_md")


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


TEST_SENTENCES = (
    ("It was Jane’s car that got stolen last night", True),
    ("It was in the principate of Tiberius Caesar, who reigned from AD14 to AD37.",False),
    ("It wasn’t the most obvious problem.", False),
    ("It wasn’t the most obvious problem that intrigued me, it was the subtle issue of responsibility.",True),
    ("If it were John who is the candidate, I would vote for him.",True),
    ("It was a moral debt that I had inherited from my grandmother.", True),
    ("He was a man who ate my cakes.", False),
    ("It was in the principate of Tiberius Caesar that their druids and prophets and healers of this type were abolished.",True),
    ("It was during the principate of Tiberius Caesar when their druids and prophets and healers of this type were abolished.",True),
)
sentences = [TestSentence(text, truth) for text, truth in TEST_SENTENCES]

ZMD = "[\(.*\)]*" # zero or more dependents
NH = "[^-]*"      # any symbol that is not hyphen (-)
ANY = "\(.*\)"    # any symbol b/w two parenthesis

v1 = "\([^-]*-be-[^-]*-ROOT[\(.*\)]*\([i|I]t-it-PRP-nsubj\)[\(.*\)]*\([^-]*-[^-]*-NN[^-]*-attr\(.*\)\([^-]*-[^-]*-VB[^-]*-relcl"
v2 = "\([^-]*-be-[^-]*-ROOT\(.*\)*\(i|It-it-PRP-nsubj\)\(.*\)*\([^-]*-[^-]*-NN[^-]*-attr\(.*\)\([^-]*-[^-]*-VB[^-]*-relcl"
v3 = "\([^-]*-be-[^-]*-ROOT.*\([iI]t-it-PRP-nsubj\).*\([^-]*-[^-]*-NN[^-]*-attr.*\([^-]*-[^-]*-VB[^-]*-relcl"

test = "\([^-]*-be-[^-]*-ROOT(.*)\((i|I)t-it-PRP-nsubj\)(.*)\([^-]*-[^-]*-NN[^-]*-attr(.*)\([^-]*-[^-]*-VB[^-]*-relcl"

# https://regex101.com/r/HKhj03/1

# part 1: captures the ROOT verb
# \([^-]*-be-[^-]*-ROOT

# part 2: captures anything between the ROOT and it
# [\(.*\)]*

# part 3: captured it
#\((i|I)t-it-PRP-nsubj\)


# part ?
# [\(.*\)]*\([^-]*-[^-]*-NN[^-]*-attr\(.*\)\([^-]*-[^-]*-VB[^-]*-relcl

"""
TRUE
It was Jane’s car that got stolen last night

(was-be-VBD-ROOT(It-it-PRP-nsubj)(car-car-NN-attr(Jane-Jane-NNP-poss(’s-’s-POS-case)(stolen-steal-VBN-relcl(that-that-WDT-nsubjpass)(got-got-VBD-auxpass)(night-night-NN-npadvmod(last-last-JJ-amod))))))



TRUE
It wasn’t the most obvious problem that intrigued me, it was the subtle issue of responsibility.

(was-be-VBD-ROOT(was-be-VBD-ccomp(It-it-PRP-nsubj)(n’t-not-RB-neg)(problem-problem-NN-attr(the-the-DT-det)(obvious-obvious-JJ-amod(most-most-RBS-advmod)(intrigued-intrigue-VBD-relcl(that-that-WDT-nsubj)(me-I-PRP-dobj)(,-,-,-punct)(it-it-PRP-nsubj)(issue-issue-NN-attr(the-the-DT-det)(subtle-subtle-JJ-amod)(of-of-IN-prep(responsibility-responsibility-NN-pobj)(.-.-.-punct))))))))



TRUE
It was a moral debt that I had inherited from my grandmother.

(was-be-VBD-ROOT(It-it-PRP-nsubj)(debt-debt-NN-attr(a-a-DT-det)(moral-moral-JJ-amod)(inherited-inherit-VBN-relcl(that-that-WDT-dobj)(I-I-PRP-nsubj)(had-have-VBD-aux)(from-from-IN-prep(grandmother-grandmother-NN-pobj(my-my-PRP$-poss)(.-.-.-punct))))))



TRUE
It was in the principate of Tiberius Caesar that their druids and prophets and healers of this type were abolished.

(was-be-VBD-ROOT(It-it-PRP-nsubj)(in-in-IN-prep(principate-principate-NN-pobj(the-the-DT-det)(of-of-IN-prep(Caesar-Caesar-NNP-pobj(Tiberius-Tiberius-NNP-compound)(abolished-abolish-VBN-ccomp(that-that-IN-mark)(druids-druid-NNS-nsubjpass(their-their-PRP$-poss)(and-and-CC-cc)(prophets-prophet-NNS-conj(and-and-CC-cc)(healers-healer-NNS-conj)(of-of-IN-prep(type-type-NN-pobj(this-this-DT-det)(were-be-VBD-auxpass)(.-.-.-punct)))))))))))



TRUE
It was during the principate of Tiberius Caesar when their druids and prophets and healers of this type were abolished.

(was-be-VBD-ROOT(It-it-PRP-nsubj)(during-during-IN-prep(principate-principate-NN-pobj(the-the-DT-det)(of-of-IN-prep(Caesar-Caesar-NNP-pobj(Tiberius-Tiberius-NNP-compound)(abolished-abolish-VBN-advcl(when-when-WRB-advmod)(druids-druid-NNS-nsubjpass(their-their-PRP$-poss)(and-and-CC-cc)(prophets-prophet-NNS-conj(and-and-CC-cc)(healers-healer-NNS-conj)(of-of-IN-prep(type-type-NN-pobj(this-this-DT-det)(were-be-VBD-auxpass)(.-.-.-punct)))))))))))



TRUE (SKIP FOR NOW)
If it were John who is the candidate, I would vote for him."

(vote-vote-VB-ROOT(were-be-VBD-advcl(If-if-IN-mark)(it-it-PRP-nsubj)(John-John-NNP-attr)(is-be-VBZ-ccomp(who-who-WP-nsubj)(candidate-candidate-NN-attr(the-the-DT-det)(,-,-,-punct)(I-I-PRP-nsubj)(would-would-MD-aux)(for-for-IN-prep(him-he-PRP-pobj))))))



FALSE
He was a man who ate my cakes.

(was-be-VBD-ROOT(He-he-PRP-nsubj)(man-man-NN-attr(a-a-DT-det)(ate-eat-VBD-relcl(who-who-WP-nsubj)(cakes-cake-NNS-dobj(my-my-PRP$-poss)(.-.-.-punct)))))



FALSE
It was in the principate of Tiberius Caesar, who reigned from AD14 to AD37.

(was-be-VBD-ROOT(It-it-PRP-nsubj)(in-in-IN-prep(principate-principate-NN-pobj(the-the-DT-det)(of-of-IN-prep(Caesar-Caesar-NNP-pobj(Tiberius-Tiberius-NNP-compound)(,-,-,-punct)(reigned-reign-VBD-relcl(who-who-WP-nsubj)(from-from-IN-prep(AD14-AD14-NNP-pobj)(to-to-IN-prep(AD37-AD37-NNP-pobj)(.-.-.-punct)))))))))



FALSE
It wasn’t the most obvious problem.

(was-be-VBD-ROOT(It-it-PRP-nsubj)(n’t-not-RB-neg)(problem-problem-NN-attr(the-the-DT-det)(obvious-obvious-JJ-amod(most-most-RBS-advmod)(.-.-.-punct))))


"""





it_cleft_regex = re.compile(test)
def search_tree_with_pattern(tree_string:str, regex:re.Pattern) -> bool:
    found = regex.findall(tree_string)
    return len(found) >= 1

def get_incorrect_sentence_predictions(sentences, regex:re.Pattern):
    
    incorrect = []
    for sentence in sentences:
        doc = list(nlp(sentence.text).sents)[0]
        tree_string = tree_to_string(doc)
        result = search_tree_with_pattern(tree_string, regex) == sentence.truth
        if not result:
            incorrect.append((sentence.text, tree_string, sentence.truth))
    return incorrect

def display_incorrect_predictions(sentences, regex:re.Pattern):
    
    incorrect_preds = get_incorrect_sentence_predictions(sentences, regex)
    for text, tree_string, truth in incorrect_preds:
        print(f"{text}\n{tree_string}\n{truth}\n")
        
        
 
# display_incorrect_predictions(sentences, it_cleft_regex)


#doc = nlp("If it were John who is the candidate, I would vote for him.")
# displacy.serve(doc, auto_select_port=True)
s = "It was yesterday that John saw me last."
s = "It's only recently that he started playing guitar."
s = "If it were John who is the candidate, I would vote for him."
s = "If it was butter that I bought yesterday, you should buy cheese."
doc = list(nlp(s).sents)[0]
tree_string = tree_to_string(doc)
displacy.serve(doc, auto_select_port=True)



