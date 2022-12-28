# TODO
This file contains ideas for future additions/improvements
------

## Additions:
- add functions to get information about the vectors
- add glove lexical vector to the feature vector
    - in document, get glove vecs for each word and average them -> stick into grammar vector
- add syntactic stuff
- fix odd parenthesis issue in `preproccess.py` in **fix_data**
- Ablation studies
- move bin dev into a shell script
- syntax feats: counts of certain dependency labels, finite verb subject ommision, type of rel clause, 
- morph: embedded finite vs non-finite clauses (I hope that I can leave tomorrow vs I hope to leave tomorrow)
- prodosy: do people have a stress (melodic) preference?

## Ideas
- token level punc marks VS. string level (this does not account for text-based emojis, etc..)
- add accented letters to letter counts?
- add sentence boundary syms for ngrams
- k-NN: if no majority vote, switch k value??
- KNN vs NN (1 vector/document VS 1 vector/author k=1)