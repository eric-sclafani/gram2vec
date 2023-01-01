# TODO
This file contains ideas for future additions/improvements
------

## Additions:
- add functions to get information about the vectors
- add glove lexical vector to the feature vector https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
    - in document, get glove vecs for each word and average them -> stick into grammar vector
- add syntactic stuff
- Ablation studies
- move bin dev into a shell script
- syntax feats: counts of certain dependency labels, finite verb subject ommision, type of rel clause, 
- morph: embedded finite vs non-finite clauses (I hope that I can leave tomorrow vs I hope to leave tomorrow)
- prosody: do people have a stress (melodic) preference?

## Ideas
- token level punc marks VS. string level (this does not account for text-based emojis, etc..)
- add accented letters to letter counts?
- k-NN: if no majority vote, switch k value??
- KNN vs NN (1 vector/document VS 1 vector/author k=1)

## Meeting notes:
- sentence vector
- mixed bigram
- MUD
- word stat normalizations