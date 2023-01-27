# TODO
This file contains ideas for future additions/improvements
------

## Additions:
- Add discourse type evaluation for PAN
- look into adding special rules for dep parser for PAN data (so I dont have to do information injection)
- syntax feats: counts of certain dependency labels, finite verb subject ommision, type of rel clause, 
- morph: embedded finite vs non-finite clauses (I hope that I can leave tomorrow vs I hope to leave tomorrow)
- prosody: do people have a stress (melodic) preference?
- NEED TO LOOK AT NORMALIZING AGAIN
- emojis: look for ASCII emojis

## Ideas
- k-NN: if no majority vote, switch k value??
- KNN vs NN (1 vector/document VS 1 vector/author k=1)


## Code rework next steps (1/26)
- figure out vocab generation and streamline the process
- Get FeatureVector working
- Remake GrammarVectorizer and config file handling

## To check out
- https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
- https://huggingface.co/datasets/guardian_authorship
- https://www.jmlr.org/papers/volume21/19-678/19-678.pdf
- More metric learning stuff: https://www.semanticscholar.org/paper/Metric-Learning%3A-A-Support-Vector-Approach-Nguyen-Guo/51d8dabeb6d4aad285f5f5765de8baff771f5693
