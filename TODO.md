# TODO
This file contains ideas for future additions/improvements
------

## Additions:
- Add discourse type evaluation for PAN
- syntax feats: counts of certain dependency labels, finite verb subject ommision, type of rel clause, 
- morph: embedded finite vs non-finite clauses (I hope that I can leave tomorrow vs I hope to leave tomorrow)
- prosody: do people have a stress (melodic) preference?
- NEED TO LOOK AT NORMALIZING AGAIN
- emojis: look for ASCII emojis

## Ideas
- k-NN: if no majority vote, switch k value??
- KNN vs NN (1 vector/document VS 1 vector/author k=1)
- look into adding special rules for dep parser for PAN data (so I dont have to do information injection)


## Code rework next steps
- move counter functions to own module?
- Get FeatureVector working
- Remake GrammarVectorizer and config file handling

## To check out

### Corpora
- https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
- https://huggingface.co/datasets/guardian_authorship
- https://www.jmlr.org/papers/volume21/19-678/19-678.pdf
### Metric learning
- More metric learning stuff: https://www.semanticscholar.org/paper/Metric-Learning%3A-A-Support-Vector-Approach-Nguyen-Guo/51d8dabeb6d4aad285f5f5765de8baff771f5693
### Other
- https://www.cs.waikato.ac.nz/ml/weka/index.html