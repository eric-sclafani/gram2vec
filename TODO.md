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
- `NEW EVAL PLAN`:
    - Regarding making the document pairs, I should just create them from my already existing train, dev, test splits instead of using the raw data. Why? Because of the information injection actually.

    - Because the texts in my splits are modified (from the injection), it’ll actually be more involved to check if the document in the RAW data is in train, dev or test. The strings won't be the same because of the modification.


## When changing data sets, the following need to be altered manually:
- POS bigrams & mixed bigrams in `featurizers.py` vocabulary paths need to be set to new pickle file


## Code rework next steps
- debug feaurizers.py
- KNN:
    - make result file output better

## To check out

### Corpora
- https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
- https://huggingface.co/datasets/guardian_authorship
- https://www.jmlr.org/papers/volume21/19-678/19-678.pdf
### Metric learning
- More metric learning stuff: https://www.semanticscholar.org/paper/Metric-Learning%3A-A-Support-Vector-Approach-Nguyen-Guo/51d8dabeb6d4aad285f5f5765de8baff771f5693
### Other
- https://www.cs.waikato.ac.nz/ml/weka/index.html