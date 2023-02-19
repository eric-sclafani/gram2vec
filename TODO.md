# TODO
This file contains ideas for future additions/improvements
------

## Additions:
- Add discourse type evaluation for PAN
- syntax feats: counts of certain dependency labels, finite verb subject ommision, type of rel clause, 
- morph: embedded finite vs non-finite clauses (I hope that I can leave tomorrow vs I hope to leave tomorrow)
- prosody: do people have a stress (melodic) preference?
- emojis: look for ASCII emojis

## Improvements:

- Overhaul `pan_preprocess.py`
- Document preprocessing and vocab scripts
- change type annotations to use Typing module (for compatibility purposes)

## Issues:
- odd crash when turning certain features off while using cosine metric

## When changing data sets, the following need to be altered manually:
- get_dataset_name in `utils.py` has to be updated with a new dataset name conditional
- After generating vocabulary via `generate_non_static_vocab.py`, POS bigrams & mixed bigrams in `featurizers.py` vocabulary paths need to be set to new pickle file generated for the new dataset
- 

## To check out

### Corpora
- https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
- https://huggingface.co/datasets/guardian_authorship
- https://www.jmlr.org/papers/volume21/19-678/19-678.pdf
### Metric learning
- More metric learning stuff: https://www.semanticscholar.org/paper/Metric-Learning%3A-A-Support-Vector-Approach-Nguyen-Guo/51d8dabeb6d4aad285f5f5765de8baff771f5693
### Other
- https://www.cs.waikato.ac.nz/ml/weka/index.html

## Before I gradiate:
- add very detailed documentation of project