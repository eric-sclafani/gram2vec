# TODO
This file contains ideas for future additions/improvements
------

## `pan_preprocess.py` overhaul:
- BEFORE OVERHAUL: write a quick script to get all discourse type counts. Then, write a better one later
- From the raw data, create a new jsonlines file with the following format: 
```python
{"author_id": id, "text": document, "discourse_type":discourse_type},
...             
```
- Create a "sorted" and "fixed_sorted" version of the above. This is essentially merging pairs.jsonl and truths.jsonl
- In `knn_classifier.py`, write a function that sorts by author_id, 


## Additions:
- syntax feats: counts of certain dependency labels, finite verb subject ommision, type of rel clause, 
- morph: embedded finite vs non-finite clauses (I hope that I can leave tomorrow vs I hope to leave tomorrow)
- prosody: do people have a stress (melodic) preference?
- emojis: look for ASCII emojis

## Improvements:

- Document preprocessing and vocab scripts
- change type annotations to use Typing module (for compatibility purposes)

## Issues:
- odd crash when turning certain features off while using cosine metric
- odd parenthesis crash happening in fix_data in pan_preprocess

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