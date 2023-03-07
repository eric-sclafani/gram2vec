# TODO
This file contains ideas for future additions/improvements
------

## Before PR:

- write a readme inside of vocab/ describing static vs non-static
- make sure metric learn splitting works correctly
- discourse: just look at eval data

## Features:

- syntax feats: counts of certain dependency labels, finite verb subject ommision, type of rel clause, 
- morph: embedded finite vs non-finite clauses (I hope that I can leave tomorrow vs I hope to leave tomorrow)
- prosody: do people have a stress (melodic) preference?
- emojis: look for ASCII emojis
- John David’s distinction between post-speech, co-speech and pro-speech emoji could be a cool feature
- https://aclanthology.org/2021.findings-emnlp.359/
- relook at Document statistics feature (split into two vectors)
- contractions, emphasized text?, 



## Improvements:

- Document preprocessing and vocab scripts
- change type annotations to use Typing module (for compatibility purposes)

## Issues:
- odd crash when turning certain features off while using cosine metric (may be fixed? need to check)
- odd parenthesis crash happening in fix_data in pan_preprocess

## When changing data sets, the following need to be altered manually:
- get_dataset_name function has to be updated with a new dataset name conditional
- After generating vocabulary via `generate_non_static_vocab.py`, POS bigrams & mixed bigrams in `featurizers.py` vocabulary paths need to be set to new pickle file generated for the new dataset


## To check out

### Corpora
- https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
- https://huggingface.co/datasets/guardian_authorship

## PAN22 paper
https://publications.aston.ac.uk/id/eprint/44368/1/Stamatatos_2022_VoR.pdf
