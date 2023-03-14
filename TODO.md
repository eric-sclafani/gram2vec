# TODO
This file contains ideas for future additions/improvements

------

## Future repo improvements:
- Improve dependency handling, maybe bundle this code as a package?
- Add some kind of uniform data set loader that makes it easier to load / concatenate different data sets

## Evals:
- start playing with TA1 eval script


## Features:

- syntax feats: counts of certain dependency labels, finite verb subject ommision, type of rel clause, 
- morph: embedded finite vs non-finite clauses (I hope that I can leave tomorrow vs I hope to leave tomorrow)
- character n-grams
- prosody: do people have a stress (melodic) preference?
- emojis: look for ASCII emojis
- John David’s distinction between post-speech, co-speech and pro-speech emoji could be a cool feature
- https://aclanthology.org/2021.findings-emnlp.359/
- relook at Document statistics feature (split into two vectors)
- contractions, emphasized text?, 
- https://onlinelibrary.wiley.com/doi/full/10.1002/asi.21001?casa_token=yw5ePLow8pMAAAAA%3Al7T1qxyPzwjoxBcAc3uDE9RqFUSZTqaSJVoAmTkY3sUTy5iCAaWSF3dIe3YKmX1PLUaRceSH1QnvSP3Z
- https://ceur-ws.org/Vol-1178/CLEF2012wn-PAN-TanguyEt2012.pdf




## Improvements:

- Document preprocessing script
- change type annotations to use Typing module (for compatibility purposes)
- In generate_tabular_vector_dataset: normalize counts if needed

## Issues:
- odd crash when turning certain features off while using cosine metric (may be fixed? need to check)

## When changing data sets, the following need to be altered manually:
- get_dataset_name function has to be updated with a new dataset name conditional
- After generating vocabulary via `generate_non_static_vocab.py`, POS bigrams & mixed bigrams in `featurizers.py` vocabulary paths need to be manually set to new pickle file generated for the new dataset


## To check out

### Corpora
- https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm

## PAN22 paper
https://publications.aston.ac.uk/id/eprint/44368/1/Stamatatos_2022_VoR.pdf
