# TODO
This file contains ideas for future additions/improvements

------

## Future repo improvements:
- Use a config file for G2V again and store it in ../../../config/

## Evals:
- dim red

## Features:

- work more on syntactic feats
- John David’s distinction between post-speech, co-speech and pro-speech emoji could be a cool feature
- relook at Document statistics feature (split into two vectors)
- https://aclanthology.org/2021.findings-emnlp.359/
- https://onlinelibrary.wiley.com/doi/full/10.1002/asi.21001?casa_token=yw5ePLow8pMAAAAA%3Al7T1qxyPzwjoxBcAc3uDE9RqFUSZTqaSJVoAmTkY3sUTy5iCAaWSF3dIe3YKmX1PLUaRceSH1QnvSP3Z
- https://ceur-ws.org/Vol-1178/CLEF2012wn-PAN-TanguyEt2012.pdf


## Code improvements:
- Make `featurizers.py` call `generate_non_static_vocab` on the fly

## When changing data sets, the following need to be altered manually:
- get_dataset_name function has to be updated with a new dataset name conditional
- After generating vocabulary via `generate_non_static_vocab.py`, POS bigrams & mixed bigrams in `featurizers.py` vocabulary paths need to be manually set to new pickle file generated for the new dataset


## To check out
- None

## PAN22 paper
https://publications.aston.ac.uk/id/eprint/44368/1/Stamatatos_2022_VoR.pdf
