# TODO
This file contains ideas for future additions/improvements

------

## Verbalizer
- Generate a quantitative description of count, OR
- Compare each feature to each feature in an averaged document feature vector (measure with how many standard deviations)(only extract features with n amount of stds from mean)

## Evals:
- dim reduction experiments

## Features:

- Contraction counts
- John David’s distinction between post-speech, co-speech and pro-speech emoji could be a cool feature
- relook at Document statistics feature (split into two vectors)
- https://aclanthology.org/2021.findings-emnlp.359/
- https://onlinelibrary.wiley.com/doi/full/10.1002/asi.21001?casa_token=yw5ePLow8pMAAAAA%3Al7T1qxyPzwjoxBcAc3uDE9RqFUSZTqaSJVoAmTkY3sUTy5iCAaWSF3dIe3YKmX1PLUaRceSH1QnvSP3Z
- https://ceur-ws.org/Vol-1178/CLEF2012wn-PAN-TanguyEt2012.pdf


## Code improvements:
- Make `featurizers.py` call `generate_non_static_vocab` on the fly for when new datasets are used
- Revise all doc strings to use numpy format
- 

## Other:
- Start preparing the features for when they need to be multilingual (likely requires a significant refactor)



## To check out
- None

## PAN22 paper
https://publications.aston.ac.uk/id/eprint/44368/1/Stamatatos_2022_VoR.pdf
