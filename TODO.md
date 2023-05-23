# TODO
This file contains ideas for future additions/improvements

------

## Evals:
- dim reduction experiments

## Features:

- Contraction counts
- John David’s distinction between post-speech, co-speech and pro-speech emoji could be a cool feature
- relook at Document statistics feature (split into two vectors)
- https://aclanthology.org/2021.findings-emnlp.359/
- https://onlinelibrary.wiley.com/doi/full/10.1002/asi.21001?casa_token=yw5ePLow8pMAAAAA%3Al7T1qxyPzwjoxBcAc3uDE9RqFUSZTqaSJVoAmTkY3sUTy5iCAaWSF3dIe3YKmX1PLUaRceSH1QnvSP3Z
- https://ceur-ws.org/Vol-1178/CLEF2012wn-PAN-TanguyEt2012.pdf
- mispellings?

## CURRENT REFACTOR:
- Make `vectorizer.py` call `generate_non_static_vocab` on the fly for when new datasets are used
- Revise all doc strings to use numpy format
- Look into better ways to package the code
- For the **Document** class in `vectorizer.py`, change the attributes to be properties or methods instead
- Option to vectorize a list of documents (independent of IDs) OR a directory of JSONL files (will expect IDs)
- Vocab: change non-static handling and convert pickle to text file delimited by newlines?
- check out those odd edge cases Zack found
- 


## Code improvements:
- Verbalizer is fine for now. Touch base with Ansh in near future

## Other:
- Start preparing the features for when they need to be multilingual



## To check out
- None

## PAN22 paper
https://publications.aston.ac.uk/id/eprint/44368/1/Stamatatos_2022_VoR.pdf
