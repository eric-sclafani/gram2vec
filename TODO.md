# TODO
This file contains ideas for future additions/improvements

------

## Evals:
- dim reduction experiments

## CURRENT REFACTOR:
- Revise doc strings
- add polars into the string calculations
- Option to vectorize a list of documents (independent of IDs) OR a directory of JSONL files (will expect IDs)
- At the end of refactor, if I use nlp.pipe, finetune the multiprocessing options
- add unit tests
- Look into better ways to package the code
    - make gram2vec pip installable?
- test time and performance between small and medium spacy english models
- experiment with size of n for mixed bigrams and pos bigrams
- add subsection about vocab to the readme
    - also, mentioned that vocab files have to have the same name as counting function
- add subsection in readme about how some features need access to vocab, while some others dont

## Code improvements:
- Verbalizer is fine for now. Touch base with Ansh in near future

## Other:
- Start preparing the features for when they need to be multilingual



## To check out
- None

## PAN22 paper
https://publications.aston.ac.uk/id/eprint/44368/1/Stamatatos_2022_VoR.pdf
