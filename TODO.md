# TODO
This file contains ideas for future additions/improvements

------

## Evals:
- dim reduction experiments

## CURRENT REFACTOR:
- move config to json
- Revise doc strings
- Option to vectorize a list of documents (independent of IDs) OR a directory of JSONL files (will expect IDs)
- At the end of refactor, if I use nlp.pipe, finetune the multiprocessing options
- Possibly add unit tests?
- Look into better ways to package the code
    - make gram2vec pip installable?
- make grammarvectorize a function instead?
- test time and performance between small and medium spacy english models
- after code is working, convert the lists in `load_spacy` to generators instead and see if that provides a speed boost
- refresh non_static_vocab parameter
- have vectorizer import generate_non_static_vocab
- experiment with size of n for mixed bigrams and pos bigrams
- add subsection about vocab to the readme
    - also, mentioned that vocab files have to have the same name as counting function

## Code improvements:
- Verbalizer is fine for now. Touch base with Ansh in near future

## Other:
- Start preparing the features for when they need to be multilingual



## To check out
- None

## PAN22 paper
https://publications.aston.ac.uk/id/eprint/44368/1/Stamatatos_2022_VoR.pdf
