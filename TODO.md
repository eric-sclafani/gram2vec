# TODO
This file contains ideas for future additions/improvements

------

## Evals:
- dim reduction experiments

## CURRENT REFACTOR:
- move config back to a toml file
- Make `vectorizer.py` call `generate_non_static_vocab` on the fly for when new datasets are used
- Revise all doc strings to use numpy format
- Look into better ways to package the code
- For the **Document** class in `vectorizer.py`, change the attributes to be properties or methods instead
- Option to vectorize a list of documents (independent of IDs) OR a directory of JSONL files (will expect IDs)
- Vocab: change non-static handling and convert pickle to text file delimited by newlines?
- check out those odd edge cases Zack found
- At the end of refactor, if I use nlp.pipe, finetune the multiprocessing options


## Code improvements:
- Verbalizer is fine for now. Touch base with Ansh in near future

## Other:
- Start preparing the features for when they need to be multilingual



## To check out
- None

## PAN22 paper
https://publications.aston.ac.uk/id/eprint/44368/1/Stamatatos_2022_VoR.pdf
