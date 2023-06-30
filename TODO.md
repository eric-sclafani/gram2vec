# TODO
This file contains ideas for future additions/improvements

------

## Evals:
- experiment with spotify annoy

## TODO before pip installation:
- Make unit tests
- https://carpentries-incubator.github.io/python_packaging/instructor/05-publishing.html


## Treegex:
- Treegex will eventually be ported to its own repo and thus be its own module
- For now, the focus is on getting a working system for linear dependency tree matching. Code simplification and refinement can be focused on at a later time
- Experiment with spacy large en model
    - this would require updating regex to match the large model's parse trees
- Two user endpoints: 
    1. Feed in one or more spaCy docs (if user already has an nlp instance elsewhere in their code) (useful for g2v)
    2. Feed in one or more strings (this will instantiate a new nlp instance)

- Remember to import the user api into __init__ in order to have a convenient import system

### Known issues:

- `General`
    - In sentences where the _same pattern_ occurs more than once, the regex counts it as **one** occurence, _not_ two. I'm not sure if this is an issue with the regex itself, or my python implementation. This is a priority \#1 issue.

- `It-clefts`:
    - Temporal it-clefts are not parsed correctly. May be fixed when switching to large model (needs experimentation)
    - It-clefts like "If it were John who is the candidate, I would vote for him." are not captured. 
- `Pseudo-clefts`:
    - "What I need is none of your business." is a false positive
- `Passives`:
    - In the following two sentences, "book" is being mislabeled as "nsubj" when it should be "nsubjpass":
        - "The book which was given to me by Mary was really interesting."
        - "The book given to me by Mary was really interesting"


## Other:
- multilingual 


## To check out
- add polars into the string calculations https://pola-rs.github.io/polars-book/user-guide/
- check out https://github.com/vi3k6i5/flashtext for potential speed improvements

## PAN22 paper
https://publications.aston.ac.uk/id/eprint/44368/1/Stamatatos_2022_VoR.pdf
