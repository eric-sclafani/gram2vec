# TODO
This file contains ideas for future additions/improvements

------

## Evals:
- experiment with spotify annoy

## TODO before pip installation:
- Clean up vocab handling
- https://carpentries-incubator.github.io/python_packaging/instructor/05-publishing.html

## TODO eventually
- Add unit tests


## SRM (Syntax Regex Matcher):
- SRM will eventually be ported to its own repo and thus be its own module
- For now, the focus is on getting a working system for linear dependency tree matching. Code simplification and refinement can be focused on at a later time
- Have an option for each match to tell you the exact sentence index span that the match found in the original sentence (not the linear tree). This will require a lot more work, as the regex matching happens on the _linear tree_, not the original sentence 
- Documentation (including the regex pattern testing process)
- The \<word\> part of the linear tree schema (\<word\>-\<lemma\>-\<pos\>-\<deplink\>) may not be needed. Could be removed, which would make the regex slightly more readable (and possibly optimize it a bit?). I say this because it has not played a part in any of the regex I've implemented yet
- Think about adding a CLI for the pattern matching?
- Add tag questions

### Known issues:

- `General`
    - In sentences where the _same pattern_ occurs more than once, the regex counts it as **one** occurence, _not_ two. I'm not sure if this is an issue with the regex itself, or my python implementation. This is a priority \#1 issue.

- `It-clefts`:
    - Temporal it-clefts are not parsed correctly. 

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
