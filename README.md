# Gram2Vec

## Description
`Gram2Vec` is a vectorization algorithm that embeds documents into a higher dimensional space by extracting the normalized relative frequencies of stylistic features present in the text. More specifically, Gram2Vec vectorizes based off feartures pertaining to grammar, such as POS tags, punctuation, syntactic constructions (WIP), and much more.

## Setup

In your working directory, create an environment by running (I think any version > 3.9 should work, not 100% sure though):
```bash
python3.11 -m venv venv/
source venv/bin/activate
```
which will create a directory called `venv/` which will store all the dependencies. 

After cloning `gram2vec` into your working directory, run:
```bash
pip install gram2vec/
```
which will install gram2vec into your environment, as well as all of its dependencies.

## Usage

There are two options for calling the vectorizer.

The first option, `vectorizer.from_jsonlines()`, is used to generate a dataframe from **either a single .jsonl file** _OR_ **a directory of .jsonl files**.

```python
>>> from gram2vec import vectorizer
>>> my_df = vectorizer.from_jsonlines("path/to/dataset/data.jsonl")
>>> my_df = vectorizer.from_jsonlines("path/to/dataset/directory/")
```

The second option,`vectorizer.from_documents()`, is used to generate a dataframe **from a list of strings**. Note that this does NOT take into account author or document IDs, unlike the `.from_jsonlines()` function.
```python
>>> from gram2vec import vectorizer
>>> documents = [
    "This is a test string 😄!!!",
    "The string below me is false.",
    "The string above me is true 😱!"
]
>>> my_df = vectorizer.from_documents(documents)
```


You can also enable or disable select feature extractors by using the `config` parameter, which takes a dictionary of feature names mapped to 1 or 0 (1 = ON, 0 = OFF). 

By default, all are on **EXCEPT** for `mixed_bigrams`, which adds 1200 dimensions and slows the processing speed down considerably. I recommend leaving it off. Here's an example of what a configuration looks like:

```python
config = {
    "pos_unigrams":1,
    "pos_bigrams":0,
    "func_words":1,
    "punctuation":1,
    "letters":0,
    "emojis":1,
    "dep_labels":1,
    "mixed_bigrams":0,
    "morph_tags":1
    }
my_df = vectorizer.from_jsonlines("path/to/dataset/directory/", config=config)
```

Additionally, there is an option to include the document embedding produced by **word2vec**. This option should `ONLY` be used for experimenting, `NOT` official authorship attribution evaluations. 

The purpose of this is to test how well the grammatical stylistic features perform during authorship attribution with and without the embedding. The point of stylistic feature extraction is to create vectors `100% agnostic of content`, only capturing the style from documents. Since we know that **word2vec** embeddings do include content, they are useful to comparse `gram2vec` vectors to.
```python
my_df = vectorizer.from_jsonlines("path/to/dataset/directory/", include_content_embedding=True)
```

## Vocab

This section provides more details about how vocabulary works in `gram2vec` and is not needed to understand how to use the software.

In general, each feature is frequency based (I will be adding regex matching ones soon). A **vocab** is therefore the collection of items that get counted for a feature. Each vocab is stored in a local `vocab/` directory. These files are read by `gram2vec` and used in the feature extractors.

If new vocabularies are added, for the sake of consistancy, the vocabulary files should have the same name as the feature function. Examples of this can be seen in `vectorizer.py`.

Some vocabularies require more explanation. The following subsections go into more detail about them

### POS Bigrams

From the list of POS tags from <a href="https://universaldependencies.org/u/pos/">Universal dependencies</a> (18 total tags), I create all possible combinations. 
> So $18^2$ = 324 possible POS bigrams

### Mixed bigrams

From a list of closed class words aggregated from the internet, I take all possible combinations of them (100 words) with open class POS tags (6 tags). Order is also important, i.e. ("the", "NOUN") is _not_ the same as ("NOUN", "the")
> So 100 closed class words * 6 POS tags * 2 possible orderings per bigram = 1200 possible mixed bigrams 

`NOTE`: this feature is heavily WIP and is disabled by default due to it's size. It increases the processing time significantly. 
