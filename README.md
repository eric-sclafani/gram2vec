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

If you'd like to play around with the source code, I recommend installing it with the `-e` flag which installs an editable version of the package (so you don't have to pip install every time you modify the source code):
```bash
pip install -e gram2vec/
```

## Usage

### Vectorizer

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
>>> vectorizer.from_documents(documents)
```
|   pos_unigrams:ADJ   |   pos_unigrams:ADP   |   pos_unigrams:ADV   | ... |   sentences:obj-relcl   |   sentences:tag-question   |   sentences:coordinate-clause   |
|----------------------|----------------------|----------------------|-----|------------------------|---------------------------|-------------------------------|
|           0.000000   |           0.000000   |           0.0        | ... |           0.0          |           0.0             |           0.0                 |
|           0.142857   |           0.142857   |           0.0        | ... |           0.0          |           0.0             |           0.0                 |
|           0.142857   |           0.142857   |           0.0        | ... |           0.0          |           0.0             |           0.0                 |

You can also enable or disable select feature extractors by using the `config` parameter, which takes a dictionary of feature names mapped to 1 or 0 (1 = ON, 0 = OFF). 

By default, `all features are activated`. Here's an example of what a configuration looks like:

```python
>>> config = {
    "pos_unigrams":1,
    "pos_bigrams":0,
    "func_words":1,
    "punctuation":1,
    "letters":0,
    "emojis":1,
    "dep_labels":1,
    "morph_tags":1,
    "sentences":1
    }
>>> my_df = vectorizer.from_jsonlines("path/to/dataset/directory/", config=config)
```

Additionally, there is an option to include the document embedding produced by **word2vec**. This option should `ONLY` be used for experimenting, `NOT` official authorship attribution evaluations. 

The purpose of this is to test how well the grammatical stylistic features perform during authorship attribution with and without the embedding. The point of stylistic feature extraction is to create vectors `completely independent of content`, only capturing the style from documents. Since we know that **word2vec** embeddings do include content, they are useful to compare `gram2vec` vectors to.

```python
>>> my_df = vectorizer.from_jsonlines("path/to/dataset/directory/", include_content_embedding=True)
```

### Verbalizer

The goal of the verbalizer is to calculate zscores from a given grammatical feature vector dataframe and produce string representations for the most "salient" features. Given a dataset, `Verbalizer` will calculate and store the zscores for each row. This works on the document and author levels.

The theory behind this is that when calculating the zscores for each feature, that will tell us `how many standard deviations away that feature is from the average feature`. We can then answer questions like: "which features does author A use frequently that sets them apart from authors B, C, and D?". This also works on the document level too. Given an **unseen document vector**, we can also answer: "how much does it's features deviate from the other document vectors?"

Additionally, a **threshold** value is used to select the zscores that deviate from the mean that many times. By default, this value is `2.0`

To get started, import both vectorizer and verbalizer from gram2vec. `Verbalizer` needs a dataframe with **authorIDS** and **documentID** fields included. You can use `vectorizer.from_jsonlines()` which will include them automatically, or use `vectorizer.from_documents()` and manually add those required columns to the dataframe.
```python
>>> from gram2vec import vectorizer, verbalizer

>>> my_df = vectorizer.from_jsonlines("path/to/dataset/directory/") 
>>> verbalized = verbalizer.Verbalizer(my_df)
```
You can also change the zscore threshold if desired:
```python
>>> verbalized = verbalizer.Verbalizer(my_df, zscore_threshold=2.5)
```

Zscores and verbalizations can be done on the `author` and `document` levels. For the author level, the `.verbalize_author()` method is used. It accepts a unique author id and returns a dataframe with the feature names, zscores, and verbalizations as columns:
```python

>>> verbalized.verbalize_author_id("en_112")
>>> verbalized.head(5)
```
| index |      feature_name     |  zscore  |                                     verbalized                                          |
|-------|-----------------------|----------|-------------------------------------------------|
|   0   | pos_unigrams:ADV      | 2.437037 | This author uses more pos_unigrams:ADV than the average author                                 
|   1   | pos_bigrams:ADV ADP   | 2.759779 | This author uses more pos_bigrams:ADV ADP than the average author                             
|   2   | pos_bigrams:ADP SYM   | 2.192766 | This author uses more pos_bigrams:ADP SYM than the average author                              
|   3   | pos_bigrams:PUNCT AUX | 2.714269 | This author uses more pos_bigrams:PUNCT AUX than the average author                        
|   4   | pos_bigrams:DET SCONJ | 7.416198 | This author uses more pos_bigrams:DET SCONJ than the average author                       
|   5   | pos_bigrams:DET SYM   | 2.238211 | This author uses more pos_bigrams:DET SYM than the average author 

To verbalize **unseen documents**, use the `.verbalize_document_vector()` method. This function takes an unseen *document vector* as input and calculates the zscores and verbalized string for it with respect to the data the `Verbalizer` data is initially fit with:
```python
>>> my_df = vectorizer.from_jsonlines("path/to/dataset/directory/") # this is essentially the "training data"
>>> verbalized = verbalizer.Verbalizer(my_df)
>>> verbalized.verbalize_document_vector(my_unseed_doc_vector_here) # unseen document vector
```

| index |         feature_name        |  zscore   |                                    verbalized                                   |
|-------|----------------------------|-----------|---------------------------------------------------------------------------------|
|  30   |          emojis:🥰            | 5.518523  | This document uses more emojis:🥰 than the average document            |
|  31   |       dep_labels:cc          | 2.400670  | This document uses more dep_labels:cc than the average document       |
|  32   |     dep_labels:meta        | 4.329617  | This document uses more dep_labels:meta than the average document    |
|  33   | morph_tags:ConjType=Cmp | 2.118285  | This document uses more morph_tags:ConjType=Cmp than the average document |
|  34   | sentences:coordinate-clause | 2.517907  | This document uses more sentences:coordinate-clause than the average document |

## Vocab

This section provides more details about how vocabulary works in `gram2vec` and is not needed to understand how to use the software.

In general, each feature is frequency based. A **vocab** is therefore the collection of items that get counted for a feature. Each vocab is stored in a local `vocab/` directory. These files are read by `gram2vec` and used in the feature extractors.

If new vocabularies are added, for the sake of consistency, the vocabulary files should have the same name as the feature function. Examples of this can be seen in `vectorizer.py`.

Some features, in particular the `sentences` feature produced by `SyntaxRegexMatcher`, do not require a vocab (at least, not in the same way the others do).

Some vocabularies require more explanation. The following subsections go into more detail about them

### POS Bigrams

From the list of POS tags from <a href="https://universaldependencies.org/u/pos/">Universal dependencies</a> (18 total tags), I create all possible combinations. 
> So $18^2$ = 324 possible POS bigrams

## Adding more features

If you'd like to extend the code and add more countable features, here is a detailed guide on how to do so.

### Step 1

Define what you want to count and why. What is the intuition behind it and what could it tell you about authors' writing styles?

### Step 2

Define a vocabulary. This will be some collection of countable objects. In the [vocab](src/gram2vec/vocab) directory, the vocabularies are just text files of countable items. Alternatively, you may want to add a regex matching feature similar to how my other package, [Syntax Regex Matcher](https://github.com/eric-sclafani/syntax-regex-matcher), works.

## Step 2.5 (SECTION WORK IN PROGRESS)

If possible, create a custom spaCy extension for your countable items. It makes the code cleaner (imo) and works well. See the [load_spacy.py](src/gram2vec/load_spacy.py) functions as examples. This functionality is not a requirement, as not all of my features make use of spaCy extensions.

## Step 3

Inside of [vectorizer.py](src/gram2vec/vectorizer.py)