# Gram2Vec

## Description
`Gram2Vec` is a vectorization algorithm that embeds documents into a higher dimensional space by extracting the normalized relative frequencies of stylistic features present in the text. More specifically, Gram2Vec vectorizes based off feartures pertaining to grammar, such as POS tags, syntactic constructions, and much more.

## Setup

Create an environment by running:
```bash
python3.11 -m venv venv/
source venv/bin/activate
```
Which will create a directory called `venv/` which will store all the dependencies. Now run:
```bash
pip3 install -r requirements.txt
```
Which will install all the dependencies for the project.

Note: for the `spacy` installation, I have the M1 mac version installed. If spacy throws you an error, you may need to install the version specific to your PC [https://spacy.io/usage](https://spacy.io/usage)

## Usage

Before using the vectorizer, you can enable or disable select feature extractors. By default, the `config.json` file to do this is located in `./gram2vec/gram2vec/`. 1 indicates the feature is activated, 0 means its deactivated:

<small>config.json</small>
```json
{
    "pos_unigrams":1,
    "pos_bigrams":0,
    "func_words":1,
    "punctuation":0,
    "letters":1,
    "emojis":1,
    "dep_labels":0,
    "mixed_bigrams":1,
    "morph_tags":1
}  
```

There are two options for calling the vectorizer.

The first option, `vectorizer.from_jsonlines()`, can be used to generate a matrix from either a single .jsonl file OR a directory of .jsonl files:

```python
>>> from gram2vec.gram2vec import vectorizer # exact import may vary depending on where you're calling this module from
>>> my_matrix = vectorizer.from_jsonlines("path/to/dataset/data.jsonl")
>>> my_matrix = vectorizer.from_jsonlines("path/to/dataset/directory/")
```



For example:
```python
>>> G2V_CONFIG = {
    "pos_unigrams":0,
    "pos_bigrams":1,
    "func_words":1,
    "punc":0,
    "letters":1,
    "common_emojis":0,
    "embedding_vector":0,
    "document_stats":0,
    "dep_labels":1,
    "mixed_bigrams":1,
} 
>>> g2v = GrammarVectorizer(G2V_CONFIG)
>>> g2v.get_config()
["pos_bigrams", "func_words", "letters", "dep_labels", "mixed_bigrams"]
```

From here, you can use the **g2v.create_vector_df()** method to vectorize a list of documents and store everything in a dataframe:
```python
>>> docs = [
    "This is an extremely wonderful string",
    "The string below me is false.",
    "The string above me is true"
    ]
>>> df = g2v.create_vector_df(docs)
>>> df.shape
(3, 429) # 3 document vectors, 429 features each
```

## Vocab

### POS Bigrams

From the list of POS tags from <a href="https://universaldependencies.org/u/pos/">Universal dependencies</a> (18 total tags), I create all possible combinations. 
> So $18^2$ = 324 possible POS bigrams

### Mixed bigrams

From a list of closed class words aggregated from the internet, I take all possible combinations of them (100 words) with open class POS tags (6 tags). Order is also important, i.e. ("the", "NOUN") is _not_ the same as ("NOUN", "the")
> So 100 closed class words * 6 POS tags * 2 possible orderings per bigram = 1200 possible mixed bigrams 
