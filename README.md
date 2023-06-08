# Gram2Vec

## Description
`Gram2Vec` is a feature extraction algorithm that extracts grammatical properties from a given document and returns vectors representing that document's grammatical footprint. This is one part of PAUSIT team's stylistic feature vectors for TA1. 

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


Next, you need to download spacy's medium size English language model:
```bash
python3 -m spacy download en_core_web_md   
```
## Usage

### `GrammarVectorizer`

Import the **GrammarVectorizer** class and create an instance like so:
```python
>>> from gram2vec.vectorizer import GrammarVectorizer # exact import may vary depending on where you're calling this module from
>>> g2v = GrammarVectorizer() 
```

The **GrammarVectorizer** instance can also be supplied a configuration to disable or enable features from activating. This configuration is a dictionary that maps `activated` features to `1` and `deactivated` features to `0`. 

By default, all features are activated. You can access the currently activated features through the `.config` attribute, which returns a list of strings.

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
