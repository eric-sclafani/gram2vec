# Gram2Vec

## Description
`Gram2Vec` is a feature extraction algorithm that extracts grammatical and syntactic properties from a given document and returns a 1-dimensional vector. This is one part of the PAUSIT team's **Linguistic Indicator Vector** (LIV). 


## Setup

If you use poetry, run `poetry install`
inside **gram2vec/**. It should install all dependencies in `pyproject.toml`. Then run `poetry shell`  to spawn the venv.

Alternatively, if you don't use poetry, the bare minimum dependencies are:
```toml
python = "^3.8"
demoji = "^1.1.0"
numpy = "^1.24.0"
pandas = "^1.5.2"
scikit-learn = "^1.2.0"
metric-learn = "^0.6.2"
jsonlines = "^3.1.0"
spacy = "^3.4.4"
nltk = "^3.8"
toml = "^0.10.2"
ijson = "^3.1.4"
```
## Usage

### `GrammarVectorizer`

Before using the vectorizer, you can disable or enable features from activating. The `config.toml` configuration file maps `activated` features to `1` and deactivated features to `0`.
```toml
[Features]
pos_unigrams=1
pos_bigrams=1
func_words=0
punc=1
letters=0
common_emojis=1
embedding_vector=1
document_stats=0
dep_labels=1
mixed_bigrams=1
```
Import the **GrammarVectorizer** class and create an instance like so:
```python3
>>> from gram2vec.featurizers import GrammarVectorizer
>>> g2v = GrammarVectorizer()
```
From here, use the **g2v.vectorize()** method on an input string. By default, this will return a `numpy array` of all feature vectors concatenated into one:
```python
>>> my_vector = g2v.vectorize("""Four score and seven years ago our fathers brought forth, upon this continent, a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived, and so dedicated, can long endure. We are met on a great battle field of that war. We come to dedicate a portion of it, as a final resting place for those who died here, that the nation might live""")
>>> my_vector
array([ 6.93069307e-02,  7.92079208e-02,  6.93069307e-02,  
        4.95049505e-02, 3.96039604e-02,  1.18811881e-01, 
        0.00000000e+00,  1.78217822e-01, 1.98019802e-02,
        ...
      ])
>>> my_vector.shape
(707,)
```

Optionally, the *return_vector* parameter can be switched to **False** in order to return a **FeatureVector** object instead:
```python
>>> my_vector = g2v.vectorize("""Four score and seven years ago our fathers brought forth, upon this continent, a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived, and so dedicated, can long endure. We are met on a great battle field of that war. We come to dedicate a portion of it, as a final resting place for those who died here, that the nation might live""", return_vector=False)
>>> my_vector
'<gram2vec.featurizers.FeatureVector object at 0x15aacc6d0>'
```
The **FeatureVector** class gives access to the following methods:

- `.vector` - returns the concatenated 1D vector
- `.get_vector_by_feature` (*feature_name*) - returns the vector of a specific feature given that feature's name
- `.get_counts_by_feature` (*feature_name*) - returns the count dict of a specific feature given that feature's name

```python
>>> my_vector.vector
array([ 6.93069307e-02,  7.92079208e-02,  6.93069307e-02,  
        4.95049505e-02, 3.96039604e-02,  1.18811881e-01, 
        0.00000000e+00,  1.78217822e-01, 1.98019802e-02,
        ...
      ])
>>> my_vector.get_vector_by_feature("pos_unigrams")

array([0.06930693, 0.07920792, 0.06930693, 0.04950495, 0.03960396,
       0.11881188, 0.        , 0.17821782, 0.01980198, 0.00990099,
       0.07920792, 0.        , 0.12871287, 0.02970297, 0.        ,
       0.12871287, 0.        , 0.        ])

>>> my_vector.get_counts_by_feature("pos_unigrams")

{'ADJ': 7, 'ADP': 8, 'ADV': 7, 'AUX': 5, 'CCONJ': 4, 'DET': 12, 'INTJ': 0, 'NOUN': 18, 'NUM': 2, 'PART': 1, 'PRON': 8, 'PROPN': 0, 'PUNCT': 13, 'SCONJ': 3, 'SYM': 0, 'VERB': 13, 'X': 0, 'SPACE': 0}
```

### `Internal KNN Evaluation`

There are two ways to evaluate the PAN 2022 data set using `K-NN`:

### **1. Overall accuracy**

To get the overall accuracy, use the `knn_classifier.py` module:
```
usage: knn_classifer.py [-h] [-k K_VALUE] [-m METRIC] [-train TRAIN_PATH] [-eval EVAL_PATH]

options:
  -h, --help            show this help message and exit


  -k K_VALUE, --k_value K_VALUE
                        k value for K-NN
                        DEFAULT = 7

  -m METRIC, --metric   METRIC
                        distance metric
                        DEFAULT = "cosine"

  -train TRAIN_PATH, --train_path TRAIN_PATH
                        Path to train data
                        DEFAULT = "data/pan/train_dev_test/train.json"

  -eval EVAL_PATH, --eval_path EVAL_PATH
                        Path to eval data
                        DEFAULT = "data/pan/train_dev_test/dev.json"
```

Running `python3 knn_classifier.py` will give you the overall accuracy score, which currently sits at **21.07%**.  I found `cosine similarity` and `k = 7` to be a good sweetspot. 


### **2. Bin accuracy**

Instead of overall accuracy, you can run `./evaluate_bins.sh` to evaluate the bins. The script just loops through the dev bins in the `data/pan/dev_bins/sorted_by_docfreq/` directory and applies K-NN classification. You can modify the arguments inside the script if desired. 


## Slides
https://drive.google.com/file/d/1BQ0vffPEvFJsoAR_lO3oZ-YvOCpo596E/view
