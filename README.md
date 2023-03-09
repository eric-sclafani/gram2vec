# Gram2Vec

## Description
`Gram2Vec` is a feature extraction algorithm that extracts grammatical properties from a given document and returns vectors representing that document's grammatical footprint. This is one part of PAUSIT team's stylistic feature vectors for TA1. 


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
spacy = "^3.4.4"
nltk = "^3.8"
ipdb = "^0.13.11"
ijson = "^3.1.4"
metric-learn = "^0.6.2"
jsonlines = "^3.1.0"
more-itertools = "^9.0.0"
matplotlib = "^3.7.0"
seaborn = "^0.12.2"
```
## Usage

### `GrammarVectorizer`

Import the **GrammarVectorizer** class and create an instance like so:
```python3
>>> from gram2vec.featurizers import GrammarVectorizer
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
>>> g2v.config
["pos_bigrams", "func_words", "letters", "dep_labels", "mixed_bigrams"]
```

From here, use the **g2v.vectorize()** method on an input string. By default, this will return a `numpy array`:
```python
>>> my_document = """Four score and seven years ago our fathers brought forth, upon this continent, a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived, and so dedicated, can long endure. We are met on a great battle field of that war. We come to dedicate a portion of it, as a final resting place for those who died here, that the nation might live"""
>>> my_vector = g2v.vectorize(my_document)
>>> my_vector
array([ 6.93069307e-02,  7.92079208e-02,  6.93069307e-02,  
        4.95049505e-02, 3.96039604e-02,  1.18811881e-01, 
        0.00000000e+00,  1.78217822e-01, 1.98019802e-02,
        ...
      ])
>>> my_vector.shape
(707,)
```
You can also use the **g2v.vectorize_episode()** method to vectorize a list of documents. This will return a 2D matrix of document vectors:
```python
>>> docs = [
    "This is an extremely wonderful string",
    "The string below me is false.",
    "The string above me is true"
    ]
>>> g2v.vectorize_episode(docs)
array([[0.16666667, 0., 0.16666667, ..., 0., 0.,0.],
       [0.14285714, 0.14285714, 0., ..., 0., 0.,0. ],
       [0.16666667, 0.16666667, 0., ..., 0., 0.,0.]
       ])
```

Optionally, the *return_obj* parameter can be switched to **True** in order to return a **FeatureVector** object instead:
```python
>>> my_vector = g2v.vectorize(my_document, return_obj=True)
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

There are two ways to evaluate using `kNN` currently. The second one is specific to PAN 2022.

### **1. Overall scores**

To get the R@1 or R@8 scores, use the `knn_classifier.py` module:
```
python3 eval/knn_classifer.py -h

usage: knn_classifer.py [-h] [-k K_VALUE] [-m {R@1,R@8}] [-train TRAIN_PATH] [-eval EVAL_PATH]

optional arguments:
  -h, --help            show this help message and exit

  -k K_VALUE, --k_value K_VALUE
                        k value to calculate R@1. Is ignored when --metric == R@8
                        default=6

  -m {R@1,R@8}, --metric {R@1,R@8}
                        Metric to calculate
                        default="R@1"

  -train TRAIN_PATH, --train_path TRAIN_PATH
                        Path to train directory
                        default="eval/pan22_splits/knn/train.json"

  -eval EVAL_PATH, --eval_path EVAL_PATH
                        Path to eval directory
                        default="eval/pan22_splits/knn/dev.json"
```




### **2. PAN2022 Bin scores**

Additionally, you can run `./eval/evaluate_bins.sh` to evaluate the development bins. The script just loops through the bins located in the **eval_bins/sorted_by_doc_freq/** directory and applies kNN classification. You can modify the arguments inside the script if desired. The results are stored in **eval/results/pan_dev_bin_results.csv**. 

<div class="alert alert-block alert-danger"><b>Note:</b> Each time the script is ran, <b>eval/results/pan_dev_bin_results.csv</b> gets overridden with the newest 8 bin evaluation scores.</div>



## Slides
(from site visit)

https://docs.google.com/presentation/d/1zAbZ7iwBXwmOo-y-QxSauo8PD5CwHpueToR6yXpnmLM/edit#slide=id.p
