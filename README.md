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




### `Internal KNN Evaluation`

--------------

NOTE: The evaluation setup inside this repository is no longer being used for TA1. That has been moved to the `pausit-eval` repository.

---------------


There are two ways to evaluate using `kNN` currently. The second one is specific to PAN 2022. 

 > **Note**: Both evaluation scripts below `must` be ran from the **~/gram2vec/gram2vec/** directory

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

> **Note**: Each time the script is ran, **eval/results/pan_dev_bin_results.csv** gets overridden with the newest 8 bin evaluation scores.



## Slides
(from site visit)

https://docs.google.com/presentation/d/1zAbZ7iwBXwmOo-y-QxSauo8PD5CwHpueToR6yXpnmLM/edit#slide=id.p
