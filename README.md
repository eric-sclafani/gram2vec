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

```python3
from gram2vec.featurizers import GrammarVectorizer

g2v = GrammarVectorizer()

```




### `Internal KNN Evaluation`
Before using `K-NN`, you can play with the `config.toml` file. This file acts as an "on/off" switch for each feature (`1 = ON`, `0 = OFF`). I've left everything ON for now. The point of this is to perform ablation studies, examing the effects that turning certain features off has on the accuracy scores. The file gets automatically read by `featurizers.py`.


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

Running `python3 knn_classifier.py` will give you the overall accuracy score, which currently sits at **20.35%**. Changing the `k-value` doesn't change the accuracy too much. I found `7` to be a good sweetspot. 


### **2. Bin accuracy**

Instead of overall accuracy, you can run `./evaluate_bins.sh` to evaluate the bins. The script just loops through the dev bins in the `data/pan/dev_bins/sorted_by_docfreq/` directory and applies K-NN classification. You can modify the arguments inside the script if desired. 


## Slides
https://drive.google.com/file/d/1BQ0vffPEvFJsoAR_lO3oZ-YvOCpo596E/view
