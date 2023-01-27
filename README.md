# Gram2Vec

## Note
This branch is stable. See `dev` branch for most recent changes and updates.

## Description
`Gram2Vec` is a feature extraction algorithm that extracts grammatical and syntactic properties from a given document and returns a 1-dimensional vector. This is one part of the PAUSIT team's **Linguistic Indicator Vector** (LIV).

## Disclaimer

The codebase is due to be refactored, although the functionality will remain the same. 

Please excuse the messy code in some parts of the project :) 

Also, this is my first time sharing code for someone else's consumption. **All feedback is highly welcome and encouraged!**

## Setup

I use Poetry to manage my dependencies. Run:
```bash
poetry install
```
in the directory of this project. It should install all dependencies in `pyproject.toml`. Then run:
```
poetry shell
```
to create a virtual environment for the project.

Alternatively, if you don't want to use poetry, the dependencies are as follows:
```toml
python = "^3.8"
demoji = "^1.1.0"
numpy = "^1.24.0"
pandas = "^1.5.2"
scikit-learn = "^1.2.0"
spacy = "^3.4.4"
matplotlib = "^3.6.2"
seaborn = "^0.12.1"
ipykernel = "^6.19.4" # <- only needed if you want to use visualize.ipynb
nltk = "^3.8"
toml = "^0.10.2"
pytest = "^7.2.0"
more-itertools = "^9.0.0"
ipdb = "^0.13.11"
ijson = "^3.1.4"
plotly = "5.11.0"
nbformat = "^5.7.1"
```
(Note: I haven't tested these instructions yet, so please tell me if they're incorrect/unclear!)

## Package structure

Below is a list of my files/directories inside of `gram2vec/`:

- `data/` - contains the PAN 2022 data (raw, preprocessed, train/dev/test splits, and dev bins), as well as my proprocessing scripts

- `logs/` - contains log files for each featurizer. Lets you see what each featurizer is capturing exactly

- `results/` - contains the results in for the form of json files (testing data has not been used yet)

- `vocab/` - contains vocabulary needed for some featurizers. They are generated automatically by `featurizers.py`

- `clear_logs.sh` - removes the log files. Only used when they get big (I plan to automate their removal every time the classifier is ran)

- `config.toml` - the desired configuration of features to use during K-NN classification

- `evaluate_bins.sh` - bash script that evaluates each bin (see the `Bin accuracy` subsection for more info)

- `featurizers.py` - module which houses all the features. This module gets called directly by the K-NN classifier

- `knn_clasifier.py` - the K-NN classifier. This module vectorizes input data and applies K-NN classification. Results are recorded in the `results/` directory

- `utils.py` - small module of miscellaneous helper functions

- `visualize.ipynb` - notebook used to create data visualizations


## Usage

Before using `K-NN`, you can play with the `config.toml` file. This file acts as an "on/off" switch for each feature (`1 = ON`, `0 = OFF`). I've left everything ON for now. The point of this is to perform ablation studies, examing the effects that turning certain features off has on the accuracy scores. The file gets automatically read by `featurizers.py`.


There are two ways to evaluate the PAN 2022 data set using `K-NN`:

### 1. Overall accuracy

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

(**Note**: `Cosine` does give the best accuracy so far. However, I developed my code base using `Euclidean` and only recently switched to `cosine`. There may be crashes when turning certain features off)

### 2. Bin accuracy

Instead of overall accuracy, you can run `./evaluate_bins.sh` to evaluate the bins. The script just loops through the dev bins in the `data/pan/dev_bins/sorted_by_docfreq/` directory and applies K-NN classification. You can modify the arguments inside the script if desired. 


## Normalization

The issue of normalization came up in my presentation. I have not addressed this issue yet. Currently, I am just using `sklearn's StandardScaler` to normalize before plugging everything into K-NN. Whether this is the right thing to do, I'm not sure. 

## Slides
https://drive.google.com/file/d/1BQ0vffPEvFJsoAR_lO3oZ-YvOCpo596E/view

## 