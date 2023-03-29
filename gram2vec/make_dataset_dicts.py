import datasets
import os
import pandas as pd
import json
from contextlib import contextmanager

from featurizers import GrammarVectorizer
g2v = GrammarVectorizer()

CANDIDATES_PATH = "hiatus-metrics/eval_samples_HRSBackground/mode_multi-genre/sample2_randomSeed1266677/TA1/multi-genre/multi-genre_TA1_candidates.jsonl"
QUERIES_PATH = "hiatus-metrics/eval_samples_HRSBackground/mode_multi-genre/sample2_randomSeed1266677/TA1/multi-genre/multi-genre_TA1_queries.jsonl"



@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)

def load_hrs(path):
    """Loads in the .jsonl Q or C documents"""
    assert path.endswith(".jsonl")
    with open(path) as fin:
        return [json.loads(line) for line in fin]


with cwd("../../../hiatus/"):
    queries = pd.DataFrame(load_hrs(QUERIES_PATH))
    candidates = pd.DataFrame(load_hrs(CANDIDATES_PATH))

queries_docs = queries["fullText"]
features_q = g2v.vectorize_episode(queries_docs).tolist()
queries['features'] = features_q

candidates_docs = candidates["fullText"]
features_c = g2v.vectorize_episode(candidates_docs).tolist()
candidates['features'] = features_c

datadict = {'queries': datasets.Dataset.from_pandas(queries), 'candidates': datasets.Dataset.from_pandas(candidates)}
data = datasets.DatasetDict(datadict)

with cwd("../../../hiatus/hiatus-metrics/"):
    data.save_to_disk('TA1_features')
