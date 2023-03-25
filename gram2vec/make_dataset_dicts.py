import datasets
import os
import pandas as pd
import json

from featurizers import GrammarVectorizer
g2v = GrammarVectorizer()

queries = []
candidates = []
with open('data/hrs/queries_test.jsonl', 'r') as f:
    for line in f:
        queries.append(json.loads(line))
        
with open('data/hrs/candidates_test.jsonl', 'r') as f:
    for line in f:
        candidates.append(json.loads(line))
        
queries = pd.DataFrame(queries)
candidates = pd.DataFrame(candidates)



queries_docs = queries["fullText"]
features_q = g2v.vectorize_episode(queries_docs).tolist()
queries['features'] = features_q

candidates_docs = candidates["fullText"]
features_c = g2v.vectorize_episode(candidates_docs).tolist()
candidates['features'] = features_c

datadict = {'queries': datasets.Dataset.from_pandas(queries), 'candidates': datasets.Dataset.from_pandas(candidates)}
data = datasets.DatasetDict(datadict)

os.chdir("../../hiatus-metrics/")
data.save_to_disk('TA1_features')
