import pandas as pd
import numpy as np
from scipy.stats import zscore
from pathlib import Path
from typing import List
import pickle as pkl
import os

# project imports
from vectorizer import GrammarVectorizer


def save_to_disk():
    pass


class Verbalizer:
    
    def __init__(self, data_dir:str):
        
        self._text_df = self._load_data(data_dir)
        self.docs_df = self._make_docs_df()
        self.author_df = self._make_author_df()
        
    def _load_data(self, data_dir:str) -> pd.DataFrame:
        """Loads a directory of .jsonl files"""
        dfs = []
        for filename in Path(data_dir).glob("*.jsonl"):
            df = pd.read_json(filename, lines=True)
            dfs.append(df)
        return pd.concat(dfs)
    
    def _exclude_columns(self, df:pd.DataFrame, exclude:List[str]) -> pd.DataFrame:
        """Excludes given columns from a dataframe. Used when calculating statistics in a df with text columns"""
        return df.loc[:, ~df.columns.isin(exclude)]
    
    def _get_author_docs(self, author_id:str):
        return self.docs_df.loc[self.docs_df['authorIDs'] == author_id]
    
    def _make_docs_df(self) -> pd.DataFrame:
        """Creates a feature vector dataframe and preserves documentIDs"""
        g2v = GrammarVectorizer()
        all_documents = self._text_df["fullText"]
        author_ids = self._text_df["authorIDs"]
        doc_ids = self._text_df["documentID"]
        
        vector_df = g2v.create_vector_df(all_documents.to_list())
        vector_df.insert(0, "authorIDs", author_ids)
        vector_df.insert(1, "documentID", doc_ids)
        return vector_df

    def _make_author_df(self) -> pd.DataFrame:
        
        author_ids = set(self._text_df["authorIDs"])
        author_ids_to_avs = {}
        
        for author_id in author_ids:
            author_doc_entries = self._get_author_docs(author_id)
            author_doc_vectors = self._exclude_columns(author_doc_entries, ['documentID', 'authorIDs'])
            author_ids_to_avs[author_id] = np.mean(author_doc_vectors, axis=0)

        return pd.DataFrame(author_ids_to_avs).T
        

    
    def _get_threshold_zscores_idxs(zscores, threshold:float):
        """Gets indices for abs(zscores) that meet a threshold"""
        selected = []
        for i, zscore in enumerate(zscores):
            if abs(zscore) > threshold:
                selected.append(i)
        return selected 
    
    def verbalize_document(self, document:str) -> List[str]:
        pass
    
    def verbalize_author(self, author_id:str) -> List[str]:
        pass
    

    
        
        
    #strategy = threshold or n_best
    # add a warning statement if an author or document has no indicative features with threshold
    
    
    



def get_identifying_features(author_id:str, threshold=2.0):
    """
    Given an author, calculates their zscores for all features and selects the ones that deviate the most from the 
    mean. These features are what separate this author from the average author
    """
    zscores = zscore(author_vectors)
    author_idx = authors_df.loc[authors_df["author_id"] == author_id].index[0]    
    author_zscores = zscores.iloc[author_idx]
    selected_zscores = get_threshold_zscores_idxs(author_zscores, threshold)
    return author_zscores.iloc[selected_zscores]

def features_to_show(author_id:str) -> List[str]:
    """Given an author id, returns n amount of this author's most identifying features"""
    features = get_identifying_features(author_id).index.to_list()
    if len(features) > 10:
        return features[:10]
    return features



def main():

    
    datapaths = {
        "hrs": "data/hrs_release_03-20-23/raw",
        "pan":"data/pan22/preprocessed"
    }
    
    verb = Verbalizer(datapaths["pan"])
    print(verb._make_author_df().info())

    
    
if __name__ == "__main__":
    main()