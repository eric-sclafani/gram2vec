import pandas as pd
import numpy as np
from scipy.stats import zscore
from pathlib import Path
from typing import List
import pickle as pkl

# project imports
from vectorizer import GrammarVectorizer

class Verbalizer:
    
    def __init__(self, data_dir:str):
        
        self.text_df = self._load_data(data_dir)
        self.docs_df = self._make_docs_df()
        self.author_df = self._make_author_df()
        
    def _load_data(self, data_dir:str) -> pd.DataFrame:
        """Loads a directory of .jsonl files"""
        dfs = []
        for filename in Path(data_dir).glob("*.jsonl"):
            df = pd.read_json(filename, lines=True)
            dfs.append(df)
        return pd.concat(dfs)
    
    def _exclude_columns(self, df:pd.DataFrame, cols:List[str]) -> pd.DataFrame:
        """Excludes given columns from a dataframe. Used when doing numerical operations"""
        return df.loc[:, ~df.columns.isin(cols)]
    
    def get_author_docs(self, author_id:str) -> pd.DataFrame:
        """Retrieves an authors documents from self.docs_df"""
        return self.docs_df.loc[self.docs_df['authorIDs'] == author_id]
    
    def _make_docs_df(self) -> pd.DataFrame:
        """Creates a document-level feature vector dataframe and preserves documentIDs"""
        g2v = GrammarVectorizer()
        all_documents = self.text_df["fullText"]
        author_ids = self.text_df["authorIDs"]
        doc_ids = self.text_df["documentID"]
        
        vector_df = g2v.create_vector_df(all_documents.to_list())
        vector_df.insert(0, "authorIDs", author_ids)
        vector_df.insert(1, "documentID", doc_ids)
        return vector_df

    def _make_author_df(self) -> pd.DataFrame:
        """Creates an author level dataframe. Each author entry is the average of that author's document vectors"""
        author_ids = set(self.text_df["authorIDs"])
        author_ids_to_avs = {}
        
        for author_id in author_ids:
            author_doc_entries = self.get_author_docs(author_id)
            author_doc_vectors = self._exclude_columns(author_doc_entries, cols=['documentID', 'authorIDs'])
            author_ids_to_avs[author_id] = np.mean(author_doc_vectors, axis=0)

        df = pd.DataFrame(author_ids_to_avs).T      
        df.reset_index(inplace=True)
        df = df.rename(columns = {'index':'authorIDs'})
        return df
    
    def _get_threshold_zscores_idxs(self, zscores:np.ndarray, threshold=2.0) -> List[int]:
        """Gets indices for zscores that are +- threshold from the mean"""
        selected = []
        for i, zscore in enumerate(zscores):
            if abs(zscore) > threshold:
                selected.append(i)
        return selected 
    
    def _get_record_from_id(self, df:pd.DataFrame, column:str, id:str) -> int:
        """Retrieves the index of a record from a given df according to a column value"""
        try:
            return df.loc[df[column] == id].index[0] 
        except (KeyError, IndexError):
            raise Exception(f"id '{id}' not found in dataframe. Check if provided id or to_verbalize flag is correct ")
    
    def _get_zscores(self, df:pd.DataFrame) -> np.ndarray:
        """Calculates zscores for a given df"""
        return zscore(self._exclude_columns(df, cols=["documentID", "authorIDs"]))
    
    def _get_identifying_features(self, id:str, df:pd.DataFrame, to_verbalize:str) -> pd.Series:
        """
        Given a unique id (document or author), calculate the feature zscores for that id and keep
        the ones that meet a given threshold (> 2 by default)
        """
        all_zscores = self._get_zscores(df)
        idx = self._get_record_from_id(df, to_verbalize, id) 
        extracted_zscores_from_idx = all_zscores.iloc[idx]
        chosen_zscores = self._get_threshold_zscores_idxs(extracted_zscores_from_idx)
        return extracted_zscores_from_idx.iloc[chosen_zscores]

    def _template(self, doc_or_author:str, feat_name:str, direction:str) -> str:
        """Template for the zscore verbalizer"""
        return f"This {doc_or_author} uses {direction} {feat_name} than the average"
    
    def _verbalize_zscores(self, zscores:pd.DataFrame, to_verbalize:str) -> List[str]:
        """Creates a list of verbalized zscores"""
        converted = []
        doc_or_author = "author" if to_verbalize == "authorIDs" else "document"
        for feat_name, zscore in zscores.items():
            direction = "less" if zscore < 0 else "more"
            converted.append(self._template(doc_or_author, feat_name, direction))
        return converted
    
    def verbalize(self, id:str, to_verbalize="authorIDs") -> pd.DataFrame:
        """
        
        """
        if to_verbalize not in ["authorIDs", "documentID"]:
            raise ValueError("Only accepted values for to_verbalize: {'authorIDs', 'documentID'} ")
        
        df = self.author_df if to_verbalize == "authorIDs" else self.docs_df
        selected_id_zscores = self._get_identifying_features(id, df, to_verbalize)
        verbs = self._verbalize_zscores(selected_id_zscores, to_verbalize)
        
        verbalized_df = pd.DataFrame({
            "feature_name":selected_id_zscores.index,
            "zscore": selected_id_zscores.values,
            "verbalized":verbs
        })
        return verbalized_df

        
        
        
        
        

        
    #strategy = threshold or n_best
    # add a warning statement if an author or document has no indicative features with threshold
    
    
    


def main():

    
    datapaths = {
        "hrs": "data/hrs_release_03-20-23/raw",
        "pan":"data/pan22/preprocessed"
    }
    
    verb = Verbalizer(datapaths["pan"])
    v = verb.verbalize("en_110", to_verbalize="authorIDs")
    print(v["verbalized"])


    
    
if __name__ == "__main__":
    main()