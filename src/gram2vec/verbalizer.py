import pandas as pd
import numpy as np
from scipy.stats import zscore
from typing import List

# code is messy and should be refactored
class Verbalizer:
    """
    This class encapsulates the zscore verbalization functionality of gram2vec
    """
    def __init__(self, docs_df:pd.DataFrame, zscore_threshold=2.0):
        
        self.threshold = zscore_threshold
        self.docs_df = docs_df
        self.author_df = self._make_author_df(self.docs_df)

    def _exclude_columns(self, df:pd.DataFrame, cols:List[str]) -> pd.DataFrame:
        """Excludes given columns from a dataframe. Used when doing numerical operations"""
        return df.loc[:, ~df.columns.isin(cols)]
    
    def _get_author_docs(self, author_id:str) -> pd.DataFrame:
        """Retrieves an authors documents from self.docs_df"""
        return self.docs_df.loc[self.docs_df["authorIDs"] == author_id]

    def _make_author_df(self, docs_df:pd.DataFrame) -> pd.DataFrame:
        """Creates an author level dataframe. Each author entry is the average of that author's document vectors"""
        author_ids = docs_df["authorIDs"].unique()
        author_ids_to_avs = {}
        
        for author_id in author_ids:
            author_doc_entries = self._get_author_docs(author_id)
            author_doc_vectors = self._exclude_columns(author_doc_entries, cols=['documentID', 'authorIDs'])
            author_ids_to_avs[author_id] = author_doc_vectors.mean(axis=0)

        df = pd.DataFrame(author_ids_to_avs).T      
        return df
    
    def _get_threshold_zscores_idxs(self, zscores:np.ndarray) -> List[int]:
        """Gets indices for zscores that are +- threshold from the mean"""
        selected = []
        for i, zscore in enumerate(zscores):
            if abs(zscore) >= self.threshold:
                selected.append(i)
        return selected 
    
    def _get_record_from_id(self, df:pd.DataFrame, id:str) -> int:
        """Retrieves the index of a record from a given df according to a column value"""
        
        if id in df.index:
            return df.index.to_list().index(id) # dumb solution but it works
        else:
            raise ValueError(f"id '{id}' not found in dataframe")

    def _get_zscores(self, df:pd.DataFrame) -> np.ndarray:
        """Calculates zscores for a given df"""
        return zscore(self._exclude_columns(df, cols=["authorIDs"]))
    
    def _get_identifying_features(self, id:str, df:pd.DataFrame) -> pd.Series:
        """Calculates the feature zscores for an id and keeps the ones that meet the given threshold"""
        all_zscores = self._get_zscores(df)
        idx = self._get_record_from_id(df, id) 
        extracted_zscores_from_idx = all_zscores.iloc[idx]
        chosen_zscores = self._get_threshold_zscores_idxs(extracted_zscores_from_idx)
        return extracted_zscores_from_idx.iloc[chosen_zscores]

    def _template(self, doc_or_author:str, feat_name:str, direction:str) -> str:
        """Template for the zscore verbalizer"""
        return f"This {doc_or_author} uses {direction} {feat_name} than the average"
    
    def _verbalize_zscores(self, zscores:pd.Series, to_verbalize:str) -> List[str]:
        """Creates a list of verbalized zscores"""
        converted = []
        for feat_name, zscore in zscores.items():
            direction = "less" if zscore < 0 else "more"
            converted.append(self._template(to_verbalize, feat_name, direction))
        return converted    
    
    def verbalize_document(self, document_vector:np.ndarray) -> pd.DataFrame:
        """
        Given a unqiue document id, retrieves that document's most distinguishing features. 
        
        Args:
        ----
            - doc_id (str) - document id to verbalize
            
        Returns
        ------
            - pd.DataFrame - dataframe of which features meet the threshold, their zscores, and their verbalized forms
        """
        
        fit_matrix = self.docs_df.select_dtypes(include=np.number).values.tolist()
        fit_matrix.append(document_vector.tolist())
        fit_matrix = np.array(fit_matrix)
        
        feature_names = self.docs_df.select_dtypes(include=np.number).columns.to_list()
        unseen_doc_zscores = pd.Series(data = zscore(fit_matrix)[-1], index=feature_names)
        chosen_zscores = self._get_threshold_zscores_idxs(unseen_doc_zscores)
        
        selected_zscores = unseen_doc_zscores.iloc[chosen_zscores]
        templated_strings = self._verbalize_zscores(selected_zscores, "document")
        
        verbalized_df = pd.DataFrame({
            "feature_name":[feature_names[i] for i in chosen_zscores],
            "zscores" : selected_zscores.values,
            "verbalized" : templated_strings
        })
       
        return verbalized_df
    
    def verbalize_author(self, author_id:str) -> pd.DataFrame:
        """
        Given a unqiue author id, retrieves that author's most distinguishing features. 
        
        Args:
        ----
            - author_id (str) - author id to verbalize
            
        Returns
        ------
            - pd.DataFrame - dataframe of which features meet the threshold, their zscores, and their verbalized forms
        """
        selected_id_zscores = self._get_identifying_features(author_id, self.author_df)
        templated_strings = self._verbalize_zscores(selected_id_zscores, "author")
        
        verbalized_df = pd.DataFrame({
            "feature_name":selected_id_zscores.index,
            "zscore": selected_id_zscores.values,
            "verbalized":templated_strings
        })
        return verbalized_df