import pandas as pd
import numpy as np
import os
from typing import List, Set
from time import time
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

def timer(func):
    """This decorator shows the execution time of the function object passed"""
    # Credits: https://www.geeksforgeeks.org/timing-functions-with-decorators-python/
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

@timer
def save_preprocessed_blogs(path:str):
    """Applies various preprocessing steps to data and saves to disk"""
    blogs_df = pd.read_csv("blogs/raw/blogtext.csv")
    blogs_df.drop(columns=["gender", "age", "topic", "sign", "date"], inplace=True)
    blogs_df["tkn_count"] = blogs_df["text"].apply(lambda x: get_text_tkn_count(x))
    blogs_df.to_csv(path, index=False)

def get_text_tkn_count(doc:str) -> int:
    """Returns the token count given a doc"""
    return len(word_tokenize(doc))

def select_from_threshold(series:pd.Series, threshold:int) -> Set[int]:
    """Takes a pandas Series and selects indices that meet a given threshold"""
    return set(series[series > threshold].index.to_list())

def extract_blog_authors(df:pd.DataFrame, avg_tok_threshold:int, doc_threshold:int) -> pd.DataFrame:
    """Selects author_ids from the blogs df that meet an avg token count threshold and doc frequency threshold"""
    
    author_avg_tkns = df.groupby("author_id")["tkn_count"].mean()
    author_doc_cnts= df.groupby("author_id")["author_id"].count()

    selected_avg_tkn_authors = select_from_threshold(author_avg_tkns, avg_tok_threshold)
    selected_doc_cnt_authors = select_from_threshold(author_doc_cnts, doc_threshold)
    selected_ids = selected_avg_tkn_authors.intersection(selected_doc_cnt_authors)
    
    return df.loc[df["author_id"].isin(selected_ids)]

def remove_entries_with_zero_tkns(df:pd.DataFrame) -> pd.DataFrame:
    """Filters out dataframe entries with 0 token counts"""
    return df[df["tkn_count"] > 0]

def main():
    
    os.chdir("../")
    PREPROCESS_PATH = "blogs/preprocessed/blogs_preprocessed.csv"
    if not os.path.exists(PREPROCESS_PATH):
        save_preprocessed_blogs(PREPROCESS_PATH)
    
    blogs_preprocessed = pd.read_csv(PREPROCESS_PATH)
    
    blogs_preprocessed = blogs_preprocessed.rename(columns={"id":"author_id"})
    
    df_sample = extract_blog_authors(df = blogs_preprocessed, 
                                     avg_tok_threshold = 250, 
                                     doc_threshold = 10)

    df_sample = remove_entries_with_zero_tkns(df_sample)
    
    print(df_sample.shape)
    # print(df_sample[df_sample["tkn_count"] < 10].count())
   

    

    #df_sample.groupby('author_id', group_keys=False).apply(lambda x: x.sample(min(len(x), 2)))


        
    
    
if __name__ == "__main__":
    main()