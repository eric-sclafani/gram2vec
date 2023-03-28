import pandas as pd
import numpy as np
import os
import re
from typing import List
from time import time
from nltk.tokenize import word_tokenize

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
def save_preprocessed_blogs(path):
    """Applies various preprocessing steps to data and saves to disk"""
    blogs_df = pd.read_csv("blogs/raw/blogtext.csv")
    blogs_df.drop(columns=["gender", "age", "topic", "sign", "date"], inplace=True)
    blogs_df["tkn_count"] = blogs_df["text"].apply(lambda x: get_text_tkn_count(x))
    blogs_df.to_csv(path, index=False)

def get_text_tkn_count(doc:str) -> int:
    """Returns the token count given a doc"""
    return len(word_tokenize(doc))

def extract_authors(blogs_df:pd.DataFrame, 
                    desired_authors:int, 
                    doc_count:range, 
                    tok_threshold:int, 
                    verbose=False) -> dict:
    pass

    
def main():
    
    os.chdir("../")
    PREPROCESS_PATH = "blogs/preprocessed/blogs_preprocessed.csv"
    if not os.path.exists(PREPROCESS_PATH):
        save_preprocessed_blogs(PREPROCESS_PATH)
    
    blogs_preprocessed = pd.read_csv(PREPROCESS_PATH, )
    
    
    
    
if __name__ == "__main__":
    main()