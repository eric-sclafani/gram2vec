import numpy as np
import pandas as pd
import os
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


def get_text_tkn_count(text:str) -> int:
    print(f"Processing text: {text[0:30]}")
    return len(word_tokenize(text))

@timer
def main():
    
    os.chdir("../")
    
    blogs_df = pd.read_csv("blogs/raw/blogtext.csv")
    blogs_df["tkn_count"] = blogs_df["text"].apply(lambda x: get_text_tkn_count(x))
    blogs_df.to_csv("blogs/preprocessed/blogtext.csv")
    
    
if __name__ == "__main__":
    main()