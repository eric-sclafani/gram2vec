

import numpy as np
import json

AuthorToDocsMapping = tuple[str,list[str]]

def save_json(data:dict, path, mode="w"):
    """Saves a dict as a JSON"""
    with open(path, mode) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)

def sort_by_token_avg(author_to_avg_tokens:dict[str, float]) -> dict[str, int]:
    """Sorts a dictionary of author id to average token count mappings"""
    author_avg_token_pairs = author_to_avg_tokens.items()
    sorted_pairs:tuple[str,int] = sorted(author_avg_token_pairs, key=lambda x: x[1])
    return dict(sorted_pairs)

def sort_by_doc_freq(train:dict) -> list[tuple]:
    """Sorts a dictionary by the amount of documents of each author"""
    author_docs_pairs:list[AuthorToDocsMapping] = train.items()
    return sorted(author_docs_pairs, key=lambda pairs: len(pairs[1]))


def sort_authors_by_avg_tokens(dev:dict, train:dict) -> list[tuple]:
    """Sorts the authors in DEV by average token count in TRAIN. Used for making dev bins"""
    author_to_avg_tokens = {}
    for author_id, documents in train.items():
        token_count = lambda x: len(x.split())  
        author_to_avg_tokens[author_id] = np.mean([token_count(doc) for doc in documents], axis=0)
    
    train_sorted = sort_by_token_avg(author_to_avg_tokens) # (low -> high)
    train_sorted_authors = train_sorted.keys()
    
    # https://stackoverflow.com/questions/21773866/how-to-sort-a-dictionary-based-on-a-list-in-python
    # sorting DEV by sorted list of authors in TRAIN
    index_map = {author_id: i for i, author_id in enumerate(train_sorted_authors)}
    dev_sorted = sorted(dev.items(), key=lambda pair: index_map[pair[0]])
    return dev_sorted

def sort_authors_by_doc_freq(dev:dict, train:dict):
    """Sorts the authors in DEV by document frequency in TRAIN. Used for making dev bins."""
    train_sorted = sort_by_doc_freq(train)
    
    # sorts dev by document occurence in train
    dev_sorted = sorted(dev.items(), key=lambda x:len(train[x[0]]))
    
    assert [i[0] for i in train_sorted] == [i[0] for i in dev_sorted], "Sorting incorrect"
    for devitems, trainitems in zip(dev_sorted, train_sorted):
        assert trainitems[0] == devitems[0]
    
    return dev_sorted
        
def save_dev_bins(sorted_data:list[tuple]):
    """Saves dev bins to directory. #! MAKE FUNCTIONN ABLE TO CREATE TEST BINS"""
    i = 0
    for bin_num in range(1,9):
        partition = dict(sorted_data[i:i+7])
        save_json(data=partition, path=f"pan/dev_bins/sorted_by_docfreq/bin_{bin_num}_dev.json")
        i += 7
        
        
        
def main():
    
    # print("Saving development bins...")
    # save_dev_bins(sort_authors_by_doc_freq(dev, train))
    # print("Done!")
    pass
    



if __name__ == "__main__":
    pass