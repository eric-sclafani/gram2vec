#!/usr/bin/env python3

import ijson


def extract_n_authors_k_entries(raw_path, n, k):
    """This function extracts n amount of authors from MUD with k amount of documents"""
    data = ijson.parse(open(raw_path), multiple_values=True)

    seen_authors = 0

    data = {}
    posts = []
    for prefix, _, value in data:
        
        if prefix == "syms.item":
            posts.append(value)
            
        if prefix.startswith("author_id") and len(posts) >= k:
            data[value] = posts
            posts = []
            seen_authors += 1

        if seen_authors == n:
            break
        
    return data



def main():
    pass




if __name__ == "__main__":
    main()