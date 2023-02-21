#!/usr/bin/env bash

# this script evaluates the knn classifier on development bins for PAN 2022

# clear the current eight results
rm results/pan_dev_bin_results.csv

for json in data/pan/dev_bins/sorted_by_docfreq/*; do

    if [ -f "$json" ]; then
        python3 knn_classifer.py --eval_path "$json"
    fi
done