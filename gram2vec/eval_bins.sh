#!/bin/env bash

# this script evaluates the knn classifier on development bins for PAN 2022


for json in data/pan/dev_bins/*; do

    if [ -f "$json" ]; then
        python3 knn_classifer.py --eval_path "$json"
    fi
done