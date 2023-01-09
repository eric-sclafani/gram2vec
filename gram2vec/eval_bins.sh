#!/bin/env bash

for json in data/pan/dev_bins/*; do

    if [ -f "$json" ]; then
        python3 knn_classifer.py --eval_path "$json"
    fi
done