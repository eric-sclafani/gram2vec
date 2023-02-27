#!/usr/bin/env bash

for i in $(seq 8 $END); do 
    python3 eval/knn_classifer.py --eval_function "majority_vote" --eval_dir "eval/eval_bins/devbin$i"; 
done

