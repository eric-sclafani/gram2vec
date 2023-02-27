#!/usr/bin/env bash

rm eval/results/pan_dev_bin_results.csv

for i in $(seq 8 $END); do 
    python3 eval/knn_classifer.py --eval_function "majority_vote" --eval_dir "eval/eval_bins/sorted_by_avg_tokens/devbin$i"; 
done

