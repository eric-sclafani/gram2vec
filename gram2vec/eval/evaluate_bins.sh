#!/usr/bin/env bash

rm eval/results/pan_dev_bin_results.csv
sorted_option="sorted_by_doc_freq"

for i in $(seq 8 $END); do 
    python3 eval/knn_classifer.py --eval_function "R@8" --eval_path "eval/eval_bins/$sorted_option/devbin$i.json"; 
done

