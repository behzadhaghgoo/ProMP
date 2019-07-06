#!/usr/bin/env bash

# Usage:
# ./baby_eval.sh eval_script.py config_index experiment_dir

for f in find $3 -type f -iname "itr_*.pkl" | sort -V
do
  printf "Processing $f"
  python $1 $2 --pkl $f
done
