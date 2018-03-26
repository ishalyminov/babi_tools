#!/usr/bin/env bash

RESULT_FOLDER=$1

mkdir .tmp

for config in `ls ablation_study_datasets`; do
  echo $config
  config_filename=$(basename "$config")
  config_filename="${config_filename%.*}"
  python babi_plus.py dialog-bAbI-tasks .tmp/$config_filename --config "ablation_study_datasets/$config"
  python make_parallel_dataset.py dialog-bAbI-tasks .tmp/$config_filename $RESULT_FOLDER/$config_filename --output_format csv 
done

rm -rf .tmp
