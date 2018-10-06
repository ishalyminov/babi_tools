#!/usr/bin/env bash

RESULT_FOLDER=$1
CONFIG_FOLDER='2018_generalization_study_configs'
mkdir .tmp

for config in `ls $CONFIG_FOLDER`; do
  echo $config
  config_filename=$(basename "$config")
  config_filename="${config_filename%.*}"
  python babi_plus.py dialog-bAbI-tasks .tmp/$config_filename --config "$CONFIG_FOLDER/$config"
  python make_parallel_dataset.py dialog-bAbI-tasks .tmp/$config_filename $RESULT_FOLDER/$config_filename --output_format csv 
done

rm -rf .tmp
