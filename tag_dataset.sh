#!/usr/bin/env bash

FOLDER_NAME=$1
CONFIG_NAME=$2
for file in `ls $FOLDER_NAME/*.txt`
do
  filename=$(basename "$file")
  filename="${filename%.*}"
  python make_deep_disfluency_tagging.py $file $FOLDER_NAME/${filename}.tagged.json $CONFIG_NAME dialog-bAbI-tasks
done
