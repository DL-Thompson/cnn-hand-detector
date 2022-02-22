#!/bin/bash

for ((i=1; i<=$#; i++))
do
  echo 'Deleting training and test data in model folder: ' ${!i}
  rm ./${!i}/data/train/*.jpg
  rm ./${!i}/data/train/annotations.json
  rm ./${!i}/data/test/*.jpg
  rm ./${!i}/data/test/annotations.json
  rm ./${!i}/data/validation/*.jpg
  rm ./${!i}/data/validation/annotations.json
done

